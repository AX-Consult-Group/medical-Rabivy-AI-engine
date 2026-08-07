# llm_client.py
# -------------------------------------------------------------------
# The ONE place the agent talks to a language model, behind a single
# interface, so the rest of the agent never knows (or cares) which
# model is on the other end:
#
#   AnthropicLLM - real Claude calls via the Anthropic Messages API.
#                  Used when ANTHROPIC_API_KEY is set.
#   MockLLM      - a deterministic, zero-cost stand-in that emulates
#                  the tool-use protocol with keyword rules. Used when
#                  no key is set (or AGENT_LLM=mock is forced), so the
#                  whole agent stack - tool routing, the loop, the eval
#                  harness - can run and be tested without spending a
#                  single token. Its final answers are labeled [MOCK],
#                  never passed off as real model output.
#
# Both return Anthropic-shaped responses:
#   {"stop_reason": "tool_use" | "end_turn",
#    "content": [{"type": "text", ...} | {"type": "tool_use", ...}]}
# so agent.py contains exactly one loop, not one per backend.
# -------------------------------------------------------------------

import json
import os
import random
import re
import time

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-5")
# 2026-08-04: retry/fallback settings. Added because a single timeout on
# the hallucination auditor's LLM call (agent.py's _verify()) was landing
# as verdict="audit_error" - which chat_ui.py quarantines exactly like a
# real hallucination. That made "red" noisy: a dropped connection looked
# identical to a genuine failed audit. Fix belongs here, not in agent.py,
# since retries are a concern of talking to the API, not of what the
# audit logic does with the answer.
FALLBACK_MODEL = os.environ.get("AGENT_FALLBACK_MODEL", "claude-haiku-4-5")
PRIMARY_RETRIES = 2       # attempts on the primary model before falling back
RETRY_BASE_DELAY = 1.0    # seconds; doubles each retry, plus jitter


# =====================================================================
# REAL BACKEND
# =====================================================================

class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model=DEFAULT_MODEL, fallback_model=FALLBACK_MODEL):
        import anthropic  # deferred so mock mode needs no SDK installed
        self.model = model
        self.fallback_model = fallback_model
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def complete(self, system, messages, tools=None, max_tokens=2000,
                 temperature=None):
        kwargs = dict(system=system, messages=messages, max_tokens=max_tokens)
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        return self._call_with_retries(kwargs)

    def _call_with_retries(self, kwargs):
        """Tries the primary model up to PRIMARY_RETRIES times (short
        backoff + jitter between attempts, for transient timeouts/rate
        limits), then falls back to a second model once before finally
        raising. Only a failure that survives all of that reaches the
        caller - so agent.py's _verify() only ever sees verdict="audit_error"
        for a genuinely broken call, not a single dropped connection."""
        last_err = None
        models_to_try = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models_to_try.append(self.fallback_model)

        for model in models_to_try:
            attempts = PRIMARY_RETRIES if model == self.model else 1
            for attempt in range(attempts):
                try:
                    resp = self._client.messages.create(model=model, **kwargs)
                    return {
                        "stop_reason": resp.stop_reason,
                        "content": [b.model_dump() for b in resp.content],
                    }
                except Exception as e:
                    last_err = e
                    on_last_attempt_for_this_model = attempt == attempts - 1
                    if not on_last_attempt_for_this_model:
                        delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                        time.sleep(delay)
        raise last_err


# =====================================================================
# MOCK BACKEND
# ---------------------------------------------------------------------
# Emulates the agent's tool-use protocol deterministically. The policy:
# derive a small PLAN (list of tool calls) from the question, then on
# each turn issue the next unexecuted step; when the plan is done,
# assemble a template answer from the tool results seen so far.
# Question parsing reuses ask_a_question.py's battle-tested detectors
# rather than re-inventing (and re-bugging) them here.
# =====================================================================

class MockLLM:
    name = "mock"

    def complete(self, system, messages, tools=None, max_tokens=2000,
                 temperature=None):
        # A call with no tools is a plain-completion call (the agent
        # uses those for answer verification) -> return a canned verdict.
        if not tools:
            return self._plain_completion()

        question = self._current_question(messages)
        plan = self._make_plan(question)
        done = self._executed_tool_calls(messages)

        if done < len(plan):
            tool_name, tool_input = plan[done]
            return {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text",
                     "text": f"[MOCK plan step {done + 1}/{len(plan)}] calling {tool_name}"},
                    {"type": "tool_use", "id": f"mock_call_{done + 1}",
                     "name": tool_name, "input": tool_input},
                ],
            }

        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text",
                         "text": self._final_answer(question, messages)}],
        }

    # ---- verification / plain calls --------------------------------
    @staticmethod
    def _plain_completion():
        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": json.dumps({
                "verdict": "not_checked",
                "issues": ["Mock mode cannot verify claims - run with a real "
                           "LLM for genuine verification."],
            })}],
        }

    # ---- conversation-state helpers ---------------------------------
    @staticmethod
    def _current_question(messages):
        """Last user message that is plain text (not a tool_result batch)."""
        for m in reversed(messages):
            if m["role"] != "user":
                continue
            c = m["content"]
            if isinstance(c, str):
                return c
            texts = [b.get("text", "") for b in c if b.get("type") == "text"]
            if texts:
                return " ".join(texts)
        return ""

    @staticmethod
    def _executed_tool_calls(messages):
        """How many tool calls already ran SINCE the last plain-text user
        message (i.e. within the current question's turn)."""
        n = 0
        for m in reversed(messages):
            c = m["content"]
            if m["role"] == "user" and isinstance(c, str):
                break  # reached the current question - stop counting
            if isinstance(c, list):
                if m["role"] == "user" and any(b.get("type") == "tool_result" for b in c):
                    n += sum(1 for b in c if b.get("type") == "tool_result")
                elif m["role"] == "user" and any(b.get("type") == "text" for b in c):
                    break
        return n

    # ---- planning ----------------------------------------------------
    def _make_plan(self, question):
        """Turn a question into an ordered list of (tool, input) steps.
        Reuses ask_a_question.py's detectors for filter parsing."""
        import ask_a_question as aq  # heavy import, but agent loads engines anyway

        ql = question.lower()
        plan = []

        npi_m = re.search(r"\b(\d{10})\b", question)

        multi_part = bool(re.search(r"what (?:should i|to) say|messaging|talking points", ql))

        # NPI-specific questions -> exact lookup first.
        if npi_m:
            plan.append(("lookup_hcp", {"npi": npi_m.group(1)}))
            if re.search(r"\b(compare|typical|benchmark|average|versus|vs\.?)\b", ql):
                plan.append(("search_documents",
                             {"query": "typical specialty benchmark profile", "top_k": 5}))
            return plan

        # Counting questions.
        if (any(w in ql for w in ("how many", "number of", "count")) or "active" in ql) \
                and any(w in ql for w in ("writer", "prescriber", "hcp")):
            st = aq._find_state(ql)
            if st:
                return [("count_active_writers", {"state": st})]

        # States aggregate.
        if any(p in ql for p in ("which states", "what states", "top states",
                                 "states with the most", "states have the most")):
            return [("states_summary", {})]

        # Structured filter/ranking signals (same signal set as the
        # baseline router in ask_a_question.py).
        targeted = aq._detect_targeted(ql)
        competitor = aq._detect_competitor(ql)
        # Reconciled 2026-07-22: _detect_min_switching was renamed to
        # _detect_switching AND its return shape changed to
        # (level_word_or_None, explicit_range_or_None) - same new shape
        # _detect_pa_burden ALSO silently picked up under its unchanged
        # name (found while fixing this - it doesn't crash, so CI never
        # caught it: min_pa/max_pa below used to silently become a
        # string like "high" and a tuple like (0.7, None) instead of
        # numbers, which would have produced wrong/empty filter results
        # or a confusing crash deep inside pandas, not here).
        switching_level, switching_range = aq._detect_switching(ql)
        formulary = aq._detect_formulary(ql)
        sample = aq._detect_sample_request(ql)
        pa_level, pa_range = aq._detect_pa_burden(ql)
        has_rank = any(p.search(ql) for p in aq.RANK_PATTERNS)
        hcp_ctx = aq._has_hcp_context(ql)
        # Reconciled 2026-07-23: structured_signal was missing 3 real
        # triggers entirely - tier, region, and generic numeric filters
        # (e.g. "days since contact"). A question that ONLY mentions one
        # of these (no targeted/switching/competitor/etc alongside it)
        # fell all the way through to search_documents with nothing to
        # find, silently. Caught by CI once the matching edge-case tests
        # were added to test_the_agent.py - these 3 questions never had
        # any other signal to lean on, so the gap had nothing to hide
        # behind. tier_signal checks BOTH a real tier and an invalid
        # region attempt (e.g. "Southeast") - the tool itself validates
        # and rejects invalid values correctly (see filter_hcps()), so
        # passing an invalid one through is the right behaviour, not a
        # bug to special-case here.
        tier_signal = aq._detect_tier(ql)
        region_signal = aq._detect_region(ql) or aq._detect_invalid_region_attempt(ql)
        numeric_filters = aq._detect_generic_numeric_filters(ql)
        structured_signal = (targeted is not None
                             or switching_level is not None or switching_range is not None
                             or (competitor and hcp_ctx) or (formulary and hcp_ctx)
                             or (sample is not None and hcp_ctx)
                             or ((pa_level is not None or pa_range is not None) and hcp_ctx)
                             or (has_rank and hcp_ctx)
                             or aq._is_forward_targeting_question(ql)
                             or (multi_part and "target" in ql)
                             or tier_signal is not None
                             or region_signal is not None
                             or bool(numeric_filters))
        if structured_signal:
            import query_spreadsheet as qs  # for HIGH_CUTOFF/LOW_CUTOFF - level word -> real number
            table_input = {}
            st = aq._find_state(ql)
            spec = aq._detect_specialty(ql)
            tier = tier_signal
            if st: table_input["state"] = st
            elif region_signal: table_input["region"] = region_signal
            if spec: table_input["specialty"] = spec
            if tier: table_input["tier"] = tier
            if targeted is not None: table_input["targeted"] = bool(targeted)
            if competitor: table_input["dominant_competitor"] = competitor
            if formulary: table_input["formulary_tier"] = formulary
            if sample is not None: table_input["recent_sample_request"] = bool(sample)

            # switching_score and pa_burden both come back as EITHER a
            # level word ("high"/"low"/"moderate") or an explicit
            # numeric range now - handle both identically, through
            # extra_filters, instead of the old separate min_switching/
            # min_pa_burden/max_pa_burden scalar fields (which only ever
            # supported a MIN threshold - "low switching score" had no
            # way to be expressed correctly before this fix either).
            extra_filters = []

            def _add_level_filter(column, level, explicit_range):
                if explicit_range is not None:
                    lo, hi = explicit_range
                    extra_filters.append({"column": column, "min": lo, "max": hi})
                elif level == "high":
                    extra_filters.append({"column": column, "min": qs.HIGH_CUTOFF})
                elif level == "low":
                    extra_filters.append({"column": column, "max": qs.LOW_CUTOFF})
                elif level == "moderate":
                    extra_filters.append({"column": column, "min": qs.LOW_CUTOFF, "max": qs.HIGH_CUTOFF})

            _add_level_filter("switching_score", switching_level, switching_range)
            _add_level_filter("pa_burden", pa_level, pa_range)
            # generic numeric filters (e.g. days_since_contact) - a
            # column: (min, max) dict, same shape _add_level_filter
            # produces per-item, just merged in directly here
            for col, (lo, hi) in numeric_filters.items():
                extra_filters.append({"column": col, "min": lo, "max": hi})
            if extra_filters:
                table_input["extra_filters"] = extra_filters

            # Reconciled 2026-07-22: _detect_sort_by no longer exists -
            # replaced by _decide_sort(ql, levels, extra_filters), which
            # needs dict-shaped context (not just the question text) to
            # correctly pick between multiple numeric columns mentioned
            # in the same question. Build that from the same two signals
            # above (a small {column: level_word} dict - a different
            # shape from the extra_filters LIST above, which matches
            # this tool's own schema instead).
            levels_dict = {}
            if switching_level: levels_dict["switching_score"] = switching_level
            if pa_level: levels_dict["pa_burden"] = pa_level
            extra_filters_dict = {f["column"]: (f.get("min"), f.get("max")) for f in extra_filters}
            sort_by, ascending, sort_reason = aq._decide_sort(ql, levels_dict, extra_filters_dict)
            table_input["sort_by"] = sort_by
            table_input["ascending"] = ascending

            top = aq._resolve_top(ql)
            if top is not None: table_input["top"] = min(top, 10)
            plan.append(("query_hcp_table", table_input))
            # Multi-part question: also fetch the messaging guidance.
            if multi_part:
                plan.append(("search_documents",
                             {"query": "recommended messaging talking points by segment",
                              "top_k": 5}))
            return plan

        # Default: narrative document search. top_k matches
        # agent_tools.py's own default (5, changed from 4 on 2026-08-04)
        # so mock mode's plan mirrors what a real call would do when it
        # doesn't override top_k itself - previously these were three
        # different hardcoded numbers (2/3/4) with no shared reasoning.
        return [("search_documents", {"query": question, "top_k": 5})]

    # ---- answer assembly ----------------------------------------------
    def _final_answer(self, question, messages):
        """Template answer stitched from this turn's tool results. Honest
        about being a mock - the real prose quality comes from Claude."""
        results = []
        for m in reversed(messages):
            if m["role"] == "user" and isinstance(m["content"], str):
                break
            if m["role"] == "user" and isinstance(m["content"], list):
                for b in m["content"]:
                    if b.get("type") == "tool_result":
                        results.append(b)
        results.reverse()

        lines = [f"[MOCK MODE ANSWER - deterministic draft, no LLM used]",
                 f"Question: {question}", ""]
        for r in results:
            content = r.get("content", "")
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content)
            snippet = str(content)[:600]
            lines.append(f"- Evidence: {snippet}")
        lines.append("")
        lines.append("(Run with ANTHROPIC_API_KEY set for a real synthesized, "
                     "verified answer.)")
        return "\n".join(lines)


# =====================================================================
# FACTORY
# =====================================================================

def get_llm():
    """Real Claude when a key is present, mock otherwise - stated loudly
    either way, never a silent substitution (same principle as
    embedding_backend.py)."""
    forced = os.environ.get("AGENT_LLM", "").lower()
    if forced == "mock":
        print("LLM backend: MOCK (forced via AGENT_LLM=mock)")
        return MockLLM()
    if os.environ.get("ANTHROPIC_API_KEY"):
        print(f"LLM backend: Anthropic ({DEFAULT_MODEL})")
        return AnthropicLLM()
    print("LLM backend: MOCK (no ANTHROPIC_API_KEY found)")
    return MockLLM()