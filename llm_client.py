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
import re

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-5")


# =====================================================================
# REAL BACKEND
# =====================================================================

class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model=DEFAULT_MODEL):
        import anthropic  # deferred so mock mode needs no SDK installed
        self.model = model
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def complete(self, system, messages, tools=None, max_tokens=2000,
                 temperature=None):
        kwargs = dict(model=self.model, system=system, messages=messages,
                      max_tokens=max_tokens)
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._client.messages.create(**kwargs)
        return {
            "stop_reason": resp.stop_reason,
            "content": [b.model_dump() for b in resp.content],
        }


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
                             {"query": "typical specialty benchmark profile", "top_k": 2}))
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
        min_switching = aq._detect_min_switching(ql)
        formulary = aq._detect_formulary(ql)
        sample = aq._detect_sample_request(ql)
        min_pa, max_pa = aq._detect_pa_burden(ql)
        has_rank = any(p.search(ql) for p in aq.RANK_PATTERNS)
        hcp_ctx = aq._has_hcp_context(ql)
        structured_signal = (targeted is not None or min_switching is not None
                             or (competitor and hcp_ctx) or (formulary and hcp_ctx)
                             or (sample is not None and hcp_ctx)
                             or ((min_pa is not None or max_pa is not None) and hcp_ctx)
                             or (has_rank and hcp_ctx)
                             or aq._is_forward_targeting_question(ql)
                             or (multi_part and "target" in ql))
        if structured_signal:
            table_input = {}
            st = aq._find_state(ql)
            spec = aq._detect_specialty(ql)
            tier = aq._detect_tier(ql)
            if st: table_input["state"] = st
            if spec: table_input["specialty"] = spec
            if tier: table_input["tier"] = tier
            if targeted is not None: table_input["targeted"] = bool(targeted)
            if competitor: table_input["dominant_competitor"] = competitor
            if min_switching is not None: table_input["min_switching"] = min_switching
            if formulary: table_input["formulary_tier"] = formulary
            if sample is not None: table_input["recent_sample_request"] = bool(sample)
            if min_pa is not None: table_input["min_pa_burden"] = min_pa
            if max_pa is not None: table_input["max_pa_burden"] = max_pa
            table_input["sort_by"] = aq._detect_sort_by(ql)
            top = aq._resolve_top(ql)
            if top is not None: table_input["top"] = min(top, 10)
            plan.append(("query_hcp_table", table_input))
            # Multi-part question: also fetch the messaging guidance.
            if multi_part:
                plan.append(("search_documents",
                             {"query": "recommended messaging talking points by segment",
                              "top_k": 3}))
            return plan

        # Default: narrative document search.
        return [("search_documents", {"query": question, "top_k": 4})]

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
