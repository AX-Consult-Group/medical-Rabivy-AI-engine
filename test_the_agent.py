# test_the_agent.py
# =====================================================================
# AGENT-LEVEL EVALUATION - the SAME 30-question set as test_the_system.py
# (same groups, same order, same questions), run through the FULL AGENT
# (LLM + tools), not the plain regex router. THREE LAYERS, same design:
#
#   LAYER 1 - ROUTING     Did the agent call the correct tool(s)? Fixed
#                         forever - doesn't depend on live data.
#
#   LAYER 2 - RULES       Did the agent call that tool with the correct
#                         ARGUMENTS (state/tier/specialty/targeted for
#                         query_hcp_table; the right chunk(s) retrieved
#                         for search_documents)? Also fixed forever.
#
#   LAYER 3 - ANSWER      Does the output match the CURRENT correct
#                         answer? Computed LIVE via ground_truth.py,
#                         using the AGENT's own default row limit (10 -
#                         NOT the router's 20, see AGENT_DEFAULT_TOP).
#                         IMPORTANT: a question whose correct answer is
#                         "ask for clarification" or "say this is
#                         invalid" is JUST AS determinable as a numeric
#                         lookup - so these are checked at Layer 3 too,
#                         never skipped just because the "value" is a
#                         fixed phrase rather than a number.
#
# GROUPING: edge cases (not-found, invalid input, AND the two
# unresolved-referent clarification questions) are grouped together at
# the end, same reorganisation as test_the_system.py.
#
# MOCK mode (no ANTHROPIC_API_KEY): MockLLM's keyword planner has no
# real referent-resolution logic, so the two unresolved-referent
# questions WILL get a spurious Layer 1 FAIL in mock mode - flagged
# inline on those questions. Trust Layer 1 on those two only in real mode.
# =====================================================================

import json
import os
import sys
import time

from agent import RabivyAgent
from llm_client import MockLLM
import ground_truth as gt

AGENT_TOP = gt.AGENT_DEFAULT_TOP  # this file always tests the AGENT's default (10), not the router's (20)


# =====================================================================
# THE 30 QUESTIONS (same set, same order, as test_the_system.py)
# =====================================================================
QUESTIONS = [

    # =================================================================
    # GROUP 1: STRUCTURED - exact lookup, ranking & counting
    # =================================================================
    {
        "group": "STRUCTURED - exact lookup, ranking & counting", "cat": "structured lookup",
        "q": "Who is the top GLP-1 prescriber in New York?",
        "expect_tools": ["query_hcp_table"],
        "expect_filters": {"state": "New York"},
        "ground_truth": lambda: gt.gt_top_prescriber("New York"),
    },
    {
        "group": "STRUCTURED - exact lookup, ranking & counting", "cat": "structured lookup",
        "q": "How many GLP-1 scripts did NPI 1344001929 write last month?",
        "expect_tools": ["lookup_hcp"],
        "expect_filters": {"npi": "1344001929"},
        "ground_truth": lambda: gt.gt_scripts_for_npi("1344001929"),
    },
    {
        "group": "STRUCTURED - exact lookup, ranking & counting", "cat": "structured ranking",
        "q": "List the top 10 High-tier prescribers nationally by propensity score.",
        "expect_tools": ["query_hcp_table"],
        "expect_filters": {"tier": "High"},
        # Was checking only the #1 NPI ([0]) even though the question
        # explicitly asks for 10 - fixed 2026-07-28, same bug as
        # test_the_system.py's version.
        "ground_truth": lambda: gt.gt_top_n_national_npis(tier="High", n=10),
    },
    {
        "group": "STRUCTURED - exact lookup, ranking & counting", "cat": "structured ranking",
        "q": "Which states have the most High-tier HCPs?",
        "expect_tools": ["states_summary"],
        "rules_check": lambda evidence: any(e.get("result", {}).get("tier") == "High" for e in evidence),
        "note": "Checkable as of the agent_tools.py fix (2026-07-28): "
                "_states_summary() used to silently drop the 'tier' field its "
                "own underlying function already computed, so there was "
                "nothing here for Layer 2 to check. Now fixed at the source.",
        "ground_truth": lambda: gt.gt_top_state_by_tier("High"),
    },
    {
        "group": "STRUCTURED - exact lookup, ranking & counting", "cat": "structured count",
        "q": "How many active GLP-1 writers are in Texas?",
        "expect_tools": ["count_active_writers"],
        "expect_filters": {"state": "Texas"},
        "ground_truth": lambda: gt.gt_count_writers("Texas", "active"),
    },

    # =================================================================
    # GROUP 2: STRUCTURED - filtered targeting (multi-field)
    # =================================================================
    {
        "group": "STRUCTURED - filtered targeting (multi-field)", "cat": "filtered targeting",
        "q": "Show me High-propensity endocrinologists in Florida who are not currently targeted.",
        "expect_tools": ["query_hcp_table"],
        "expect_filters": {"state": "Florida", "specialty": "Endocrinology", "tier": "High", "targeted": False},
        "ground_truth": lambda: gt.gt_filter_npis(top=AGENT_TOP, state="Florida", specialty="Endocrinology",
                                                   tier="High", targeted=False),
    },
    {
        "group": "STRUCTURED - filtered targeting (multi-field)", "cat": "filtered targeting",
        "q": "Which Novo-heavy prescribers in California have a high switching score?",
        "expect_tools": ["query_hcp_table"],
        "expect_filters": {"state": "California", "dominant_competitor": "Novo Nordisk"},
        # HIGH_CUTOFF verified directly from query_spreadsheet.py = 0.75
        # (was guessed as 0.6). sort_by="switching_score" matches what
        # this question naturally sorts by - without it, ground truth's
        # default propensity-sorted top-N and the agent's switching-
        # sorted top-N are different slices of the same match set.
        "ground_truth": lambda: gt.gt_filter_npis(top=AGENT_TOP, state="California",
                                                   dominant_competitor="Novo Nordisk",
                                                   extra_filters={"switching_score": (0.75, None)},
                                                   sort_by="switching_score"),
    },
    {
        "group": "STRUCTURED - filtered targeting (multi-field)", "cat": "filtered targeting",
        "q": "Find high-volume writers in Illinois with a preferred formulary status.",
        "expect_tools": ["query_hcp_table"],
        "expect_filters": {"state": "Illinois", "formulary_tier": "Preferred"},
        "ground_truth": lambda: gt.gt_filter_npis(top=AGENT_TOP, state="Illinois", formulary_tier="Preferred",
                                                   sort_by="rx_volume_monthly"),
    },
    {
        "group": "STRUCTURED - filtered targeting (multi-field)", "cat": "filtered targeting",
        "q": "Which untargeted HCPs have a recent sample request and a High tier?",
        "expect_tools": ["query_hcp_table"],
        "expect_filters": {"targeted": False, "recent_sample_request": True, "tier": "High"},
        "ground_truth": lambda: gt.gt_filter_npis(top=AGENT_TOP, targeted=False, recent_sample_request=True,
                                                   tier="High"),
    },

    # =================================================================
    # GROUP 3: RAG - HCP cards & comparisons
    # Only genuinely-answerable lookups live here - the two
    # unresolved-referent questions moved to EDGE CASES.
    # =================================================================
    {
        "group": "RAG - HCP cards & comparisons", "cat": "card summary",
        "q": "Give me a summary of NPI 1344001929.",
        "expect_tools": ["lookup_hcp"],
        "expect_tag": ["card_1344001929"],
        "note": "A specific-NPI lookup is ALWAYS fully determinable, whatever "
                "the data says - never skipped at Layer 3.",
        "ground_truth": lambda: gt.gt_hcp_card_facts("1344001929"),
    },
    {
        "group": "RAG - HCP cards & comparisons", "cat": "card (chained)",
        "q": "What's the access situation for the top prescriber in Missouri?",
        "expect_tools": ["query_hcp_table"],
        "expect_filters": {"state": "Missouri"},
        "note": "Unlike the plain regex router, query_hcp_table's row already "
                "includes formulary_tier/pa_burden/dominant_competitor, so the "
                "agent can answer this in ONE tool call.",
        "ground_truth": lambda: gt.gt_top_prescriber("Missouri"),
    },
    {
        "group": "RAG - HCP cards & comparisons", "cat": "card narrative",
        "q": "What's the story on GLP-1 writers in Arizona - who should I know about?",
        "expect_tools_any_of": [["search_documents"], ["query_hcp_table"], ["count_active_writers"]],
        "rules_check": lambda evidence: (
            _filters_match_anywhere(evidence, {"state": "Arizona"})
            or "arizona" in json.dumps(evidence, default=str).lower()
        ),
        "note": "TWO genuinely valid approaches exist here: the static "
                "state_market_summary.md doc (via search_documents), OR live "
                "structured data filtered to Arizona (via query_hcp_table / "
                "count_active_writers) - arguably the better answer, since "
                "it's live rather than a potentially-stale summary. Neither "
                "is penalised; Layer 3 accepts whichever fact matches the "
                "path actually taken. MOCK MODE CAVEAT: MockLLM's fallback "
                "planner never extracts a 'state' parameter for "
                "search_documents specifically - a FAIL via that path in "
                "mock mode is expected, not a regression.",
        "ground_truth": lambda evidence: (
            [gt.gt_state_market_fact("Arizona")]
            if any(e["tool"] == "search_documents" for e in evidence)
            else [gt.gt_count_writers("Arizona", "active")]
        ),
    },
    {
        "group": "RAG - HCP cards & comparisons", "cat": "comparison",
        "q": "Compare NPI 1344001929 to a typical endocrinologist.",
        "expect_tools_any_of": [["lookup_hcp", "search_documents"], ["lookup_hcp", "aggregate_hcp_stats"]],
        "rules_check": lambda evidence: gt.any_present(json.dumps(evidence, default=str).lower(), ["1344001929"])
                        and (gt.any_present(json.dumps(evidence, default=str).lower(),
                                            gt.NARRATIVE_FACTS["typical_endocrinologist"]["tag"])
                             or any(e["tool"] == "aggregate_hcp_stats"
                                    and str(e.get("input", {}).get("specialty", "")).lower() == "endocrinology"
                                    for e in evidence)),
        "note": "TWO sources must both be present, EITHER the narrative "
                "benchmark doc OR the new aggregate_hcp_stats tool (added "
                "2026-07-28) - a real run (2026-07-29) found the agent now "
                "prefers the more precise aggregate tool over the narrative "
                "doc for this comparison, which is a genuine improvement, "
                "not a regression - the eval was updated to accept it rather "
                "than penalise a better answer. Layer 3 checks a fact unique "
                "to EACH side (the NPI's real script count, and the live "
                "specialty-wide mean propensity) regardless of which path was "
                "used. MOCK MODE CAVEAT: MockLLM's generic search query often "
                "retrieves an adjacent but wrong benchmark section - a Layer "
                "3 FAIL here in mock mode is expected, not a regression.",
        "ground_truth": lambda: [gt.gt_scripts_for_npi("1344001929")] + gt.gt_specialty_benchmark_facts("Endocrinology"),
    },

    # =================================================================
    # GROUP 4: RAG - narrative knowledge base
    # =================================================================
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "What's our recommended messaging for competitive switchers?",
        "expect_tools": ["search_documents"], "narrative_key": "competitive_switchers",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "What is Rabivy's key differentiator versus Zepbound?",
        "expect_tools": ["search_documents"], "narrative_key": "zepbound_differentiator",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "How should a rep handle 'I'm already happy with Ozempic'?",
        "expect_tools": ["search_documents"], "narrative_key": "ozempic_objection",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "What's the Medicaid coverage outlook for GLP-1 obesity drugs?",
        "expect_tools": ["search_documents"], "narrative_key": "medicaid_access",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative (state filter)",
        "q": "What does the GLP-1 market look like in Missouri?",
        "expect_tools": ["search_documents"], "narrative_key": "missouri_market",
        "note": "Layer 3 parses the real doc LIVE, not a hardcoded number. "
                "MOCK MODE CAVEAT: same state-extraction limitation as the "
                "Arizona question above - a Layer 2/3 FAIL here in mock mode "
                "is expected, not a regression. Trust this in real mode.",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "How is Rabivy's mechanism different from tirzepatide?",
        "expect_tools": ["search_documents"], "narrative_key": "tirzepatide_mechanism",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "What is Rabivy's main dosing advantage?",
        "expect_tools": ["search_documents"], "narrative_key": "dosing_advantage",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative (benchmark)",
        "q": "What does a typical endocrinologist look like?",
        "expect_tools": ["search_documents"], "narrative_key": "typical_endocrinologist",
        "note": "Tag verified 2026-07-28. Layer 3 computed LIVE from the "
                "dataframe (population + mean propensity), not read from the "
                "doc - the doc's own 'active writers' figure has drifted.",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "How is prior authorization affecting access?",
        "expect_tools": ["search_documents"], "narrative_key": "prior_auth_access",
        "note": "Layer 3 facts '31'/'41' are the PA initial-approval-rate "
                "range for anti-obesity GLP-1s, confirmed from "
                "real_world_evidence_brief.md directly.",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "Why do patients stop taking GLP-1s after a year?",
        "expect_tools": ["search_documents"], "narrative_key": "discontinuation",
    },

    # =================================================================
    # GROUP 5: MULTI-SOURCE (structured + narrative + LLM, all at once)
    # Unlike the regex router (one call, one route), the agent can and
    # SHOULD call both engines here.
    # =================================================================
    {
        "group": "MULTI-SOURCE (structured + narrative + LLM)", "cat": "multi-source showpiece",
        "q": "Who should I target next month in New York, and what should I say to them?",
        "expect_tools": ["query_hcp_table", "search_documents"],
        "expect_filters": {"state": "New York"},
        "note": "The showpiece: needs BOTH engines. Layer 3 requires BOTH the "
                "top-propensity NY targeting NPI AND the rep talking-points "
                "chunk tag to be present - a result that only pulled one "
                "side fails here, same rigor as test_the_system.py's version.",
        "ground_truth": lambda: [(gt.gt_filter_npis(top=AGENT_TOP, state="New York") or [None])[0],
                                  "rep_talking_points_by_segment"],
    },

    # =================================================================
    # GROUP 6: EDGE CASES - not-found, nonsense input, synonym mapping,
    # and unresolved referents. ALL of these must correctly REFUSE or
    # ASK rather than guess - that correct behaviour is itself the
    # fixed, always-determinable "answer", so Layer 3 checks it directly
    # instead of skipping.
    # =================================================================
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "unresolved referent",
        "q": "Why is this High-tier doctor not converting?",
        "expect_tools": [],
        "rules_check": lambda evidence: len(evidence) == 0,
        "mock_known_limitation": True,
        "note": "No NPI given, no prior context - correct behaviour is asking "
                "for clarification, not guessing. The RULE here is simple and "
                "directly checkable: no tool should be called at all - so "
                "Layer 2 checks that directly and shows PASS/FAIL rather than "
                "skipping. MOCK MODE CAVEAT: MockLLM's keyword planner has no "
                "real referent-resolution logic and WILL guess a tool call "
                "here - a FAIL in mock mode on this specific question is an "
                "expected mock limitation, not a real bug, and is EXCLUDED "
                "from the CI pass/fail gate for that reason (see "
                "mock_known_limitation handling in run()). Only trust this in "
                "real mode.",
        "ground_truth": lambda: gt.CLARIFICATION_FACTS,
    },
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "unresolved referent",
        "q": "Is this doctor a high or low prescriber for their specialty?",
        "expect_tools": [],
        "rules_check": lambda evidence: len(evidence) == 0,
        "mock_known_limitation": True,
        "note": "Same rule, MOCK MODE CAVEAT, and CI-gate exclusion as above - only trust this in real mode.",
        "ground_truth": lambda: gt.CLARIFICATION_FACTS,
    },
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "not-found (fake NPI)",
        "q": "Tell me about NPI 0000000000.",
        "expect_tools": ["lookup_hcp"],
        "expect_filters": {"npi": "0000000000"},
        "note": "Layer 2 confirms the SPECIFIC invalid NPI from the question "
                "was actually passed to the tool (not a generic canned "
                "response) - reliable to check here since the tool schema "
                "requires an npi argument.",
        "ground_truth": lambda: gt.FAKE_NPI_REJECTION_FACTS,
    },
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "not-found (fake region)",
        "q": "Which HCPs are in the Southeast region?",
        "expect_tools": [],
        "rules_check": lambda evidence: len(evidence) == 0,
        "mock_known_limitation": True,
        "note": "UPDATED 2026-07-28: system prompt now instructs a clean "
                "rejection with NO tool call, instead of the old per-state "
                "substitution fallback - a real run found that fallback "
                "produces a genuinely incomplete/debatable state list (common "
                "'Southeast' definitions disagree on 3+ states), misleading a "
                "rep into thinking it's authoritative. Same rule shape as the "
                "unresolved-referent questions: correct behaviour is to "
                "refuse/clarify without calling a tool at all. MOCK MODE "
                "CAVEAT: MockLLM never reads the system prompt, so it will "
                "still guess a tool call here - excluded from the CI gate for "
                "the same reason as the referent questions. Only trust this "
                "in real mode.",
        "ground_truth": lambda: gt.FAKE_REGION_REJECTION_FACTS,
    },
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "tier synonym mapping",
        "q": "Which HCPs have a Low tier?",
        "expect_tools": ["query_hcp_table"],
        "expect_filters": {"tier": "Watch"},
        "note": "'Low' isn't a real tier value - must map to 'Watch' and "
                "return ACTUAL matching HCPs.",
        "ground_truth": lambda: gt.gt_filter_npis(top=AGENT_TOP, tier="Watch"),
    },
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "generic numeric threshold",
        "q": "Which HCPs have days since contact over 90?",
        "expect_tools": ["query_hcp_table"],
        "rules_check": lambda evidence: any(
            any(f.get("column") == "days_since_contact" and f.get("min") == 90.0
                for f in (e.get("input", {}).get("extra_filters") or []))
            for e in evidence),
        "note": "The rule is 'days_since_contact > 90, no upper bound' - Layer "
                "3 checks the TOTAL match count (computed live), not any "
                "specific NPI in the top-N slice.",
        "ground_truth": lambda: gt.gt_filter_count(extra_filters={"days_since_contact": (90, None)}),
    },
]

# Conversation-memory scenario, run as a sequence in ONE agent session -
# NOT part of the 30-question golden set, kept separate on purpose. The
# second question is unanswerable without remembering the first.
MEMORY_SCENARIO = [
    {"q": "Give me a summary of NPI 1344001929.", "expect_facts": ["1344001929"]},
    {"q": "Is this doctor a high or low prescriber for their specialty?",
     "expect_facts": [["1344001929", "endocrinolog", "high"]],
     "note": "must resolve 'this doctor' from conversation memory"},
]


# =====================================================================
# EVAL MACHINERY
# =====================================================================

def _tools_called(evidence):
    return {e["tool"] for e in evidence}


def _filters_match_anywhere(evidence, expected):
    """True if ANY single tool call's input matches every expected
    key/value (bool/int compared loosely, e.g. False == 0). Also
    returns the actual input in play - the matching call's input on a
    PASS, every call's input on a FAIL - so a failure record shows
    what was actually sent, not just that it didn't match."""
    for e in evidence:
        inp = e.get("input", {})
        if all(inp.get(k) == v or (isinstance(v, bool) and bool(inp.get(k)) == v) for k, v in expected.items()):
            return True, inp
    return False, [e.get("input", {}) for e in evidence]


def _narrative_retrieval_detail(evidence, expected_tags):
    """For a narrative question, pulls the ranked/scored chunk list
    straight out of the agent's OWN search_documents call - not a new
    lookup, just reading data the agent already produced to answer the
    question (agent_tools.py's _search_documents already returns each
    chunk's similarity score, in rank order). Returns None if the
    agent never called search_documents at all for this question."""
    search_calls = [e for e in evidence if e["tool"] == "search_documents"]
    if not search_calls:
        return None
    sections = search_calls[-1].get("result", {}).get("sections", [])
    rank = None
    top_results = []
    for i, s in enumerate(sections, start=1):
        chunk_id = s.get("chunk_id", "")
        is_expected = gt.any_present(chunk_id.lower(), expected_tags)
        if is_expected and rank is None:
            rank = i
        top_results.append({"chunk_id": chunk_id, "similarity": s.get("similarity"),
                            "is_expected": is_expected})
    return {"expected_tag": expected_tags, "rank": rank, "top_results": top_results}


def _haystack(result, is_mock):
    """Mock mode: judge the raw evidence only (MockLLM's prose isn't
    real). Real mode: judge the answer text AND the evidence together."""
    ev_text = json.dumps(result["evidence"], default=str).lower()
    if is_mock:
        return ev_text
    return (result["answer"] or "").lower() + " " + ev_text


def _layer1(t, evidence):
    """Returns (ok, detail). detail always carries BOTH what was
    expected and what actually got called - previously only the bool
    survived into the record, so a FAIL gave no clue what was chosen
    instead of the right tool."""
    called = sorted(_tools_called(evidence))
    if "expect_tools_any_of" in t:
        # Several DIFFERENT tool-call patterns are all legitimately
        # correct for this question - PASS if the agent's actual call
        # matches any one of them.
        expected = t["expect_tools_any_of"]
        ok = any(set(pattern).issubset(called) for pattern in expected)
    else:
        expected = sorted(t.get("expect_tools", []))
        ok = (len(called) == 0) if not expected else set(expected).issubset(called)
    return ok, {"expected": expected, "actual": called}


def _layer2(t, evidence, haystack):
    if "rules_check" in t:
        ok = t["rules_check"](evidence)
        return ("PASS" if ok else "FAIL"), "rules_check"
    if "expect_filters" in t:
        ok, actual = _filters_match_anywhere(evidence, t["expect_filters"])
        return ("PASS" if ok else "FAIL"), {"expected": t["expect_filters"], "actual": actual}
    if "expect_tag" in t:
        ok = gt.any_present(haystack, t["expect_tag"])
        return ("PASS" if ok else "FAIL"), t["expect_tag"]
    if "narrative_key" in t:
        spec = gt.NARRATIVE_FACTS[t["narrative_key"]]
        if spec["tag"] is None:
            return "SKIP", "no verified tag yet"
        ok = gt.any_present(haystack, spec["tag"])
        # Rank + score for every retrieved chunk, correct one flagged -
        # this is the piece that lets a dashboard tell "Yes" (rank 1)
        # apart from "Yes, but" (present, not first) apart from "No"
        # (missing) for the golden-test decision tree, with no extra
        # API/embedding cost - it's the agent's own search_documents
        # call, already made to answer this question.
        retrieval = _narrative_retrieval_detail(evidence, spec["tag"])
        return ("PASS" if ok else "FAIL"), (retrieval if retrieval is not None else {"expected_tag": spec["tag"]})
    return "SKIP", "no separate rule beyond correct routing/refusal for this question"


def _layer3(t, haystack, evidence=None, answer_text=None):
    """Returns (status, detail). detail is {"expected": ..., "actual":
    answer_text} wherever there's a live ground-truth comparison - the
    actual final answer used to be printed to the console (see run())
    and nowhere else; it now travels with the record."""
    if "narrative_key" in t:
        spec = gt.NARRATIVE_FACTS[t["narrative_key"]]
        if spec["facts"] is not None:
            ok = gt.all_present(haystack, spec["facts"])
            return ("PASS" if ok else "FAIL"), {"expected": spec["facts"], "actual": answer_text}
        if spec.get("state"):
            fact = gt.gt_state_market_fact(spec["state"])
            if fact is None:
                return "SKIP", "could not parse the live doc"
            ok = fact in haystack
            return ("PASS" if ok else "FAIL"), {"expected": fact, "actual": answer_text}
        if spec.get("specialty"):
            facts = gt.gt_specialty_benchmark_facts(spec["specialty"])
            if facts is None:
                return "SKIP", "could not compute live specialty facts"
            ok = gt.all_present(haystack, facts)
            return ("PASS" if ok else "FAIL"), {"expected": facts, "actual": answer_text}
    gt_fn = t.get("ground_truth")
    if gt_fn is None:
        return "SKIP", "no live ground truth for this question"
    try:
        truth = gt_fn(evidence)
    except TypeError:
        truth = gt_fn()  # most ground_truth lambdas take no arguments
    if truth is None:
        return "SKIP", "ground truth computation returned nothing"
    if isinstance(truth, list):
        ok = gt.all_present(haystack, truth)
        return ("PASS" if ok else "FAIL"), {"expected": truth, "actual": answer_text}
    ok = str(truth).lower() in haystack
    return ("PASS" if ok else "FAIL"), {"expected": truth, "actual": answer_text}


_BAR = "=" * 72


def run():
    agent = RabivyAgent()
    is_mock = isinstance(agent.llm, MockLLM)
    mode = "MOCK (Layers 1-2 mostly meaningful; see per-question mock caveats)" if is_mock else "REAL (all 3 layers meaningful)"
    print(f"\nAgent eval mode: {mode}")

    tally = {"layer1": {"PASS": 0, "FAIL": 0}, "layer2": {"PASS": 0, "FAIL": 0, "SKIP": 0},
             "layer3": {"PASS": 0, "FAIL": 0, "SKIP": 0}, "CRASH": 0}
    gating_layer1_fails = 0  # excludes known mock-mode limitations - see mock_known_limitation below
    records = []
    last_group = None

    for i, t in enumerate(QUESTIONS, start=1):
        if t["group"] != last_group:
            print(f"\n{_BAR}\n  {t['group']}\n{_BAR}")
            last_group = t["group"]

        print(f"\n{'=' * 8} QUESTION {i} {'=' * 8}")
        print(f"Q: {t['q']}")
        print(f"[{t['cat']}]")
        if t.get("note"):
            print(f"NOTE: {t['note']}")

        fresh = RabivyAgent(llm=agent.llm)
        fresh._log = lambda *a, **k: None
        try:
            result = fresh.ask(t["q"])
        except Exception as e:
            tally["CRASH"] += 1
            print(f"  CRASH: {type(e).__name__}: {e}")
            records.append({"q": t["q"], "crash": str(e)})
            continue

        for e in result["evidence"]:
            print(f"  -> tool: {e['tool']}({json.dumps(e['input'])[:120]})")

        haystack = _haystack(result, is_mock)

        l1_ok, l1_detail = _layer1(t, result["evidence"])
        tally["layer1"]["PASS" if l1_ok else "FAIL"] += 1
        is_known_mock_limitation = is_mock and t.get("mock_known_limitation")
        if not l1_ok and not is_known_mock_limitation:
            gating_layer1_fails += 1
        gate_note = "  (excluded from CI gate - known mock-mode limitation)" if (not l1_ok and is_known_mock_limitation) else ""
        print(f"  LAYER 1 (routing) : {'PASS' if l1_ok else 'FAIL'}  "
              f"(called: {l1_detail['actual']}){gate_note}")

        l2_status, l2_detail = _layer2(t, result["evidence"], haystack)
        tally["layer2"][l2_status] += 1
        print(f"  LAYER 2 (rules)   : {l2_status}  ({l2_detail})")

        l3_status, l3_detail = _layer3(t, haystack, result["evidence"], result["answer"])
        tally["layer3"][l3_status] += 1
        print(f"  LAYER 3 (answer)  : {l3_status}  ({l3_detail})")

        print(f"  ANSWER: {result['answer']}")

        records.append({
            "q": t["q"], "answer": result["answer"],
            "layer1": l1_ok, "layer1_detail": l1_detail,
            "layer2": l2_status, "layer2_detail": l2_detail,
            "layer3": l3_status, "layer3_detail": l3_detail,
            # Found 2026-08-04: agent.py's ask() already computes a
            # hallucination-audit verdict for every question - real
            # tokens already paid for it - but it was never written
            # here, so a real run's audit data was silently discarded.
            # Not needed for the golden-tree leaf (that's layer2 rank +
            # layer3 correctness alone), but worth keeping since it's
            # free at this point and directly relevant to comparing
            # ground-truth correctness against the audit's own verdict.
            "verdict": (result.get("verification") or {}).get("verdict"),
            "audit_trail": result.get("audit_trail", []),
        })

    # ---- MEMORY_SCENARIO: separate, not part of the 3-layer 30-question set ----
    print(f"\n{_BAR}\n  MEMORY SCENARIO (conversation continuity - not part of the 30)\n{_BAR}")
    mem_agent = RabivyAgent(llm=agent.llm)
    mem_agent._log = lambda *a, **k: None
    mem_ok = True
    for i, step in enumerate(MEMORY_SCENARIO, start=1):
        print(f"\n{'=' * 8} MEMORY STEP {i} {'=' * 8}")
        print(f"Q: {step['q']}")
        if step.get("note"):
            print(f"NOTE: {step['note']}")
        try:
            result = mem_agent.ask(step["q"])
        except Exception as e:
            print(f"  CRASH: {type(e).__name__}: {e}")
            mem_ok = False
            continue
        haystack = _haystack(result, is_mock)
        ok = gt.all_present(haystack, step["expect_facts"])
        status = "PASS" if ok else ("N/A IN MOCK" if is_mock else "FAIL")
        print(f"  STATUS: {status}")
        print(f"  ANSWER: {result['answer']}")
        if status == "FAIL":
            mem_ok = False

    print(f"\n{_BAR}\nSUMMARY\n{_BAR}")
    print(f"Layer 1 (routing) : {tally['layer1']['PASS']} PASS / {tally['layer1']['FAIL']} FAIL")
    print(f"Layer 2 (rules)   : {tally['layer2']['PASS']} PASS / {tally['layer2']['FAIL']} FAIL "
          f"/ {tally['layer2']['SKIP']} SKIP")
    print(f"Layer 3 (answer)  : {tally['layer3']['PASS']} PASS / {tally['layer3']['FAIL']} FAIL "
          f"/ {tally['layer3']['SKIP']} SKIP")
    print(f"Crashes           : {tally['CRASH']}")
    print(f"Memory scenario   : {'PASS' if mem_ok else 'FAIL'}")
    print(f"{_BAR}\nA data refresh should only ever move Layer 3 numbers. If Layer 1 or\n"
          f"Layer 2 fail after a data refresh, that's a REAL bug in the agent's\n"
          f"tool selection or argument parsing.\n{_BAR}")

    os.makedirs("eval_runs", exist_ok=True)
    path = os.path.join("eval_runs", f"agent_eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tally": tally, "records": records, "memory_scenario_pass": mem_ok}, f, indent=2, default=str)
    print(f"\nSaved this run to {path}")
    if is_mock and tally["layer1"]["FAIL"] != gating_layer1_fails:
        print(f"NOTE: {tally['layer1']['FAIL']} raw Layer-1 fail(s) in mock mode, but "
              f"{gating_layer1_fails} count toward the CI gate - the difference is the "
              f"known unresolved-referent mock limitation (see mock_known_limitation).")

    sys.exit(1 if (gating_layer1_fails + tally["CRASH"]) > 0 else 0)


if __name__ == "__main__":
    run()