# test_the_agent.py
# ===================================================================
# AGENT-LEVEL EVALUATION - the SAME 30-question set as test_the_system.py
# (same groups, same order), but now the whole thing runs THROUGH THE
# AGENT, so questions that were "waiting on AI" before (NEEDS-LLM /
# ROUTES->LLM in test_the_system.py) are now expected to come back as
# complete written answers.
#
# Kept in sync with test_the_system.py on purpose (2026-07-22): this
# file used to only have 22 of the 30 questions - missing the whole
# "edge cases" group and a couple of the card questions - so this file
# and test_the_system.py were quietly testing different things under
# the same name. Same 30 questions, same groups, same order in both
# files now; same boxed print style too, so a run of either is easy to
# read side by side.
#
# Two evaluation modes, chosen automatically by which LLM backend is
# active (see llm_client.py):
#
#   MOCK mode  - free, deterministic. Checks the agent's BEHAVIOUR:
#                did it pick the right tool(s), and did the right facts
#                come back in the tool results? (The mock's template
#                answers aren't judged for prose quality - there is none.)
#   REAL mode  - with ANTHROPIC_API_KEY set. Additionally checks the
#                final ANSWER TEXT contains the expected facts, and
#                reports the verification pass's verdict per question.
#
# Outcomes:
#   PASS  - expected tool used AND expected facts present (in the
#           evidence in mock mode; in the final answer in real mode)
#   WEAK  - right tool, but the expected fact didn't surface (usually a
#           retrieval-quality miss, e.g. the offline fallback embedder)
#   FAIL  - wrong tool entirely, or nothing relevant retrieved
#   CRASH - the agent threw
#
# Each run is saved to eval_runs/agent_eval_*.json, alongside the
# test_the_system.py runs, so the before/after story is diffable.
# ===================================================================

import json
import os
import sys
import time

from agent import RabivyAgent
from llm_client import MockLLM

# expect_tools: at least one call to each listed tool must happen.
#               An EMPTY list means no tool call is expected at all -
#               used for the two unresolved-referent questions below,
#               where the correct behaviour is asking for
#               clarification, not guessing at a tool call.
# expect_facts: each item is a string (or list of acceptable
#               alternatives) that must appear in the evidence (mock)
#               / answer (real). An empty list means "just check tool
#               selection, don't check for a specific fact" - used for
#               the generic numeric threshold question, whose exact
#               row count depends on live data.
TESTS = [
    # =================================================================
    # GROUP 1: STRUCTURED - exact lookup, ranking & counting
    # =================================================================
    {"group": "STRUCTURED - exact lookup, ranking & counting",
     "q": "Who is the top GLP-1 prescriber in New York?",
     "cat": "structured lookup",
     "expect_tools": ["query_hcp_table"], "expect_facts": ["1658907316"]},
    {"group": "STRUCTURED - exact lookup, ranking & counting",
     "q": "How many GLP-1 scripts did NPI 1344001929 write last month?",
     "cat": "structured lookup",
     "expect_tools": ["lookup_hcp"], "expect_facts": ["55"]},
    {"group": "STRUCTURED - exact lookup, ranking & counting",
     "q": "List the top 10 High-tier prescribers nationally by propensity score.",
     "cat": "structured ranking",
     "expect_tools": ["query_hcp_table"], "expect_facts": ["1184547828"]},
    {"group": "STRUCTURED - exact lookup, ranking & counting",
     "q": "Which states have the most High-tier HCPs?",
     "cat": "structured ranking",
     "expect_tools": ["states_summary"], "expect_facts": ["california"]},
    {"group": "STRUCTURED - exact lookup, ranking & counting",
     "q": "How many active GLP-1 writers are in Texas?",
     "cat": "structured count",
     "expect_tools": ["count_active_writers"], "expect_facts": ["576"]},

    # =================================================================
    # GROUP 2: STRUCTURED - filtered targeting (multi-field)
    # =================================================================
    {"group": "STRUCTURED - filtered targeting (multi-field)",
     "q": "Show me High-propensity endocrinologists in Florida who are not currently targeted.",
     "cat": "filtered targeting",
     "expect_tools": ["query_hcp_table"],
     "expect_facts": ["florida", "endocrinology"]},
    {"group": "STRUCTURED - filtered targeting (multi-field)",
     "q": "Which Novo-heavy prescribers in California have a high switching score?",
     "cat": "filtered targeting",
     "expect_tools": ["query_hcp_table"],
     "expect_facts": ["california", "novo"]},
    {"group": "STRUCTURED - filtered targeting (multi-field)",
     "q": "Find high-volume writers in Illinois with a preferred formulary status.",
     "cat": "filtered targeting",
     "expect_tools": ["query_hcp_table"],
     "expect_facts": ["illinois", "preferred"]},
    {"group": "STRUCTURED - filtered targeting (multi-field)",
     "q": "Which untargeted HCPs have a recent sample request and a High tier?",
     "cat": "filtered targeting",
     "expect_tools": ["query_hcp_table"], "expect_facts": ["high"]},

    # =================================================================
    # GROUP 3: RAG - HCP cards & comparisons
    # =================================================================
    {"group": "RAG - HCP cards & comparisons",
     "q": "Give me a summary of NPI 1344001929.",
     "cat": "card summary",
     "expect_tools": ["lookup_hcp"], "expect_facts": ["card_1344001929"]},
    {"group": "RAG - HCP cards & comparisons",
     "q": "What's the access situation for the top prescriber in Missouri?",
     "cat": "card (chained)",
     "expect_tools": ["query_hcp_table"], "expect_facts": ["missouri"],
     "note": "unlike the plain regex router (where this chained lookup "
             "isn't wired), query_hcp_table's row already includes "
             "formulary_tier/pa_burden/dominant_competitor - the agent "
             "can answer this in one tool call, no separate card lookup "
             "needed"},
    {"group": "RAG - HCP cards & comparisons",
     "q": "Why is this High-tier doctor not converting?",
     "cat": "card read",
     "expect_tools": [], "expect_facts": [["npi", "which", "clarif"]],
     "note": "no NPI given, no prior context - correct behaviour is "
             "asking for clarification, not calling a tool with a "
             "guessed/wrong NPI. Standalone (single-turn) version of "
             "the same question tested with memory below in "
             "MEMORY_SCENARIO - flag for review if the agent's actual "
             "clarification wording differs from what's checked here"},
    {"group": "RAG - HCP cards & comparisons",
     "q": "What's the story on GLP-1 writers in Arizona - who should I know about?",
     "cat": "card narrative",
     "expect_tools": ["search_documents"], "expect_facts": ["arizona"]},
    {"group": "RAG - HCP cards & comparisons",
     "q": "Compare NPI 1344001929 to a typical endocrinologist.",
     "cat": "comparison",
     "expect_tools": ["lookup_hcp", "search_documents"],
     "expect_facts": ["1344001929"]},
    {"group": "RAG - HCP cards & comparisons",
     "q": "Is this doctor a high or low prescriber for their specialty?",
     "cat": "comparison",
     "expect_tools": [], "expect_facts": [["npi", "which", "clarif"]],
     "note": "same unresolved-referent case as 'this High-tier doctor' "
             "above - standalone, no NPI, no context. This exact "
             "question is ALSO the second step of MEMORY_SCENARIO below, "
             "testing the opposite thing: that it correctly RESOLVES "
             "from conversation memory when context IS available"},

    # =================================================================
    # GROUP 4: RAG - narrative knowledge base
    # =================================================================
    {"group": "RAG - narrative knowledge base",
     "q": "What's our recommended messaging for competitive switchers?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["competitive_switchers", "competitive switcher"]]},
    {"group": "RAG - narrative knowledge base",
     "q": "What is Rabivy's key differentiator versus Zepbound?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["differentiator", "monthly", "dosing"]]},
    {"group": "RAG - narrative knowledge base",
     "q": "How should a rep handle 'I'm already happy with Ozempic'?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["objection_handling", "happy with ozempic", "switch"]]},
    {"group": "RAG - narrative knowledge base",
     "q": "What's the Medicaid coverage outlook for GLP-1 obesity drugs?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["payer_access", "medicaid"]]},
    {"group": "RAG - narrative knowledge base",
     "q": "What does the GLP-1 market look like in Missouri?",
     "cat": "narrative (state filter)",
     "expect_tools": ["search_documents"],
     "expect_facts": [["state_market_summary__missouri", "missouri"]]},
    {"group": "RAG - narrative knowledge base",
     "q": "How is Rabivy's mechanism different from tirzepatide?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["mechanism", "molecule"]]},
    {"group": "RAG - narrative knowledge base",
     "q": "What is Rabivy's main dosing advantage?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["monthly", "dosing", "where_rabivy_wins"]]},
    {"group": "RAG - narrative knowledge base",
     "q": "What does a typical endocrinologist look like?",
     "cat": "narrative (benchmark)",
     "expect_tools": ["search_documents"],
     "expect_facts": [["endocrinolog"]]},
    {"group": "RAG - narrative knowledge base",
     "q": "How is prior authorization affecting access?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["prior_auth", "prior auth", "access"]]},
    {"group": "RAG - narrative knowledge base",
     "q": "Why do patients stop taking GLP-1s after a year?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["discontinu", "persistence", "adherence"]]},

    # =================================================================
    # GROUP 5: MULTI-SOURCE (structured + narrative + LLM, all at once)
    # =================================================================
    {"group": "MULTI-SOURCE (structured + narrative + LLM)",
     "q": "Who should I target next month in New York, and what should I say to them?",
     "cat": "multi-source showpiece",
     "expect_tools": ["query_hcp_table", "search_documents"],
     "expect_facts": ["new york"],
     "note": "the showpiece: needs both engines joined in one answer - "
             "the whole reason the agent exists over the plain regex "
             "router, which can't synthesise across sources"},

    # =================================================================
    # GROUP 6: EDGE CASES - not-found, nonsense input & synonym mapping
    # =================================================================
    {"group": "EDGE CASES - not-found, nonsense input & synonym mapping",
     "q": "Tell me about NPI 0000000000.",
     "cat": "not-found (fake NPI)",
     "expect_tools": ["lookup_hcp"],
     "expect_facts": [["not found", "no hcp", "no record", "doesn't exist"]],
     "note": "fake NPI - the agent must say clearly not found, never "
             "hallucinate a plausible-sounding HCP"},
    {"group": "EDGE CASES - not-found, nonsense input & synonym mapping",
     "q": "Which HCPs are in the Southeast region?",
     "cat": "not-found (fake region)",
     "expect_tools": ["query_hcp_table"],
     "expect_facts": [["not one of", "not a real region", "not an official",
                        "isn't a defined region", "4 real regions",
                        "midwest, northeast, south", "not a real value for",
                        "valid values are"]],
     "note": "'Southeast' isn't a real region in this data (only Midwest/"
             "Northeast/South/West exist) - tightened 2026-07-22: this "
             "used to just check the word 'southeast' appeared anywhere, "
             "which trivially passed even when the agent silently queried "
             "8 individual states as its own stand-in with no disclosure. "
             "Now checks the answer actually SAYS it isn't one of the 4 "
             "real regions - see the VALID CATEGORIES addition in "
             "agent.py's SYSTEM_PROMPT"},
    {"group": "EDGE CASES - not-found, nonsense input & synonym mapping",
     "q": "Which HCPs have a Low tier?",
     "cat": "tier synonym mapping",
     "expect_tools": ["query_hcp_table"],
     "expect_facts": [["watch"], ["npi 1", "propensity"]],
     "note": "'Low' isn't a real tier value - should map to 'Watch' (the "
             "real bottom tier) and return ACTUAL matching HCPs, not just "
             "an explanation that Low doesn't exist. Tightened 2026-07-22: "
             "this used to only check the word 'watch' appeared, which "
             "trivially passed even when the agent just explained the "
             "tier system without returning any data. Now also checks a "
             "real NPI/propensity value is present, confirming actual "
             "rows came back - see the VALID CATEGORIES addition in "
             "agent.py's SYSTEM_PROMPT"},
    {"group": "EDGE CASES - not-found, nonsense input & synonym mapping",
     "q": "Which HCPs have days since contact over 90?",
     "cat": "generic numeric threshold",
     "expect_tools": ["query_hcp_table"], "expect_facts": [],
     "note": "exercises the ascending/extra_filters fix from 2026-07-22 "
             "reconciliation - expect_facts intentionally empty since "
             "the exact row count depends on live data, this only "
             "checks the right tool was called"},
]

# Conversation-memory scenario, run as a sequence in ONE agent session:
# the second question is unanswerable without remembering the first.
MEMORY_SCENARIO = [
    {"q": "Give me a summary of NPI 1344001929.", "expect_facts": ["1344001929"]},
    {"q": "Is this doctor a high or low prescriber for their specialty?",
     "expect_facts": [["1344001929", "endocrinolog", "high"]],
     "note": "must resolve 'this doctor' from conversation memory"},
]


def _contains(haystack, expected):
    """expected: string, or list of acceptable alternatives."""
    alts = expected if isinstance(expected, list) else [expected]
    return any(a.lower() in haystack for a in alts)


def _evidence_text(result):
    """Full tool results (not just the one-line digests) - a fact check
    against a digest would miss facts that were in fact retrieved."""
    return json.dumps(result["evidence"], default=str).lower()


_BAR = "=" * 72
_THIN = "-" * 72


def _print_group_banner(group_name):
    """Printed once, the first time a question from a new group appears
    - same convention as test_the_system.py, so a run of either file
    reads as matching, clearly labelled sections."""
    print("\n" + _BAR)
    print(f"  {group_name}")
    print(_BAR)


def _print_result(t, status, tools_used, steps, verdict, revised, note, answer_text, missing_info=None):
    """One question, one fully enclosed box - same visual convention as
    test_the_system.py. Tool-call lines are printed HERE, inside the
    box, reconstructed from res['steps'] - not left to agent.py's own
    live logging (self._log), which prints them immediately as they
    happen and would otherwise dangle above the header line, outside
    any box, before the question is even shown."""
    print(f"\n{_THIN}")
    print(f"{t['q']} - {status}")
    print(_THIN)
    for s in steps:
        print(f"  -> tool: {s['tool']}({json.dumps(s['input'])[:120]})")
    print(f"  TOOLS    : {tools_used}")
    if verdict is not None:
        print(f"  VERIFY   : {verdict}" + ("  (auto-revised)" if revised else ""))
    if note:
        print(f"  NOTE     : {note}")
    if missing_info:
        print(f"  MISSING  : {missing_info}")
    if answer_text is not None:
        print(f"  ANSWER   :")
        for line in str(answer_text).splitlines():
            print(f"    {line}")
    print(_THIN)


def run():
    agent = RabivyAgent()
    is_mock = isinstance(agent.llm, MockLLM)
    mode = "MOCK (behavioural checks)" if is_mock else "REAL (answer + verification checks)"
    print(f"\nAgent eval mode: {mode}")

    tally = {"PASS": 0, "WEAK": 0, "FAIL": 0, "CRASH": 0, "N/A IN MOCK": 0}
    records = []
    last_group = None

    for t in TESTS:
        group = t.get("group")
        if group and group != last_group:
            _print_group_banner(group)
            last_group = group

        # Fresh session per question - memory is tested separately below.
        agent_q = RabivyAgent(llm=agent.llm)
        agent_q._log = lambda *a, **k: None  # silence live printing - we
                                              # reconstruct tool-call lines
                                              # ourselves, inside the box,
                                              # from res["steps"] instead
        try:
            res = agent_q.ask(t["q"])
        except Exception as e:
            tally["CRASH"] += 1
            _print_result(t, "CRASH", [], [], None, False, t.get("note"),
                          f"!! {type(e).__name__}: {e}")
            records.append({**t, "status": "CRASH", "error": str(e)})
            continue

        tools_used = [s["tool"] for s in res["steps"]]
        expected_tools = t["expect_tools"]
        # Reconciled 2026-07-23: expect_tools=[] means "the correct
        # behaviour is asking for clarification, not calling a tool" -
        # but that requires actually understanding the question is
        # ambiguous, which MockLLM's simple keyword-based planner has no
        # way to do. It always calls SOME tool (whichever keyword signal
        # matched, or search_documents as the fallback) - it can never
        # genuinely produce "no tool call" in mock mode. So in mock
        # mode, these 2 tests are marked informational (N/A), not scored
        # as FAIL - same treatment MEMORY_SCENARIO's second step already
        # gets for the identical reason. Real mode (actual Claude) can
        # and should still be held to this expectation properly.
        if is_mock and not expected_tools:
            tally["N/A IN MOCK"] += 1
            _print_result(t, "N/A IN MOCK", tools_used, res["steps"], None, False,
                          (t.get("note") or "") + " | mock mode can't recognise "
                          "ambiguity without calling a tool - needs a real LLM "
                          "to test this properly",
                          _evidence_text(res)[:800])
            records.append({**t, "status": "N/A IN MOCK", "tools_used": tools_used})
            continue
        # Empty expect_tools means NO tool call is the correct behaviour
        # (the two unresolved-referent questions) - right_tools is True
        # only if tools_used is ALSO empty in that case.
        if not expected_tools:
            right_tools = not tools_used
        else:
            right_tools = all(tool in tools_used for tool in expected_tools)

        # Mock mode judges the evidence; real mode judges the final answer.
        judged_text = (_evidence_text(res) if is_mock
                       else (res["answer"].lower() + " " + _evidence_text(res)))
        facts_found = all(_contains(judged_text, f) for f in t["expect_facts"])

        if right_tools and facts_found:
            status = "PASS"
        elif right_tools:
            status = "WEAK"
        else:
            status = "FAIL"
        tally[status] += 1

        verdict = res["verification"].get("verdict", "-") if not is_mock else None
        missing_info = None
        if status != "PASS":
            missing_tools = [tool for tool in expected_tools if tool not in tools_used]
            missing_facts = [str(f) for f in t["expect_facts"] if not _contains(judged_text, f)]
            parts = []
            if missing_tools: parts.append(f"tools: {missing_tools}")
            if missing_facts: parts.append(f"facts: {missing_facts}")
            missing_info = " | ".join(parts) if parts else None

        answer_preview = res["answer"] if not is_mock else _evidence_text(res)[:800]
        _print_result(t, status, tools_used, res["steps"], verdict, res.get("revised", False),
                      t.get("note"), answer_preview, missing_info)
        records.append({**t, "status": status, "tools_used": tools_used,
                        "verification": res["verification"],
                        "answer": res["answer"][:1500]})

    # ---- conversation-memory scenario (one shared session) ----
    _print_group_banner("CONVERSATION MEMORY SCENARIO (one session, sequential)")
    mem_agent = RabivyAgent(llm=agent.llm)
    mem_agent._log = lambda *a, **k: None
    mem_status = "PASS"
    for step in MEMORY_SCENARIO:
        try:
            res = mem_agent.ask(step["q"])
        except Exception as e:
            mem_status = "CRASH"
            print(f"\n{_THIN}\n{step['q']} - CRASH\n{_THIN}\n  ERROR: {e}\n{_THIN}")
            break
        judged = res["answer"].lower() + " " + _evidence_text(res)
        ok = all(_contains(judged, f) for f in step["expect_facts"])
        # In mock mode the second hop legitimately can't resolve "this
        # doctor" (no LLM reading the history), so mark informational.
        label = "PASS" if ok else ("N/A IN MOCK" if is_mock else "FAIL")
        if not ok and not is_mock:
            mem_status = "FAIL"
        _print_result(step, label, [s["tool"] for s in res["steps"]], res["steps"], None, False,
                      step.get("note"), res["answer"] if not is_mock else _evidence_text(res)[:800])
    print(f"\n  memory scenario overall: {mem_status}")

    # ---- summary ----
    total = len(TESTS)
    print("\n" + _BAR)
    print("AGENT EVAL SUMMARY")
    print(_BAR)
    print(f"  PASS  (right tools, right facts)   : {tally['PASS']} / {total}")
    print(f"  WEAK  (right tools, fact missing)  : {tally['WEAK']} / {total}")
    print(f"  FAIL  (wrong tool selection)       : {tally['FAIL']} / {total}")
    print(f"  CRASH                              : {tally['CRASH']} / {total}")
    if tally["N/A IN MOCK"]:
        print(f"  N/A IN MOCK (needs real LLM)       : {tally['N/A IN MOCK']} / {total}")
    print(f"  memory scenario                    : {mem_status}")
    if is_mock:
        print("\n  NOTE: mock mode checks agent behaviour (tool selection +")
        print("  retrieval), not prose. Run with ANTHROPIC_API_KEY set for")
        print("  full answer-quality + verification evaluation.")

    os.makedirs("eval_runs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("eval_runs", f"agent_eval_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "mode": mode, "tally": tally,
                   "memory_scenario": mem_status, "results": records},
                  f, indent=2, default=str)
    print(f"\nSaved this run to {path}")
    return tally


if __name__ == "__main__":
    tally = run()
    # Proper exit code so CI (and any script) can gate on this eval:
    # FAIL or CRASH -> nonzero. WEAK is reported but tolerated - it
    # usually reflects retrieval-quality variance, not broken machinery.
    sys.exit(1 if (tally.get("FAIL", 0) + tally.get("CRASH", 0)) > 0 else 0)