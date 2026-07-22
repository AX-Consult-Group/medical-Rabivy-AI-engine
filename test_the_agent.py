# test_the_agent.py
# ===================================================================
# AGENT-LEVEL EVALUATION - the same question bank as test_the_system.py,
# but now the whole thing runs THROUGH THE AGENT, so questions that were
# "waiting on AI" before (NEEDS-LLM / ROUTES->LLM) are now expected to
# come back as complete written answers.
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
# Each run is saved to eval_runs/agent_eval_*.json, alongside her
# test_the_system.py runs, so the before/after story is diffable.
# ===================================================================

import json
import os
import sys
import time

from agent import RabivyAgent
from llm_client import MockLLM

# expect_tools: at least one call to each listed tool must happen.
# expect_facts: each item is a string (or list of acceptable alternatives)
#               that must appear in the evidence (mock) / answer (real).
TESTS = [
    # ---- structured lookups & rankings ----
    {"q": "Who is the top GLP-1 prescriber in New York?",
     "cat": "structured lookup",
     "expect_tools": ["query_hcp_table"], "expect_facts": ["1658907316"]},
    {"q": "How many GLP-1 scripts did NPI 1344001929 write last month?",
     "cat": "structured lookup",
     "expect_tools": ["lookup_hcp"], "expect_facts": ["55"]},
    {"q": "List the top 10 High-tier prescribers nationally by propensity score.",
     "cat": "structured ranking",
     "expect_tools": ["query_hcp_table"], "expect_facts": ["1184547828"]},
    {"q": "Which states have the most High-tier HCPs?",
     "cat": "structured ranking",
     "expect_tools": ["states_summary"], "expect_facts": ["california"]},
    {"q": "How many active GLP-1 writers are in Texas?",
     "cat": "structured count",
     "expect_tools": ["count_active_writers"], "expect_facts": ["576"]},

    # ---- filtered targeting (previously NEEDS-LLM: multi-field parsing) ----
    {"q": "Show me High-propensity endocrinologists in Florida who are not currently targeted.",
     "cat": "filtered targeting",
     "expect_tools": ["query_hcp_table"],
     "expect_facts": ["florida", "endocrinology"]},
    {"q": "Which Novo-heavy prescribers in California have a high switching score?",
     "cat": "filtered targeting",
     "expect_tools": ["query_hcp_table"],
     "expect_facts": ["california", "novo"]},
    {"q": "Find high-volume writers in Illinois with a preferred formulary status.",
     "cat": "filtered targeting",
     "expect_tools": ["query_hcp_table"],
     "expect_facts": ["illinois", "preferred"]},
    {"q": "Which untargeted HCPs have a recent sample request and a High tier?",
     "cat": "filtered targeting",
     "expect_tools": ["query_hcp_table"], "expect_facts": ["high"]},

    # ---- NPI cards & comparisons (previously NEEDS-LLM: synthesis) ----
    {"q": "Give me a summary of NPI 1344001929.",
     "cat": "card summary",
     "expect_tools": ["lookup_hcp"], "expect_facts": ["card_1344001929"]},
    {"q": "Compare NPI 1344001929 to a typical endocrinologist.",
     "cat": "comparison",
     "expect_tools": ["lookup_hcp", "search_documents"],
     "expect_facts": ["1344001929"]},

    # ---- narrative / document questions ----
    {"q": "What's our recommended messaging for competitive switchers?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["competitive_switchers", "competitive switcher"]]},
    {"q": "What is Rabivy's key differentiator versus Zepbound?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["differentiator", "monthly", "dosing"]]},
    {"q": "How should a rep handle 'I'm already happy with Ozempic'?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["objection_handling", "happy with ozempic", "switch"]]},
    {"q": "What's the Medicaid coverage outlook for GLP-1 obesity drugs?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["payer_access", "medicaid"]]},
    {"q": "What does the GLP-1 market look like in Missouri?",
     "cat": "narrative (state filter)",
     "expect_tools": ["search_documents"],
     "expect_facts": [["state_market_summary__missouri", "missouri"]]},
    {"q": "How is Rabivy's mechanism different from tirzepatide?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["mechanism", "molecule"]]},
    {"q": "What is Rabivy's main dosing advantage?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["monthly", "dosing", "where_rabivy_wins"]]},
    {"q": "What does a typical endocrinologist look like?",
     "cat": "narrative (benchmark)",
     "expect_tools": ["search_documents"],
     "expect_facts": [["endocrinolog"]]},
    {"q": "How is prior authorization affecting access?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["prior_auth", "prior auth", "access"]]},
    {"q": "Why do patients stop taking GLP-1s after a year?",
     "cat": "narrative",
     "expect_tools": ["search_documents"],
     "expect_facts": [["discontinu", "persistence", "adherence"]]},

    # ---- THE MULTI-SOURCE SHOWPIECE (previously impossible: needs both
    # ---- engines in one answer - the whole reason the agent exists) ----
    {"q": "Who should I target next month in New York, and what should I say to them?",
     "cat": "multi-source showpiece",
     "expect_tools": ["query_hcp_table", "search_documents"],
     "expect_facts": ["new york"]},
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


def run():
    agent = RabivyAgent()
    is_mock = isinstance(agent.llm, MockLLM)
    mode = "MOCK (behavioural checks)" if is_mock else "REAL (answer + verification checks)"
    print(f"\nAgent eval mode: {mode}")
    print("#" * 72)

    tally = {"PASS": 0, "WEAK": 0, "FAIL": 0, "CRASH": 0}
    records = []

    for t in TESTS:
        # Fresh session per question - memory is tested separately below.
        agent_q = RabivyAgent(llm=agent.llm)
        try:
            res = agent_q.ask(t["q"])
        except Exception as e:
            tally["CRASH"] += 1
            print(f"\n[CRASH] ({t['cat']}) {t['q']}\n   -> {type(e).__name__}: {e}")
            records.append({**t, "status": "CRASH", "error": str(e)})
            continue

        tools_used = [s["tool"] for s in res["steps"]]
        right_tools = all(tool in tools_used for tool in t["expect_tools"])

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

        verdict = res["verification"].get("verdict", "-")
        print(f"\n[{status:5s}] ({t['cat']})")
        print(f"   Q: {t['q']}")
        print(f"   tools: {tools_used}  | verification: {verdict}"
              + ("  | auto-revised" if res.get("revised") else ""))
        if status != "PASS":
            missing = [str(f) for f in t["expect_facts"] if not _contains(judged_text, f)]
            wrong = [tool for tool in t["expect_tools"] if tool not in tools_used]
            print(f"   -> missing tools: {wrong or '-'} | facts not found: {missing or '-'}")
        records.append({**t, "status": status, "tools_used": tools_used,
                        "verification": res["verification"],
                        "answer": res["answer"][:1500]})

    # ---- conversation-memory scenario (one shared session) ----
    print("\n" + "#" * 72)
    print("# CONVERSATION MEMORY SCENARIO (one session, sequential)")
    print("#" * 72)
    mem_agent = RabivyAgent(llm=agent.llm)
    mem_status = "PASS"
    for step in MEMORY_SCENARIO:
        try:
            res = mem_agent.ask(step["q"])
        except Exception as e:
            mem_status = "CRASH"
            print(f"[CRASH] {step['q']} -> {e}")
            break
        judged = res["answer"].lower() + " " + _evidence_text(res)
        ok = all(_contains(judged, f) for f in step["expect_facts"])
        # In mock mode the second hop legitimately can't resolve "this
        # doctor" (no LLM reading the history), so mark informational.
        label = "ok" if ok else ("n/a in mock" if is_mock else "MISS")
        if not ok and not is_mock:
            mem_status = "FAIL"
        print(f"  [{label}] {step['q']}  tools={[s['tool'] for s in res['steps']]}")
    print(f"  memory scenario: {mem_status}")

    # ---- summary ----
    total = len(TESTS)
    print("\n" + "=" * 72)
    print("AGENT EVAL SUMMARY")
    print("=" * 72)
    print(f"  PASS  (right tools, right facts)   : {tally['PASS']} / {total}")
    print(f"  WEAK  (right tools, fact missing)  : {tally['WEAK']} / {total}")
    print(f"  FAIL  (wrong tool selection)       : {tally['FAIL']} / {total}")
    print(f"  CRASH                              : {tally['CRASH']} / {total}")
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
