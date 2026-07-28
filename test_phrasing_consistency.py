# test_phrasing_consistency.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# Checks PHRASING ROBUSTNESS - does the agent give the same CORRECT
# answer regardless of how a question is worded? test_the_agent.py
# checks "did something break" on 30 FIXED wordings; this file checks
# the harder, different question: does a rephrased version of a
# question already known to work still pass the SAME checks?
#
# ZERO DUPLICATED LOGIC: this file does not redefine what "correct"
# means anywhere. It imports the real QUESTIONS list and the real
# _layer1/_layer2/_layer3 functions straight from test_the_agent.py,
# and runs each ANCHOR + its 2 variant phrasings through those exact
# same checks. If test_the_agent.py's checks improve (as they just
# did - live ground truth, tighter facts, no more npi/medicaid
# leakage), this file gets that improvement for free, automatically,
# with no separate maintenance.
#
# THE CANONICAL WORDING IS RE-RUN HERE TOO (not reused from an old
# test_the_agent.py run) - "consistent across phrasings" is a claim
# about all 3 wordings behaving the same way IN THIS SESSION, and
# ground truth is computed live at test time, so an old cached result
# could be comparing against stale ground truth. All 3 wordings
# (original + 2 variants) are always run fresh, every time.
#
# 10 ANCHORS (not all 30), 2 variants each = 30 real agent calls, not
# 90 - kept affordable. Spans all 6 groups. Chosen to include Q28
# (Southeast region) deliberately, since that question's correct
# behaviour just changed (2026-07-28 system prompt fix) and is the
# most valuable one to phrasing-stress right now.
#
# CONSISTENT means: all 3 wordings independently PASS (or legitimately
# SKIP) on EVERY layer - not just "the 3 wordings agree with each
# other". Because the per-layer checks are absolute (live ground
# truth), this is a stronger claim than simple agreement: it means all
# 3 phrasings are actually correct, not just consistently wrong the
# same way.
#
# MOCK mode: MockLLM's keyword planner doesn't meaningfully vary
# tool-calling behaviour with phrasing the way a real model does -
# this file's real value is in REAL mode (ANTHROPIC_API_KEY set).
# =====================================================================

import json
import os
import time

from agent import RabivyAgent
from llm_client import MockLLM
import test_the_agent as core
import ground_truth as gt


def _find_question(q_text):
    for t in core.QUESTIONS:
        if t["q"] == q_text:
            return t
    raise KeyError(f"Question not found in test_the_agent.QUESTIONS: {q_text!r}")


# =====================================================================
# 10 ANCHORS - question text must match test_the_agent.QUESTIONS
# exactly (looked up via _find_question) so every check is reused,
# never redefined. Variants are genuinely different phrasings
# (structure/register/abbreviation), not synonym swaps.
# =====================================================================
ANCHOR_QUESTIONS = [
    "Who is the top GLP-1 prescriber in New York?",
    "How many active GLP-1 writers are in Texas?",
    "Show me High-propensity endocrinologists in Florida who are not currently targeted.",
    "Give me a summary of NPI 1344001929.",
    "Compare NPI 1344001929 to a typical endocrinologist.",
    "What is Rabivy's key differentiator versus Zepbound?",
    "How should a rep handle 'I'm already happy with Ozempic'?",
    "Who should I target next month in New York, and what should I say to them?",
    "Is this doctor a high or low prescriber for their specialty?",
    "Which HCPs are in the Southeast region?",
]

VARIANTS = {
    "Who is the top GLP-1 prescriber in New York?": [
        "Who's writing the most GLP-1 scripts in NY?",
        "Which doctor prescribes the most GLP-1 medications in New York state?",
    ],
    "How many active GLP-1 writers are in Texas?": [
        "How many prescribers in Texas are actively writing GLP-1s?",
        "What's the count of active GLP-1 writers in TX?",
    ],
    "Show me High-propensity endocrinologists in Florida who are not currently targeted.": [
        "Any good endo targets in FL we're not calling on?",
        "Which untargeted Florida endocrinologists have high propensity scores?",
    ],
    "Give me a summary of NPI 1344001929.": [
        "Pull up NPI 1344001929 for me.",
        "Tell me about the HCP with NPI 1344001929.",
    ],
    "Compare NPI 1344001929 to a typical endocrinologist.": [
        "How does NPI 1344001929 stack up against an average endo?",
        "Is NPI 1344001929 above or below typical for their specialty?",
    ],
    "What is Rabivy's key differentiator versus Zepbound?": [
        "Why should a doctor pick Rabivy over Zepbound?",
        "What makes Rabivy different from Zepbound?",
    ],
    "How should a rep handle 'I'm already happy with Ozempic'?": [
        "What do I say to someone who thinks their current GLP-1 is fine?",
        "A doctor says they're happy with Ozempic - how do I respond?",
    ],
    "Who should I target next month in New York, and what should I say to them?": [
        "Who should I see in New York this week and what's my pitch?",
        "Give me my New York call list for next month plus talking points.",
    ],
    "Is this doctor a high or low prescriber for their specialty?": [
        "Is he a high or low volume writer compared to peers?",
        "How does this doctor compare to others in the same specialty?",
    ],
    "Which HCPs are in the Southeast region?": [
        "Show me HCPs in the Southeastern US.",
        "Can I get a list of prescribers in the Southeast states?",
    ],
}


def _ok(status):
    """PASS or SKIP both count as 'not a problem' - only FAIL/CRASH
    break consistency. Matches the SKIP semantics already established
    in test_the_agent.py (a legitimate skip isn't a failure)."""
    return status in ("PASS", "SKIP")


_BAR = "=" * 72


def run():
    agent = RabivyAgent()
    is_mock = isinstance(agent.llm, MockLLM)
    mode = "MOCK (weak signal - see file header)" if is_mock else "REAL (the real check)"
    print(f"\nPhrasing consistency eval mode: {mode}")
    print(f"{len(ANCHOR_QUESTIONS)} anchors x 3 phrasings each (original + 2 variants) "
          f"= {len(ANCHOR_QUESTIONS) * 3} total agent calls")

    results = []
    consistent_count = 0
    last_group = None

    for i, q_text in enumerate(ANCHOR_QUESTIONS, start=1):
        t = _find_question(q_text)
        if t["group"] != last_group:
            print(f"\n{_BAR}\n  {t['group']}\n{_BAR}")
            last_group = t["group"]

        print(f"\n{'=' * 8} ANCHOR {i} {'=' * 8}")
        print(f"[{t['cat']}]")
        if t.get("note"):
            print(f"NOTE (from test_the_agent.py): {t['note']}")

        wordings = [q_text] + VARIANTS[q_text]
        per_wording = []

        for w in wordings:
            fresh = RabivyAgent(llm=agent.llm)
            fresh._log = lambda *a, **k: None
            try:
                result = fresh.ask(w)
            except Exception as e:
                print(f"  CRASH [{w!r}]: {type(e).__name__}: {e}")
                per_wording.append({"q": w, "l1": "CRASH", "l2": "CRASH", "l3": "CRASH"})
                continue

            haystack = core._haystack(result, is_mock)
            l1_status = "PASS" if core._layer1(t, result["evidence"]) else "FAIL"
            l2_status, _ = core._layer2(t, result["evidence"], haystack)
            l3_status, _ = core._layer3(t, haystack, result["evidence"])

            print(f"  [{w!r}]")
            print(f"    L1={l1_status}  L2={l2_status}  L3={l3_status}  "
                  f"(tools: {sorted(core._tools_called(result['evidence']))})")

            per_wording.append({"q": w, "l1": l1_status, "l2": l2_status, "l3": l3_status})

        layer_consistent = {
            "l1": all(_ok(pw["l1"]) for pw in per_wording),
            "l2": all(_ok(pw["l2"]) for pw in per_wording),
            "l3": all(_ok(pw["l3"]) for pw in per_wording),
        }
        anchor_ok = all(layer_consistent.values())

        verdict = "CONSISTENT" if anchor_ok else "INCONSISTENT"
        print(f"  -> {verdict}  (L1 consistent: {layer_consistent['l1']}, "
              f"L2: {layer_consistent['l2']}, L3: {layer_consistent['l3']})")
        if anchor_ok:
            consistent_count += 1

        results.append({"anchor": q_text, "group": t["group"], "consistent": anchor_ok,
                        "layer_consistent": layer_consistent, "wordings": per_wording})

    print(f"\n{_BAR}\nSUMMARY\n{_BAR}")
    print(f"{consistent_count} / {len(ANCHOR_QUESTIONS)} anchors CONSISTENT across all 3 phrasings, all 3 layers")
    if consistent_count < len(ANCHOR_QUESTIONS):
        print("An INCONSISTENT anchor means at least one rephrasing failed a layer that the")
        print("canonical wording passes - see the per-wording L1/L2/L3 lines above for which")
        print("layer and which specific phrasing tripped it.")
    if is_mock:
        print("\nNOTE: mock mode's signal is weak here - MockLLM's keyword planner doesn't")
        print("meaningfully vary tool selection with phrasing. Run with ANTHROPIC_API_KEY")
        print("set for the real check.")

    os.makedirs("eval_runs", exist_ok=True)
    path = os.path.join("eval_runs", f"phrasing_consistency_eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"consistent": consistent_count, "total": len(ANCHOR_QUESTIONS),
                   "results": results}, f, indent=2, default=str)
    print(f"\nSaved this run to {path}")
    return results


if __name__ == "__main__":
    run()
