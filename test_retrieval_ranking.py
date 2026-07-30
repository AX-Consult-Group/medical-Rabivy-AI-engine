# test_retrieval_ranking.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# Checks RETRIEVAL RANKING QUALITY - not just "was the correct chunk
# retrieved somewhere in the results" (already checked by test_the_
# system.py / test_the_agent.py's Layer 2), but "was it ranked FIRST,
# not buried behind a worse match".
#
# WHY THIS IS A SEPARATE FILE, NOT MORE FIELDS BOLTED ONTO THE OTHER
# TWO: search_documents.semantic_search() is pure embedding similarity
# - local computation, no LLM call, no API cost. This file calls it
# DIRECTLY, bypassing the agent/LLM entirely - it tests the SEARCH
# ENGINE itself, not the LLM's judgment about when/how to call it
# (that part is already covered by the existing Layer 2 checks). That
# means this file is free to run, any time, independently of the other
# two - no shared run needed, no token cost either way.
#
# WHY RANK MATTERS, NOT JUST PRESENCE: the LLM reads retrieved results
# in order and weighs the top one most heavily. A chunk that's
# technically "in the results" but ranked 4th is a meaningfully worse
# outcome than the same chunk ranked 1st - and the existing Layer 2
# checks (any_present across all returned sections) can't tell the two
# apart. This file can.
#
# REUSES ground_truth.NARRATIVE_FACTS directly - the same tag list
# already verified against real chunk content earlier in this project.
# No new facts/tags defined here.

# =====================================================================
# CONFIRMED FINDING (2026-07-28, real embeddings, real run): 4 of 10
# narrative questions did not rank their correct chunk first. One
# ("Why do patients stop taking GLP-1s after a year?") missed
# retrieving it entirely, even at top_k=6.
#
# This is what this file exists to catch - test_the_system.py and
# test_the_agent.py's Layer 2 checks only verify presence, not rank,
# and (before a fix made the same day) had a tag-matching bug that
# masked this specific miss as a false PASS.
#
# Not fixed further - a genuine all-MiniLM-L6-v2 embedding-model
# limitation.
# =====================================================================

import json
import os
import time

import search_documents as sd
import ground_truth as gt

# The canonical question wording for each narrative question, keyed the
# same way as ground_truth.NARRATIVE_FACTS - reused from test_the_system.py
# / test_the_agent.py so there's exactly one place these questions live.
NARRATIVE_QUESTIONS = {
    "competitive_switchers": "What's our recommended messaging for competitive switchers?",
    "zepbound_differentiator": "What is Rabivy's key differentiator versus Zepbound?",
    "ozempic_objection": "How should a rep handle 'I'm already happy with Ozempic'?",
    "medicaid_access": "What's the Medicaid coverage outlook for GLP-1 obesity drugs?",
    "missouri_market": "What does the GLP-1 market look like in Missouri?",
    "tirzepatide_mechanism": "How is Rabivy's mechanism different from tirzepatide?",
    "dosing_advantage": "What is Rabivy's main dosing advantage?",
    "typical_endocrinologist": "What does a typical endocrinologist look like?",
    "prior_auth_access": "How is prior authorization affecting access?",
    "discontinuation": "Why do patients stop taking GLP-1s after a year?",
}

TOP_K = 5  # how many results to inspect per question - matches the agent's default top_k


def _rank_of_correct_chunk(results, expected_tags):
    """Returns the 1-indexed rank of the first result whose chunk_id
    matches any of the expected tags, or None if it's not in the
    returned results at all."""
    for i, r in enumerate(results, start=1):
        chunk_id = r["chunk"].get("chunk_id", "")
        if gt.any_present(chunk_id.lower(), expected_tags):
            return i
    return None


_BAR = "=" * 72


def run():
    print(f"\n{_BAR}\nRETRIEVAL RANKING QUALITY\n{_BAR}")
    print(f"Checking {len(NARRATIVE_QUESTIONS)} narrative questions - is the correct chunk")
    print(f"ranked #1, or merely present somewhere in the top {TOP_K} results?\n")

    results_log = []
    rank1_count = 0
    present_not_rank1_count = 0
    missing_count = 0

    for key, question in NARRATIVE_QUESTIONS.items():
        spec = gt.NARRATIVE_FACTS[key]
        if spec["tag"] is None:
            print(f"[{key}] SKIP - no verified tag yet (see ground_truth.py)")
            continue

        data = sd.semantic_search(question, top_k=TOP_K)
        if not data.get("found", True) or not data.get("results"):
            print(f"[{key}] NO RESULTS RETURNED - {data.get('error', 'unknown reason')}")
            results_log.append({"question": question, "key": key, "rank": None, "status": "NO_RESULTS"})
            missing_count += 1
            continue

        rank = _rank_of_correct_chunk(data["results"], spec["tag"])
        top_chunk_id = data["results"][0]["chunk"].get("chunk_id", "?")
        top_score = data["results"][0]["score"]

        if rank == 1:
            status = "RANK 1 (best outcome)"
            rank1_count += 1
        elif rank is not None:
            status = f"PRESENT BUT RANKED #{rank} (not first)"
            present_not_rank1_count += 1
        else:
            status = f"NOT IN TOP {TOP_K} AT ALL"
            missing_count += 1

        print(f"[{key}]")
        print(f"  Q: {question}")
        print(f"  Expected tag(s): {spec['tag']}")
        print(f"  Top result: {top_chunk_id}  (score {top_score:.3f})")
        print(f"  -> {status}")

        results_log.append({"question": question, "key": key, "rank": rank,
                            "top_chunk_id": top_chunk_id, "top_score": top_score, "status": status})

    total_checked = rank1_count + present_not_rank1_count + missing_count
    print(f"\n{_BAR}\nSUMMARY\n{_BAR}")
    print(f"Ranked #1 (best outcome)        : {rank1_count} / {total_checked}")
    print(f"Present but NOT ranked #1        : {present_not_rank1_count} / {total_checked}")
    print(f"Not in top {TOP_K} results at all      : {missing_count} / {total_checked}")
    print(f"\nA question in the middle category (present but not #1) is NOT currently")
    print(f"caught by test_the_system.py / test_the_agent.py's Layer 2 checks - those")
    print(f"only check presence, not rank. This file is the only place that distinction shows up.")

    os.makedirs("eval_runs", exist_ok=True)
    path = os.path.join("eval_runs", f"retrieval_ranking_eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"rank1_count": rank1_count, "present_not_rank1_count": present_not_rank1_count,
                   "missing_count": missing_count, "total": total_checked, "results": results_log},
                  f, indent=2, default=str)
    print(f"\nSaved this run to {path}")
    return results_log


if __name__ == "__main__":
    run()
