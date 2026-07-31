# test_label_loop.py
# =====================================================================
# Unit tests for the eval label loop's PLUMBING - the parts that must
# be correct for any rating to mean anything: the kappa computation,
# the gold-answer planting, the mock judge's determinism, the vote
# supersede rule, and the report's decision matrix. Pure-code tests, no
# LLM calls, no retrieval - runs anywhere, free, in seconds.
#
# (The RATERS themselves are examined by the entrance exam and gold
# items at runtime - that's label_loop.py --exam, not this file.)
# =====================================================================

import json
import os
import sys
import tempfile

import label_loop as ll

RESULTS = {"PASS": 0, "FAIL": 0}


def check(name, ok, detail=""):
    RESULTS["PASS" if ok else "FAIL"] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


print("\n--- Cohen's kappa ---")
# Hand-computed example: 10 items, raters agree on 8.
# Rater A: 6x"x", 4x"y". Rater B: 6x"x", 4x"y".  po=0.8
# pe = 0.6*0.6 + 0.4*0.4 = 0.52 -> kappa = (0.8-0.52)/0.48 = 0.58333...
pairs = [("x", "x")] * 5 + [("y", "y")] * 3 + [("x", "y"), ("y", "x")]
k = ll.cohens_kappa(pairs)
check("hand-computed value", abs(k - 0.5833333) < 1e-6, f"got {k}")
check("perfect agreement -> 1.0", ll.cohens_kappa([("a", "a"), ("b", "b")]) == 1.0)
check("empty input -> None", ll.cohens_kappa([]) is None)
check("single shared category -> 1.0 (pe==1 guard)",
      ll.cohens_kappa([("a", "a"), ("a", "a")]) == 1.0)
# Kappa can be negative (worse than chance) - must not crash or clamp.
neg = ll.cohens_kappa([("a", "b"), ("b", "a")])
check("worse-than-chance is negative", neg is not None and neg < 0, f"got {neg}")

print("\n--- Stable candidate shuffling ---")
r1 = ll._stable_rng("some question")
r2 = ll._stable_rng("some question")
seq = list(range(10))
a, b = seq[:], seq[:]
r1.shuffle(a)
r2.shuffle(b)
check("same question -> same order (human and judge see identical layouts)", a == b)
r3 = ll._stable_rng("a different question")
c = seq[:]
r3.shuffle(c)
check("different question -> different order", a != c)

print("\n--- Mock judge ---")
cands = [
    {"chunk_id": "why_patients_discontinue", "snippet": "patients discontinue because of side effects and cost"},
    {"chunk_id": "unrelated_dosing", "snippet": "monthly dosing schedule for the product"},
]
v1 = ll.MockJudge().choose("Why do patients discontinue - side effects?", cands)
v2 = ll.MockJudge().choose("Why do patients discontinue - side effects?", cands)
check("deterministic across calls", v1 == v2)
check("picks the overlapping chunk", v1["choice"] == "why_patients_discontinue", str(v1))
v3 = ll.MockJudge().choose("zzz qqq xyzzy plugh", cands)
check("no overlap at all -> none-of-these", v3["choice"] == ll.NONE_OF_THESE, str(v3))

print("\n--- Judge JSON extraction ---")
check("plain object", ll._extract_json('{"choice": "abc", "why": "w"}')["choice"] == "abc")
check("commentary around object",
      ll._extract_json('Sure! Here: {"choice": "abc", "why": "w"} hope that helps')["choice"] == "abc")
check("garbage -> None", ll._extract_json("no json here at all") is None)

print("\n--- Vote supersede rule + report decision matrix (temp files) ---")
tmp = tempfile.mkdtemp()
orig_cases, orig_votes, orig_labels = ll.CASES_PATH, ll.VOTES_PATH, ll.LABELS_PATH
ll.CASES_PATH = os.path.join(tmp, "cases.jsonl")
ll.VOTES_PATH = os.path.join(tmp, "votes.jsonl")
ll.LABELS_PATH = os.path.join(tmp, "labels.jsonl")
try:
    cases = [
        # live, raters agree on a NEW chunk (differs from existing tag) -> proposal
        {"case_id": "live_a", "kind": "live", "narrative_key": "a", "question": "qa",
         "candidates": [{"chunk_id": "right_chunk", "snippet": ""},
                        {"chunk_id": "old_chunk", "snippet": ""}],
         "existing_tag": ["old_tag"], "retriever_rank_of_tag": 3},
        # live, raters disagree -> adjudication queue (never auto-resolved)
        {"case_id": "live_b", "kind": "live", "narrative_key": "b", "question": "qb",
         "candidates": [{"chunk_id": "c1", "snippet": ""}, {"chunk_id": "c2", "snippet": ""}],
         "existing_tag": ["c1"], "retriever_rank_of_tag": 2},
        # gold, judge correct, human wrong -> feeds gold accuracy only
        {"case_id": "gold_c", "kind": "gold", "narrative_key": "c", "question": "qc",
         "candidates": [{"chunk_id": "truth", "snippet": ""}, {"chunk_id": "decoy", "snippet": ""}],
         "gold_answer": "truth", "existing_tag": ["truth"]},
        # live, both say none-of-these -> retrieval hole, NOT a label
        {"case_id": "live_d", "kind": "live", "narrative_key": "d", "question": "qd",
         "candidates": [{"chunk_id": "c9", "snippet": ""}],
         "existing_tag": None, "retriever_rank_of_tag": None},
    ]
    os.makedirs(tmp, exist_ok=True)
    with open(ll.CASES_PATH, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    # Supersede: human first votes old_chunk on live_a, then corrects to right_chunk.
    ll.append_vote("live_a", "human", "old_chunk")
    ll.append_vote("live_a", "human", "right_chunk")
    ll.append_vote("live_a", "judge", "right_chunk", "matches", "test-model")
    ll.append_vote("live_b", "human", "c1")
    ll.append_vote("live_b", "judge", "c2", "disagrees", "test-model")
    ll.append_vote("gold_c", "human", "decoy")
    ll.append_vote("gold_c", "judge", "truth", "", "test-model")
    ll.append_vote("live_d", "human", ll.NONE_OF_THESE)
    ll.append_vote("live_d", "judge", ll.NONE_OF_THESE, "", "test-model")

    votes = ll.load_votes()
    check("latest vote supersedes", votes[("live_a", "human")]["choice"] == "right_chunk")

    ll.run_report()  # prints its own output; we verify the written labels file
    with open(ll.LABELS_PATH) as f:
        labels = {r["narrative_key"]: r for r in map(json.loads, f)}

    check("agreed live case -> final_label set", labels["a"]["final_label"] == "right_chunk")
    check("agreed live case keeps retriever rank for fine-tune set",
          labels["a"]["retriever_rank_of_label"] == 3)
    check("disagreement -> NO final label (adjudication, not auto-trust)",
          labels["b"]["final_label"] is None and labels["b"]["agreed"] is False)
    check("gold case flagged gold with planted answer",
          labels["c"]["gold"] is True and labels["c"]["gold_answer"] == "truth")
    check("none-of-these consensus -> label is 'none', a hole not a tag",
          labels["d"]["final_label"] == ll.NONE_OF_THESE)
finally:
    ll.CASES_PATH, ll.VOTES_PATH, ll.LABELS_PATH = orig_cases, orig_votes, orig_labels

print(f"\n{'=' * 40}\n{RESULTS['PASS']} PASS / {RESULTS['FAIL']} FAIL\n{'=' * 40}")
sys.exit(1 if RESULTS["FAIL"] else 0)
