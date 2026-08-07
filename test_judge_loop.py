# test_judge_loop.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# Free, no-API-key tests over judge_loop.py's pure logic: the mock
# judge backend and verdict parsing, the entrance exam's scoring/pass-
# bar math, the kappa comparison math, and - most important - a direct
# check that --build-gold's construction genuinely never mutates the
# real index files, not just a claim in a comment. Plus human_review_ui.py's
# vote-loading/supersede rule, since it's the other half of the same
# feature. Same spirit as label_loop.py's test_label_loop.py on
# eval-label-loop: pure code, no LLM, no real retrieval needed, runs
# free anywhere.
#
# Renamed from test_label_review.py on 2026-08-07 when the five
# judge_loop.py-related files merged into one - see judge_loop.py's own
# header for why.
# =====================================================================

import hashlib
import json
import os

from judge_loop import MockJudge, _parse_verdict, run_exam, GOLD_ITEMS_PATH, PASS_BAR


# =====================================================================
# The judge itself
# =====================================================================

def test_mock_judge_is_deterministic():
    j = MockJudge()
    a = j.judge("q", "side effects and cost", "the answer is side effects and cost")
    b = j.judge("q", "side effects and cost", "the answer is side effects and cost")
    assert a["verdict"] == b["verdict"]


def test_mock_judge_flags_unrelated_answer_as_unsupported():
    j = MockJudge()
    result = j.judge("q", "side effects and cost drive discontinuation",
                      "patients discontinue due to unrelated travel logistics")
    assert result["verdict"] == "unsupported"


def test_mock_judge_supports_overlapping_answer():
    j = MockJudge()
    result = j.judge("q", "discontinuation is driven by side effects and cost",
                      "discontinuation is driven by side effects and cost")
    assert result["verdict"] == "supported"


def test_parse_verdict_valid_json():
    parsed = _parse_verdict('{"verdict": "supported", "reasoning": "matches"}')
    assert parsed["verdict"] == "supported"
    assert parsed["reasoning"] == "matches"


def test_parse_verdict_malformed_reply_is_treated_as_unsupported():
    # A judge whose reply we can't even parse should never silently
    # count as a pass - erring toward "flag it," not toward "trust it."
    parsed = _parse_verdict("not valid json at all")
    assert parsed["verdict"] == "unsupported"


def test_parse_verdict_rejects_unexpected_verdict_value():
    parsed = _parse_verdict('{"verdict": "maybe", "reasoning": "unsure"}')
    assert parsed["verdict"] == "unsupported"


# =====================================================================
# --build-gold / --exam - scoring / pass-bar math
# =====================================================================

def test_exam_runs_end_to_end_in_mock_mode(monkeypatch):
    monkeypatch.setenv("JUDGE_BACKEND", "mock")
    if not os.path.exists(GOLD_ITEMS_PATH):
        import judge_loop
        judge_loop.cmd_build_gold()
    summary = run_exam()
    assert summary["total_items"] > 0
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert summary["passed"] == (summary["accuracy"] >= PASS_BAR)


def test_exam_fails_closed_with_zero_items(tmp_path, monkeypatch):
    empty_path = tmp_path / "empty_gold_items.jsonl"
    empty_path.write_text("")
    monkeypatch.setattr("judge_loop.GOLD_ITEMS_PATH", str(empty_path))
    # Also redirect where the result gets written - this test deliberately
    # produces a FAILING exam, and without this it would overwrite the
    # real output/LABEL_LOOP/exam_result.json with that failure, which
    # would incorrectly block --review from running for real (found by
    # hand: running this suite once did exactly that, before this fix).
    monkeypatch.setattr("judge_loop.EXAM_RESULT_PATH", str(tmp_path / "exam_result.json"))
    monkeypatch.setenv("JUDGE_BACKEND", "mock")
    summary = run_exam()
    # Zero items must never read as "passed" - an empty exam proves
    # nothing, and a silent pass here would let an unproven judge
    # straight through to live traffic.
    assert summary["passed"] is False


def test_gold_items_include_conversation_memory_case():
    # Added 2026-08-07 alongside the preceding_context feature - the
    # exam needs at least one case testing that the judge correctly
    # credits an answer grounded in PRIOR conversation, not just fresh
    # evidence, or the exam isn't testing the new capability at all.
    import judge_loop
    items = judge_loop.build_gold_items()
    memory_items = [i for i in items if i["method"] == "conversation_memory"]
    assert len(memory_items) > 0
    assert all(i["true_verdict"] == "supported" for i in memory_items)


# =====================================================================
# --build-gold - the safety guarantee itself
# =====================================================================

def _hash_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_gold_item_construction_never_mutates_the_real_index():
    chunks_path = os.path.join("output", "chunks_tagged.json")
    embeddings_path = os.path.join("output", "embeddings.npy")
    before_chunks = _hash_file(chunks_path)
    before_embeddings = _hash_file(embeddings_path)

    import judge_loop
    judge_loop.build_gold_items()  # the actual construction logic, not just cmd_build_gold()'s I/O

    after_chunks = _hash_file(chunks_path)
    after_embeddings = _hash_file(embeddings_path)
    assert before_chunks == after_chunks, "chunks_tagged.json changed - this must never happen"
    assert before_embeddings == after_embeddings, "embeddings.npy changed - this must never happen"


# =====================================================================
# human_review_ui.py - vote loading / supersede rule
# =====================================================================

def test_human_votes_keep_latest_per_query_id(tmp_path, monkeypatch):
    import human_review_ui as hui
    votes_path = tmp_path / "human_votes.jsonl"
    with open(votes_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": "q1", "rater": "human", "verdict": "supported",
                             "voted_at": "t1"}) + "\n")
        f.write(json.dumps({"query_id": "q1", "rater": "human", "verdict": "unsupported",
                             "voted_at": "t2"}) + "\n")  # re-rated, should win
        f.write(json.dumps({"query_id": "q2", "rater": "human", "verdict": "supported",
                             "voted_at": "t3"}) + "\n")
    monkeypatch.setattr(hui, "VOTES_PATH", str(votes_path))
    latest = hui.load_human_votes()
    assert latest == {"q1": "unsupported", "q2": "supported"}


def test_pending_cases_excludes_already_voted(tmp_path, monkeypatch):
    import human_review_ui as hui
    cases_path = tmp_path / "judge_review.jsonl"
    votes_path = tmp_path / "human_votes.jsonl"
    with open(cases_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": "q1", "question": "a", "evidence_text": "e",
                             "given_answer": "ans"}) + "\n")
        f.write(json.dumps({"query_id": "q2", "question": "b", "evidence_text": "e",
                             "given_answer": "ans"}) + "\n")
    with open(votes_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": "q1", "rater": "human", "verdict": "supported",
                             "voted_at": "t1"}) + "\n")
    monkeypatch.setattr(hui, "CASES_PATH", str(cases_path))
    monkeypatch.setattr(hui, "VOTES_PATH", str(votes_path))
    cases, pending = hui.pending_cases()
    assert len(cases) == 2
    assert [c["query_id"] for c in pending] == ["q2"]


# =====================================================================
# --report - the comparison math itself
# =====================================================================

def test_kappa_is_one_for_perfect_agreement():
    from judge_loop import cohens_kappa
    pairs = [("supported", "supported"), ("unsupported", "unsupported"),
             ("supported", "supported"), ("unsupported", "unsupported")]
    assert cohens_kappa(pairs) == 1.0


def test_kappa_is_zero_for_chance_level_agreement():
    from judge_loop import cohens_kappa
    # Both raters split 50/50 independently of each other - textbook
    # zero-kappa construction: agreement matches what the marginals alone predict.
    pairs = [("supported", "supported"), ("supported", "unsupported"),
             ("unsupported", "supported"), ("unsupported", "unsupported")]
    assert cohens_kappa(pairs) == 0.0


def test_kappa_none_when_no_shared_cases():
    from judge_loop import cohens_kappa
    assert cohens_kappa([]) is None


def test_report_only_scores_cases_both_raters_touched(monkeypatch):
    import judge_loop as jl
    import human_review_ui as hui

    # build_kappa_report() imports load_human_votes INSIDE the function
    # body (not at module top), so patching the SOURCE module
    # (human_review_ui.load_human_votes) is picked up fresh on each
    # call - unlike a top-of-file `from X import Y`, which binds a
    # reference once at import time and would need the caller's own
    # copy patched instead. Worth the comment since this exact gotcha
    # bit the old report_kappa.py version of this test.
    fake_judge = {"q1": "supported", "q2": "unsupported", "q3": "supported"}
    fake_human = {"q1": "supported", "q2": "supported"}  # q3 not yet reviewed by a human

    monkeypatch.setattr(jl, "_load_judge_verdicts", lambda: fake_judge)
    monkeypatch.setattr(hui, "load_human_votes", lambda: fake_human)

    report = jl.build_kappa_report()

    assert report["compared_n"] == 2
    assert report["judge_only_ids"] == ["q3"]
    assert len(report["disagreements"]) == 1  # q2: judge said unsupported, human said supported


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
