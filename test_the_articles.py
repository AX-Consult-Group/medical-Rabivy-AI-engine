# test_the_articles.py
# =====================================================================
# AGENT-LEVEL EVALUATION for the real-article questions in
# article_ground_truth.py - same 3-layer design AND same matching
# METHOD as test_the_agent.py:
#
#   LAYER 1 - ROUTING     Did the agent call search_documents at all?
#   LAYER 2 - RETRIEVAL   Is the correct chunk_id present ANYWHERE in
#                         this turn's evidence, checked the same way
#                         test_the_agent.py's _layer2/_haystack do it -
#                         json.dumps() the whole evidence list, lower-
#                         case it, and substring-search for the
#                         chunk_id/tag. Deliberately NOT structural
#                         parsing of the tool result's exact shape.
#   LAYER 3 - ANSWER      Does the final answer contain the real facts?
#
# WHY THE HAYSTACK APPROACH, NOT STRUCTURAL PARSING (fixed after a
# real bug): an earlier draft of this file hand-parsed
# evidence[i]["result"]["sections"][j]["chunk_id"], assuming a specific
# tool-result shape. That assumption was WRONG - agent_tools.py's
# _search_documents() reshapes semantic_search()'s raw output
# ("results"/"chunk") into a different shape ("sections", chunk_id
# flattened out) before the LLM ever sees it, and the hand-parsed
# version silently found 0 chunks on every single question as a
# result. The REAL test_the_agent.py never has this problem because it
# never assumes a shape at all - it just stringifies everything and
# does a substring search, which works regardless of exactly how any
# given tool happens to nest its result. Copied that exact method here
# instead of re-guessing.
#
# WHY THIS FILE EXISTS, GIVEN test_article_retrieval.py ALREADY CHECKS
# RETRIEVAL: that file tests the search engine in isolation - useful,
# free, fast, but it can't tell you whether the LLM found a correct
# answer DESPITE imperfect ranking (exactly the "discontinuation"
# pattern from the eval-label-loop thread: bad retrieval, fine answer,
# no real problem for the rep). This file is what actually fills in
# Jürgen's matrix cells with real evidence instead of hypothetical
# broken cases.
#
# IMPORTANT CAVEAT UNIQUE TO REAL PAPERS (not an issue with the
# original simulated docs): these are real, published papers Claude
# may already know facts from via pretraining. A "Layer 2 FAIL, Layer
# 3 PASS" here is genuinely ambiguous in a way it wasn't before - it
# could mean "retrieval was bad but the LLM synthesized correctly from
# whatever it DID get" (the good outcome), or it could mean "the LLM
# already knew this fact from training data and never needed retrieval
# at all" (tells you nothing about your pipeline). If this fires on
# chen_pooled_or or chen_sensitivity_or (the two confirmed retrieval
# MISSES in test_article_retrieval.py), it's worth re-asking the same
# question in a fresh, tool-less conversation to rule out pretraining
# recall before trusting the result as evidence your system worked.
# =====================================================================

import json
import os
import sys
import time

from agent import RabivyAgent
from llm_client import MockLLM
import ground_truth as gt
import article_ground_truth as agt_gt


def _haystack(result, is_mock):
    """Same convention as test_the_agent.py's _haystack(): mock mode
    judges the raw evidence only (MockLLM's prose isn't real text to
    check); real mode judges the answer AND the evidence together."""
    ev_text = json.dumps(result["evidence"], default=str).lower()
    if is_mock:
        return ev_text
    return (result["answer"] or "").lower() + " " + ev_text


def _layer1_routing(evidence):
    called = {e.get("tool") for e in evidence}
    ok = "search_documents" in called
    return ("PASS" if ok else "FAIL"), ("search_documents called" if ok else "search_documents NEVER called")


def _layer2_single(haystack, expected_chunk_ids):
    if not expected_chunk_ids:
        return "SKIP", "no resolvable ground truth"
    ok = gt.any_present(haystack, expected_chunk_ids)
    return ("PASS" if ok else "FAIL"), expected_chunk_ids


def _layer2_synthesis(haystack, expected_by_doc):
    if not all(expected_by_doc.values()):
        return "SKIP", "no resolvable ground truth for one or more documents"
    per_doc = {doc: gt.any_present(haystack, [cid]) for doc, cid in expected_by_doc.items()}
    n_found = sum(per_doc.values())
    if n_found == len(expected_by_doc):
        return "PASS", per_doc
    if n_found > 0:
        return f"PARTIAL({n_found}/{len(expected_by_doc)})", per_doc
    return "FAIL", per_doc


def _layer3_answer(haystack, facts):
    ok = gt.all_present(haystack, facts)
    return ("PASS" if ok else "FAIL"), facts


_BAR = "=" * 72


def run(use_mock=None):
    llm = MockLLM() if use_mock else None  # None -> agent.py's get_llm() decides (real key if set)

    print(f"\n{_BAR}\nARTICLE AGENT EVAL - 3 layers, real agent.py\n{_BAR}")
    print(f"{len(agt_gt.QUESTIONS)} questions\n")

    layer_counts = {"L1": {}, "L2": {}, "L3": {}}
    results_log = []

    for i, q in enumerate(agt_gt.QUESTIONS, start=1):
        agent = RabivyAgent(llm=llm)  # fresh agent per question - no cross-question memory bleed
        result = agent.ask(q["q"])
        is_mock = isinstance(agent.llm, MockLLM)
        haystack = _haystack(result, is_mock)

        l1_status, l1_detail = _layer1_routing(result["evidence"])

        if q["type"] == "single_source":
            l2_status, l2_detail = _layer2_single(haystack, q["expected_chunk_ids"])
        else:
            l2_status, l2_detail = _layer2_synthesis(haystack, q["expected_chunk_ids_by_doc"])

        l3_status, l3_detail = _layer3_answer(haystack, q["facts"])

        layer_counts["L1"][l1_status] = layer_counts["L1"].get(l1_status, 0) + 1
        layer_counts["L2"][l2_status] = layer_counts["L2"].get(l2_status, 0) + 1
        layer_counts["L3"][l3_status] = layer_counts["L3"].get(l3_status, 0) + 1

        pretraining_flag = ""
        if l2_status == "FAIL" and l3_status == "PASS":
            pretraining_flag = ("  ! L2 FAIL + L3 PASS on a REAL paper - verify this isn't just "
                                 "pretraining recall (re-ask with no tools) before trusting it")

        print(f"[{i}/{len(agt_gt.QUESTIONS)}] [{q['key']}] ({q['type']}) - mock={is_mock}")
        print(f"  Q: {q['q']}")
        print(f"  L1 routing  : {l1_status:5} - {l1_detail}")
        print(f"  L2 retrieval: {l2_status:12} - expected {l2_detail}")
        print(f"  L3 answer   : {l3_status:5} - expected facts {l3_detail}")
        if pretraining_flag:
            print(pretraining_flag)
        answer_preview = (result["answer"] or "")[:200]
        print(f"  Answer given: {answer_preview}{'...' if len(result['answer'] or '') > 200 else ''}")
        print()

        results_log.append({
            "n": i, "key": q["key"], "type": q["type"], "question": q["q"], "is_mock": is_mock,
            "l1": l1_status, "l2": l2_status, "l3": l3_status,
            "answer": result["answer"], "pretraining_flag": bool(pretraining_flag),
        })

    print(f"{_BAR}\nSUMMARY\n{_BAR}")
    for layer, counts in layer_counts.items():
        print(f"  {layer}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    n_pretraining_flags = sum(1 for r in results_log if r["pretraining_flag"])
    if n_pretraining_flags:
        print(f"\n! {n_pretraining_flags} question(s) flagged for pretraining-recall check - see above")

    os.makedirs("eval_runs", exist_ok=True)
    path = os.path.join("eval_runs", f"article_agent_eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"layer_counts": layer_counts, "results": results_log}, f, indent=2, default=str)
    print(f"\nSaved this run to {path}")
    return results_log


if __name__ == "__main__":
    mock_flag = "--mock" in sys.argv or os.environ.get("AGENT_LLM", "").lower() == "mock"
    run(use_mock=mock_flag)