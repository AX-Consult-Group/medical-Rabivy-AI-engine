# test_article_retrieval.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# Same method as test_retrieval_ranking.py: calls sd.semantic_search()
# DIRECTLY - the real production search function, bypassing only the
# routing/LLM layer (not the search engine itself). An earlier draft
# of this file reimplemented the embedding + cosine-similarity math by
# hand, loading embeddings.npy directly - that was a mistake, since it
# meant testing a hand-built approximation of retrieval rather than
# the actual code path the real system uses. Fixed here to match
# test_retrieval_ranking.py's method exactly.
#
# DELIBERATELY SEARCHES THE FULL CORPUS, not just article chunks -
# sd.semantic_search() naturally does this already (it has no idea
# these are "the article questions"), which is exactly the point: a
# real rep's question competes against ALL indexed content - HCP
# cards, strategic docs, competitive intel, rep_field material, the
# other narrative docs - not a clean, pre-filtered subset. If the
# correct article chunk still ranks well amid all that real noise,
# that's a much more meaningful result than ranking well in isolation.
#
# Ground truth comes from article_ground_truth.py (live substring
# lookup against chunks_tagged.json), same as before - only the
# RETRIEVAL METHOD changed, not how "correct" is determined.
# =====================================================================

import json
import os
import time

import search_documents as sd
import article_ground_truth as agt

TOP_K = 5  # same bar as test_retrieval_ranking.py


def _rank_of_any(results, expected_chunk_ids):
    """1-indexed rank of the FIRST result matching any of the expected
    chunk_ids, or None if none of them appear in the results at all."""
    if not expected_chunk_ids:
        return None
    expected_set = set(expected_chunk_ids)
    for i, r in enumerate(results, start=1):
        if r["chunk"].get("chunk_id", "") in expected_set:
            return i
    return None


_BAR = "=" * 72


def run_single_source(q, results):
    expected = q["expected_chunk_ids"]
    if not expected:
        return "SKIP", "no resolvable ground truth (see article_ground_truth.py)", None

    rank = _rank_of_any(results, expected)
    top_chunk_id = results[0]["chunk"].get("chunk_id", "?") if results else "?"
    top_score = results[0]["score"] if results else None

    if rank == 1:
        return "RANK1", f"top result: {top_chunk_id}", top_score
    if rank is not None:
        return f"PRESENT_NOT_RANK1(#{rank})", f"top result instead: {top_chunk_id}", top_score
    return f"MISSING", f"top result instead: {top_chunk_id}", top_score


def run_synthesis(q, results):
    expected_by_doc = q["expected_chunk_ids_by_doc"]
    if not all(expected_by_doc.values()):
        return "SKIP", "no resolvable ground truth for one or more documents", None

    per_doc_rank = {}
    for doc, cid in expected_by_doc.items():
        per_doc_rank[doc] = _rank_of_any(results, [cid])

    docs_found = sum(1 for r in per_doc_rank.values() if r is not None)
    detail = f"per-document rank in top {TOP_K}: {per_doc_rank}"
    top_score = results[0]["score"] if results else None

    if docs_found == len(expected_by_doc):
        return "ALL_SOURCES_PRESENT", detail, top_score
    if docs_found > 0:
        return f"PARTIAL({docs_found}/{len(expected_by_doc)})", detail, top_score
    return "NO_SOURCES_PRESENT", detail, top_score


def run():
    print(f"\n{_BAR}\nARTICLE RETRIEVAL - via real search_documents.semantic_search()\n{_BAR}")
    print(f"Searching the FULL corpus (all doc_types) for each of the "
          f"{len(agt.QUESTIONS)} article questions -\n"
          f"real production noise included, not a pre-filtered subset.\n")

    results_log = []
    counts = {}

    for i, q in enumerate(agt.QUESTIONS, start=1):
        data = sd.semantic_search(q["q"], top_k=TOP_K)
        if not data.get("found", True) or not data.get("results"):
            print(f"[{i}/{len(agt.QUESTIONS)}] [{q['key']}] NO RESULTS RETURNED - {data.get('error', 'unknown reason')}")
            results_log.append({"n": i, "key": q["key"], "status": "NO_RESULTS"})
            counts["NO_RESULTS"] = counts.get("NO_RESULTS", 0) + 1
            continue

        results = data["results"]
        if q["type"] == "single_source":
            status, detail, top_score = run_single_source(q, results)
        else:
            status, detail, top_score = run_synthesis(q, results)

        counts[status] = counts.get(status, 0) + 1

        print(f"[{i}/{len(agt.QUESTIONS)}] [{q['key']}] ({q['type']})")
        print(f"  Q: {q['q']}")
        print(f"  {detail}" + (f"  (score {top_score:.3f})" if top_score is not None else ""))
        print(f"  -> {status}")

        results_log.append({"n": i, "key": q["key"], "type": q["type"], "question": q["q"],
                            "status": status, "detail": detail, "top_score": top_score})

    print(f"\n{_BAR}\nSUMMARY\n{_BAR}")
    for label, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:26}: {n} / {len(agt.QUESTIONS)}")

    os.makedirs("eval_runs", exist_ok=True)
    path = os.path.join("eval_runs", f"article_retrieval_eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"counts": counts, "results": results_log}, f, indent=2, default=str)
    print(f"\nSaved this run to {path}")
    return results_log


if __name__ == "__main__":
    run()