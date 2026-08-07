# label_loop.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# The EVAL LABEL LOOP (see eval_label_loop_spec.md): turns retrieval-
# ranking failures into VERIFIED ground-truth labels using two
# independent raters - one human (rater_ui.py) and one LLM judge from a
# DIFFERENT model family than the agent (OpenAI-class, so its errors
# don't correlate with Claude's) - then measures the raters themselves
# (Cohen's kappa for agreement, gold items for accuracy against known
# truth) before any label is trusted.
#
# Four subcommands, run in this order:
#
#   python label_loop.py --build-cases   builds output/label_cases.jsonl:
#                                        live cases (questions whose
#                                        correct chunk is NOT ranked #1)
#                                        + gold items (planted known
#                                        answers - the raters' exam)
#   python label_loop.py --exam          judge entrance exam on gold
#                                        items ONLY - pass bar 90%.
#                                        Run this BEFORE any live rating.
#   python label_loop.py --judge         judge votes on every case ->
#                                        output/label_votes.jsonl
#   (human votes come from rater_ui.py -> same votes file)
#   python label_loop.py --report        joins votes: kappa, gold
#                                        accuracy, adjudication queue,
#                                        proposed NARRATIVE_FACTS
#                                        changes, consolidated
#                                        output/eval_labels.jsonl
#
# DESIGN RULES BAKED IN (from the spec - do not quietly relax these):
#   - Candidate sets are the UNION of retrievers (semantic today; the
#     keyword retriever plugs into keyword_search() below when Tilabo's
#     hybrid experiment lands). Never one retriever's list alone.
#   - Every case carries a "none of these" option. A none-of-these
#     consensus is a retrieval hole, not a label.
#   - Candidates are shown in RANDOMIZED order (stable per question, so
#     human and judge see the same order) with no scores and no
#     retriever-of-origin - anti-anchoring.
#   - Gold items are UNMARKED in the live stream. Only this file's
#     internal records know which is which.
#   - Disagreement NEVER auto-resolves toward either rater - it goes to
#     the adjudication queue in the report. At disagreement time, truth
#     is unknown: only votes exist.
#   - The report PROPOSES NARRATIVE_FACTS changes as a printed diff -
#     it never edits ground_truth.py itself.
#
# MOCK MODE: with no OPENAI_API_KEY set (or JUDGE_LLM=mock), a
# deterministic keyword-overlap MockJudge stands in - free, useful for
# testing this file's plumbing end to end, and clearly labeled. The
# real inter-family signal only exists with the real judge.
# =====================================================================

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request

import ground_truth as gt
import search_documents as sd
from test_retrieval_ranking import NARRATIVE_QUESTIONS

CASES_PATH = os.path.join("output", "label_cases.jsonl")
VOTES_PATH = os.path.join("output", "label_votes.jsonl")
LABELS_PATH = os.path.join("output", "eval_labels.jsonl")

TOP_K = 5            # per retriever, before the union
SNIPPET_CHARS = 700  # how much chunk text the raters see
NONE_OF_THESE = "none"
EXAM_PASS_BAR = 0.9  # entrance-exam pass bar (spec section 5b)
KAPPA_TRIAGE_BAR = 0.8   # triage-mode eligibility (spec section 5)
KAPPA_TRIAGE_MIN_N = 30

_BAR = "=" * 72


# =====================================================================
# RETRIEVERS - the union candidate set
# =====================================================================

def semantic_candidates(question, top_k=TOP_K):
    data = sd.semantic_search(question, top_k=top_k)
    if not data.get("results"):
        return []
    return [r["chunk"] for r in data["results"]]


def keyword_search(question, top_k=TOP_K):
    """PLUG-IN POINT for the keyword retriever (Tilabo's hybrid
    experiment). Until that lands, returns [] - the candidate union is
    semantic-only, and this function is the ONLY line that changes when
    hybrid retrieval arrives. Deliberately not implemented here: the
    retrieval engine layer is not this file's to build."""
    return []


def _stable_rng(question):
    """Deterministic per-question randomizer - human and judge must see
    the SAME shuffled order (anti-anchoring, but reproducible), and
    re-running --build-cases must not silently reshuffle everything."""
    seed = int(hashlib.md5(question.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed)


def build_candidate_set(question):
    """Union of all retrievers' top-k, deduplicated by chunk_id,
    shuffled in a stable order. Returns a list of {chunk_id, snippet}."""
    seen, merged = set(), []
    for chunk in semantic_candidates(question) + keyword_search(question):
        cid = chunk.get("chunk_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            merged.append({"chunk_id": cid,
                           "snippet": (chunk.get("text") or "")[:SNIPPET_CHARS]})
    _stable_rng(question).shuffle(merged)
    return merged


def _chunk_lookup():
    """chunk_id -> chunk dict, from the tagged chunk store - used to
    PLANT a known-correct chunk into a gold item's candidates when the
    retriever itself didn't surface it."""
    with open(os.path.join("output", "chunks_tagged.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c for c in chunks}


def _correct_chunk_id(candidates_or_chunks, tags):
    """First chunk_id (among dicts with a chunk_id) matching the
    verified tag list - same matching rule as test_retrieval_ranking."""
    for c in candidates_or_chunks:
        cid = c.get("chunk_id", "")
        if cid and gt.any_present(cid.lower(), tags):
            return cid
    return None


# =====================================================================
# --build-cases
# =====================================================================

def build_cases(extra_questions_path=None):
    """Live cases = verified narrative questions whose correct chunk is
    NOT currently ranked #1 (the existing tag becomes a HYPOTHESIS the
    raters test blind - convergence confirms it, divergence flags a
    possible mislabel). Gold items = verified questions the retriever
    already ranks correctly, with the known answer planted among
    distractors - these examine the RATERS, not the retriever.
    Optionally appends tag-less questions from a JSONL file (one
    {"key":..., "question":...} per line) as pure live cases - that is
    the file new golden-set candidates arrive through."""
    lookup = _chunk_lookup()
    cases, n_live, n_gold = [], 0, 0

    for key, question in NARRATIVE_QUESTIONS.items():
        spec = gt.NARRATIVE_FACTS[key]
        if not spec.get("tag"):
            continue
        candidates = build_candidate_set(question)
        results = sd.semantic_search(question, top_k=TOP_K).get("results") or []
        ranked_ids = [r["chunk"].get("chunk_id", "") for r in results]
        rank = None
        for i, cid in enumerate(ranked_ids, start=1):
            if gt.any_present(cid.lower(), spec["tag"]):
                rank = i
                break

        if rank == 1:
            # Retriever agrees with verified truth -> safest gold item.
            gold_answer = _correct_chunk_id(candidates, spec["tag"])
            if gold_answer is None:
                # Not in the union (can't happen when rank==1, but be
                # safe): plant it from the chunk store, replacing the
                # last (weakest, post-shuffle arbitrary) candidate.
                planted = next((c for cid, c in lookup.items()
                                if gt.any_present(cid.lower(), spec["tag"])), None)
                if planted is None:
                    print(f"[{key}] WARNING: verified tag matches no chunk in the store - skipped")
                    continue
                gold_answer = planted["chunk_id"]
                candidates = candidates[:-1] + [{"chunk_id": gold_answer,
                                                 "snippet": (planted.get("text") or "")[:SNIPPET_CHARS]}]
                _stable_rng(question + "|gold").shuffle(candidates)
            cases.append({"case_id": f"gold_{key}", "kind": "gold", "narrative_key": key,
                          "question": question, "candidates": candidates,
                          "gold_answer": gold_answer, "existing_tag": spec["tag"]})
            n_gold += 1
        else:
            cases.append({"case_id": f"live_{key}", "kind": "live", "narrative_key": key,
                          "question": question, "candidates": candidates,
                          "existing_tag": spec["tag"], "retriever_rank_of_tag": rank})
            n_live += 1

    if extra_questions_path:
        with open(extra_questions_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                q = json.loads(line)
                cases.append({"case_id": f"live_{q['key']}", "kind": "live",
                              "narrative_key": q["key"], "question": q["question"],
                              "candidates": build_candidate_set(q["question"]),
                              "existing_tag": None, "retriever_rank_of_tag": None})
                n_live += 1

    os.makedirs("output", exist_ok=True)
    with open(CASES_PATH, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"{_BAR}\nBUILT {len(cases)} CASES -> {CASES_PATH}\n{_BAR}")
    print(f"  live cases (need labels)      : {n_live}")
    print(f"  gold items (examine the raters): {n_gold}")
    print("\nGold items are UNMARKED in the rating stream - rater_ui.py and the")
    print("judge see live and gold cases identically. Only the report knows which")
    print("is which.")
    return cases


def load_cases():
    if not os.path.exists(CASES_PATH):
        sys.exit(f"No {CASES_PATH} - run: python label_loop.py --build-cases")
    with open(CASES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# =====================================================================
# THE JUDGE - independent second rater, different model family
# =====================================================================

JUDGE_PROMPT = """You are an independent rater for a pharma commercial-intelligence
retrieval system. Below is a QUESTION a field rep asked, and a list of CANDIDATE
document chunks. Choose the single candidate that best and most directly answers
the question. If NO candidate genuinely answers it, choose "none".

Do not reward surface word overlap - choose the chunk whose CONTENT answers the
question, even if it shares few words with it.

QUESTION: {question}

CANDIDATES:
{candidates}

Respond with ONLY a JSON object and NOTHING else:
{{"choice": "<chunk_id or none>", "why": "<one sentence>"}}"""


def _extract_json(text):
    """First balanced JSON object in the reply - same defensive pattern
    as agent.py's auditor parsing (judges also add commentary)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class OpenAIJudge:
    """The real judge: an OpenAI-class model (a DIFFERENT family from the
    Claude agent, so rating errors don't correlate). stdlib-only HTTP -
    no new package dependency. Needs OPENAI_API_KEY; model overridable
    via JUDGE_MODEL."""

    name = "openai"

    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")

    def choose(self, question, candidates):
        listing = "\n\n".join(f"[{c['chunk_id']}]\n{c['snippet']}" for c in candidates)
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user",
                          "content": JUDGE_PROMPT.format(question=question, candidates=listing)}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            reply = json.load(resp)["choices"][0]["message"]["content"]
        parsed = _extract_json(reply) or {}
        choice = str(parsed.get("choice", NONE_OF_THESE)).strip()
        valid = {c["chunk_id"] for c in candidates} | {NONE_OF_THESE}
        if choice not in valid:
            choice = NONE_OF_THESE  # an invented chunk_id counts as "none"
        return {"choice": choice, "why": str(parsed.get("why", ""))[:300]}


_STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "what",
              "whats", "which", "who", "whos", "how", "why", "when", "where", "our", "we",
              "i", "you", "your", "s", "to", "of", "in", "on", "for", "and", "or", "vs",
              "versus", "after", "with", "should", "can", "their", "they", "it", "its"}


class MockJudge:
    """Deterministic keyword-overlap stand-in - free, offline, useful ONLY
    for testing this loop's plumbing. Its ratings carry no independent
    signal (it is exactly the surface-overlap heuristic the real judge is
    told to avoid), so a report produced in mock mode is a plumbing test,
    never evidence."""

    name = "mock"

    def choose(self, question, candidates):
        q_tokens = {t for t in "".join(ch if ch.isalnum() else " " for ch in question.lower()).split()
                    if t not in _STOPWORDS}
        best_choice, best_score = NONE_OF_THESE, 0
        for c in sorted(candidates, key=lambda c: c["chunk_id"]):  # stable tie-break
            text = (c["chunk_id"] + " " + c["snippet"]).lower()
            c_tokens = {t for t in "".join(ch if ch.isalnum() else " " for ch in text).split()
                        if t not in _STOPWORDS}
            score = len(q_tokens & c_tokens)
            if score > best_score:
                best_choice, best_score = c["chunk_id"], score
        return {"choice": best_choice,
                "why": f"mock keyword-overlap pick (score {best_score})"}


def get_judge():
    forced = os.environ.get("JUDGE_LLM", "").lower()
    if forced != "mock" and os.environ.get("OPENAI_API_KEY"):
        j = OpenAIJudge()
        print(f">>> Judge backend: OpenAI ({j.model}) - real cross-family signal")
        return j
    print(">>> Judge backend: MOCK keyword-overlap (no OPENAI_API_KEY, or JUDGE_LLM=mock)")
    print(">>> Mock ratings test the plumbing only - they are NOT independent evidence.")
    return MockJudge()


# =====================================================================
# --exam  (spec section 5b - run BEFORE any live rating)
# =====================================================================

def run_exam():
    judge = get_judge()
    gold = [c for c in load_cases() if c["kind"] == "gold"]
    if not gold:
        sys.exit("No gold items in the case file - run --build-cases first.")
    print(f"\n{_BAR}\nJUDGE ENTRANCE EXAM - {len(gold)} gold items, pass bar "
          f"{EXAM_PASS_BAR:.0%}\n{_BAR}")
    correct = 0
    for c in gold:
        verdict = judge.choose(c["question"], c["candidates"])
        ok = verdict["choice"] == c["gold_answer"]
        correct += ok
        print(f"[{c['narrative_key']}] {'CORRECT' if ok else 'WRONG':7s} "
              f"chose {verdict['choice']}" + ("" if ok else f"  (truth: {c['gold_answer']})"))
    score = correct / len(gold)
    passed = score >= EXAM_PASS_BAR
    print(f"\nSCORE: {correct}/{len(gold)} ({score:.0%}) -> "
          f"{'PASSED - judge may rate live cases' if passed else 'FAILED - do NOT use this judge on live cases'}")
    if not passed:
        print("Fix before proceeding: re-prompt the judge, add domain context, or swap")
        print("models. A judge that can't find planted known answers can't be trusted")
        print("to find unknown ones.")
    if judge.name == "mock":
        print("\nNOTE: mock judge - this exam only proves the exam harness runs.")
    return passed


# =====================================================================
# VOTES
# =====================================================================

def append_vote(case_id, actor, choice, why="", model=""):
    os.makedirs("output", exist_ok=True)
    with open(VOTES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"case_id": case_id, "actor": actor, "choice": choice,
                            "why": why, "model": model,
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")


def load_votes():
    """Latest vote per (case_id, actor) - a re-vote supersedes."""
    votes = {}
    if os.path.exists(VOTES_PATH):
        with open(VOTES_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    v = json.loads(line)
                    votes[(v["case_id"], v["actor"])] = v
    return votes


def run_judge():
    judge = get_judge()
    cases = load_cases()
    votes = load_votes()
    done = 0
    for c in cases:
        if (c["case_id"], "judge") in votes:
            continue  # already voted - re-run only adds what's missing
        verdict = judge.choose(c["question"], c["candidates"])
        append_vote(c["case_id"], "judge", verdict["choice"], verdict["why"],
                    model=getattr(judge, "model", judge.name))
        done += 1
        print(f"[{c['narrative_key']}] judge -> {verdict['choice']}")
    print(f"\n{done} new judge vote(s) appended to {VOTES_PATH} "
          f"({len(cases) - done} already voted).")


# =====================================================================
# --report
# =====================================================================

def cohens_kappa(pairs):
    """pairs: list of (choice_a, choice_b). Plain-code Cohen's kappa -
    chance-corrected agreement over the observed category set."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    cats = {c for pair in pairs for c in pair}
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n)
             for c in cats)
    if pe == 1.0:
        return 1.0  # only one category ever used, and they always agreed
    return (po - pe) / (1 - pe)


def run_report():
    cases = load_cases()
    votes = load_votes()
    both, labels = [], []
    gold_correct = {"human": [0, 0], "judge": [0, 0]}  # [correct, seen]
    adjudication_queue, holes, proposals, confirmations = [], [], [], []

    for c in cases:
        h = votes.get((c["case_id"], "human"))
        j = votes.get((c["case_id"], "judge"))
        if c["kind"] == "gold":
            for actor, v in (("human", h), ("judge", j)):
                if v:
                    gold_correct[actor][1] += 1
                    gold_correct[actor][0] += v["choice"] == c["gold_answer"]
        if not (h and j):
            continue
        both.append((c, h, j))

        agreed = h["choice"] == j["choice"]
        record = {"question": c["question"], "narrative_key": c["narrative_key"],
                  "kind": c["kind"], "candidates": [x["chunk_id"] for x in c["candidates"]],
                  "human_choice": h["choice"], "judge_choice": j["choice"],
                  "judge_model": j.get("model", ""), "agreed": agreed,
                  "adjudicated_by": None, "adjudication_reason": None,
                  "final_label": h["choice"] if agreed else None,
                  "gold": c["kind"] == "gold",
                  "gold_answer": c.get("gold_answer"),
                  "retriever_rank_of_label": c.get("retriever_rank_of_tag"),
                  "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        labels.append(record)

        if c["kind"] != "live":
            continue
        if not agreed:
            adjudication_queue.append((c, h, j))
        elif h["choice"] == NONE_OF_THESE:
            holes.append(c)
        else:
            tags = c.get("existing_tag")
            if tags and gt.any_present(h["choice"].lower(), tags):
                confirmations.append((c, h["choice"]))
            else:
                proposals.append((c, h["choice"]))

    print(f"{_BAR}\nLABEL LOOP REPORT\n{_BAR}")
    print(f"Cases: {len(cases)} total | jointly voted (human + judge): {len(both)}")

    pairs = [(h["choice"], j["choice"]) for _, h, j in both]
    kappa = cohens_kappa(pairs)
    print(f"\nINTER-RATER AGREEMENT (human vs judge)")
    if kappa is None:
        print("  No jointly-voted cases yet - collect votes from both raters first.")
    else:
        agree_n = sum(a == b for a, b in pairs)
        print(f"  Raw agreement : {agree_n}/{len(pairs)}")
        print(f"  Cohen's kappa : {kappa:.3f}  (chance-corrected)")
        eligible = kappa >= KAPPA_TRIAGE_BAR and len(pairs) >= KAPPA_TRIAGE_MIN_N
        print(f"  Triage mode   : {'ELIGIBLE' if eligible else 'NOT YET'} "
              f"(needs kappa >= {KAPPA_TRIAGE_BAR} over >= {KAPPA_TRIAGE_MIN_N} items; "
              f"have {len(pairs)})")

    print(f"\nGOLD-ITEM ACCURACY (validity - correctness against planted truth)")
    for actor in ("human", "judge"):
        correct, seen = gold_correct[actor]
        print(f"  {actor:6s}: {correct}/{seen} correct" if seen else
              f"  {actor:6s}: no gold votes yet")

    if adjudication_queue:
        print(f"\nADJUDICATION QUEUE ({len(adjudication_queue)} case(s)) - expert decides, with sources open:")
        for c, h, j in adjudication_queue:
            print(f"  [{c['narrative_key']}] human: {h['choice']}  vs  judge: {j['choice']}")
            if j.get("why"):
                print(f"      judge's reason: {j['why']}")
    if holes:
        print(f"\nRETRIEVAL HOLES ({len(holes)}) - both raters chose 'none of these':")
        for c in holes:
            print(f"  [{c['narrative_key']}] {c['question']}")
    if confirmations:
        print(f"\nCONFIRMED EXISTING TAGS ({len(confirmations)}) - raters independently "
              f"converged on the already-verified chunk:")
        for c, choice in confirmations:
            print(f"  [{c['narrative_key']}] {choice}")
    if proposals:
        print(f"\nPROPOSED NARRATIVE_FACTS CHANGES ({len(proposals)}) - review, then edit "
              f"ground_truth.py by hand (this report never edits it):")
        for c, choice in proposals:
            print(f"  [{c['narrative_key']}] tag {c.get('existing_tag')} -> \"{choice}\"")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")
    print(f"\nConsolidated labels -> {LABELS_PATH}")
    print("Settled live labels with retriever_rank_of_label != 1 are the fine-tune")
    print("training set (spec section 7) - worthwhile from ~50 triplets.")


# =====================================================================

def main():
    p = argparse.ArgumentParser(description="Eval label loop (see eval_label_loop_spec.md)")
    p.add_argument("--build-cases", action="store_true")
    p.add_argument("--questions-file", help="optional JSONL of extra tag-less questions "
                                            '({"key":..., "question":...} per line)')
    p.add_argument("--exam", action="store_true")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    if args.build_cases:
        build_cases(args.questions_file)
    elif args.exam:
        ok = run_exam()
        sys.exit(0 if ok else 1)
    elif args.judge:
        run_judge()
    elif args.report:
        run_report()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
