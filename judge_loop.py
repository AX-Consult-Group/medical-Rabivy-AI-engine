# judge_loop.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# The contrarian judge, end to end, in one file with four subcommands -
# mirroring the shape of Jürgen's label_loop.py on the eval-label-loop
# branch (PR #15: --build-cases / --exam / --judge / --report), just
# renamed to match what this tool actually does. Originally this was
# five separate files (judge.py, build_gold_items.py, run_judge_exam.py,
# review_live_batch.py, report_kappa.py) - consolidated into one on
# 2026-08-07 because seven new files for one feature was genuinely hard
# for anyone new to the repo to make sense of at a glance. The human
# rating page stays its own file (human_review_ui.py) because it's a
# fundamentally different kind of thing - a running HTTP server, not a
# one-shot script - same reason Jürgen kept rater_ui.py separate from
# label_loop.py too. Tests stay separate as test_judge_loop.py.
#
#   python judge_loop.py --build-gold   builds the entrance exam (gold_items.jsonl)
#   python judge_loop.py --exam         scores the judge against it (exam_result.json)
#   python judge_loop.py --review       (if the judge passed) reviews a live sample (judge_review.jsonl)
#   python judge_loop.py --report       Cohen's kappa vs. human_review_ui.py's votes (kappa_report.json)
#
# WHY THIS EXISTS AT ALL: agent.py's own hallucination audit is one
# model checking its own retrieval. Nothing today catches the case
# where that check itself is wrong. This is a second, independent
# model doing the same evidence-vs-answer check blind, on a sample of
# real traffic, calibrated against known-answer cases first so we know
# whether to trust it before pointing it at anything real.
#
# UPDATED 2026-08-07 alongside three fixes on main:
#   1. chat_ui.py now logs preceding_context (the model's conversation
#      memory) on every query. --review and --build-gold both use it now
#      - a live question answered from memory, with no fresh tool call,
#      is legitimate (see agent.py's SYSTEM_PROMPT: "Follow-up questions
#      ... resolve them from context"), and the judge needs to see that
#      prior context to fairly judge it as supported, not just flag
#      every memory-only answer as unsupported for lack of fresh evidence.
#   2. build_dashboard.py's _GATE_FAIL_COLOR now has Source citations at
#      amber, not red - --review imports _gate_color/_worst_color
#      directly from build_dashboard.py rather than reimplementing the
#      rule, so this fix (and the audit_error fix) are already reflected
#      here automatically, with nothing to change in this file for that
#      part.
# =====================================================================

import argparse
import json
import os
import random
import re
import urllib.request

# =====================================================================
# SECTION 1 - THE JUDGE ITSELF (was judge.py)
# =====================================================================
# Same "one interface, pluggable backends" shape as llm_client.py:
#
#   ClaudeJudge - real Claude calls via the Anthropic Messages API.
#                 Used when ANTHROPIC_API_KEY is set. DEFAULT today
#                 because that's the key actually available - worth
#                 being upfront this is NOT the "different model
#                 family" setup label_loop.py's original design called
#                 for (PR #15: "an OpenAI-class model, different family
#                 from the agent on purpose, so rating errors don't
#                 correlate"). agent.py also runs on Claude, so a Claude
#                 judge can share blind spots with the exact thing it's
#                 checking. Swap to OpenAIJudge (JUDGE_BACKEND=openai)
#                 once that key exists.
#   OpenAIJudge - stdlib-only HTTP, same pattern Jürgen's label_loop.py
#                 used. Wired up and ready, but UNTESTED - no
#                 OPENAI_API_KEY in this environment yet.
#   MockJudge   - deterministic, zero-cost, same purpose as
#                 llm_client.py's MockLLM - lets everything below be
#                 built and tested without spending a token.
#
# All three return: {"verdict": "supported" | "unsupported", "reasoning": str}
# =====================================================================

DEFAULT_CLAUDE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-5")
DEFAULT_OPENAI_MODEL = os.environ.get("JUDGE_OPENAI_MODEL", "gpt-4o-mini")

JUDGE_SYSTEM = """You are an independent fact-checker. You will be shown a \
question, EVIDENCE, and an ANSWER that was given to that question. Your ONLY \
job is to decide whether the answer's claims are actually supported by the \
evidence shown - nothing else. Do not use outside knowledge, and do not \
reward a well-written answer that isn't backed by the evidence text in \
front of you.

The evidence may contain two parts, clearly labelled:
- "Retrieved this turn" - fresh tool results.
- "Established earlier in this conversation" - real question/answer pairs \
from earlier in the SAME session, kept as conversation memory. A claim that \
matches something already correctly established there is JUST AS supported \
as one from fresh evidence - the assistant is allowed to build on what it \
already correctly said a turn or two ago without re-fetching it. Only mark \
a claim unsupported if it isn't backed by EITHER part.

Respond with ONLY a JSON object, no other text:
{"verdict": "supported" or "unsupported", "reasoning": "one sentence why"}"""


def _build_prompt(question, evidence_text, given_answer):
    return (f"QUESTION:\n{question}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            f"ANSWER GIVEN:\n{given_answer}\n\n"
            f"Is the answer actually supported by the evidence above?")


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _parse_verdict(text):
    """Both real backends funnel through here so a malformed reply fails
    the same way regardless of which model sent it. An unparseable
    reply is treated as 'unsupported' rather than silently dropped or
    treated as a pass - erring toward flagging for review, not toward
    trusting a judge whose output we couldn't even read.

    FIXED 2026-08-11: real Claude replies sometimes wrap the JSON in a
    ```json ... ``` markdown fence despite JUDGE_SYSTEM saying "ONLY a
    JSON object, no other text" - json.loads() chokes on the backticks
    and this used to fall straight into the unparseable/unsupported
    branch below. That silently passed the real exam run (74%, bar 90%)
    because the "default to unsupported" fallback happened to match
    true_verdict on every method except conversation_memory (the only
    one whose true_verdict is "supported") - which is why that one
    category alone came back 0/8. Strip a fence, if present, before
    parsing."""
    text = text.strip()
    fence_match = _CODE_FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        data = json.loads(text)
        verdict = data.get("verdict")
        if verdict not in ("supported", "unsupported"):
            raise ValueError(f"unexpected verdict value: {verdict!r}")
        return {"verdict": verdict, "reasoning": data.get("reasoning", "")}
    except (json.JSONDecodeError, ValueError) as e:
        return {"verdict": "unsupported",
                "reasoning": f"[unparseable judge reply, treated as unsupported: {e}] {text[:200]}"}


class ClaudeJudge:
    name = "claude"

    def __init__(self, model=DEFAULT_CLAUDE_MODEL):
        import anthropic  # deferred import, same reason as llm_client.py
        self.model = model
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def judge(self, question, evidence_text, given_answer):
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=300,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user",
                       "content": _build_prompt(question, evidence_text, given_answer)}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _parse_verdict(text)


class OpenAIJudge:
    """Stdlib-only HTTP, same shape as label_loop.py's judge on the
    eval-label-loop branch (no SDK dependency, just urllib). Not yet
    tested against a real key - verify this path once OPENAI_API_KEY
    exists before trusting it the way ClaudeJudge has been exercised."""
    name = "openai"

    def __init__(self, model=DEFAULT_OPENAI_MODEL):
        self.model = model
        self.api_key = os.environ["OPENAI_API_KEY"]  # raises loudly if missing

    def judge(self, question, evidence_text, given_answer):
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": _build_prompt(question, evidence_text, given_answer)},
            ],
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        text = body["choices"][0]["message"]["content"]
        return _parse_verdict(text)


class MockJudge:
    """Deterministic word-overlap check between the evidence text and
    the given answer - free, no network. Lets every subcommand below be
    run end to end without spending a token. Not a real judgement -
    just enough signal to exercise the plumbing, same spirit as
    llm_client.py's MockLLM."""
    name = "mock"
    THRESHOLD = 0.15

    def judge(self, question, evidence_text, given_answer):
        overlap = self._word_overlap(evidence_text, given_answer)
        verdict = "supported" if overlap >= self.THRESHOLD else "unsupported"
        return {"verdict": verdict,
                "reasoning": f"[MOCK] word overlap between evidence and answer = {overlap:.2f}"}

    @staticmethod
    def _word_overlap(evidence_text, given_answer):
        ev_words = set(evidence_text.lower().split())
        ans_words = set(w.strip(".,;:") for w in given_answer.lower().split())
        ans_words.discard("")
        if not ans_words:
            return 0.0
        return len(ev_words & ans_words) / len(ans_words)


def get_judge():
    """Real Claude when a key is present, mock otherwise - same loud,
    never-silent-substitution principle as llm_client.py's get_llm().
    JUDGE_BACKEND=mock forces the mock; JUDGE_BACKEND=openai (with
    OPENAI_API_KEY set) uses the OpenAI backend once that key exists."""
    forced = os.environ.get("JUDGE_BACKEND", "").lower()
    if forced == "mock":
        print("Judge backend: MOCK (forced via JUDGE_BACKEND=mock)")
        return MockJudge()
    if forced == "openai" and os.environ.get("OPENAI_API_KEY"):
        print(f"Judge backend: OpenAI ({DEFAULT_OPENAI_MODEL})")
        return OpenAIJudge()
    if os.environ.get("ANTHROPIC_API_KEY"):
        print(f"Judge backend: Claude ({DEFAULT_CLAUDE_MODEL}) - same model "
              f"family as the agent itself, see file header for why that matters")
        return ClaudeJudge()
    print("Judge backend: MOCK (no ANTHROPIC_API_KEY or OPENAI_API_KEY found)")
    return MockJudge()


# =====================================================================
# Shared paths + shared evidence formatting - both --build-gold and
# --review need to show the judge evidence in EXACTLY the same shape,
# or the exam isn't actually testing the judge on what it'll really see.
# =====================================================================

OUTPUT_DIR = os.path.join("output", "LABEL_LOOP")
GOLD_ITEMS_PATH = os.path.join(OUTPUT_DIR, "gold_items.jsonl")
EXAM_RESULT_PATH = os.path.join(OUTPUT_DIR, "exam_result.json")
JUDGE_REVIEW_PATH = os.path.join(OUTPUT_DIR, "judge_review.jsonl")
KAPPA_REPORT_PATH = os.path.join(OUTPUT_DIR, "kappa_report.json")
QUERY_LOG_PATH = os.path.join("output", "QUERY_LOG", "query_log.jsonl")

# History logs, added 2026-08-11 - EXAM_RESULT_PATH/JUDGE_REVIEW_PATH/
# KAPPA_REPORT_PATH above are each OVERWRITTEN by every run (by design -
# they're "latest snapshot" files other code reads, e.g.
# human_review_ui.py reads JUDGE_REVIEW_PATH as the current case pool).
# That's fine for driving the next step of the pipeline, but it means
# nothing is kept once a run is superseded - no trend, no "what did the
# exam score last week" for a future dashboard. These three append one
# JSON line per run instead, full detail, nothing summarized away, so a
# dashboard can be built against them later without re-instrumenting -
# same mistake we don't want to repeat from the golden-test dashboard.
EXAM_HISTORY_PATH = os.path.join(OUTPUT_DIR, "exam_history.jsonl")
REVIEW_HISTORY_PATH = os.path.join(OUTPUT_DIR, "review_history.jsonl")
REPORT_HISTORY_PATH = os.path.join(OUTPUT_DIR, "report_history.jsonl")

PASS_BAR = float(os.environ.get("EXAM_PASS_BAR", "0.90"))
SAMPLE_SIZE_PER_COLOR = int(os.environ.get("LIVE_JUDGE_SAMPLE_SIZE", "5"))
SAMPLE_SEED = int(os.environ.get("LIVE_JUDGE_SAMPLE_SEED", "42"))  # fixed -> reproducible sampling


def _now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _append_history(path, record):
    """Appends one full JSON record as a line - never rewrites or prunes
    earlier lines, so every past run stays on disk exactly as it
    happened. Used by --exam/--review/--report so a future dashboard has
    a genuine history to read, not just whatever the most recent run
    happened to leave behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _format_evidence_block(fresh_evidence_text, preceding_context):
    """The one place both --build-gold and --review turn (fresh
    evidence, prior conversation) into the text the judge actually
    reads - see JUDGE_SYSTEM above for how it's told to weigh the two
    parts. preceding_context is agent.py's trimmed history shape:
    [{"role": "user"/"assistant", "content": "..."}, ...]."""
    parts = [f"--- Retrieved this turn ---\n{fresh_evidence_text}"]
    if preceding_context:
        convo = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in preceding_context)
        parts.append(f"--- Established earlier in this conversation ---\n{convo}")
    return "\n\n".join(parts)


# =====================================================================
# SECTION 2 - BUILD THE ENTRANCE EXAM (was build_gold_items.py)
# =====================================================================
# Why not just use the golden 30 as-is: every narrative question in the
# golden set currently passes Layer 3 (test_the_agent.py / evals.qmd) -
# the evidence already backs the answer every time. An exam built only
# from cases that already pass can't prove the judge would CATCH a bad
# one. So this takes the golden 30's real, verified answers and chunks
# (the only place in the system with confirmed ground truth) and
# deliberately constructs broken variants - Jürgen's four methods from
# PR #15's eval-label-loop discussion, retargeted at this exam, plus a
# fifth added 2026-08-07 to test the new conversation-memory handling:
#
#   1. RETRIEVAL GAP        - real search results with the correct chunk
#                              removed from OUR OWN COPY of the list.
#   2. WRONG CHUNK           - the correct chunk swapped for a real,
#                              topically plausible, but wrong one.
#   3. SYNTHESIS ISSUE       - correct chunk kept, paired with a wrong
#                              answer (borrowed from a different golden
#                              question - guaranteed unrelated).
#   4. NATURAL FAILURE       - today's real semantic_search() result,
#                              for any question where the correct chunk
#                              genuinely isn't in the top results right
#                              now. Computed live, not hardcoded.
#   5. CONVERSATION MEMORY   - no fresh evidence at all, but the correct
#                              answer already established in a
#                              constructed "earlier turn" - true_verdict
#                              is "supported". Tests that the judge
#                              correctly credits prior-conversation
#                              grounding instead of flagging every
#                              memory-only answer as unsupported for
#                              lack of a fresh tool call.
#
# SAFETY - why this can't break the real system: only ever CALLS
# search_documents.semantic_search(), a pure function that reads
# chunks_tagged.json / embeddings.npy and returns a fresh Python list -
# never touches that module's internal `chunks` / `embeddings` globals,
# never writes to those files. Every "removal" or "swap" happens on OUR
# OWN local copy of the list that function handed back. The only file
# this writes to is output/LABEL_LOOP/gold_items.jsonl - nothing else
# on disk is touched.
# =====================================================================

def _facts_as_answer(facts):
    """Turns a NARRATIVE_FACTS 'facts' list (ground_truth.py's format -
    a mix of plain strings and [alternative, alternative] lists) into
    one representative answer sentence."""
    return f"Based on the evidence: {'; '.join(_answer_terms(facts))}."


def _answer_terms(facts):
    """The flat term list behind a NARRATIVE_FACTS 'facts' list - split
    out so _facts_as_answer() and the evidence-overlap checks below
    (_kept_without_answer_evidence, _find_synthesis_donor) all use the
    exact same terms, not independently-written extractions that could
    quietly drift apart."""
    return [f[0] if isinstance(f, list) else f for f in facts]


def _term_words(facts):
    """Lowercased word set behind a facts list's terms - deliberately
    NOT built from _facts_as_answer()'s full sentence, since its
    'Based on the evidence:' boilerplate words ('the', 'evidence', ...)
    would otherwise pollute an overlap check against ordinary prose."""
    return set(w for t in _answer_terms(facts) for w in t.lower().split())


def _resolve_facts(nf):
    """Three NARRATIVE_FACTS entries (missouri_market, arizona_market,
    typical_endocrinologist) deliberately store facts=None - their real
    facts are computed LIVE from a doc/spreadsheet instead of hardcoded
    (see ground_truth.py's own comments on those entries). Mirrors
    test_the_agent.py's _layer3() fallback chain exactly, so gold items
    are built from the same facts the golden 30 is actually scored
    against. Returns None if the live computation itself fails (e.g. the
    source doc or spreadsheet is missing/unparseable) - caller should
    skip that item, same as a missing correct_chunk already does."""
    if nf["facts"] is not None:
        return nf["facts"]
    import ground_truth as gt
    if nf.get("state"):
        fact = gt.gt_state_market_fact(nf["state"])
        return [fact] if fact is not None else None
    if nf.get("specialty"):
        return gt.gt_specialty_benchmark_facts(nf["specialty"])
    return None


def _format_chunk_results(results):
    if not results:
        return "(no relevant evidence retrieved)"
    return "\n\n---\n\n".join(
        f"[{r['chunk']['chunk_id']}]\n{r['chunk']['text']}" for r in results
    )


def _find_correct_chunk(chunks, tag):
    """The one real chunk whose chunk_id contains every tag string -
    same matching rule test_the_agent.py's Layer 2 uses."""
    for c in chunks:
        if all(t in c["chunk_id"] for t in tag):
            return c
    return None


def _find_distractor(chunks, exclude_chunk_id, answer_text, doc_type):
    """First chunk, same doc_type as the correct one (topically
    plausible), that isn't the correct chunk and doesn't already
    contain the real answer terms - deterministic (first match, not
    random), so exam cases are reproducible run to run."""
    answer_words = set(answer_text.lower().split())
    for c in chunks:
        if c["chunk_id"] == exclude_chunk_id or c["doc_type"] != doc_type:
            continue
        if not (answer_words & set(c["text"].lower().split())):
            return c
    return None


def _find_synthesis_donor(narrative_qs, gt, key, correct_chunk_text):
    """First OTHER narrative question (in QUESTIONS order - deterministic,
    reproducible run to run) whose real facts DON'T already appear
    somewhere in correct_chunk's own text. The old version just took
    the next question in rotation with no such check - a real problem
    because 'monthly dosing' is the company's single core commercial
    message and repeats across nearly every doc, so a fixed +1 donor
    could easily land on facts that are, coincidentally, also sitting
    right there in the 'correct' chunk. Found 2026-08-11 via a real
    exam run: competitive_switchers__synthesis_issue was marked
    'unsupported' but the judge correctly called it supported, because
    the borrowed donor answer (zepbound_differentiator's 'monthly
    dosing') genuinely was backed by the shown evidence too - the gold
    item's premise was wrong, not the judge. Returns (None, None) if no
    non-overlapping donor exists among the golden 30."""
    chunk_words = set(correct_chunk_text.lower().split())
    for other in narrative_qs:
        other_key = other["narrative_key"]
        if other_key == key:
            continue
        other_nf = gt.NARRATIVE_FACTS.get(other_key)
        if other_nf is None:
            continue
        other_facts = _resolve_facts(other_nf)
        if other_facts is None:
            continue
        if _term_words(other_facts) & chunk_words:
            continue
        return other_key, other_facts
    return None, None


def _established_sentence(nf, resolved_facts, correct_chunk):
    """A natural, source-grounded sentence standing in for what a REAL
    prior turn's assistant message would have looked like - not just
    _facts_as_answer()'s bare term-list ('283.' / '1536; 37.'), which a
    rigorous judge correctly finds too cryptic to trust as 'established'
    on its own (found 2026-08-11: this was most of the
    conversation_memory 0/8 -> 2/8 failure - real judge_reasoning:
    'the number 283 without context', '1536; 37 without any supporting
    evidence').

    For hardcoded facts, the correct chunk's own real text already
    states them in context, so reuse it verbatim - genuinely sourced,
    nothing invented.

    For the two 'computed LIVE, not hardcoded' cases (state market
    figures, specialty benchmark - see ground_truth.py's own comments
    on missouri_market/arizona_market/typical_endocrinologist), the
    correct chunk's TEXT does NOT state the live number, and per
    ground_truth.py's own documented doc-vs-live drift may state a
    stale DIFFERENT one - quoting it here would be actively misleading.
    Those two instead get a short factual sentence built only from the
    fact's own governing label (state or specialty - both already in
    NARRATIVE_FACTS) plus the live number itself. No outside claim is
    added beyond what NARRATIVE_FACTS and the live computation already
    assert."""
    if nf["facts"] is not None:
        return correct_chunk["text"]
    if nf.get("state"):
        return (f"{nf['state']}'s HCP universe, per the state market summary, "
                f"is {resolved_facts[0]}.")
    if nf.get("specialty"):
        population, mean_propensity = resolved_facts
        pop_str = population[-1] if isinstance(population, list) else population
        return (f"For {nf['specialty']}, the benchmark population is {pop_str} "
                f"with a mean propensity score of {mean_propensity}.")
    return _facts_as_answer(resolved_facts)  # fallback - shouldn't happen given the two checks above


def _make_gold_item(item_id, method, question, evidence_text, given_answer, true_verdict, note):
    return {"item_id": item_id, "method": method, "question": question,
            "evidence_text": evidence_text, "given_answer": given_answer,
            "true_verdict": true_verdict, "note": note}


def build_gold_items():
    import search_documents
    import ground_truth as gt
    from test_the_agent import QUESTIONS

    chunks = search_documents.chunks
    narrative_qs = [q for q in QUESTIONS if "narrative_key" in q]
    items = []

    for q in narrative_qs:
        key = q["narrative_key"]
        nf = gt.NARRATIVE_FACTS.get(key)
        if nf is None:
            continue
        correct_chunk = _find_correct_chunk(chunks, nf["tag"])
        if correct_chunk is None:
            continue
        resolved_facts = _resolve_facts(nf)
        if resolved_facts is None:
            continue  # live fact computation failed (doc/spreadsheet missing) - skip, same as a missing chunk

        question_text = q["q"]
        true_answer = _facts_as_answer(resolved_facts)
        term_words = _term_words(resolved_facts)
        real_results = search_documents.semantic_search(question_text, top_k=6)
        # Correct chunk removed AND any other surviving chunk that
        # happens to already state the same facts also stripped -
        # otherwise "retrieval gap" isn't a genuine gap when a repeated
        # message (e.g. "monthly dosing") also appears verbatim in a
        # different chunk that's still in the list. Found 2026-08-11:
        # the judge was correctly reading real supporting text still in
        # "kept" and calling those items supported - the gold item's
        # premise was wrong, not the judge.
        kept = [r for r in real_results.get("results", [])
                if r["chunk"]["chunk_id"] != correct_chunk["chunk_id"]
                and not (term_words & set(r["chunk"]["text"].lower().split()))]

        # ---- 1. retrieval gap ----
        items.append(_make_gold_item(
            f"{key}__retrieval_gap", "retrieval_gap", question_text,
            _format_evidence_block(_format_chunk_results(kept), None),
            true_answer, "unsupported",
            "Correct chunk removed from our own copy of the results; the "
            "true answer is given, but nothing shown actually backs it.",
        ))

        # ---- 2. wrong chunk ----
        distractor = _find_distractor(chunks, correct_chunk["chunk_id"], true_answer,
                                       correct_chunk["doc_type"])
        if distractor is not None:
            items.append(_make_gold_item(
                f"{key}__wrong_chunk", "wrong_chunk", question_text,
                _format_evidence_block(
                    _format_chunk_results([{"chunk": distractor, "score": 0.0}]), None),
                true_answer, "unsupported",
                "A real, topically-plausible chunk substituted for the correct "
                "one; the true answer is given, but the evidence shown is "
                "about something else.",
            ))

        # ---- 3. synthesis issue (borrow another Q's real facts) ----
        donor_key, donor_facts = _find_synthesis_donor(narrative_qs, gt, key, correct_chunk["text"])
        if donor_key is not None:
            wrong_answer = _facts_as_answer(donor_facts)
            items.append(_make_gold_item(
                f"{key}__synthesis_issue", "synthesis_issue", question_text,
                _format_evidence_block(
                    _format_chunk_results([{"chunk": correct_chunk, "score": 0.0}]), None),
                wrong_answer, "unsupported",
                f"Correct evidence kept as-is, but the answer shown is "
                f"actually the real facts for a different golden question "
                f"('{donor_key}'), confirmed to not already appear in the "
                f"correct chunk's own text.",
            ))

        # ---- 4. natural failure, computed live ----
        top_chunk_ids = [r["chunk"]["chunk_id"] for r in real_results.get("results", [])]
        if correct_chunk["chunk_id"] not in top_chunk_ids:
            items.append(_make_gold_item(
                f"{key}__natural_failure", "natural_failure", question_text,
                _format_evidence_block(
                    _format_chunk_results(real_results.get("results", [])), None),
                true_answer, "unsupported",
                "Today's REAL semantic_search() result, unmodified - the "
                "correct chunk genuinely isn't in the top results right now.",
            ))

        # ---- 5. conversation memory (added 2026-08-07, fixed 2026-08-11) ----
        established_text = _established_sentence(nf, resolved_facts, correct_chunk)
        prior_turn = [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": established_text},
        ]
        items.append(_make_gold_item(
            f"{key}__conversation_memory", "conversation_memory", question_text,
            _format_evidence_block("(no fresh evidence retrieved this turn)", prior_turn),
            true_answer, "supported",
            "No fresh tool call this turn - the answer is a legitimate "
            "restatement of what was already correctly established earlier "
            "in the same conversation (shown here as real, source-grounded "
            "text, not a bare term list - see _established_sentence()). "
            "Should be judged supported, not flagged just because nothing "
            "was retrieved THIS turn.",
        ))

    return items


def cmd_build_gold():
    items = build_gold_items()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(GOLD_ITEMS_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    by_method = {}
    for item in items:
        by_method[item["method"]] = by_method.get(item["method"], 0) + 1
    print(f"Wrote {len(items)} gold items to {GOLD_ITEMS_PATH}")
    for method, count in sorted(by_method.items()):
        print(f"  {method}: {count}")
    print("\nSource files read only - chunks_tagged.json and embeddings.npy were never modified.")


# =====================================================================
# SECTION 3 - THE ENTRANCE EXAM (was run_judge_exam.py)
# =====================================================================
# Shows each gold item to the judge BLIND (question + evidence + answer
# only - never the planted true_verdict, method, or item_id). Pass bar
# is 90%, matching Jürgen's label_loop.py bar on eval-label-loop. Below
# that, --review refuses to run at all - a judge that hasn't proven it
# can spot a planted bad case has no business being trusted on a real
# one where we don't already know the answer.
# =====================================================================

def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_exam():
    # Deliberately checks EXISTENCE, not emptiness - a missing file is a
    # real setup error (forgot --build-gold), but a file that exists
    # with zero items should still produce a real (failing) exam result,
    # not an exception. Conflating the two here regressed
    # test_exam_fails_closed_with_zero_items during the 2026-08-07
    # file consolidation - caught by the test suite, fixed by splitting
    # the checks back apart.
    if not os.path.exists(GOLD_ITEMS_PATH):
        raise FileNotFoundError(f"{GOLD_ITEMS_PATH} not found - run --build-gold first.")
    items = _read_jsonl(GOLD_ITEMS_PATH)
    judge = get_judge()

    results = []
    correct = 0
    for item in items:
        verdict = judge.judge(item["question"], item["evidence_text"], item["given_answer"])
        is_correct = verdict["verdict"] == item["true_verdict"]
        correct += is_correct
        results.append({
            "item_id": item["item_id"], "method": item["method"],
            "question": item["question"], "evidence_text": item["evidence_text"],
            "given_answer": item["given_answer"],
            # question/evidence_text/given_answer added 2026-08-11 - without
            # them, a past exam run in EXAM_HISTORY_PATH only shows verdicts
            # and reasoning, not what the judge actually looked at. That's
            # invisible today (gold_items.jsonl has it), but gold_items.jsonl
            # is a "latest" snapshot --build-gold OVERWRITES every run, so
            # it can't be trusted to still match an OLDER exam run's items -
            # each history record needs to be self-contained.
            "true_verdict": item["true_verdict"], "judge_verdict": verdict["verdict"],
            "judge_reasoning": verdict["reasoning"], "correct": is_correct,
        })

    total = len(items)
    accuracy = correct / total if total else 0.0
    summary = {
        "run_at": _now_iso(), "judge_backend": judge.name,
        "total_items": total, "correct": correct,
        "accuracy": accuracy, "pass_bar": PASS_BAR,
        "passed": accuracy >= PASS_BAR and total > 0,
        "results": results,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(EXAM_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _append_history(EXAM_HISTORY_PATH, summary)  # full run, kept forever - see EXAM_HISTORY_PATH's own comment
    return summary


def cmd_exam():
    summary = run_exam()
    verdict_word = "PASSED" if summary["passed"] else "FAILED"
    print(f"Judge exam {verdict_word}: {summary['correct']}/{summary['total_items']} "
          f"correct ({summary['accuracy']:.0%}), bar = {summary['pass_bar']:.0%}, "
          f"backend = {summary['judge_backend']}")
    if not summary["passed"]:
        print("This judge is NOT cleared to review live traffic. --review will refuse to run.")
    by_method = {}
    for r in summary["results"]:
        b = by_method.setdefault(r["method"], {"correct": 0, "total": 0})
        b["total"] += 1
        b["correct"] += r["correct"]
    for method, b in sorted(by_method.items()):
        print(f"  {method}: {b['correct']}/{b['total']}")


# =====================================================================
# SECTION 4 - REVIEW A LIVE BATCH (was review_live_batch.py)
# =====================================================================
# Once (and only once) --exam says the judge passed, samples a batch of
# REAL live queries - Amber and Green only - and has the judge review
# them blind, same way it reviewed the gold items.
#
# Why not Red: a Red live query is already quarantined by chat_ui.py
# (verdict in ("fail", "audit_error")) and held for mandatory human
# review before a rep ever sees it. Running the judge on it first would
# just have it re-derive "this failed," which the gate that failed
# already told us. No value added, so Red is skipped entirely.
#
# Colour classification imports build_dashboard.py's OWN _gate_color /
# _worst_color directly (not reimplemented), so this file's idea of
# "Amber"/"Green" can never quietly drift from the real dashboard - and
# automatically picks up the 2026-08-07 fixes there (Source citations
# now amber not red; audit_error now correctly red not amber) with
# nothing to change here.
#
# SAFETY: only READS query_log.jsonl. Writes ONLY to judge_review.jsonl.
# query_log.jsonl itself is never modified.
# =====================================================================

def _row_color(row):
    from build_dashboard import _gate_color, _worst_color
    gate_rows = [{"name": g["name"], "ok": g["ok"], "color": _gate_color(g)}
                 for g in row.get("gates", [])]
    return _worst_color([g["color"] for g in gate_rows])


def _format_row_evidence(row):
    parts = []
    for ev in row.get("evidence", []):
        for section in ev.get("sections", []):
            parts.append(f"[{section.get('chunk_id', '?')}]\n{section.get('text', '')}")
    fresh = "\n\n---\n\n".join(parts) if parts else "(no evidence sections logged)"
    # preceding_context added 2026-08-07 (chat_ui.py) - a row with no
    # fresh evidence but real prior context is a legitimate follow-up,
    # not an automatic unsupported. See JUDGE_SYSTEM above.
    return _format_evidence_block(fresh, row.get("preceding_context"))


def sample_batch(rows, rng):
    green = [r for r in rows if _row_color(r) == "green"]
    amber = [r for r in rows if _row_color(r) == "amber"]
    return (rng.sample(green, min(SAMPLE_SIZE_PER_COLOR, len(green))),
            rng.sample(amber, min(SAMPLE_SIZE_PER_COLOR, len(amber))))


def review_batch(rows, judge):
    reviewed_at = _now_iso()  # one timestamp for the whole batch - they're reviewed together, in one run
    reviewed = []
    for row in rows:
        color = _row_color(row)
        evidence_text = _format_row_evidence(row)
        given_answer = row.get("answer", "")
        verdict = judge.judge(row["question"], evidence_text, given_answer)
        agrees = (verdict["verdict"] == "supported") == (color == "green")
        reviewed.append({
            "query_id": row.get("query_id"), "question": row.get("question"),
            "evidence_text": evidence_text, "given_answer": given_answer,
            "gate_color": color, "judge_verdict": verdict["verdict"],
            "judge_reasoning": verdict["reasoning"],
            "agrees_with_gates": agrees, "flagged_for_human": not agrees,
            # judge_backend/reviewed_at added 2026-08-11 so a case in
            # judge_review.jsonl or a future dashboard can be read
            # standalone, without cross-referencing a separate run log
            # for which model/when produced it.
            "judge_backend": judge.name, "reviewed_at": reviewed_at,
        })
    return reviewed


def cmd_review():
    if not os.path.exists(EXAM_RESULT_PATH):
        print("Judge is NOT cleared to review live traffic: no exam result found - run --exam first.")
        return
    with open(EXAM_RESULT_PATH, "r", encoding="utf-8") as f:
        exam = json.load(f)
    if not exam.get("passed"):
        print(f"Judge is NOT cleared to review live traffic: scored {exam.get('accuracy', 0):.0%} "
              f"on its entrance exam, below the {exam.get('pass_bar', 0):.0%} bar.")
        return

    rows = _read_jsonl(QUERY_LOG_PATH)
    if not rows:
        print(f"No rows found in {QUERY_LOG_PATH} - nothing to sample.")
        return

    rng = random.Random(SAMPLE_SEED)
    sampled_green, sampled_amber = sample_batch(rows, rng)
    judge = get_judge()
    reviewed = review_batch(sampled_green + sampled_amber, judge)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(JUDGE_REVIEW_PATH, "w", encoding="utf-8") as f:
        for r in reviewed:
            f.write(json.dumps(r) + "\n")
    # Full batch, kept forever - see REVIEW_HISTORY_PATH's own comment above.
    _append_history(REVIEW_HISTORY_PATH, {
        "run_at": _now_iso(), "judge_backend": judge.name,
        "sampled_green_n": len(sampled_green), "sampled_amber_n": len(sampled_amber),
        "rows": reviewed,
    })

    flagged = sum(1 for r in reviewed if r["flagged_for_human"])
    print(f"Judge backend: {judge.name}")
    print(f"Reviewed {len(sampled_green)} Green + {len(sampled_amber)} Amber live queries -> {len(reviewed)} total.")
    print(f"{flagged} disagreed with the gate outcome - flagged for human adjudication "
          f"(never auto-resolved, same principle as label_loop.py's --report).")
    print(f"Full results: {JUDGE_REVIEW_PATH}")
    print("query_log.jsonl itself was NOT modified.")


# =====================================================================
# SECTION 5 - JUDGE VS. HUMAN COMPARISON (was report_kappa.py)
# =====================================================================
# Cohen's kappa between the judge (judge_review.jsonl) and a human
# (human_review_ui.py's human_votes.jsonl) on the SAME sampled batch.
# Kappa, not raw agreement %, because raw agreement is misleading when
# most traffic is the easy/obvious kind - two raters who both just say
# "supported" every time would score high on raw agreement while
# telling you nothing. Disagreements are NEVER auto-resolved toward
# either rater - reported only, same as label_loop.py's adjudication
# queue; deciding who was right is a human task.
# =====================================================================

VERDICTS = ("supported", "unsupported")


def cohens_kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    a_counts = {v: sum(1 for a, _ in pairs if a == v) for v in VERDICTS}
    b_counts = {v: sum(1 for _, b in pairs if b == v) for v in VERDICTS}
    pe = sum((a_counts[v] / n) * (b_counts[v] / n) for v in VERDICTS)
    if pe == 1.0:
        return 1.0 if po == 1.0 else None
    return (po - pe) / (1 - pe)


def _load_judge_verdicts():
    return {row["query_id"]: row["judge_verdict"] for row in _read_jsonl(JUDGE_REVIEW_PATH)}


def _load_judge_backend():
    """Which backend/model produced the batch in JUDGE_REVIEW_PATH - all
    rows in one --review run share it (see review_batch()), so the
    first row's is enough. Kept separate from _load_judge_verdicts() so
    tests can monkeypatch that one alone without also having to fake
    this."""
    rows = _read_jsonl(JUDGE_REVIEW_PATH)
    return rows[0].get("judge_backend") if rows else None


def build_kappa_report():
    from human_review_ui import load_human_votes
    judge_verdicts = _load_judge_verdicts()
    human_verdicts = load_human_votes()

    shared_ids = sorted(set(judge_verdicts) & set(human_verdicts))
    pairs = [(judge_verdicts[qid], human_verdicts[qid]) for qid in shared_ids]
    disagreements = [
        {"query_id": qid, "judge_verdict": judge_verdicts[qid], "human_verdict": human_verdicts[qid]}
        for qid in shared_ids if judge_verdicts[qid] != human_verdicts[qid]
    ]
    return {
        "run_at": _now_iso(), "judge_backend": _load_judge_backend(),
        "judge_only_ids": sorted(set(judge_verdicts) - set(human_verdicts)),
        "human_only_ids": sorted(set(human_verdicts) - set(judge_verdicts)),
        "compared_n": len(shared_ids),
        "agreement_pct": (sum(1 for a, b in pairs if a == b) / len(pairs)) if pairs else None,
        "cohens_kappa": cohens_kappa(pairs),
        "disagreements": disagreements,
    }


def cmd_report():
    report = build_kappa_report()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(KAPPA_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _append_history(REPORT_HISTORY_PATH, report)  # full run, kept forever - see REPORT_HISTORY_PATH's own comment above

    n = report["compared_n"]
    if n == 0:
        print("No cases have both a judge verdict and a human vote yet - run --review then human_review_ui.py first.")
        return
    kappa = report["cohens_kappa"]
    kappa_str = f"{kappa:.2f}" if kappa is not None else "undefined (no variance in either rater)"
    print(f"Compared {n} case(s) both judge and human rated.")
    print(f"Raw agreement: {report['agreement_pct']:.0%}")
    print(f"Cohen's kappa: {kappa_str}")
    print(f"{len(report['disagreements'])} disagreement(s) - see {KAPPA_REPORT_PATH}, not auto-resolved toward either rater.")
    if report["judge_only_ids"]:
        print(f"{len(report['judge_only_ids'])} case(s) judged but not yet human-reviewed.")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build-gold", action="store_true", help="Build the entrance exam from the golden 30")
    group.add_argument("--exam", action="store_true", help="Score the judge against the entrance exam")
    group.add_argument("--review", action="store_true", help="Review a sampled batch of live queries (needs a passed exam)")
    group.add_argument("--report", action="store_true", help="Cohen's kappa vs. human_review_ui.py's votes")
    args = parser.parse_args()

    if args.build_gold:
        cmd_build_gold()
    elif args.exam:
        cmd_exam()
    elif args.review:
        cmd_review()
    elif args.report:
        cmd_report()


if __name__ == "__main__":
    main()
