# article_ground_truth.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# Same design principle as ground_truth.py's gt_state_market_fact:
# ground truth = a live lookup against the real source, never a
# hand-guessed value. For real journal articles, chunk_ids are
# machine-numbered paragraphs (e.g. "..._introduction_p5"), not
# human-named tags like "competitive_switchers" - so instead of
# hardcoding which paragraph number holds a fact, this finds the
# chunk LIVE by searching chunks_tagged.json for a distinctive
# substring unique to that fact.
#
# gt_article_chunk_for_fact() is deliberately STRICT: if a substring
# matches zero or more-than-one chunk, it returns None rather than
# guessing - an ambiguous substring is a real signal (either the fact
# genuinely isn't there, or the substring needs tightening), not
# something to silently paper over. Import this into
# test_the_agent.py's QUESTIONS the same way NARRATIVE_FACTS is used.
# =====================================================================

import json

with open("output/chunks_tagged.json", "r", encoding="utf-8") as f:
    _ALL_CHUNKS = json.load(f)

ARTICLE_CHUNKS = [c for c in _ALL_CHUNKS if c["doc_type"] == "articles"]


# ---- CHUNK: gt_article_chunk_for_fact ----
def gt_article_chunk_for_fact(distinctive_substring, doc_filter=None):
    """Finds the chunk_id containing a distinctive fact - live, not
    hardcoded. Returns a single chunk_id string, or None if the
    substring is missing or ambiguous (matches != 1 chunk)."""
    subs = distinctive_substring.lower()
    matches = [
        c for c in ARTICLE_CHUNKS
        if subs in c["text"].lower()
        and (doc_filter is None or doc_filter.lower() in c["source_doc"].lower())
    ]
    if len(matches) != 1:
        return None
    return matches[0]["chunk_id"]


def gt_article_chunks_for_fact_any_of(substrings, doc_filter=None):
    """For facts that are LEGITIMATELY stated in more than one real
    paragraph (confirmed by reading the source, not a search bug) -
    e.g. Meier's paper introduces 'Ominous Octet' in one paragraph and
    elaborates on it in the next. Returns the list of valid chunk_ids
    (an OR-group, same convention gt.any_present already uses) rather
    than forcing a single answer where the source itself has two."""
    ids = []
    for s in substrings:
        cid = gt_article_chunk_for_fact(s, doc_filter=doc_filter)
        if cid:
            ids.append(cid)
    return ids or None


# =====================================================================
# ARTICLE_FACTS - single-source, single-fact questions (Group A)
# Each entry: distinctive substring(s) to locate the RIGHT chunk live,
# doc_filter to make sure we're searching within the correct paper,
# and the fact(s) that must appear in the final answer (Layer 3).
# =====================================================================
ARTICLE_FACTS = {
    "flow_egfr_decline": {
        "doc_filter": "Cooper",
        "substring": "1.16 mL/min",
        "facts": ["1.16 ml/min"],  # was bare "1.16" - tie to the unit so a stray "1.16" elsewhere can't false-match
        "q": "What eGFR decline benefit did semaglutide show in the FLOW trial, per Cooper & van Raalte?",
    },
    "discontinuation_rate": {
        "doc_filter": "Cooper",
        "substring": "5% and 10%",
        # FIXED: was ["5", "10"] - bare digits ALWAYS collide (page numbers,
        # scores, even the DOI "10.1172" literally contains "10"). Confirmed
        # on a real run: agent cited the WRONG source with the WRONG number
        # (28.2%, from an unrelated doc) and this still scored L3 PASS,
        # since "5" and "10" both trivially appear elsewhere regardless.
        # Now requires the phrase together, not two independent bare digits.
        "facts": ["5% and 10%"],
        "q": "What percentage of subjects stopped GLP-1 receptor agonist treatment due to side effects, per Cooper & van Raalte?",
    },
    "receptor_location_cooper": {
        "doc_filter": "Cooper",
        "substring": "major site of expression appears to be in the renin-expressing juxtaglomerular",
        "facts": ["juxtaglomerular"],
        "q": "Where is the GLP-1 receptor mainly expressed in the kidney, per Cooper & van Raalte?",
    },
    "select_trial_reduction": {
        "doc_filter": "Cooper",
        "substring": "reduced the composite kidney endpoint by 22%",
        "facts": ["22%"],  # was bare "22" - tightened with the percent sign
        "q": "What kidney endpoint reduction did semaglutide show in the SELECT trial (non-diabetic patients), per Cooper & van Raalte?",
    },
    "ominous_octet": {
        "doc_filter": "Meier",
        "substrings_any": ["Ominous Octet\"", "DeFronzo's \"Ominous Octet\""],  # legitimately in 2 real paragraphs
        "facts": ["Ominous Octet"],
        "q": "What did DeFronzo call the 8 defects behind type 2 diabetes, per Meier et al.?",
    },
    "receptor_location_meier": {
        "doc_filter": "Meier",
        "substring": "proximal tubular cells",
        "facts": ["proximal tubular"],
        "q": "Where are GLP-1 receptors predominantly expressed in the kidney, per Meier et al.?",
        # DELIBERATE CROSS-DOCUMENT TRAP: Cooper says "juxtaglomerular",
        # Meier says "proximal tubular" - same general topic, two
        # different papers' specific claims. Tests whether retrieval
        # pulls THIS paper's claim, not a blended/contaminated answer.
    },
    "rage_deletion_effect": {
        "doc_filter": "Meier",
        "substring": "RAGE protein levels",
        # FIXED: "increased" alone is a common English word (same class of
        # bug as ground_truth.py's own documented "persistence"/"discontinue"
        # fix) - tied to "RAGE" as one phrase instead of two independent facts.
        "facts": ["increased RAGE"],
        "q": "What happens to RAGE protein levels when GLP-1 receptor is deleted, per Meier et al.?",
    },
    "chen_pooled_or": {
        "doc_filter": "Chen",
        "substring": "OR, 0.85 [95% CI, 0.77-0.94]",
        "facts": ["0.85"],  # a decimal this specific has low collision risk - left as is
        "q": "What was the pooled odds ratio for composite kidney outcome in Chen et al.'s meta-analysis?",
    },
    "chen_patient_count": {
        "doc_filter": "Chen",
        "substring": "17,996",
        # FIXED: was ["17,996", "17996"] as two SEPARATE top-level facts -
        # gt.all_present treats top-level items as AND (both required
        # simultaneously), but real text only ever uses ONE format at a
        # time. Meant as an OR-group (either format acceptable), same
        # convention as ground_truth.py's own _with_comma_variant - now
        # correctly nested as one list. Confirmed on a real run: the
        # agent's answer literally said "17,996 patients" and still
        # scored a false L3 FAIL under the old (broken) structure.
        "facts": [["17,996", "17996"]],
        "q": "How many patients did Chen et al. include with baseline eGFR under 60?",
    },
    "chen_sensitivity_or": {
        "doc_filter": "Chen",
        "substring": "0.83",
        "facts": ["0.83"],  # same reasoning as chen_pooled_or - specific decimal, low risk
        "q": "What did excluding exendin-4-backbone agents do to the kidney-outcome odds ratio, per Chen et al.?",
    },
}


# =====================================================================
# SYNTHESIS_FACTS - cross-document questions (Group B). Ground truth
# here is a SET of facts, each anchored to a DIFFERENT source document -
# Layer 2 for these checks that chunks from MULTIPLE documents got
# retrieved (not just one), Layer 3 checks the answer reflects the
# shared conclusion across sources.
# =====================================================================
# =====================================================================
# SYNTHESIS_FACTS - cross-document questions (Group B). Ground truth
# here is a SET of facts, each anchored to a DIFFERENT source document -
# Layer 2 for these checks that chunks from MULTIPLE documents got
# retrieved (not just one), Layer 3 checks the answer reflects the
# shared conclusion across sources.
#
# HONEST LIMITATION, not fixed: unlike the single-source facts above,
# these can't be tightened the same way - they're testing a SYNTHESIZED
# conclusion, not a verbatim quote, so topic words like "diabetes" or
# "type 2 diabetes" are unavoidably common throughout ANY GLP-1/kidney
# evidence blob regardless of whether the agent's answer actually
# reflects the right synthesized conclusion. Treat L3 PASS/FAIL on
# these two specifically as a weaker signal than the single-source
# questions - worth eyeballing the actual answer text yourself rather
# than trusting the automated check alone, same caution as the
# pretraining-recall flag elsewhere in this file.
# =====================================================================
SYNTHESIS_FACTS = {
    "diabetic_vs_nondiabetic_evidence": {
        "q": "Across Cooper, Meier, and Chen, is the kidney-protective benefit of "
             "GLP-1 receptor agonists more established in patients with diabetes or without diabetes?",
        "per_document_substrings": {
            "Cooper": "extended to individuals without T2D",
            "Meier": "role in nondiabetic CKD is less clear and is not well established",
            "Chen": "the only large GLP-1 receptor agonist study enrolling nondiabetic patients",
        },
        "facts": [["diabetes", "diabetic"], ["stronger", "established", "more evidence", "well established"]],
    },
    "dominant_evidence_population": {
        "q": "What population does most current GLP-1 receptor agonist clinical trial "
             "evidence for kidney outcomes focus on, according to these three papers?",
        "per_document_substrings": {
            "Cooper": "initial clinical studies focused on patients with T2D",
            "Meier": "a dedicated renal outcomes trial in nondiabetic CKD remains an important unmet need",
            "Chen": "all others were conducted in diabetic populations",
        },
        "facts": [["type 2 diabetes", "diabetic kidney disease", "t2d", "diabetic populations"]],
    },
}


# =====================================================================
# QUESTIONS - the actual list to import into test_the_agent.py (Group A
# questions) or feed into the retrieval eval (test_article_retrieval.py).
# Each entry resolves its own ground-truth chunk_id(s) live via the
# functions above, so nothing here is a frozen/guessed value.
# =====================================================================
def _build_questions():
    questions = []
    for key, spec in ARTICLE_FACTS.items():
        if "substrings_any" in spec:
            expected_ids = gt_article_chunks_for_fact_any_of(spec["substrings_any"], doc_filter=spec["doc_filter"])
        else:
            cid = gt_article_chunk_for_fact(spec["substring"], doc_filter=spec["doc_filter"])
            expected_ids = [cid] if cid else None
        questions.append({
            "key": key, "type": "single_source", "q": spec["q"],
            "expected_chunk_ids": expected_ids, "facts": spec["facts"],
        })
    for key, spec in SYNTHESIS_FACTS.items():
        expected_ids = {}
        for doc, substr in spec["per_document_substrings"].items():
            expected_ids[doc] = gt_article_chunk_for_fact(substr, doc_filter=doc)
        questions.append({
            "key": key, "type": "synthesis", "q": spec["q"],
            "expected_chunk_ids_by_doc": expected_ids, "facts": spec["facts"],
        })
    return questions


QUESTIONS = _build_questions()


if __name__ == "__main__":
    # Standalone verification: proves every question's ground truth
    # actually resolves against the real chunks_tagged.json BEFORE
    # anything gets wired into a live agent/retrieval test - the same
    # "trust but verify" spirit as ground_truth.py, just runnable on
    # its own since there's no live agent call involved at this stage.
    print(f"Loaded {len(ARTICLE_CHUNKS)} article chunks from output/chunks_tagged.json\n")
    n_pass, n_fail = 0, 0
    for q in QUESTIONS:
        if q["type"] == "single_source":
            ok = bool(q["expected_chunk_ids"])
            detail = q["expected_chunk_ids"]
        else:
            ok = all(q["expected_chunk_ids_by_doc"].values())
            detail = q["expected_chunk_ids_by_doc"]
        n_pass += ok
        n_fail += not ok
        print(f"{'PASS' if ok else 'FAIL'} [{q['type']:13}] {q['key']}")
        if not ok:
            print(f"       -> {detail}")
    print(f"\n{n_pass}/{len(QUESTIONS)} questions have resolvable ground truth.")
    if n_fail:
        print(f"! {n_fail} question(s) need a tighter/different substring - see FAIL lines above.")