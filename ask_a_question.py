# ask_a_question.py
# -------------------------------------------------------------------
# The single entry point. Looks at a question and decides which engine
# should answer it:
#   - STRUCTURED engine  -> ranking / counting / filtering (the spreadsheet)
#   - RAG (search_documents.py) -> narrative / card / characterisation questions
# This is the top-level router that ties both engines into one system.
#
# Return contract: ask() always returns (route_label: str, data: dict).
# Both engines hand back plain data here, never a pre-formatted
# sentence - formatting happens once, at the very end, in
# format_answer(). This is what makes Q20-style multi-source synthesis
# possible later: you can't cleanly merge a structured result + RAG
# text + CRM data into one answer if any of them already locked
# themselves into unrelated string formats.
#
# NOTE on warm state: like search_documents.py, query_spreadsheet.py and
# search_documents.py both load their data (Excel / embedding
# model+vectors) once, at import time. That's correct for a long-lived
# process - it only becomes a problem if ask_a_question.py itself gets
# re-imported per request once this is behind an API. Keep this a
# single warm process.
# -------------------------------------------------------------------

import re
import query_spreadsheet
import search_documents   # your existing RAG router (NPI lookup / state / semantic)

# ---- Rank / count / list cue words --------------------------------
RANK_WORDS = ["top ", "highest", "most", "rank", "leading", "biggest"]

# Word-boundary versions - same fix as search_documents.py's
# RANKING_PATTERNS. Bare substrings here ("most", "rank", "leading")
# risk matching inside unrelated words ("almost", "misleading",
# "outrank"), and "top " alone matches inside "stop " ("Why do patients
# STOP taking..." - a real bug found via an actual eval run). Use
# these patterns, not the raw list.
RANK_PATTERNS = [re.compile(r"\btop\b"), re.compile(r"\bhighest\b"),
                  re.compile(r"\bmost\b"), re.compile(r"\brank\b"),
                  re.compile(r"\bleading\b"), re.compile(r"\bbiggest\b")]
COUNT_WORDS = ["how many", "number of", "count"]
LIST_WORDS = ["list", "which ", "show me", "find ", "who are", "give me"]

# Explicit phrases for the states-aggregate report, instead of a bare
# "state" substring (which only worked before because "states" happens
# to contain it - fragile, and the same accidental-match risk flagged
# in 1_chunk_documents.py/2_tag_chunks.py/search_documents.py elsewhere
# in this project).
STATE_AGG_PHRASES = ["which states", "what states", "states with the most",
                     "states have the most", "top states"]

SPECIALTIES = {"endo": "Endocrinology", "endocrinolog": "Endocrinology", "primary care": "Primary Care",
               "obesity medicine": "Obesity Medicine", "obesity": "Obesity Medicine"}

# Casual abbreviations reps actually use. Word-boundary matched so
# "endo"/"endos" don't accidentally match inside unrelated words like
# "endorse" or "endocrine system" mentioned in passing. Found via a
# real run: "endos" matched nothing in SPECIALTIES above, so the
# specialty filter silently dropped out entirely and Primary Care /
# Obesity Medicine rows leaked into a question that only asked about
# endocrinologists.
SPECIALTY_ABBREVIATIONS = [
    (re.compile(r"\bendos?\b"), "Endocrinology"),
    (re.compile(r"\bpcps?\b"), "Primary Care"),
    (re.compile(r"\bob[\s-]?med\b"), "Obesity Medicine"),
]

# Same abbreviation set added to search_documents.py, kept consistent
# here so state detection doesn't silently work in one engine's router
# and not the other's.
STATE_ALIASES = {
    "ny": "New York", "tx": "Texas", "fl": "Florida", "ca": "California",
    "il": "Illinois", "pa": "Pennsylvania", "oh": "Ohio", "ga": "Georgia",
    "nc": "North Carolina", "mi": "Michigan", "nj": "New Jersey",
    "va": "Virginia", "wa": "Washington", "az": "Arizona", "ma": "Massachusetts",
    "mo": "Missouri", "ar": "Arkansas",
}

# Tier language - "high-propensity"/"high tier" etc, not a bare "high"
# (which would also match "high switching score" and misfire).
TIER_PATTERNS = [
    (re.compile(r"\bhigh[- ]?(?:propensity|tier)\b"), "High"),
    (re.compile(r"\bmedium[- ]?(?:propensity|tier)\b"), "Medium"),
    (re.compile(r"\blow[- ]?(?:propensity|tier)\b"), "Low"),
]

# Competitor language, same shape as 2_tag_chunks.py's
# COMPETITOR_KEYWORDS so the two stay easy to keep in sync if a third
# competitor is added.
COMPETITOR_KEYWORDS = {
    "novo_nordisk": {"words": ["semaglutide", "ozempic", "wegovy", "novo nordisk"],
                     "label": "Novo Nordisk"},
    "eli_lilly": {"words": ["tirzepatide", "mounjaro", "zepbound", "eli lilly", "lilly"],
                  "label": "Eli Lilly"},
}

# Explicit switching-score threshold ("switching score above 0.7") takes
# priority; "high switching" alone falls back to an assumed cutoff.
# NOTE: 0.6 is a placeholder assumption, not a value confirmed against
# your actual switching_score distribution - worth checking against the
# real data before relying on this default in production questions.
_SWITCHING_EXPLICIT = re.compile(r"switching[^0-9]{0,15}([\d.]+)")
_SWITCHING_HIGH = re.compile(r"\bhigh[- ]?switching\b")
DEFAULT_HIGH_SWITCHING_THRESHOLD = 0.6

_TARGETED_NOT = re.compile(r"\b(?:not\s+(?:currently\s+)?targeted|untargeted)\b")
_TARGETED_YES = re.compile(r"\b(?:currently\s+|already\s+)?targeted\b")

# A bare competitor mention isn't enough on its own to mean "filter HCPs
# by dominant_competitor" - "How is Rabivy different from Zepbound?" also
# mentions a competitor brand, but it's a narrative comparison question,
# not an HCP-targeting one. Require actual HCP/targeting context before
# treating a competitor mention as a structured-filter trigger.
HCP_CONTEXT_WORDS = ["hcp", "prescriber", "writer", "doctor", "physician",
                      "clinician", "targeted", "target"]


def _has_hcp_context(ql):
    return any(w in ql for w in HCP_CONTEXT_WORDS) or _detect_specialty(ql) is not None


# Default result-list size when the question doesn't say. 20 is a
# working compromise: generous enough that narrow filters (state +
# specialty + tier + targeted) almost never get silently cut short,
# but not so large that a very broad filter dumps hundreds of rows
# into an answer nobody asked to see the whole of.
DEFAULT_RESULT_LIMIT = 20

_ALL_PATTERN = re.compile(r"\ball\b|\bevery(one)?\b")


def _resolve_top(ql):
    """Decide how many results to return: an explicit number ('top 30')
    always wins: 'all'/'every' explicitly means no limit; otherwise the
    default above applies. This is the one place that policy lives, so
    both the filter path and the propensity-ranking path stay
    consistent rather than drifting apart."""
    top_match = re.search(r"top\s+(\d+)", ql)
    if top_match:
        return int(top_match.group(1))
    if _ALL_PATTERN.search(ql):
        return None
    return DEFAULT_RESULT_LIMIT


def _find_state(ql):
    for s in query_spreadsheet.df["state"].unique():
        if s in ql:
            # df's state column is lowercase (asserted in
            # query_spreadsheet.py); return the title-cased form for
            # display/matching consistency.
            return s.title()
    for abbr, full in STATE_ALIASES.items():
        if re.search(r"\b" + abbr + r"\b", ql):
            return full
    return None


def _detect_specialty(ql):
    for key, val in SPECIALTIES.items():
        if key in ql:
            return val
    for pattern, val in SPECIALTY_ABBREVIATIONS:
        if pattern.search(ql):
            return val
    return None


def _detect_tier(ql):
    for pattern, label in TIER_PATTERNS:
        if pattern.search(ql):
            return label
    return None


def _detect_targeted(ql):
    if _TARGETED_NOT.search(ql):
        return 0
    if _TARGETED_YES.search(ql):
        return 1
    return None


def _detect_competitor(ql):
    for info in COMPETITOR_KEYWORDS.values():
        for w in info["words"]:
            if re.search(r"\b" + re.escape(w) + r"\b", ql):
                return info["label"]
    return None


def _detect_min_switching(ql):
    m = _SWITCHING_EXPLICIT.search(ql)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    if _SWITCHING_HIGH.search(ql):
        return DEFAULT_HIGH_SWITCHING_THRESHOLD
    return None


# PA burden (pa_burden column, 0-1 continuous) is a DIFFERENT concept from
# formulary_tier="PARequired" (a categorical status) - both can involve the
# words "PA"/"prior auth", so this specifically requires the word "burden"
# to avoid the two colliding. "prior authorization burden" or "PA burden"
# -> this; "PA required"/"prior auth required" (no "burden") -> formulary.
_PA_BURDEN_EXPLICIT_ABOVE = re.compile(
    r"(?:pa|prior auth\w*)\s*burden[^0-9]{0,15}(?:above|over|greater than|>)\s*([\d.]+)")
_PA_BURDEN_EXPLICIT_BELOW = re.compile(
    r"(?:pa|prior auth\w*)\s*burden[^0-9]{0,15}(?:below|under|less than|<)\s*([\d.]+)")
_PA_BURDEN_HIGH = re.compile(r"\bhigh[- ]?(?:pa|prior auth\w*)[- ]?burden\b")
_PA_BURDEN_LOW = re.compile(r"\blow[- ]?(?:pa|prior auth\w*)[- ]?burden\b")
DEFAULT_HIGH_PA_BURDEN_THRESHOLD = 0.6
DEFAULT_LOW_PA_BURDEN_THRESHOLD = 0.3


def _detect_pa_burden(ql):
    """Returns (min_pa_burden, max_pa_burden) - at most one will be set.
    An explicit number ('PA burden above 0.7') always wins over the
    'high'/'low' assumed-threshold fallback."""
    m = _PA_BURDEN_EXPLICIT_ABOVE.search(ql)
    if m:
        try:
            return float(m.group(1)), None
        except ValueError:
            pass
    m = _PA_BURDEN_EXPLICIT_BELOW.search(ql)
    if m:
        try:
            return None, float(m.group(1))
        except ValueError:
            pass
    if _PA_BURDEN_HIGH.search(ql):
        return DEFAULT_HIGH_PA_BURDEN_THRESHOLD, None
    if _PA_BURDEN_LOW.search(ql):
        return None, DEFAULT_LOW_PA_BURDEN_THRESHOLD
    return None, None


# =====================================================================
# GENERIC NUMERIC-COLUMN FALLBACK
# -------------------------------------------------------------------
# Hand-built detectors above (tier, targeted, competitor, switching,
# formulary, PA burden) exist for the columns a rep would naturally ask
# about in plain English - each needed its own translation from casual
# phrasing to the exact column/value ("high propensity" -> tier="High").
#
# The spreadsheet has ~30 columns total, and most of the rest (e.g.
# days_since_contact, rep_engagement_score, rx_volume_z, logit_score)
# don't have a natural-language phrasing worth hand-building - nobody
# says "logit score" in conversation. Rather than writing 20+ more
# one-off regexes, this is a single generic mechanism: name the real
# column, add a comparison and a number, and it works for ANY numeric
# column without new code. The tradeoff is it needs the literal column
# name (with spaces/hyphens/underscores all accepted) rather than
# natural phrasing - e.g. "days_since_contact over 30" or "rep
# engagement score above 0.5", not a casual paraphrase of the concept.
# =====================================================================
_GENERIC_EXCLUDE_COLUMNS = {"npi", "hcp_id", "targeted", "sample_request_recent", "zero_writer"}
_OP_ABOVE = r"(?:above|over|greater than|more than|higher than|>)"
_OP_BELOW = r"(?:below|under|less than|fewer than|lower than|<)"


def _column_phrase_pattern(col):
    """Turn a real column name like 'days_since_contact' into a regex
    that matches it however someone might type it: with underscores,
    spaces, or hyphens."""
    parts = col.split("_")
    return r"[ _-]+".join(re.escape(p) for p in parts)


def _build_generic_numeric_patterns():
    try:
        numeric_cols = query_spreadsheet.df.select_dtypes(include="number").columns
    except Exception:
        numeric_cols = []
    patterns = {}
    for col in numeric_cols:
        if col in _GENERIC_EXCLUDE_COLUMNS:
            continue
        phrase = _column_phrase_pattern(col)
        patterns[col] = (
            re.compile(rf"\b{phrase}\b[^0-9]{{0,20}}{_OP_ABOVE}\s*([\d.]+)"),
            re.compile(rf"\b{phrase}\b[^0-9]{{0,20}}{_OP_BELOW}\s*([\d.]+)"),
        )
    return patterns


# Built once, at import time, directly from the real spreadsheet's
# columns - if a new numeric column is ever added to the spreadsheet,
# this fallback covers it automatically, with no code change needed.
GENERIC_NUMERIC_PATTERNS = _build_generic_numeric_patterns()


def _detect_generic_numeric_filters(ql):
    """Returns {column: (min_val, max_val)} for any numeric column
    explicitly named in the question with a comparison. Empty dict if
    none matched - this is a fallback, not a replacement for the
    hand-built detectors above."""
    found = {}
    for col, (above_pat, below_pat) in GENERIC_NUMERIC_PATTERNS.items():
        m_above = above_pat.search(ql)
        m_below = below_pat.search(ql)
        min_v = float(m_above.group(1)) if m_above else None
        max_v = float(m_below.group(1)) if m_below else None
        if min_v is not None or max_v is not None:
            found[col] = (min_v, max_v)
    return found


# Formulary status language -> the real formulary_tier column values.
# "non-preferred" is checked before bare "preferred", since "preferred"
# is a literal substring of "non-preferred" (word-boundary regex alone
# doesn't protect against that - the hyphen is a boundary character
# too) - same ordering lesson as the targeted/not-targeted detection.
def _detect_formulary(ql):
    if re.search(r"\bnon[- ]?preferred\b", ql):
        return "NonPreferred"
    if re.search(r"\bpreferred\b", ql):
        return "Preferred"
    if re.search(r"\bpa[- ]?required\b|\bprior auth", ql):
        return "PARequired"
    if re.search(r"\bnot[- ]?covered\b|\buncovered\b", ql):
        return "NotCovered"
    return None


# Recent sample-request language -> the real sample_request_recent
# column (0/1). Checked for a negative phrasing first, same reasoning
# as _detect_formulary above.
_SAMPLE_NO = re.compile(r"\bno sample request\b|\bwithout (?:a )?sample request\b|"
                         r"\bhas(?:n't| not) requested (?:a )?sample\b")
_SAMPLE_YES = re.compile(r"\bsample request(?:ed)?\b|\brequested (?:a )?sample\b")


def _detect_sample_request(ql):
    if _SAMPLE_NO.search(ql):
        return 0
    if _SAMPLE_YES.search(ql):
        return 1
    return None


# "high-volume writers" (Q8-style) means sort by actual Rx volume, not
# the default propensity_score - otherwise a "high-volume" question
# would come back ranked by the wrong column entirely.
_HIGH_VOLUME_PATTERN = re.compile(r"\bhigh[- ]?volume\b|\bhighest volume\b")


def _detect_sort_by(ql):
    if _HIGH_VOLUME_PATTERN.search(ql):
        return "rx_volume_monthly"
    return "propensity_score"


def ask(question):
    ql = question.lower()

    # 1. "how many ... writers in X" -> count. Also catches abbreviated
    # fragments like "active writers in TX?" that drop the explicit
    # how-many/count phrase entirely - there's no other structured
    # function "active <role> in <state>" could reasonably mean besides
    # this count, so it's safe to widen the trigger rather than let it
    # fall to RAG (which can't answer a numeric question at all).
    if (any(w in ql for w in COUNT_WORDS) or "active" in ql) and ("writer" in ql or "prescriber" in ql or "hcp" in ql):
        st = _find_state(ql)
        if st:
            return "STRUCTURED / count", query_spreadsheet.count_writers(st)

    # 2. "how many scripts did NPI ... write" -> single-NPI script lookup
    m = re.search(r"\b(\d{10})\b", question)
    if m and ("script" in ql or "rx" in ql or "write" in ql or "wrote" in ql):
        return "STRUCTURED / hcp scripts", query_spreadsheet.hcp_scripts(m.group(1))

    # 3. Multi-field targeting filter - THE previously-missing path.
    # Triggered by targeted-status, competitor, switching-score,
    # formulary-status, or recent-sample-request language, none of
    # which any other path (here or in search_documents.py) can
    # answer, since chunks_tagged.json has none of these fields.
    # Tier alone doesn't trigger this - see note at path 6 for why.
    targeted = _detect_targeted(ql)
    competitor = _detect_competitor(ql)
    min_switching = _detect_min_switching(ql)
    formulary_tier = _detect_formulary(ql)
    sample_request = _detect_sample_request(ql)
    min_pa_burden, max_pa_burden = _detect_pa_burden(ql)
    extra_filters = _detect_generic_numeric_filters(ql)
    competitor_signal = competitor is not None and _has_hcp_context(ql)
    # Same guard as competitor detection above, and for the same reason:
    # "prior auth" is meant to catch "PA required" as a filter value, but
    # it's also a literal substring of "prior authorization" - a general
    # narrative question like "How is prior authorization affecting
    # access?" was getting hijacked into this filter path instead of
    # going to RAG, because formulary_tier alone was enough to trigger
    # it. Require actual HCP-targeting context, same as competitor.
    formulary_signal = formulary_tier is not None and _has_hcp_context(ql)
    sample_signal = sample_request is not None and _has_hcp_context(ql)
    pa_burden_signal = (min_pa_burden is not None or max_pa_burden is not None) and _has_hcp_context(ql)
    extra_signal = bool(extra_filters) and _has_hcp_context(ql)
    if (targeted is not None or competitor_signal or min_switching is not None
            or formulary_signal or sample_signal or pa_burden_signal or extra_signal):
        # Default 20 if unspecified; "top 30" or "all"/"every" override it.
        # See _resolve_top() - one shared policy, not duplicated per path.
        return "STRUCTURED / filter", query_spreadsheet.filter_hcps(
            state=_find_state(ql),
            specialty=_detect_specialty(ql),
            tier=_detect_tier(ql),
            targeted=targeted,
            dominant_competitor=competitor,
            min_switching=min_switching,
            formulary_tier=formulary_tier,
            recent_sample_request=sample_request,
            min_pa_burden=min_pa_burden,
            max_pa_burden=max_pa_burden,
            extra_filters=extra_filters or None,
            sort_by=_detect_sort_by(ql),
            top=_resolve_top(ql),
        )

    # 4. "which states have the most high-tier HCPs" -> aggregate report.
    # Explicit phrase match instead of a bare "state" substring.
    if any(p in ql for p in STATE_AGG_PHRASES) and "high" in ql:
        return "STRUCTURED / states by high tier", query_spreadsheet.states_by_high_tier()

    # 5. "top prescriber in X" -> ranking within a state (volume, not propensity)
    if any(p.search(ql) for p in RANK_PATTERNS) and "propensity" not in ql:
        st = _find_state(ql)
        if st and ("prescriber" in ql or "writer" in ql or "volume" in ql):
            return "STRUCTURED / top prescriber", query_spreadsheet.top_prescriber(st)

    # 6. "top N by propensity[, tier]" -> ranking by propensity.
    # If a state is also named, this used to silently drop it (a real
    # bug: "top prescriber in Texas by propensity" matched here with no
    # state parameter available on top_n_by_propensity at all). Now: a
    # named state routes through filter_hcps instead, which does
    # support state + sort_by, so the state constraint is honoured.
    # Tier is only applied if the question actually named one - no more
    # silently defaulting every unscoped "top N by propensity" to High.
    if any(p.search(ql) for p in RANK_PATTERNS) and "propensity" in ql:
        n = _resolve_top(ql)
        tier = _detect_tier(ql)
        st = _find_state(ql)
        if st:
            return "STRUCTURED / top by propensity (state)", query_spreadsheet.filter_hcps(
                state=st, tier=tier, sort_by="propensity_score", top=n)
        return "STRUCTURED / top by propensity", query_spreadsheet.top_n_by_propensity(n, tier=tier)

    # 7. Otherwise -> hand it to the RAG (narrative / cards / semantic).
    # Returns actual chunk text + source metadata, not just chunk_id/score
    # - a bare ID list can't feed an answer-synthesis step that needs
    # the real retrieved content.
    route, results = search_documents.search(question)
    low_confidence = "low confidence" in route or not results
    return "RAG", {
        "route": route,
        "low_confidence": low_confidence,
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "score": score,
                "doc_type": chunk["doc_type"],
                "source_doc": chunk.get("source_doc"),
                "text": chunk["text"],
            }
            for chunk, score in results
        ],
    }


# =====================================================================
# PRESENTATION - formats the (route, data) pair for display. Kept
# separate from ask() itself, same principle as query_spreadsheet.py's
# format_* functions.
# =====================================================================

def format_answer(route, data):
    if route.startswith("STRUCTURED"):
        if "results" in data and "n" in data:
            return query_spreadsheet.format_top_n_by_propensity(data)
        if "results" in data and "count" in data:
            return query_spreadsheet.format_filter_hcps(data)
        if "results" in data:
            return query_spreadsheet.format_states_by_high_tier(data)
        if "active" in data:
            return query_spreadsheet.format_count_writers(data)
        if "rx_volume_monthly" in data and "state" in data:
            return query_spreadsheet.format_top_prescriber(data)
        if "rx_volume_monthly" in data:
            return query_spreadsheet.format_hcp_scripts(data)
        if "error" in data:
            return data["error"]
        return str(data)

    if route == "RAG":
        if data["low_confidence"]:
            header = f"(RAG route: {data['route']}) - no confident match found.\n"
        else:
            header = f"(RAG route: {data['route']})\n"
        lines = [header]
        for c in data["chunks"]:
            tag = f"[{c['score']:.3f}]" if c["score"] is not None else "[card]"
            lines.append(f"  {tag} {c['chunk_id']} ({c['doc_type']}): {c['text'][:150]}")
        return "\n".join(lines)

    return str(data)


if __name__ == "__main__":
    questions = [
        "Who is the top GLP-1 prescriber in New York?",                          # structured, rank/state
        "How many active GLP-1 writers are in Texas?",                          # structured, count
        "List the top 10 High-tier prescribers by propensity",                  # structured, national rank
        "What's the top prescriber by propensity in Texas?",                    # rank+state+propensity collision case
        "Show me High-propensity endocrinologists in Florida who are not currently targeted",  # Q6 - the previously-missing filter path
        "Which HCPs have a high switching score and are on Novo Nordisk?",       # filter, competitor+switching
        "How is Rabivy different from Zepbound?",                               # RAG
        "asdkfj nonsense question with no signal at all",                       # RAG, low confidence
    ]
    for q in questions:
        engine, data = ask(q)
        print("=" * 72)
        print(f"Q: {q}")
        print(f"   -> {engine}")
        print(format_answer(engine, data))
        print()