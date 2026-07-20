# ask_a_question.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# This is the ONLY file in the system that tries to understand plain
# English. Every other file (query_spreadsheet.py, search_documents.py)
# just does clean, mechanical lookups when told exactly what to look
# up - this file's whole job is reading a rep's real sentence and
# deciding what that actually means.
#
# It picks one of two engines:
#   - STRUCTURED (query_spreadsheet.py) - for ranking / counting /
#     filtering questions, where the answer is real spreadsheet rows.
#   - RAG (search_documents.py) - for narrative / card / "tell me
#     about" questions, where the answer is document text.
#
# THE MOST IMPORTANT RULE IN THIS FILE: if a real answer can't be
# found - a bad NPI, an unrecognized state, a filter matching nothing,
# a document search with no confident match - this file says so
# plainly. It never quietly hands back something unrelated and lets
# the rep mistake it for a real answer.
#
# Every function below returns a plain dictionary (via ask()), never a
# sentence - format_answer() at the bottom of this file is the only
# place a sentence gets written, once, in one consistent way.
# =====================================================================

import re
import query_spreadsheet as qs
import search_documents as sd


# =====================================================================
# SECTION 1 - WORD LISTS AND SYNONYM TABLES
# -----------------------------------------------------------------
# This is the "dictionary" this file uses to translate plain English
# into the exact real values query_spreadsheet.py and search_documents.py
# understand. Every value on the RIGHT-hand side of these mappings
# must be a real value that actually exists in the spreadsheet today -
# the self-check at the bottom of this section confirms that on every
# run, so a data change can never silently break these mappings again
# without anyone noticing.
# =====================================================================

# ---- Rank / count / list language -----------------------------------
# Word-boundary patterns, not plain substrings - "top " alone would
# also match inside "stop taking" (a real bug found via testing), and
# "most"/"rank"/"leading" as bare substrings would match inside
# "almost"/"misleading"/"outrank".
RANK_PATTERNS = [re.compile(r"\btop\b"), re.compile(r"\bhighest\b"),
                  re.compile(r"\bmost\b"), re.compile(r"\brank\b"),
                  re.compile(r"\bleading\b"), re.compile(r"\bbiggest\b"),
                  re.compile(r"\bgreatest\b"), re.compile(r"\bnumber[- ]?one\b")]
COUNT_WORDS = ["how many", "number of", "count"]

# ONE shared list-word vocabulary, used both for "does this question
# want a plain list" (structured filter path) and "does this question
# want a list of HCP cards" (RAG path). Previously these lived as two
# separately-typed copies in two different files and had drifted apart
# from each other - a classic way for a bug to sneak back in later.
LIST_WORDS = ["list", "which ", "show me", "find ", "who are", "give me",
              "prescribers", "doctors", "hcps", "physicians", "clinicians", "targets"]

# Market/overview language - a question with this AND a state name is
# a narrative "what does the market look like" question, not a request
# for a plain list of cards.
MARKET_WORDS = ["market", "landscape", "summary", "overview", "narrative",
                 "happening", "situation", "picture", "going on", "how is",
                 "what's the story", "whats the story"]

# ---- Specialty -------------------------------------------------------
SPECIALTIES = {"endo": "Endocrinology", "endocrinolog": "Endocrinology",
               "primary care": "Primary Care", "obesity medicine": "Obesity Medicine"}
# NOTE: bare "obesity" is deliberately NOT mapped here - it's the
# DISEASE name, not the specialty, and matches constantly in narrative
# questions ("Medicaid coverage for obesity", "obesity prevalence")
# that have nothing to do with filtering by specialty. Confirmed bug:
# "Does Medicaid cover GLP-1s for obesity?" was misread as
# specialty=Obesity Medicine and wrongly routed to the structured
# filter path instead of the real narrative answer.
SPECIALTY_ABBREVIATIONS = [
    (re.compile(r"\bendos?\b"), "Endocrinology"),
    (re.compile(r"\bpcps?\b"), "Primary Care"),
    (re.compile(r"\bob[\s-]?med\b"), "Obesity Medicine"),
]

# ---- State + region ---------------------------------------------------
STATE_ALIASES = {
    "ny": "New York", "tx": "Texas", "fl": "Florida", "ca": "California",
    "il": "Illinois", "pa": "Pennsylvania", "oh": "Ohio", "ga": "Georgia",
    "nc": "North Carolina", "mi": "Michigan", "nj": "New Jersey",
    "va": "Virginia", "wa": "Washington", "az": "Arizona", "ma": "Massachusetts",
    "mo": "Missouri", "ar": "Arkansas",
}

# Region words. Only the FOUR real Census regions in this data get a
# mapping (Midwest / Northeast / South / West) - there is no "North"
# region in the real data (Northeast and Midwest split what a rep
# might loosely call "the north"), so a bare "north"/"northern" is
# deliberately left UNMAPPED rather than guessed at.
REGION_WORDS = {
    "south": "South", "southern": "South",
    "west": "West", "western": "West",
    "midwest": "Midwest", "midwestern": "Midwest",
    "northeast": "Northeast", "northeastern": "Northeast",
}

# Compass-like words that LOOK like they might be a region but aren't
# real Census regions in this data (only South/West/Midwest/Northeast
# exist). Kept deliberately SEPARATE from REGION_WORDS (which must
# only ever map to real values - see the self-check in Section 5) so
# these can be recognized and clearly rejected instead of silently
# falling through to an unrelated document search. Found via a real
# test: "Which HCPs are in the Southeast region?" was returning
# Wyoming/Mississippi/Tennessee market summaries that happened to
# score just above the confidence threshold - a wrong answer that
# LOOKED confident, which is worse than an honest "not found."
_INVALID_REGION_WORDS = {
    "southeast": "Southeast", "southeastern": "Southeast",
    "southwest": "Southwest", "southwestern": "Southwest",
    "northwest": "Northwest", "northwestern": "Northwest",
}


def _detect_invalid_region_attempt(ql):
    for word, label in _INVALID_REGION_WORDS.items():
        if re.search(r"\b" + word + r"\b", ql):
            return label
    return None

# ---- Tier --------------------------------------------------------------
# Real tier values are High / Medium / Watch - there is no "Low" tier
# in the real data. A rep saying "low tier" or "low propensity" almost
# certainly means the bottom tier, which is actually called "Watch" -
# so "low" is mapped to Watch, not left to silently match nothing (the
# old version of this file mapped "low" to a "Low" value that never
# existed, which meant that filter always silently returned 0 rows).
TIER_PATTERNS = [
    (re.compile(r"\bhigh[- ]?(?:propensity|tier)\b"), "High"),
    (re.compile(r"\bmedium[- ]?(?:propensity|tier)\b"), "Medium"),
    (re.compile(r"\bwatch[- ]?(?:tier|list)?\b"), "Watch"),
    (re.compile(r"\blow[- ]?(?:propensity|tier)\b"), "Watch"),
]

# ---- Competitor ----------------------------------------------------
# Real dominant_competitor values are ONLY "Eli Lilly" and "Novo
# Nordisk" - there is no "Other" value anywhere in the real data (the
# old version of this file guessed one, and it could never have
# matched anything).
COMPETITOR_KEYWORDS = {
    "novo_nordisk": {"words": ["semaglutide", "ozempic", "wegovy", "novo nordisk", "novo"],
                     "label": "Novo Nordisk"},
    "eli_lilly": {"words": ["tirzepatide", "mounjaro", "zepbound", "eli lilly", "lilly", "eli"],
                  "label": "Eli Lilly"},
}

# ---- Payer -----------------------------------------------------------
# Real dominant_payer values are Commercial / Medicare / Medicaid -
# there is no "OOP" (out-of-pocket) value; no HCP has out-of-pocket as
# their DOMINANT payer in this data. A question about out-of-pocket
# exposure almost certainly means the pct_oop SCORE column instead -
# see _detect_oop_level() below, which handles that correctly.
_PAYER_PATTERNS = [
    (re.compile(r"\bmedicare\b"), "Medicare"),
    (re.compile(r"\bmedicaid\b"), "Medicaid"),
    (re.compile(r"\bcommercial[- ]?(?:payer|insurance|insured|heavy|dominant|plans?)\b"), "Commercial"),
]

# ---- AX relationship ---------------------------------------------------
# Real ax_relationship values are "One", "TwoPlus" - and "Zero", which
# query_spreadsheet.py adds to represent the blank/missing cells (see
# its comment: 41.6% of rows are blank here, almost certainly meaning
# "no AX product relationship" rather than genuinely unknown data).
_AX_RELATIONSHIP_PATTERNS = [
    (re.compile(r"\b(?:2\+|two\+|two\s*or\s*more|2\s*or\s*more)\s*(?:ax\s*)?products?\b|"
                r"\bax\s*(?:relationship|products?)\s*(?:of\s*)?(?:2\+|two\+)\b"), "TwoPlus"),
    (re.compile(r"\bone\s*ax\s*product\b|\bax\s*relationship\s*of\s*one\b"), "One"),
    (re.compile(r"\bno\s*ax\s*(?:relationship|products?)\b|\bzero\s*ax\s*products?\b|"
                r"\bax\s*relationship\s*of\s*zero\b"), "Zero"),
]

# ---- Formulary status --------------------------------------------------
def _detect_formulary(ql):
    """Formulary status language -> the real formulary_tier column
    values. "non-preferred" is checked before bare "preferred", since
    "preferred" is a literal substring of "non-preferred" - a
    word-boundary alone doesn't protect against that, the hyphen is a
    boundary character too."""
    if re.search(r"\bnon[- ]?preferred\b", ql):
        return "NonPreferred"
    if re.search(r"\bpreferred\b", ql):
        return "Preferred"
    if re.search(r"\bpa[- ]?required\b|\bprior auth", ql):
        return "PARequired"
    if re.search(r"\bnot[- ]?covered\b|\buncovered\b", ql):
        return "NotCovered"
    return None


# ---- Sample request status ---------------------------------------------
_SAMPLE_NO = re.compile(r"\bno sample request\b|\bwithout (?:a )?sample request\b|"
                         r"\bhas(?:n't| not) requested (?:a )?sample\b")
_SAMPLE_YES = re.compile(r"\bsample request(?:ed)?\b|\brequested (?:a )?sample\b")


def _detect_sample_request(ql):
    if _SAMPLE_NO.search(ql):
        return 0
    if _SAMPLE_YES.search(ql):
        return 1
    return None


# ---- Targeted status -----------------------------------------------
_TARGETED_NOT = re.compile(r"\b(?:not\s+(?:currently\s+)?targeted|untargeted)\b")
_TARGETED_YES = re.compile(r"\b(?:currently\s+|already\s+)?targeted\b")


def _detect_targeted(ql):
    if _TARGETED_NOT.search(ql):
        return 0
    if _TARGETED_YES.search(ql):
        return 1
    return None


# ---- Decile -----------------------------------------------------------
_DECILE_TOP = re.compile(r"\btop\s*decile\b")
_DECILE_EXACT = re.compile(r"\bdecile\s*(\d{1,2})\b")


def _detect_decile(ql):
    """'top decile' means decile 10; a bare 'decile 9' is an exact
    match, not a threshold."""
    if _DECILE_TOP.search(ql):
        return 10
    m = _DECILE_EXACT.search(ql)
    if m:
        try:
            v = int(m.group(1))
            if 1 <= v <= 10:
                return v
        except ValueError:
            pass
    return None


# ---- Zero-writer status -------------------------------------------------
_ZERO_WRITER_ONLY = re.compile(r"\bzero[- ]?writers?\s*only\b|\bonly\s*zero[- ]?writers?\b|"
                                r"\bnon[- ]?writers?\s*only\b|\bonly\s*non[- ]?writers?\b")
_ZERO_WRITER_EXCLUDE = re.compile(r"\bexclud\w*\s*(?:the\s*)?zero[- ]?writers?\b|"
                                    r"\bactive\s*writers?\s*only\b|\bonly\s*active\s*writers?\b|"
                                    r"\bexcluding\s*non[- ]?writers?\b")

# A BARE mention of "active writers/prescribers/hcps" (no "only"
# qualifier) - this is what lets a list-style question like "Show all
# active writers in Texas" apply zero_writer=False once it reaches
# PATH 3, instead of that phrase being invisible to the filter engine.
_ACTIVE_WRITER_BARE = re.compile(r"\bactive\s*(?:glp-1\s*)?(?:writers?|prescribers?|hcps?)\b")

# A BARE mention of "zero-writers"/"non-writers" (no "only" qualifier)
# - used ONLY by PATH 1 to decide which number a plain count question
# is actually asking about (see Issue 8's fix in PATH 1 below). This is
# deliberately separate from _ZERO_WRITER_ONLY, which means something
# different: "give me a FILTERED LIST of zero-writers only."
_ZERO_WRITER_MENTION = re.compile(r"\bzero[- ]?writers?\b|\bnon[- ]?writers?\b")


def _detect_zero_writer(ql):
    """True (only zero-writers), False (exclude zero-writers), or
    None (not mentioned - don't filter on it at all, same policy as
    every other filter here)."""
    if _ZERO_WRITER_EXCLUDE.search(ql) or _ACTIVE_WRITER_BARE.search(ql):
        return False
    if _ZERO_WRITER_ONLY.search(ql):
        return True
    return None


# ---- Unresolved referent guard ------------------------------------------
# Catches a question that clearly means ONE specific doctor ("this
# doctor," "that HCP," "him," "her") but never actually names which
# one via an NPI. Routing anyway means either engine ends up guessing
# - reusing an NPI from an earlier unrelated question, or matching
# whatever card happens to score highest - and answering fluently as
# if that guess were correct. A rep has no way to tell a guessed
# answer apart from a real one, so this has to be caught before ANY
# routing happens, not inside one engine's branch.
#
# This system only ever identifies a specific HCP by their 10-digit
# NPI (no proper names appear anywhere in the data), so "no NPI in
# the question" is a reliable stand-in for "no name given" here.
_UNRESOLVED_REFERENT_PATTERNS = [
    re.compile(r"\bthis (?:doctor|hcp|physician|prescriber|provider)\b"),
    re.compile(r"\bthat (?:doctor|hcp|physician|prescriber|provider)\b"),
    re.compile(r"\bhim\b"), re.compile(r"\bher\b"),
    re.compile(r"\bhe\b"), re.compile(r"\bshe\b"),
]
_NPI_PATTERN = re.compile(r"\b(\d{10})\b")


def _has_unresolved_referent(question):
    """True if the question refers to a specific-but-unnamed doctor
    and gives no NPI anywhere to resolve who that is."""
    if _NPI_PATTERN.search(question):
        return False  # a real NPI is given - fully resolved, nothing to flag
    ql = question.lower()
    return any(p.search(ql) for p in _UNRESOLVED_REFERENT_PATTERNS)


# ---- Explicit list-ACTION language (Issue 9) -----------------------
# Deliberately separate from LIST_WORDS above: LIST_WORDS includes
# plain nouns ("prescribers", "hcps", "doctors"...) that show up just
# as often in a bare COUNT question ("how many active prescribers")
# as in a real list request - so using LIST_WORDS itself to gate PATH 1
# would incorrectly block plain counts too. This is only the ACTION
# verbs that mean "show me the actual entries," not just "count them."
_LIST_ACTION_PHRASES = ["list", "show ", "give me", "find ", "which "]


def _wants_list_action(ql):
    return any(p in ql for p in _LIST_ACTION_PHRASES)


# ---- HCP-context guard --------------------------------------------------
# A bare mention of a competitor, payer, etc. isn't always about
# filtering HCPs - "How is Rabivy different from Zepbound?" mentions a
# competitor brand but is a narrative comparison question, not an
# HCP-targeting one. Require actual HCP/targeting context before
# treating these mentions as filter triggers.
HCP_CONTEXT_WORDS = ["hcp", "prescriber", "writer", "doctor", "physician",
                      "clinician", "targeted", "target"]


def _has_hcp_context(ql):
    return any(w in ql for w in HCP_CONTEXT_WORDS) or _detect_specialty(ql) is not None


# =====================================================================
# SECTION 2 - THE 0-1 SCORE COLUMNS: "high" / "moderate" / "low"
# -----------------------------------------------------------------
# This is the one place that decides which real column a "high X" /
# "low X" phrase is about, then hands that off to query_spreadsheet.py's
# `levels` system, which applies ONE standard cutoff (0.75 high / 0.25
# low) consistently everywhere.
# =====================================================================

# switching_score and pa_burden get their own hand-checked phrasing,
# since they're the two reps ask about most often, in the most varied
# ways. Every other 0-1 score column (pct_novo, obesity_prev, etc.) is
# covered by the GENERIC mechanism further down, reusing the same
# phrase list already built for COLUMN_SYNONYMS (see Section 3).

_SWITCHING_EXPLICIT = re.compile(r"switching[^0-9]{0,15}([\d.]+)")
_SWITCHING_HIGH = re.compile(r"\bhigh[- ]?switching\b")
_SWITCHING_LOW = re.compile(r"\blow[- ]?switching\b")
_COMPETITIVE_SWITCHER_PHRASE = re.compile(r"\bcompetitive\s*switchers?\b")


def _detect_switching(ql):
    """Returns (level_word_or_None, explicit_range_or_None).
    An explicit number ('switching score above 0.7') always wins over
    the 'high'/'low' word shortcut."""
    m = _SWITCHING_EXPLICIT.search(ql)
    if m:
        try:
            return None, (float(m.group(1)), None)
        except ValueError:
            pass
    if _SWITCHING_HIGH.search(ql):
        return "high", None
    if _SWITCHING_LOW.search(ql):
        return "low", None
    return None, None


def _detect_competitive_switcher_targeting(ql):
    """True only if 'competitive switcher(s)' appears ALONGSIDE a real
    targeting cue (a state, a tier, ranking language, or a list word).
    The same phrase also shows up in purely definitional/messaging
    questions ("What's our recommended messaging for competitive
    switchers?"), which should stay in RAG - so a targeting cue is
    required alongside it, not just the phrase alone."""
    if not _COMPETITIVE_SWITCHER_PHRASE.search(ql):
        return False
    return (
        _find_state(ql) is not None
        or _detect_tier(ql) is not None
        or any(p.search(ql) for p in RANK_PATTERNS)
        or any(w in ql for w in LIST_WORDS)
    )


# PA burden (pa_burden column) is a DIFFERENT concept from
# formulary_tier="PARequired" (a categorical status) - both can
# involve the words "PA"/"prior auth", so this specifically requires
# the word "burden" to avoid the two colliding.
_PA_BURDEN_EXPLICIT_ABOVE = re.compile(
    r"(?:pa|prior auth\w*)\s*burden[^0-9]{0,15}(?:above|over|greater than|>)\s*([\d.]+)")
_PA_BURDEN_EXPLICIT_BELOW = re.compile(
    r"(?:pa|prior auth\w*)\s*burden[^0-9]{0,15}(?:below|under|less than|<)\s*([\d.]+)")
_PA_BURDEN_HIGH = re.compile(r"\bhigh[- ]?(?:pa|prior auth\w*)[- ]?burden\b")
_PA_BURDEN_LOW = re.compile(r"\blow[- ]?(?:pa|prior auth\w*)[- ]?burden\b")


def _detect_pa_burden(ql):
    """Returns (level_word_or_None, explicit_range_or_None), same
    shape as _detect_switching above."""
    m = _PA_BURDEN_EXPLICIT_ABOVE.search(ql)
    if m:
        try:
            return None, (float(m.group(1)), None)
        except ValueError:
            pass
    m = _PA_BURDEN_EXPLICIT_BELOW.search(ql)
    if m:
        try:
            return None, (None, float(m.group(1)))
        except ValueError:
            pass
    if _PA_BURDEN_HIGH.search(ql):
        return "high", None
    if _PA_BURDEN_LOW.search(ql):
        return "low", None
    return None, None


# pct_oop (out-of-pocket share) - a rep asking about "out-of-pocket
# exposure" or "high patient cost" almost certainly means this SCORE
# column, not the dominant_payer category (which never equals OOP in
# this data - see the payer note above).
_OOP_HIGH = re.compile(r"\bhigh[- ]?out[- ]?of[- ]?pocket\b|\bhigh\s*oop\b|\bhigh\s*patient\s*cost\b")
_OOP_LOW = re.compile(r"\blow[- ]?out[- ]?of[- ]?pocket\b|\blow\s*oop\b|\blow\s*patient\s*cost\b")


def _detect_oop_level(ql):
    if _OOP_HIGH.search(ql):
        return "high"
    if _OOP_LOW.search(ql):
        return "low"
    return None


# =====================================================================
# SECTION 3 - NAMED-COLUMN SYNONYMS + GENERIC FALLBACKS
# -----------------------------------------------------------------
# Real questions don't use literal column names ("obesity prevalence",
# not "obesity_prev") - this section maps real English phrases to the
# real column name, for every column that doesn't already have its own
# hand-built detector above.
# =====================================================================
COLUMN_SYNONYMS = {
    "pa_burden": [r"\bpa\s*burden\b", r"\bprior\s*auth\w*\s*burden\b"],
    "obesity_prev": [r"\bobesity\s*preval\w*\b", r"\bobesity\s*rate\b",
                      r"\baddressable\s*population\b"],
    "ax_relationship": [r"\bax\s*(?:product|relationship)s?\b", r"\bax\s*adoption\b"],
    "days_since_contact": [r"\bdays?\s*since\s*(?:last\s*)?contact\w*\b",
                            r"\bhasn'?t\s*been\s*contact\w*\b",
                            r"\bnot\s*(?:been\s*)?contact\w*\b"],
    "years_practice": [r"\byears?\s*(?:in\s*)?practice\b", r"\bexperience\b",
                        r"\bhow\s*long\s*(?:in\s*)?practic\w*\b"],
    "rep_engagement_score": [r"\brep\s*engagement\b"],
    "pct_novo": [r"\b(?:percent|%|pct)\s*novo\b", r"\bnovo\s*(?:percent|%|share)\b"],
    "pct_lilly": [r"\b(?:percent|%|pct)\s*lilly\b", r"\blilly\s*(?:percent|%|share)\b"],
    "pct_other_brand": [r"\b(?:percent|%|pct)\s*other\s*brand\b",
                         r"\bother\s*brand\s*(?:percent|%|share)\b"],
    "nrx_share": [r"\bnew\s*patient\s*share\b", r"\bnrx\s*share\b"],
}


def _detect_named_column(ql):
    """Which real spreadsheet column is this question actually about,
    based on real phrasing. Returns the column name, or None."""
    for col, patterns in COLUMN_SYNONYMS.items():
        if any(re.search(p, ql) for p in patterns):
            return col
    return None


_ASCENDING_WORDS = re.compile(r"\b(?:lowest|least|fewest|smallest|worst|bottom|shortest)\b")
_DESCENDING_WORDS = re.compile(r"\b(?:highest|most|greatest|top|best|longest)\b")


def _detect_sort_direction(ql):
    """True = ascending (smallest first), False = descending (largest
    first)."""
    return bool(_ASCENDING_WORDS.search(ql))


def _column_phrase_pattern(col):
    """Turns a real column name like 'days_since_contact' into a
    regex matching it with underscores, spaces, or hyphens."""
    parts = col.split("_")
    return r"[ _-]+".join(re.escape(p) for p in parts)


# ---- Generic "high X" / "low X" for any other 0-1 score column -------
# This is what lets a score column that never got its own hand-built
# detector (pct_novo, obesity_prev, rep_engagement_score, ...) still
# support "high"/"low" language, for EVERY such column independently -
# not just the first one the question happens to mention.
#
# "high" or "low" must sit within a few words of that SPECIFIC
# column's own phrase, on either side ("high percent Novo" or "percent
# Novo is high"). This is what correctly handles a question mentioning
# two different score columns at once, like "high percent Novo and low
# rep engagement" - each column only picks up the high/low word that's
# actually next to it, instead of one column accidentally grabbing
# whichever high/low word appears first anywhere in the sentence
# (a real bug found via testing - see the note in _build_generic_
# level_patterns below).
#
# \bhigh\b / \blow\b (not just "high"/"low" as substrings) also means
# this naturally never fires on "highest"/"lowest" - those have extra
# letters right after, so the word-boundary check fails on purpose,
# leaving bare superlatives to path 2b's sort logic instead.
_NEARBY = r"[^,;]{0,20}"  # within ~20 characters, never crossing a comma/semicolon

# Splits a question into separate clauses at "and"/"but"/commas/
# semicolons. Each clause is then searched SEPARATELY for a column's
# phrase + high/low - this is what stops "high" (meant for "percent
# Novo") from bleeding across "and" into also matching near
# "rep engagement" in "high percent Novo and low rep engagement." A
# character-count window alone (the first version of this fix) wasn't
# enough - "and low " is only ~8 characters, well inside any
# reasonable window, so the two clauses have to be split apart first,
# not just kept close-but-limited.
_CLAUSE_SPLIT = re.compile(r"\band\b|\bbut\b|,|;")


def _build_generic_level_patterns():
    patterns = {}
    for col in qs.SCORE_COLUMNS:
        phrase_alternatives = COLUMN_SYNONYMS.get(col)
        if not phrase_alternatives:
            continue  # no known phrasing for this column - can't detect it in English yet
        alt = "|".join(f"(?:{p})" for p in phrase_alternatives)
        patterns[col] = (
            re.compile(rf"\bhigh\b{_NEARBY}(?:{alt})|(?:{alt}){_NEARBY}\bhigh\b"),
            re.compile(rf"\blow\b{_NEARBY}(?:{alt})|(?:{alt}){_NEARBY}\blow\b"),
        )
    return patterns


GENERIC_LEVEL_PATTERNS = _build_generic_level_patterns()


def _detect_generic_levels(ql):
    """Returns {column: "high"/"low"} for EVERY score column whose own
    phrase appears near the word high/low, WITHIN THE SAME CLAUSE of
    this question - not just the first column the question happens to
    mention, and not confused by a different clause's high/low word."""
    found = {}
    for clause in _CLAUSE_SPLIT.split(ql):
        for col, (high_pat, low_pat) in GENERIC_LEVEL_PATTERNS.items():
            if col in found:
                continue  # first clause that names this column wins
            if high_pat.search(clause):
                found[col] = "high"
            elif low_pat.search(clause):
                found[col] = "low"
    return found


# ---- Generic numeric fallback, for OTHER_NUMERIC_COLUMNS -------------
# Covers exact-number requests on any real numeric column that doesn't
# have its own hand-built rule - "days_since_contact over 30",
# "rep engagement score above 0.5". Built directly from the live data's
# real numeric columns, so a new numeric column added later is covered
# automatically.
_GENERIC_EXCLUDE_COLUMNS = {"npi", "hcp_id", "targeted", "sample_request_recent", "zero_writer"}
_OP_ABOVE = r"(?:above|over|greater than|more than|higher than|>)"
_OP_BELOW = r"(?:below|under|less than|fewer than|lower than|<)"


def _build_generic_numeric_patterns():
    patterns = {}
    for col in list(qs.SCORE_COLUMNS) + list(qs.OTHER_NUMERIC_COLUMNS):
        if col in _GENERIC_EXCLUDE_COLUMNS:
            continue
        phrase = _column_phrase_pattern(col)
        patterns[col] = (
            re.compile(rf"\b{phrase}\b[^0-9]{{0,20}}{_OP_ABOVE}\s*([\d.]+)"),
            re.compile(rf"\b{phrase}\b[^0-9]{{0,20}}{_OP_BELOW}\s*([\d.]+)"),
        )
    return patterns


GENERIC_NUMERIC_PATTERNS = _build_generic_numeric_patterns()


def _detect_generic_numeric_filters(ql):
    """Returns {column: (min_val, max_val)} for any real numeric
    column explicitly named with a comparison. Empty dict if none
    matched."""
    found = {}
    for col, (above_pat, below_pat) in GENERIC_NUMERIC_PATTERNS.items():
        m_above = above_pat.search(ql)
        m_below = below_pat.search(ql)
        min_v = float(m_above.group(1)) if m_above else None
        max_v = float(m_below.group(1)) if m_below else None
        if min_v is not None or max_v is not None:
            found[col] = (min_v, max_v)
    return found


# "high-volume writers" means sort by actual Rx volume - a pure sort
# signal, not a threshold filter (rx_volume_monthly isn't a 0-1 score
# column, so there's no "high" cutoff to apply to it).
_HIGH_VOLUME_PATTERN = re.compile(r"\bhigh[- ]?volume\b|\bhighest volume\b")


# ---------------------------------------------------------------------
# Does this question want to be ranked by prescribing VOLUME? True for
# the explicit "high-volume" phrase, OR for general ranking language
# ("top", "most", ...) combined with volume-context words
# ("prescriber", "writer", "script", ...) - the same combination path
# 5 (the plain ranking path) already understands. This is reused here
# so path 3 (the filter path) picks the SAME sort column path 5 would
# have picked - found via a real test: "top prescribers in Kansas"
# alone correctly sorted by volume (path 5), but "top prescribers in
# Kansas with recent sample request" fell into path 3 instead (because
# the sample-request filter also triggers path 3), and path 3 had no
# idea "top prescribers" meant volume at all, so it silently defaulted
# to propensity instead.
# ---------------------------------------------------------------------
def _wants_volume_sort(ql):
    if _HIGH_VOLUME_PATTERN.search(ql):
        return True
    return any(p.search(ql) for p in RANK_PATTERNS) and _has_volume_context(ql)

# Comparison language for "compare NPI X to a typical Y" - needs both
# the HCP's own card AND a benchmark chunk, not just the card alone.
_COMPARISON_WORDS = re.compile(
    r"\b(?:compare|comparison|typical|benchmark|average|versus|vs\.?|"
    r"high or low|low or high|strong or weak|weak or strong|"
    r"better or worse|worse or better|compared to|relative to)\b")

# Forward-looking targeting language ("who should I target").
_FORWARD_TARGET_PATTERN = re.compile(
    r"\bwho should i target\b|\bwho (?:should|do|would) i target\b|"
    r"\bwho to target\b|\bbest targets?\b|\bwho are (?:my|the) targets?\b|"
    r"\bwho(?:m)? should i prioriti[sz]e\b|"
    r"\bwhich (?:hcps?|prescribers?|writers?|doctors?) should i prioriti[sz]e\b|"
    r"\bshould i prioriti[sz]e\b|\bwho should i focus on\b|"
    r"\bwho should i visit\b|\bwho (?:should|do|would) i visit\b|"
    r"\bwho to visit\b|\bwho are (?:my|the) best visits?\b|"
    r"\bwho are (?:my|the) best (?:hcps?|prescribers?|writers?|doctors?) to visit\b|"
    # Noun-phrase style targeting requests, not just questions opening
    # with "who" - "top NY targets next month", "give me my top 3
    # targets" - found via testing: these were falling through to a
    # plain (unranked) card list instead.
    r"\b(?:top|best)\s+(?:\w+\s+){0,3}?targets?\b")

# Messaging language - if this is present alongside a targeting cue,
# the question needs BOTH a filtered list AND messaging guidance
# joined together, which only RAG can do (it can write prose combining
# both). Confirmed via testing: narrow phrasing here (only catching
# "talking points", plural) missed real questions like "how do I
# respond" or "what's the pitch" - this list is deliberately broader.
_MESSAGING_WORDS = re.compile(
    r"what (?:should i|to|do i|would i|can i) say|messaging|talking points?|\bmessage\b|"
    r"how (?:do|should|would) i (?:respond|reply|handle|address)|"
    r"what.?s (?:the |my )?pitch|\bobjection\b")

DEFAULT_RESULT_LIMIT = 20
_ALL_PATTERN = re.compile(r"\ball\b|\bevery(one)?\b")

# ---------------------------------------------------------------------
# Singular vs plural detection, for deciding whether a ranking question
# wants ONE result ("the top prescriber") or a list ("top prescribers").
#
# The old approach tried to match "rank-word + a few filler words +
# noun" in that exact order, which breaks the moment the words appear
# in a different order - "which DOCTOR has the HIGHEST volume" has the
# noun BEFORE the rank word, and "who WROTE the most scripts" has no
# noun naming an HCP at all. This version is much simpler and more
# robust: look for a real singular or plural noun ANYWHERE in the
# question, in either order, and if there's no noun at all, a bare
# "who" opening the question ("who wrote...", "who has...") is natural
# English for asking about ONE person, not a list.
# ---------------------------------------------------------------------
_PLURAL_HCP_NOUN = re.compile(r"\b(?:prescribers|writers|hcps|doctors|physicians|clinicians)\b")
_SINGULAR_HCP_NOUN = re.compile(r"\b(?:prescriber|writer|hcp|doctor|physician|clinician)\b")
_BARE_WHO_OPENING = re.compile(r"^\s*who\b")


def _resolve_top(ql):
    """An explicit number ('top 30') always wins; 'all'/'every' means
    no limit; a real singular noun ('the top prescriber') or a bare
    'who' opening with no noun at all ('who wrote the most scripts')
    means exactly one result was asked for; a real plural noun
    ('top prescribers') means a list; otherwise the generous default
    applies."""
    top_match = re.search(r"top\s+(\d+)", ql)
    if top_match:
        return int(top_match.group(1))
    if _ALL_PATTERN.search(ql):
        return None
    if _PLURAL_HCP_NOUN.search(ql):
        return DEFAULT_RESULT_LIMIT
    if _SINGULAR_HCP_NOUN.search(ql):
        return 1
    if _BARE_WHO_OPENING.search(ql) and any(p.search(ql) for p in RANK_PATTERNS):
        # A bare "who" is only treated as singular alongside genuine
        # ranking language ("who WROTE THE MOST scripts"). Without a
        # rank word, "who" is forward-targeting language ("who should
        # I target") instead, which wants a shortlist, not one person -
        # a real regression found via testing: this fallback was
        # wrongly forcing "who should I target in New York" down to a
        # single result.
        return 1
    return DEFAULT_RESULT_LIMIT


# =====================================================================
# SECTION 4 - STATE, REGION, SPECIALTY, TIER, COMPETITOR, PAYER, AX
# DETECTORS
# =====================================================================

def _find_state(ql):
    """Full state name match first, then abbreviation aliases.
    Aliases are word-boundary matched only, so 'ma' doesn't match
    inside 'maintenance'."""
    for s in qs.KNOWN_STATES:
        if s in ql:
            return s.title()
    for abbr, full in STATE_ALIASES.items():
        # "pa" (Pennsylvania) collides with "PA burden"/"PA
        # required"/"prior auth" - a question like "which HCPs have a
        # high PA burden nationally" (no real state named) was
        # silently matching Pennsylvania here. Skip the Pennsylvania
        # abbreviation specifically when "PA" clearly means
        # prior-authorization instead.
        if abbr == "pa" and re.search(r"\bpa\s*burden\b|\bpa[- ]?required\b|\bprior\s*auth", ql):
            continue
        if re.search(r"\b" + abbr + r"\b", ql):
            return full
    return None


def _detect_region(ql):
    """Only checked by the caller when _find_state() found nothing -
    that ordering is what correctly tells "prescribers in the South"
    (region) apart from "prescribers in South Carolina" (state): a
    full state name always wins first."""
    for word, region in REGION_WORDS.items():
        if re.search(r"\b" + word + r"\b", ql):
            return region
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


def _detect_competitor(ql):
    for info in COMPETITOR_KEYWORDS.values():
        for w in info["words"]:
            if re.search(r"\b" + re.escape(w) + r"\b", ql):
                return info["label"]
    return None


def _detect_payer(ql):
    for pattern, label in _PAYER_PATTERNS:
        if pattern.search(ql):
            return label
    return None


def _detect_ax_relationship(ql):
    for pattern, label in _AX_RELATIONSHIP_PATTERNS:
        if pattern.search(ql):
            return label
    return None


def _is_forward_targeting_question(ql):
    return (bool(_FORWARD_TARGET_PATTERN.search(ql))
            and not _MESSAGING_WORDS.search(ql))


# ---------------------------------------------------------------------
# Does this question sound like it's about PRESCRIBING VOLUME? Used to
# decide whether ranking language ("most", "top") + a state means
# "rank by rx_volume_monthly." Deliberately broader than just the
# literal words "prescriber"/"writer"/"volume" - a natural phrasing
# like "who writes the most scripts in Ohio" uses neither of those
# words at all (it says "write" and "scripts" instead), and was found
# via real testing to fall all the way through to a RAG search that
# has no way to answer a ranking question at all.
# ---------------------------------------------------------------------
_VOLUME_CONTEXT_WORDS = ["prescrib", "writ", "wrote", "volume", "script"]
_RX_WORD = re.compile(r"\brx\b")


def _has_volume_context(ql):
    return any(w in ql for w in _VOLUME_CONTEXT_WORDS) or bool(_RX_WORD.search(ql))


# =====================================================================
# SECTION 5 - SELF-CHECK: every word this file maps TO must be real
# -----------------------------------------------------------------
# This runs once, when the file is first imported. It confirms every
# value this file's synonym tables produce (TIER_PATTERNS, COMPETITOR_
# KEYWORDS, _PAYER_PATTERNS, _AX_RELATIONSHIP_PATTERNS) is a value that
# genuinely exists in the live spreadsheet right now. If someone
# changes the underlying data later and a mapped value stops existing,
# this fails loudly at startup - instead of that filter silently
# returning 0 results for months, the way the old "Other" competitor /
# "OOP" payer / "Zero" ax_relationship guesses did.
# =====================================================================
def _verify_synonym_targets(name, labels, column):
    real_values = set(qs.KNOWN_VALUES.get(column, []))
    bad = [label for label in labels if label not in real_values]
    if bad:
        raise ValueError(
            f"{name} maps to {bad}, which {'is' if len(bad) == 1 else 'are'} not real "
            f"value(s) in column '{column}'. Real values are: {sorted(real_values)}. "
            f"Fix the mapping in ask_a_question.py."
        )


_verify_synonym_targets("TIER_PATTERNS", [label for _, label in TIER_PATTERNS], "tier")
_verify_synonym_targets("COMPETITOR_KEYWORDS",
                         [info["label"] for info in COMPETITOR_KEYWORDS.values()], "dominant_competitor")
_verify_synonym_targets("_PAYER_PATTERNS", [label for _, label in _PAYER_PATTERNS], "dominant_payer")
_verify_synonym_targets("_AX_RELATIONSHIP_PATTERNS",
                         [label for _, label in _AX_RELATIONSHIP_PATTERNS], "ax_relationship")
_verify_synonym_targets("REGION_WORDS", list(REGION_WORDS.values()), "region")


# =====================================================================
# SECTION 6 - SORT DECISION
# -----------------------------------------------------------------
# Decides which column to sort a filtered list by, when more than one
# numeric measure was mentioned in the question. Deliberately
# conservative: a wrong silent guess here is worse than an honest
# "ranking is ambiguous, filtered on both."
# =====================================================================

def _detect_continuous_signals(ql, levels, extra_filters):
    """Every continuous (orderable) column this question actually
    named, as (column, ascending) pairs."""
    signals = []
    for column, level in (levels or {}).items():
        signals.append((column, level == "low"))
    for column, (min_v, max_v) in (extra_filters or {}).items():
        ascending = max_v is not None and min_v is None
        signals.append((column, ascending))
    if _wants_volume_sort(ql):
        signals.append(("rx_volume_monthly", False))
    return signals


# ---------------------------------------------------------------------
# Catches an EXPLICIT sorting instruction - "sorted by propensity
# score", "ranked by PA burden" - which should always win over any
# other rule below, since the person said exactly what they want.
# ---------------------------------------------------------------------
_RANK_OR_SORT_VERB = re.compile(r"\b(?:sort(?:ed)?|rank(?:ed)?)\b")
_BY_SPLIT = re.compile(r"\bby\s+")
_EXPLICIT_SORT_STOP_WORDS = re.compile(r"\b(?:in|for|and|who|among|within)\b|[?.,;]")


def _detect_explicit_sort_column(ql):
    # Find the rank/sort verb FIRST, then look for the next "by" AFTER
    # it - not just the first "by" anywhere in the question. This is
    # what correctly handles "Rank Novo-heavy California prescribers by
    # switching score" (words sitting between the verb and "by"), while
    # still not accidentally grabbing an unrelated earlier "by" clause
    # in a longer question (e.g. "targeted by reps, ranked by X").
    m = _RANK_OR_SORT_VERB.search(ql)
    if not m:
        return None
    after_verb = ql[m.end():]
    parts = _BY_SPLIT.split(after_verb, maxsplit=1)
    if len(parts) < 2:
        return None
    phrase = _EXPLICIT_SORT_STOP_WORDS.split(parts[1], maxsplit=1)[0].strip()
    if not phrase:
        return None
    if "propensity" in phrase:
        return "propensity_score"
    if "switching" in phrase:
        return "switching_score"
    if re.search(r"\bpa\s*burden\b|\bprior\s*auth", phrase):
        return "pa_burden"
    if re.search(r"\bvolume\b|\bscripts?\b|\brx\b", phrase):
        return "rx_volume_monthly"
    return _detect_named_column(phrase)


# ---------------------------------------------------------------------
# Where in the question's own text a given column was first mentioned -
# used to pick "the first one listed" when two or more measures are
# named and neither is propensity (see _decide_sort's rule 4). Reuses
# the same phrase patterns already built for detecting each column, so
# there's one definition of "what counts as mentioning this column,"
# not two that could drift apart.
# ---------------------------------------------------------------------
_SORT_POSITION_EXTRA_PATTERNS = {
    "switching_score": [r"\bswitching\b"],
    "pct_oop": [r"\bout[- ]?of[- ]?pocket\b", r"\boop\b"],
    "rx_volume_monthly": [r"\bvolume\b", r"\bscripts?\b", r"\brx\b"],
}


def _first_match_position(ql, column):
    patterns = list(COLUMN_SYNONYMS.get(column, [])) + _SORT_POSITION_EXTRA_PATTERNS.get(column, [])
    patterns.append(_column_phrase_pattern(column))
    best = None
    for p in patterns:
        m = re.search(p, ql)
        if m and (best is None or m.start() < best):
            best = m.start()
    return best if best is not None else 10 ** 9  # not found in text - sorts last


def _decide_sort(ql, levels, extra_filters):
    """Priority order, per an explicit decision on how ties should
    resolve:
      1. An EXPLICIT "sorted by X" / "ranked by X" instruction always
         wins over everything else.
      2. Otherwise, if "propensity" is named anywhere in the question,
         sort by propensity_score - even if other numeric measures are
         ALSO mentioned. Naming propensity directly should always win
         over an incidental mention of something else.
      3. Otherwise, exactly one other numeric measure -> sort by that,
         the one thing the question was actually about.
      4. Otherwise, two or more OTHER numeric measures (no propensity
         named) -> sort by whichever was named FIRST in the question's
         own word order, and say so explicitly - never guess silently
         between them.
      5. Nothing numeric mentioned at all -> propensity_score, the
         sensible default.
    """
    ascending_requested = _detect_sort_direction(ql)

    explicit_col = _detect_explicit_sort_column(ql)
    if explicit_col:
        direction = "lowest first" if ascending_requested else "highest first"
        return (explicit_col, ascending_requested,
                f"Sorted by {explicit_col} ({direction}) - explicitly requested in this question.")

    # "High-propensity endocrinologists" mentions propensity directly,
    # but that phrase gets fully consumed by the TIER detector
    # (tier="High") before it ever reaches the signal list below - so
    # checking the raw text here (not just the detected signals) is
    # what makes sure a direct mention of "propensity" always wins,
    # even when other numeric measures were also detected.
    if "propensity" in ql:
        direction = "lowest first" if ascending_requested else "highest first"
        return ("propensity_score", ascending_requested,
                f"Sorted by propensity_score ({direction}) - propensity was named "
                f"directly in this question.")

    signals = _detect_continuous_signals(ql, levels, extra_filters)
    if len(signals) == 0:
        return ("propensity_score", False,
                "Sorted by propensity_score (highest first) - the default ranking, "
                "since no specific numeric measure was asked about.")
    if len(signals) == 1:
        col, ascending = signals[0]
        direction = "lowest first" if ascending else "highest first"
        return (col, ascending,
                f"Sorted by {col} ({direction}) - the specific measure this question was about.")

    signals_by_position = sorted(signals, key=lambda cs: _first_match_position(ql, cs[0]))
    col, ascending = signals_by_position[0]
    direction = "lowest first" if ascending else "highest first"
    cols = ", ".join(c for c, _ in signals)
    return (col, ascending,
            f"Multiple numeric measures were mentioned ({cols}) - sorted by {col} ({direction}), "
            f"the first one named in this question. Ask for a different one specifically to rank "
            f"by that instead.")


def _has_other_filter_language(ql):
    """True if the question asks for something a plain count can't
    express. Used by path 1's guard - if true, this isn't really a
    plain count question even if it also says 'active writers in
    Texas'; let path 3 (the real filter path) handle it instead."""
    if _detect_specialty(ql) or _detect_tier(ql) or _detect_targeted(ql) is not None:
        return True
    if _detect_competitor(ql) is not None and _has_hcp_context(ql):
        return True
    switching_level, switching_exact = _detect_switching(ql)
    if switching_level or switching_exact:
        return True
    if _detect_formulary(ql) is not None and _has_hcp_context(ql):
        return True
    if _detect_payer(ql) is not None and _has_hcp_context(ql):
        return True
    if _detect_ax_relationship(ql) is not None and _has_hcp_context(ql):
        return True
    if _detect_decile(ql) is not None and _has_hcp_context(ql):
        return True
    pa_level, pa_exact = _detect_pa_burden(ql)
    if pa_level or pa_exact:
        return True
    if _detect_sample_request(ql) is not None:
        return True
    return False


# =====================================================================
# SECTION 7 - THE MAIN ROUTER
# -----------------------------------------------------------------
# Reads the question, decides which engine (and which function on
# that engine) can answer it, calls it with clean arguments, and
# returns (route_label, data). data always comes straight from
# query_spreadsheet.py or search_documents.py - never rewritten into a
# sentence here, that's format_answer()'s job at the very end.
# =====================================================================

def ask(question):
    ql = question.lower()

    # ---- PATH 0: unresolved referent -> ask, don't guess -------------
    # Runs before every other path (structured AND RAG), since both
    # engines were separately found to guess at "which doctor" instead
    # of asking. See _has_unresolved_referent's comment above for why.
    if _has_unresolved_referent(question):
        return "CLARIFICATION / unresolved referent", {
            "kind": "clarification",
            "message": ("This question seems to be about a specific doctor "
                        "(\"this doctor\" / \"that HCP\" / \"him\" / \"her\"), but no NPI is "
                        "given, so I don't know which one. Please include the 10-digit NPI."),
        }

    # ---- PATH 1: "how many ... writers in X" -> plain count --------
    if (any(w in ql for w in COUNT_WORDS) or "active" in ql) and ("writer" in ql or "prescriber" in ql or "hcp" in ql):
        st = _find_state(ql)
        if st and not _has_other_filter_language(ql) and not _wants_list_action(ql):
            result = qs.count_writers(st)
            # Which number does this question actually want? A bare
            # "zero-writers"/"non-writers" mention means the rep wants
            # the ZERO count, not the default active count.
            result["asked_about"] = "zero" if _ZERO_WRITER_MENTION.search(ql) else "active"
            return "STRUCTURED / count", result

    # ---- PATH 2: "how many scripts did NPI ... write" -> HCP lookup ----
    m = re.search(r"\b(\d{10})\b", question)
    if m and ("script" in ql or "rx" in ql or "write" in ql or "wrote" in ql):
        return "STRUCTURED / hcp lookup", qs.get_row_by_npi(m.group(1))

    # ---- PATH 2b: "lowest/highest <column> [in state]" -> named-column sort ----
    # A bare superlative with no number attached ("lowest PA burden")
    # sorts by one named column and returns the extreme end - a
    # genuinely different shape from "filter to everything above/below X."
    named_col = _detect_named_column(ql)
    if named_col and (_ASCENDING_WORDS.search(ql) or _DESCENDING_WORDS.search(ql)):
        ascending = _detect_sort_direction(ql)
        result = qs.filter_hcps(
            state=_find_state(ql),
            specialty=_detect_specialty(ql),
            sort_by=named_col,
            ascending=ascending,
            top=_resolve_top(ql),
        )
        direction = "lowest first" if ascending else "highest first"
        result["sort_reason"] = f"Sorted by {named_col} ({direction}) - the measure this question named directly."
        return "STRUCTURED / named column sort", result

    # ---- PATH 3: multi-field targeting filter -----------------------
    targeted = _detect_targeted(ql)
    competitor = _detect_competitor(ql)
    formulary_tier = _detect_formulary(ql)
    sample_request = _detect_sample_request(ql)
    payer = _detect_payer(ql)
    ax_relationship = _detect_ax_relationship(ql)
    zero_writer = _detect_zero_writer(ql)
    decile = _detect_decile(ql)
    tier = _detect_tier(ql)
    state = _find_state(ql)
    region = _detect_region(ql) if not state else None

    levels = {}
    extra_filters = {}

    switching_level, switching_exact = _detect_switching(ql)
    if switching_level is None and _detect_competitive_switcher_targeting(ql):
        switching_level = "high"
    if switching_exact:
        extra_filters["switching_score"] = switching_exact
    elif switching_level:
        levels["switching_score"] = switching_level

    pa_level, pa_exact = _detect_pa_burden(ql)
    if pa_exact:
        extra_filters["pa_burden"] = pa_exact
    elif pa_level:
        levels["pa_burden"] = pa_level

    oop_level = _detect_oop_level(ql)
    if oop_level:
        levels["pct_oop"] = oop_level

    levels.update(_detect_generic_levels(ql))

    extra_filters.update(_detect_generic_numeric_filters(ql))

    competitor_signal = competitor is not None and _has_hcp_context(ql)
    formulary_signal = formulary_tier is not None and _has_hcp_context(ql)
    sample_signal = sample_request is not None and _has_hcp_context(ql)
    payer_signal = payer is not None and _has_hcp_context(ql)
    ax_relationship_signal = ax_relationship is not None and _has_hcp_context(ql)
    decile_signal = decile is not None and _has_hcp_context(ql)
    zero_writer_signal = zero_writer is not None
    forward_target_signal = _is_forward_targeting_question(ql)
    region_signal = region is not None and _has_hcp_context(ql)
    # A BARE tier mention ("High-tier HCPs", "Low tier prescribers") on
    # its own is deliberately NOT normally enough to trigger this path -
    # "top 10 High-tier prescribers by propensity" also mentions a
    # tier, and that needs to reach path 6 (ranking) instead, not get
    # intercepted here. So this only fires when there's a genuine list
    # request ("which", "show me", ...) AND no ranking language at all
    # - that combination can only mean "just filter to this tier",
    # never a ranking request. Found via testing: without this, a
    # plain "which HCPs have a Low tier?" had nothing to catch it and
    # fell all the way through to a generic document search.
    tier_list_signal = (tier is not None and any(w in ql for w in LIST_WORDS)
                         and not any(p.search(ql) for p in RANK_PATTERNS))

    # Any messaging language anywhere in the question means this whole
    # path steps aside for RAG, which is the only place that can
    # actually join a filtered list with messaging guidance. Computed
    # once, applied to every trigger uniformly - a question like
    # "Which Novo-heavy prescribers should I message about switching,
    # and what should I say?" must not slip past this on a trigger
    # that lacks its own separate messaging guard.
    messaging_override = bool(_MESSAGING_WORDS.search(ql))

    if (not messaging_override) and (
            targeted is not None or competitor_signal or bool(levels) or bool(extra_filters)
            or formulary_signal or sample_signal or forward_target_signal or payer_signal
            or ax_relationship_signal or decile_signal or zero_writer_signal or region_signal
            or tier_list_signal):
        sort_by, ascending, sort_reason = _decide_sort(ql, levels, extra_filters)
        result = qs.filter_hcps(
            state=state,
            region=region if region_signal else None,
            specialty=_detect_specialty(ql),
            tier=tier,
            targeted=targeted,
            dominant_competitor=competitor if competitor_signal else None,
            formulary_tier=formulary_tier if formulary_signal else None,
            recent_sample_request=sample_request if sample_signal else None,
            dominant_payer=payer if payer_signal else None,
            ax_relationship=ax_relationship if ax_relationship_signal else None,
            zero_writer=zero_writer,
            decile=decile if decile_signal else None,
            levels=levels or None,
            extra_filters=extra_filters or None,
            sort_by=sort_by,
            ascending=ascending,
            top=_resolve_top(ql),
        )
        result["sort_reason"] = sort_reason
        return "STRUCTURED / filter", result

    # ---- PATH 4: "which states have the most X-tier HCPs" -----------
    STATE_AGG_PHRASES = ["which states", "what states", "states with the most",
                          "states have the most", "top states"]
    if any(p in ql for p in STATE_AGG_PHRASES):
        return "STRUCTURED / states by tier", qs.states_by_tier(tier=tier or "High")

    # ---- PATH 5: "top prescriber(s) in X" -> ranking by volume ------
    if any(p.search(ql) for p in RANK_PATTERNS) and "propensity" not in ql:
        st = _find_state(ql)
        if st and _has_volume_context(ql):
            n = _resolve_top(ql)
            if n == 1:
                return "STRUCTURED / top prescriber", qs.top_prescriber(st)
            result = qs.filter_hcps(state=st, sort_by="rx_volume_monthly", top=n)
            result["sort_reason"] = "Sorted by rx_volume_monthly (highest first) - ranked by prescribing volume, as asked."
            return "STRUCTURED / top prescribers by volume (state)", result

    # ---- PATH 6: "top N by propensity[, tier]" -----------------------
    if any(p.search(ql) for p in RANK_PATTERNS) and "propensity" in ql:
        n = _resolve_top(ql)
        tier_for_propensity = _detect_tier(ql)
        st = _find_state(ql)
        reason = "Sorted by propensity_score (highest first) - ranked by opportunity, as asked."
        if st:
            result = qs.filter_hcps(state=st, tier=tier_for_propensity, sort_by="propensity_score", top=n)
            result["sort_reason"] = reason
            return "STRUCTURED / top by propensity (state)", result
        result = qs.top_n_by_propensity(n, tier=tier_for_propensity or "High")
        result["sort_reason"] = reason
        return "STRUCTURED / top by propensity", result

    # ---- Catch a region-like word that isn't a REAL region, before ----
    # falling through to RAG - see _detect_invalid_region_attempt's
    # comment for why this matters (a misleading "confident" wrong
    # answer is worse than an honest rejection).
    if not _find_state(ql):
        invalid_region = _detect_invalid_region_attempt(ql)
        if invalid_region:
            return "STRUCTURED / filter", {
                "kind": "filtered_list", "found": False,
                "error": (f"'{invalid_region}' is not a real region in this data. "
                          f"Valid regions are: {qs.KNOWN_VALUES['region']}."),
            }

    # ---- PATH 7: everything else -> RAG (narrative / cards / semantic) ----
    return _ask_rag(question, ql)


# ---------------------------------------------------------------------
# Handles every question that reaches RAG. Always checks a named NPI
# FIRST, and always says clearly if that NPI doesn't exist - this is
# the fix for the old bug where a mistyped NPI silently fell through
# to an unrelated semantic search instead of saying "not found."
# ---------------------------------------------------------------------
def _ask_rag(question, ql):
    m_npi = re.search(r"\b(\d{10})\b", question)

    if m_npi:
        card_result = sd.lookup_card_by_npi(m_npi.group(1))
        if not card_result["found"]:
            # Clear, immediate "not found" - never falls through to a
            # generic search over unrelated documents.
            return "RAG / card lookup", card_result

        if _COMPARISON_WORDS.search(ql):
            # "Compare NPI X to a typical Y" needs the card AND a
            # benchmark chunk to actually compare against.
            specialty = card_result["chunk"].get("specialty") or None
            benchmark_query = (f"typical {specialty} benchmark profile" if specialty
                                else "typical specialty benchmark profile")
            benchmark_result = sd.semantic_search(benchmark_query, top_k=1)
            return "RAG / comparison (card + benchmark)", {
                "kind": "comparison", "found": True,
                "card": card_result["chunk"],
                "benchmark": benchmark_result.get("results", []),
            }

        return "RAG / card lookup", card_result

    # No NPI named - decide between a plain card LIST and a semantic
    # (meaning-based) search, using the same list-word vocabulary as
    # the structured filter path above, so both parts of this file
    # agree on what "give me a list" means.
    wants_list = any(w in ql for w in LIST_WORDS) and not _MESSAGING_WORDS.search(ql)
    is_market = any(w in ql for w in MARKET_WORDS)
    spec = _detect_specialty(ql)
    st = _find_state(ql)

    if wants_list and not is_market and (spec or st):
        result = sd.list_cards(state=st, specialty=spec, limit=DEFAULT_RESULT_LIMIT)
        return f"RAG / card list (state={st}, specialty={spec})", result

    if st:
        result = sd.semantic_search(question, state=st, top_k=5)
        return f"RAG / semantic (state={st})", result

    result = sd.semantic_search(question, top_k=5)
    return "RAG / semantic (general)", result


# =====================================================================
# SECTION 8 - TURNING THE ANSWER INTO A SENTENCE
# -----------------------------------------------------------------
# The ONLY place in this file that writes a sentence a rep can read.
# Dispatches on data["kind"] (a plain label, not key-guessing), and -
# the most important rule in this whole file - ALWAYS checks
# data["found"] first. If nothing was found, that's said plainly,
# before anything else happens.
# =====================================================================

CLARIFICATIONS = {
    "STRUCTURED / top prescriber": (
        "\n\n(Note: \"top\" here means highest current monthly prescription "
        "volume - not propensity/opportunity. Check the tier and propensity "
        "score above separately before treating this as your best target.)"
    ),
}


def format_answer(route, data):
    # ---- Clarification requests aren't a failed lookup - they never
    # reached a search at all, so they're returned as-is, before the
    # "was it found" check below (which is about search results).
    if data.get("kind") == "clarification":
        return data["message"]

    # ---- Rule #1, checked before anything else: was it found? -------
    if not data.get("found", True):
        return f"Answer not found. {data.get('error', 'No matching data.')}"

    kind = data.get("kind")

    if kind == "hcp_lookup":
        answer = qs.format_hcp_scripts(data)
    elif kind == "top_prescriber":
        answer = qs.format_top_prescriber(data)
    elif kind == "writer_count":
        answer = qs.format_count_writers(data)
    elif kind == "ranked_list":
        answer = qs.format_top_n_by_propensity(data)
    elif kind == "state_summary":
        answer = qs.format_states_by_tier(data)
    elif kind == "filtered_list":
        if data["count"] == 0:
            return "Answer not found. No HCPs match these filters."
        answer = qs.format_filter_hcps(data)

    elif kind == "card_lookup":
        chunk = data["chunk"]
        answer = chunk["text"]

    elif kind == "card_list":
        if data["count"] == 0:
            return "Answer not found. No HCP cards match this state/specialty."
        lines = ["Not ranked - these are simply every matching card, in no particular order "
                 "(this is a lookup, not a score-based ranking).",
                 f"{data['count']} matching HCP card(s):"]
        for c in data["chunks"]:
            lines.append(f"  NPI {c['npi']} ({c['specialty']}, {c['state']})")
        answer = "\n".join(lines)

    elif kind == "semantic_search":
        if data.get("low_confidence") or not data.get("results"):
            return "Answer not found. No confident match in the documents for this question."
        lines = ["Sorted by relevance to your question (highest similarity score first)."]
        if data.get("confidence") == "moderate":
            lines.append("(Note: confidence in this match is modest - worth checking the "
                          "source document before relying on it heavily.)")
        for r in data["results"]:
            lines.append(f"[{r['score']:.3f}] {r['chunk']['text'][:300]}")
        answer = "\n\n".join(lines)

    elif kind == "comparison":
        lines = [data["card"]["text"]]
        if data["benchmark"]:
            lines.append("\n--- Benchmark for comparison ---")
            lines.append(data["benchmark"][0]["chunk"]["text"])
        else:
            lines.append("\n(No benchmark document found to compare against.)")
        answer = "\n".join(lines)

    else:
        # Should never happen if every engine function tags "kind"
        # correctly - surfaced loudly rather than silently guessed at.
        return f"Answer not found. Unrecognized result shape (kind={kind})."

    if route in CLARIFICATIONS:
        answer += CLARIFICATIONS[route]
    return answer


# =====================================================================
# Quick manual test - only runs if you execute this file directly
# (python ask_a_question.py). Not used when imported by another script.
# =====================================================================
if __name__ == "__main__":
    questions = [
        "Who is the top GLP-1 prescriber in New York?",
        "How many active GLP-1 writers are in Texas?",
        "List the top 10 High-tier prescribers by propensity",
        "Which prescribers in the South region have a high switching score?",
        "Which prescribers in South Carolina have a high switching score?",
        "Show me High-propensity endocrinologists in Florida who are not currently targeted",
        "Which HCPs have a high percent Novo and low rep engagement?",
        "How is Rabivy different from Zepbound?",
        "Tell me about NPI 0000000000",  # deliberately fake, to test "not found"
    ]
    for q in questions:
        route, data = ask(q)
        print("=" * 72)
        print(f"Q: {q}")
        print(f"   -> {route}")
        print(format_answer(route, data))
        print()