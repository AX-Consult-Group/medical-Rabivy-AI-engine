# ask.py
# -------------------------------------------------------------------
# The single entry point. Looks at a question and decides which engine
# should answer it:
#   - STRUCTURED engine  -> ranking / counting / filtering (the spreadsheet)
#   - RAG (search.py)    -> narrative / card / characterisation questions
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
# NOTE on warm state: like search.py, structured.py and search.py both
# load their data (Excel / embedding model+vectors) once, at import
# time. That's correct for a long-lived process - it only becomes a
# problem if ask.py itself gets re-imported per request once this is
# behind an API. Keep this a single warm process.
# -------------------------------------------------------------------

import re
import structured
import search   # your existing RAG router (NPI lookup / state / semantic)

# ---- Rank / count / list cue words --------------------------------
RANK_WORDS = ["top ", "highest", "most", "rank", "leading", "biggest"]

# Word-boundary versions - same fix as search.py's RANKING_PATTERNS.
# Bare substrings here ("most", "rank", "leading") risk matching inside
# unrelated words ("almost", "misleading", "outrank"), and "top " alone
# matches inside "stop " ("Why do patients STOP taking..." - a real bug
# found via an actual eval run). Use these patterns, not the raw list.
RANK_PATTERNS = [re.compile(r"\btop\b"), re.compile(r"\bhighest\b"),
                  re.compile(r"\bmost\b"), re.compile(r"\brank\b"),
                  re.compile(r"\bleading\b"), re.compile(r"\bbiggest\b")]
COUNT_WORDS = ["how many", "number of", "count"]
LIST_WORDS = ["list", "which ", "show me", "find ", "who are", "give me"]

# Explicit phrases for the states-aggregate report, instead of a bare
# "state" substring (which only worked before because "states" happens
# to contain it - fragile, and the same accidental-match risk flagged
# in chunk.py/tag.py/search.py elsewhere in this project).
STATE_AGG_PHRASES = ["which states", "what states", "states with the most",
                     "states have the most", "top states"]

SPECIALTIES = {"endocrinolog": "Endocrinology", "primary care": "Primary Care",
               "obesity medicine": "Obesity Medicine", "obesity": "Obesity Medicine"}

# Same abbreviation set added to search.py, kept consistent here so
# state detection doesn't silently work in one engine's router and not
# the other's.
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

# Competitor language, same shape as tag.py's COMPETITOR_KEYWORDS so
# the two stay easy to keep in sync if a third competitor is added.
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


def _find_state(ql):
    for s in structured.df["state"].unique():
        if s in ql:
            # df's state column is lowercase (asserted in structured.py);
            # return the title-cased form for display/matching consistency.
            return s.title()
    for abbr, full in STATE_ALIASES.items():
        if re.search(r"\b" + abbr + r"\b", ql):
            return full
    return None


def _detect_specialty(ql):
    for key, val in SPECIALTIES.items():
        if key in ql:
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


def ask(question):
    ql = question.lower()

    # 1. "how many ... writers in X" -> count
    if any(w in ql for w in COUNT_WORDS) and ("writer" in ql or "prescriber" in ql or "hcp" in ql):
        st = _find_state(ql)
        if st:
            return "STRUCTURED / count", structured.count_writers(st)

    # 2. "how many scripts did NPI ... write" -> single-NPI script lookup
    m = re.search(r"\b(\d{10})\b", question)
    if m and ("script" in ql or "rx" in ql or "write" in ql or "wrote" in ql):
        return "STRUCTURED / hcp scripts", structured.hcp_scripts(m.group(1))

    # 3. Multi-field targeting filter - THE previously-missing path.
    # Triggered by targeted-status, competitor, or switching-score
    # language, none of which any other path (here or in search.py)
    # can answer, since chunks_tagged.json has none of these fields.
    # Tier alone doesn't trigger this - see note at path 6 for why.
    targeted = _detect_targeted(ql)
    competitor = _detect_competitor(ql)
    min_switching = _detect_min_switching(ql)
    competitor_signal = competitor is not None and _has_hcp_context(ql)
    if targeted is not None or competitor_signal or min_switching is not None:
        return "STRUCTURED / filter", structured.filter_hcps(
            state=_find_state(ql),
            specialty=_detect_specialty(ql),
            tier=_detect_tier(ql),
            targeted=targeted,
            dominant_competitor=competitor,
            min_switching=min_switching,
        )

    # 4. "which states have the most high-tier HCPs" -> aggregate report.
    # Explicit phrase match instead of a bare "state" substring.
    if any(p in ql for p in STATE_AGG_PHRASES) and "high" in ql:
        return "STRUCTURED / states by high tier", structured.states_by_high_tier()

    # 5. "top prescriber in X" -> ranking within a state (volume, not propensity)
    if any(p.search(ql) for p in RANK_PATTERNS) and "propensity" not in ql:
        st = _find_state(ql)
        if st and ("prescriber" in ql or "writer" in ql or "volume" in ql):
            return "STRUCTURED / top prescriber", structured.top_prescriber(st)

    # 6. "top N by propensity[, tier]" -> ranking by propensity.
    # If a state is also named, this used to silently drop it (a real
    # bug: "top prescriber in Texas by propensity" matched here with no
    # state parameter available on top_n_by_propensity at all). Now: a
    # named state routes through filter_hcps instead, which does
    # support state + sort_by, so the state constraint is honoured.
    # Tier is only applied if the question actually named one - no more
    # silently defaulting every unscoped "top N by propensity" to High.
    if any(p.search(ql) for p in RANK_PATTERNS) and "propensity" in ql:
        n = 10
        mn = re.search(r"top\s+(\d+)", ql)
        if mn:
            n = int(mn.group(1))
        tier = _detect_tier(ql)
        st = _find_state(ql)
        if st:
            return "STRUCTURED / top by propensity (state)", structured.filter_hcps(
                state=st, tier=tier, sort_by="propensity_score", top=n)
        return "STRUCTURED / top by propensity", structured.top_n_by_propensity(n, tier=tier)

    # 7. Otherwise -> hand it to the RAG (narrative / cards / semantic).
    # Returns actual chunk text + source metadata, not just chunk_id/score
    # - a bare ID list can't feed an answer-synthesis step that needs
    # the real retrieved content.
    route, results = search.search(question)
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
# separate from ask() itself, same principle as structured.py's
# format_* functions.
# =====================================================================

def format_answer(route, data):
    if route.startswith("STRUCTURED"):
        if "results" in data and "n" in data:
            return structured.format_top_n_by_propensity(data)
        if "results" in data and "count" in data:
            return structured.format_filter_hcps(data)
        if "results" in data:
            return structured.format_states_by_high_tier(data)
        if "active" in data:
            return structured.format_count_writers(data)
        if "rx_volume_monthly" in data and "state" in data:
            return structured.format_top_prescriber(data)
        if "rx_volume_monthly" in data:
            return structured.format_hcp_scripts(data)
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