# ground_truth.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# Computes the CURRENT correct answer for each golden question, LIVE,
# by calling the exact same real functions the system itself uses
# (query_spreadsheet.py). Nothing in here is a frozen number typed in
# by hand - every value is worked out fresh each time a test runs, the
# same way the real system is supposed to work it out. This is what
# lets test_the_system.py and test_the_agent.py survive a data refresh
# without needing to be hand-edited every time - the DATA can change
# and these tests still check the right thing.
#
# Also holds NARRATIVE_FACTS: a small table used ONLY for RAG/narrative
# questions, where "the answer" is written prose rather than a data
# value that can be computed. Each entry's facts were pulled by
# actually reading the real source chunk (docs/rep_field/,
# docs/clinical/, docs/strategic/, docs/reference/) - never guessed or
# paraphrased from the question itself, since that risks a check that
# passes just because the answer echoes the question's own wording.
# =====================================================================

import query_spreadsheet as qs

# The regex router and the LLM agent apply DIFFERENT default row
# limits when a question doesn't say how many results it wants. "The
# default" is not one shared number - ground truth must use whichever
# default belongs to the system actually being tested.
ROUTER_DEFAULT_TOP = 20
AGENT_DEFAULT_TOP = 10


def npi_str(row_or_dict):
    """Turn a raw NPI (int / float / numpy scalar) into a plain string
    for comparing against system output text."""
    return str(int(row_or_dict["npi"]))


# =====================================================================
# LIVE VALUE LOOKUPS (Layer 3 for STRUCTURED single-value questions)
# =====================================================================

def gt_top_prescriber(state):
    """Today's #1 prescriber (by monthly Rx volume) in a state."""
    data = qs.top_prescriber(state)
    return npi_str(data) if data.get("found") else None


def gt_scripts_for_npi(npi):
    """Today's monthly script count for one specific NPI."""
    data = qs.get_row_by_npi(npi)
    return str(int(data["rx_volume_monthly"])) if data.get("found") else None


def gt_count_writers(state, asked_about="active"):
    """Today's count of active (or zero) GLP-1 writers in a state."""
    data = qs.count_writers(state, asked_about=asked_about)
    return str(data[asked_about]) if data.get("found") else None


def gt_top_n_national_npis(tier="High", n=10):
    """Today's top-N NPIs nationally by propensity score, within a tier.
    Index 0 is the single #1 HCP."""
    data = qs.top_n_by_propensity(n=n, tier=tier)
    if not data.get("found") or not data["results"]:
        return None
    return [npi_str(r) for r in data["results"]]


def gt_top_state_by_tier(tier="High"):
    """Today's state with the most HCPs of a given tier."""
    data = qs.states_by_tier(n=1, tier=tier)
    if not data.get("found") or not data["results"]:
        return None
    return data["results"][0]["state"]


# =====================================================================
# LIVE ROW-SET LOOKUPS (Layer 3 for filtered / multi-row questions)
# =====================================================================

def gt_filter_npis(top=ROUTER_DEFAULT_TOP, **filter_kwargs):
    """Today's set of NPIs matching a filter recipe, using the exact
    same filter_hcps() function the real system calls. `top` should
    match whatever default the SYSTEM UNDER TEST applies (see
    ROUTER_DEFAULT_TOP / AGENT_DEFAULT_TOP above) so this check is
    neither stricter nor looser than real behaviour.

    NOTE for whoever runs this first: a couple of the filter recipes
    below (in test_the_system.py / test_the_agent.py) encode a
    reasonable GUESS at exact thresholds (e.g. what counts as "high
    switching score"). These are flagged inline where they're used -
    verify against a real run and tighten the threshold if the regex
    router or LLM resolves it differently in practice."""
    data = qs.filter_hcps(top=top, **filter_kwargs)
    if not data.get("found"):
        return None
    return [npi_str(r) for r in data["results"]]


# =====================================================================
# MATCHING HELPERS
# Shared by both test files so the pass/fail logic can't drift between
# them. `expected` items can be:
#   - a plain string            -> must appear (AND item)
#   - a list of strings         -> any ONE of them is enough (OR group)
# A list of these items, checked together, means every item (or every
# OR group) must be satisfied - same convention the old files used,
# just centralised here instead of copy-pasted three times.
# =====================================================================

def _one_matches(haystack, item):
    if isinstance(item, list):
        return any(alt.lower() in haystack for alt in item)
    return item.lower() in haystack


def all_present(haystack, expected_list):
    """True only if every item (or OR-group) in expected_list is found."""
    if not expected_list:
        return True
    return all(_one_matches(haystack, item) for item in expected_list)


def any_present(haystack, expected_list):
    """True if at least one item (or OR-group) in expected_list is found -
    used for tag/chunk_id checks where several tag names are plausible
    (search indexing can name the same chunk slightly differently)."""
    if not expected_list:
        return True
    return any(_one_matches(haystack, item) for item in expected_list)


def gt_filter_count(**filter_kwargs):
    """Today's TOTAL count of HCPs matching a filter recipe (not just
    the top-N rows) - for questions like 'days since contact over 90'
    where the meaningful ground truth is the total match count printed
    in the answer, not any particular NPI (the top-20 rows shown are an
    arbitrary slice of a much bigger true set)."""
    data = qs.filter_hcps(top=1, **filter_kwargs)  # top=1 is enough - we only need "count"
    if not data.get("found"):
        return None
    return str(data["count"])


def _apply_filters(df, filters):
    """Applies equality filters, case-insensitively for string values -
    the dataframe stores some columns lowercase (state) and others
    properly-cased (specialty, tier) - callers shouldn't need to know
    which is which."""
    sub = df
    for col, val in filters.items():
        if isinstance(val, str):
            sub = sub[sub[col].str.lower() == val.lower()]
        else:
            sub = sub[sub[col] == val]
    return sub


def gt_pct_matching(subset_filters, condition_column, condition_value):
    """Percentage of rows matching subset_filters that ALSO satisfy
    condition_column == condition_value. Returns a dict with the EXACT
    underlying counts as well as the derived percentage - the counts
    are whole numbers with zero legitimate rounding ambiguity, so they
    should be checked exactly, never with a tolerance. Only the
    derived percentage itself should get any rounding leniency."""
    sub = _apply_filters(qs.df, subset_filters)
    total = len(sub)
    if total == 0:
        return None
    if isinstance(condition_value, str):
        match = sub[condition_column].str.lower() == condition_value.lower()
    else:
        match = sub[condition_column] == condition_value
    matching = int(match.sum())
    return {"total_count": total, "matching_count": matching,
            "percentage": 100 * matching / total}  # full precision - caller decides rounding


def gt_mean_column(column, **filters):
    """Mean of a numeric column among rows matching the given filters.
    Returns a dict with the EXACT row count alongside the mean - the
    count should be checked exactly (no legitimate ambiguity), the
    mean itself can have a small rounding tolerance."""
    sub = _apply_filters(qs.df, filters)
    count = len(sub)
    if count == 0:
        return None
    # 3 decimals, not 1 - a 0-1 scale metric (propensity_score) needs more
    # precision than 1dp gives. Found 2026-07-28: 1dp rounding turned a
    # correct raw mean of 0.3548 into 0.4, making the agent's exactly-right
    # answer of 0.355 look like a false FAIL against too-tight a tolerance.
    # UPDATED: no rounding at all now - full precision, caller decides.
    return {"count": count, "mean": float(sub[column].mean())}

def gt_sum_column(column, **filters):
    """Total (sum) of a numeric column among rows matching the given
    filters - a genuinely different operation from an average (no
    division involved), so a wrong answer here means a different kind
    of mistake than a wrong mean would."""
    sub = _apply_filters(qs.df, filters)
    count = len(sub)
    if count == 0:
        return None
    return {"count": count, "sum": float(sub[column].sum())}


def gt_group_diff(filters_a, filters_b):
    """Counts for TWO separate groups plus their difference - tests
    whether the agent correctly retrieves two independent facts before
    combining them, a different failure mode from a single-group stat
    (either count alone could be wrong, or the subtraction itself)."""
    count_a = len(_apply_filters(qs.df, filters_a))
    count_b = len(_apply_filters(qs.df, filters_b))
    return {"count_a": count_a, "count_b": count_b, "difference": count_a - count_b}


def gt_hcp_card_facts(npi):
    """For a specific-NPI lookup/summary question: a specific NPI lookup
    is ALWAYS fully determinable, no matter what the data says - so
    Layer 3 here should never be a skip. Returns the NPI, tier, and
    specialty live from the row - facts that must appear in ANY correct
    summary of this HCP, whatever the current data values are."""
    data = qs.get_row_by_npi(npi)
    if not data.get("found"):
        return None
    return [str(int(data["npi"])), data["tier"], data["specialty"]]


import re as _re


def gt_state_market_fact(state, doc_path="docs/reference/state_market_summary.md"):
    """Parses the REAL state_market_summary.md file live, at test time,
    for one state's 'HCP universe' figure. This is deliberately reading
    the actual file rather than hardcoding a number pulled from it once -
    if the doc is regenerated with new figures, this check follows it
    automatically instead of going stale."""
    try:
        with open(doc_path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return None
    m = _re.search(rf"^## {_re.escape(state)}\n- HCP universe: ([\d,]+)", text, _re.MULTILINE)
    if not m:
        return None
    return m.group(1)  # e.g. "312" - kept with commas stripped by caller if needed


def _with_comma_variant(n):
    """Returns [plain, comma-formatted] so a check matches whichever
    format the source text happens to use (e.g. '1536' vs '1,536')."""
    return [str(n), f"{n:,}"]


def gt_specialty_benchmark_facts(specialty):
    """Live-computed benchmark figures for a specialty, straight from
    the same dataframe query_spreadsheet.py uses - NOT read from the
    static specialty_benchmark_profiles.md doc, because a real check
    (2026-07-28) found that doc's 'active writers' figure has already
    DRIFTED slightly from the live spreadsheet snapshot (1,519 vs a
    live-computed 1,536) - exactly the kind of silent staleness this
    whole ground-truth approach exists to catch. Only the two figures
    that verified EXACTLY against the live data are used here:
    population count and mean propensity score. Population is returned
    as a comma/no-comma OR-group since the source text writes it with
    a thousands separator ('1,536')."""
    sub = qs.df[qs.df["specialty"] == specialty]
    if sub.empty:
        return None
    population = _with_comma_variant(len(sub))
    mean_propensity = str(round(sub["propensity_score"].mean() * 100))
    return [population, mean_propensity]


# ---------------------------------------------------------------------
# CLARIFICATION / REJECTION PHRASE LISTS
# Used as Layer 3 (the ANSWER) for questions whose only correct output
# is "ask for clarification" or "say clearly this isn't valid" - these
# are NOT skipped, because the correct answer is fully determinable and
# never changes: a specific-doctor question with no NPI given must
# always ask for the NPI, regardless of what the data says. Several
# phrasings are listed as alternatives (an OR-group) since an LLM may
# word this differently each time - the router's wording is fixed, but
# checking the same broad list costs nothing and stays consistent.
# ---------------------------------------------------------------------
CLARIFICATION_FACTS = [["10-digit npi", "which npi", "provide the npi", "specify the npi",
                        "no npi is given", "don't know which", "which doctor", "which hcp", "clarif"]]

FAKE_NPI_REJECTION_FACTS = [["not found", "no hcp", "no record", "doesn't exist", "no hcp card found"]]

FAKE_REGION_REJECTION_FACTS = [["not a real region", "not one of", "not an official", "isn't a defined region",
                                "4 real regions", "midwest, northeast, south", "not a real value for",
                                "valid values are"]]



# tag   -> list of acceptable chunk_id/tag substrings (Layer 2: did it
#          retrieve the RIGHT chunk, regardless of wording)
# facts -> list of specific facts that must appear in the final answer,
#          pulled from the real chunk content (Layer 3: thin, but
#          anchored to the source, not to the question's own words)
# =====================================================================
NARRATIVE_FACTS = {
    "competitive_switchers": {
        "tag": ["competitive_switchers"],
        "facts": [["novo", "lilly"], "monthly dosing"],
        "source": "docs/rep_field/rep_talking_points_by_segment.md",
    },
    "zepbound_differentiator": {
        # Tightened 2026-07-28 (same fix as discontinuation, applied
        # preemptively): "differentiator" alone is a plain English word
        # that could accidentally match an unrelated chunk's prose. The
        # real chunk_id is
        # rabivy_product_benefits_brief__the_headline_differentiator_monthly_dosing
        # - "headline_differentiator" only ever appears as part of that
        # real chunk_id, never as ordinary body text.
        "tag": ["headline_differentiator"],
        "facts": ["monthly", "dosing"],
        "source": "docs/rep_field/rabivy_product_benefits_brief.md",
    },
    "ozempic_objection": {
        "tag": ["objection_handling"],
        "facts": ["monthly dosing", "mechanism"],
        "source": "docs/rep_field/objection_handling_guide.md",
    },
    "medicaid_access": {
        "tag": ["payer_access"],
        "facts": ["13", "medicaid"],
        "source": "docs/reference/payer_access_brief.md (uploaded copy)",
    },
    "missouri_market": {
        "tag": ["state_market_summary__missouri"],
        "facts": None,  # computed LIVE, see "state" below - not hardcoded (was [["283","136"]])
        "state": "Missouri",
        "source": "docs/reference/state_market_summary.md (parsed live at test time, not hardcoded)",
    },
    "arizona_market": {
        "tag": ["state_market_summary__arizona"],
        "facts": None,
        "state": "Arizona",
        "source": "docs/reference/state_market_summary.md (parsed live at test time, not hardcoded)",
    },
    "tirzepatide_mechanism": {
        "tag": ["mechanism_comparison", "molecule_and_mechanism", "how_is_this_different_from_ti"],
        "facts": ["antagonism", "agonism"],
        "source": "docs/rep_field/objection_handling_guide.md + docs/clinical/glp1_class_overview.md",
    },
    "dosing_advantage": {
        "tag": ["monthly_dosing", "where_rabivy_wins", "positioning_summary"],
        "facts": ["monthly", "dosing"],
        "source": "docs/rep_field/rabivy_product_benefits_brief.md (same brief as "
                  "zepbound_differentiator - not independently re-checked, "
                  "reuses an already-verified source)",
    },
    "prior_auth_access": {
        "tag": ["prior_auth"],
        "facts": [["31", "41"]],
        "source": "docs/clinical/real_world_evidence_brief.md",
    },
    "discontinuation": {
        # FIXED 2026-07-28: was ["why_patients_discontinue", "persistence",
        # "discontinue"] - the last two are common English words that
        # showed up in an UNRELATED chunk's body prose ("Reframe around
        # persistence data... patients discontinue within the first
        # year"), causing Layer 2 to falsely PASS even when the WRONG
        # chunk was retrieved (found via test_retrieval_ranking.py,
        # which checks chunk_id specifically rather than searching all
        # retrieved text). Same false-positive mechanism as the Q25
        # "npi"-as-JSON-key bug fixed earlier. Now only the real,
        # verified, precise chunk_id counts.
        "tag": ["why_patients_discontinue"],
        "facts": ["side effects", "cost"],
        "source": "docs/clinical/real_world_evidence_brief.md",
    },
    # VERIFIED 2026-07-28: real chunk_id confirmed via chunks_tagged.json.
    # Facts are computed LIVE (see gt_specialty_benchmark_facts) rather
    # than hardcoded - population count and mean propensity verified to
    # match the live dataframe exactly. NOTE: the doc's own "active
    # writers" figure (1,519) does NOT match a live filter (1,536) - a
    # real drift between this static doc and the current data snapshot,
    # so that particular figure is deliberately NOT used as a fact here.
    "typical_endocrinologist": {
        "tag": ["specialty_benchmark_profiles__typical_endocrinology_profile"],
        "facts": None,  # computed LIVE via "specialty" below
        "specialty": "Endocrinology",
        "source": "docs/reference/specialty_benchmark_profiles.md + live query_spreadsheet.df",
    },
}