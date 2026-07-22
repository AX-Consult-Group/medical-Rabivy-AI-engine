# query_spreadsheet.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# This file is the STRUCTURED data engine. It knows how to look things
# up, count things, filter things, and sort things in the master HCP
# spreadsheet - and nothing else.
#
# IMPORTANT - what this file does NOT do:
# This file does not try to understand a rep's question. It never reads
# a sentence like "who's the top prescriber in Texas" and figures out
# what that means. That job belongs somewhere else (ask_a_question.py).
# Every function in this file expects to be handed clean, already-
# decided values - a real state name, a real column name, a real
# number - and it just does the lookup.
#
# GOAL OF THIS VERSION: every real column in the spreadsheet can be
# filtered on and sorted by, in some way:
#   - Word-based ("category") columns - state, region, specialty, tier,
#     dominant_competitor, formulary_tier, dominant_payer,
#     ax_relationship - filtered by giving the exact real value.
#   - Yes/no columns - targeted, zero_writer, recent_sample_request -
#     filtered by True/False or 0/1.
#   - 0-to-1 SCORE columns (propensity_score, switching_score,
#     pa_burden, obesity_prev, rep_engagement_score, nrx_share, and the
#     pct_* payer/competitor-share columns) - filtered with a plain
#     word: "high" (>= 0.75), "low" (<= 0.25), or "moderate" (between).
#     This ONE standard cutoff is used everywhere in this file, instead
#     of every different question picking its own guessed number.
#   - ANY other numeric column (rx_volume_monthly, days_since_contact,
#     years_practice, etc.) - filtered with an exact "above this number
#     / below this number", since a flat 0.75/0.25 cutoff wouldn't mean
#     anything for a column like "days since contact" (which ranges
#     from 0 to over 1000, not 0 to 1).
#   - EVERY column, of any type, can be sorted by - highest-first or
#     lowest-first, whichever is asked for.
#
# Who calls this file:
#   - ask_a_question.py (today), which works out what the rep meant
#     and calls the right function here with clean arguments.
#   - In future, an AI agent could call these functions directly the
#     same way, IF each function's job is obvious from its name and
#     comment alone - that's why every function below has a plain-
#     English explanation above it.
#
# Every function returns a plain dictionary (never a sentence). Every
# dictionary includes:
#   - "kind"  : a short label saying what TYPE of result this is, so
#               whatever's reading it doesn't have to guess.
#   - "found" : True/False - did this actually produce a usable result?
#   - "error" : only present when found is False - a plain-English
#               reason, so a bad request never comes back looking the
#               same as a real empty answer.
#
# BUG-CLASS FIX IN THIS VERSION (see count_by_flag() below):
# A real test caught count_writers() always reporting the ACTIVE count,
# even when the question asked about ZERO-writers - the data was right,
# only the sentence was wrong, because nothing recorded which count the
# caller actually wanted highlighted. Rather than patch that one
# function, this version pulls the fix into a single reusable
# primitive (count_by_flag) that any yes/no column - zero_writer,
# targeted, sample_request_recent, or a new one added later - can use
# safely, so this exact bug can't quietly reappear somewhere else.
# =====================================================================

import glob
import os
import pandas as pd

DATA_DIR = "data"
FILE_PATTERN = "rabivy_propensity_data*.xlsx"

# The standard cutoffs used for every 0-to-1 score column in this
# file. One number, used consistently everywhere, instead of every
# question picking its own guess (the old code used 0.6 in one place
# and 0.3 in another - this replaces both with one agreed policy).
HIGH_CUTOFF = 0.75
LOW_CUTOFF = 0.25

# Columns that are IDs or internal computation steps, not real filter
# targets - excluded even if their numbers happen to fall in a 0-1
# range, since "high"/"low" wouldn't mean anything sensible for them.
_NEVER_A_SCORE_COLUMN = {"npi", "hcp_id", "targeted", "zero_writer",
                          "sample_request_recent", "decile"}


# ---------------------------------------------------------------------
# Finds the master spreadsheet file to load. Picks the most RECENT one
# if there's more than one sitting in the data folder (dates are
# written as YYYY.MM.DD in the filename, so sorting the names also
# sorts them by date correctly).
# ---------------------------------------------------------------------
def _find_data_file():
    candidates = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)))
    if not candidates:
        raise FileNotFoundError(
            f"No file matching '{FILE_PATTERN}' found in '{DATA_DIR}/'. "
            f"Expected the master propensity spreadsheet there."
        )
    return candidates[-1]


_DATA_PATH = _find_data_file()
df = pd.read_excel(_DATA_PATH)

# ---------------------------------------------------------------------
# NPI clean-up. Excel sometimes stores ID numbers as decimals (e.g.
# 1234567890.0) if any row has a blank one - this turns them all into
# consistent whole numbers, once here, so nothing downstream has to
# worry about it.
# ---------------------------------------------------------------------
df["npi"] = pd.to_numeric(df["npi"], errors="coerce")
if df["npi"].isna().any():
    _bad_npi = df["npi"].isna().sum()
    print(f"! WARNING: {_bad_npi} row(s) have a non-numeric/missing NPI - "
          f"these rows won't be reachable by NPI lookup.")
df["npi"] = df["npi"].astype("Int64")  # whole numbers that can still hold blanks safely

# ---------------------------------------------------------------------
# Safety check: every state lookup in this file assumes the "state"
# column is written in lowercase (e.g. "texas", not "Texas"). If that's
# ever not true, every state lookup below would quietly return nothing
# and nobody would know why. So: check it loudly, right now, at
# startup - not silently with `assert` (which can be switched off by
# accident), but with a real error that always fires.
# ---------------------------------------------------------------------
_non_lower = df["state"].dropna().map(lambda s: s != s.lower())
if _non_lower.any():
    raise ValueError(
        "The 'state' column contains values that are not lowercase. "
        "Every state lookup in this file assumes lowercase - fix the "
        "source data or update this file's state-handling logic."
    )


# =====================================================================
# KNOWN REAL VALUES - built directly from the spreadsheet itself
# -----------------------------------------------------------------
# Plain English: this is a "cheat sheet" of every real value that
# actually appears in each column - built fresh from the real data
# every time this file loads, not typed in by hand and guessed. It's
# used below to catch a bad request early ("Watch" is a real tier,
# "Low" is not) instead of silently returning zero results.
#
# CATEGORICAL_COLUMNS is found automatically: every text (word-based)
# column in the spreadsheet, except "state" (which has its own
# lowercase handling above) and "hcp_id" (an ID, not a real category).
# Building this list automatically - instead of typing it out by hand -
# means a brand new text column (like "region" was, until today) gets
# covered the moment it appears in the spreadsheet, with no code change
# needed here.
# =====================================================================
_id_like_columns = {"state", "hcp_id"}
CATEGORICAL_COLUMNS = sorted([
    col for col in df.select_dtypes(include=["object", "str"]).columns
    if col not in _id_like_columns
])
KNOWN_VALUES = {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_COLUMNS}

# ax_relationship has no "Zero" string value in the raw data - a
# missing/blank cell is how "no AX product relationship" is actually
# recorded (confirmed: 41.6% of all rows are blank here, which is far
# too many to be random missing data). "Zero" is added as a real,
# filterable value so those ~6,200 HCPs aren't permanently unreachable
# by any filter - filter_hcps() below checks for this label
# specifically and filters on the blank cells instead of an equality
# match, since a blank value can never equal anything by definition.
AX_RELATIONSHIP_ZERO_LABEL = "Zero"
KNOWN_VALUES["ax_relationship"].append(AX_RELATIONSHIP_ZERO_LABEL)
KNOWN_STATES = sorted(df["state"].dropna().unique().tolist())  # lowercase, as stored

# ---------------------------------------------------------------------
# Which numeric columns are real "0 to 1 scores" (like propensity_score
# or pa_burden), found automatically by checking which numeric columns
# actually only ever contain values between 0 and 1 in the real data -
# instead of a hand-typed list that could go stale. ID-like and
# yes/no columns are excluded even though they're technically 0-1,
# since "high"/"low" doesn't mean anything for a plain yes/no flag.
# ---------------------------------------------------------------------
_numeric_columns = df.select_dtypes(include="number").columns
SCORE_COLUMNS = sorted([
    col for col in _numeric_columns
    if col not in _NEVER_A_SCORE_COLUMN
    and df[col].min() >= 0 and df[col].max() <= 1
])

# Every other real numeric column - filterable by an exact number
# (above/below), and sortable, but with no "high/low" word shortcut,
# since a flat 0.75/0.25 cutoff wouldn't mean anything for these
# (e.g. days_since_contact ranges from 0 to over 1000, not 0 to 1).
OTHER_NUMERIC_COLUMNS = sorted([
    col for col in _numeric_columns
    if col not in _NEVER_A_SCORE_COLUMN and col not in SCORE_COLUMNS
])

# The real yes/no (boolean-style) columns in this spreadsheet - used
# by count_by_flag() below so a new flag column can be counted safely
# without writing a brand new one-off function each time.
FLAG_COLUMNS = ("zero_writer", "targeted", "sample_request_recent")


# ---------------------------------------------------------------------
# Checks one word-based (categorical) value - like tier="High" - against
# the real values that actually exist in that column. Returns None if
# it's fine, or a plain-English error message (listing the real valid
# values) if it isn't.
# ---------------------------------------------------------------------
def _check_categorical(column, value):
    if value is None:
        return None
    valid = KNOWN_VALUES.get(column)
    if valid is None:
        return f"'{column}' is not a recognized category column in this data."
    if value not in valid:
        return f"'{value}' is not a real value for '{column}'. Valid values are: {valid}"
    return None


# ---------------------------------------------------------------------
# Turns a state name typed in any case ("Texas", "TEXAS", "texas")
# into the lowercase form the spreadsheet actually uses, and checks
# it's a real state. Returns (clean_state_or_None, error_or_None).
# ---------------------------------------------------------------------
def _clean_state(state):
    if state is None:
        return None, None
    clean = state.lower().strip()
    if clean not in KNOWN_STATES:
        return None, f"'{state}' is not a recognized state in this data."
    return clean, None


# ---------------------------------------------------------------------
# Turns a word - "high", "moderate", or "low" - into the actual number
# range it means, using the ONE standard cutoff for this whole file
# (HIGH_CUTOFF / LOW_CUTOFF at the top of the file - same number for
# every score column, no exceptions). Returns (min_value, max_value) -
# whichever side doesn't apply is None.
# ---------------------------------------------------------------------
def _level_to_range(level):
    level = level.lower().strip()
    if level == "high":
        return HIGH_CUTOFF, None
    if level == "low":
        return None, LOW_CUTOFF
    if level == "moderate":
        return LOW_CUTOFF, HIGH_CUTOFF
    return None, None  # caller checks this case separately, see _check_level


# ---------------------------------------------------------------------
# Checks a {column: "high"/"moderate"/"low"} request is actually
# possible: is this column a real 0-1 score column, and is the word
# one of the three allowed words? Returns None if fine, or a
# plain-English error if not.
# ---------------------------------------------------------------------
def _check_level(column, level):
    if column not in SCORE_COLUMNS:
        return (f"'{column}' isn't a 0-1 score column, so 'high'/'moderate'/'low' "
                f"doesn't apply to it. Score columns you CAN use this on: {SCORE_COLUMNS}. "
                f"For other numeric columns, filter with an exact number instead "
                f"(see extra_filters).")
    if level.lower().strip() not in ("high", "moderate", "low"):
        return f"'{level}' must be one of: high, moderate, low."
    return None


# ---------------------------------------------------------------------
# Turns a {column: "high"/"moderate"/"low"} dict into a plain-English
# statement of exactly what number those words mean - "high (0.75 or
# above)", not just the bare word "high" on its own. Used every single
# time `levels` is part of a request, so nobody ever has to remember
# (or guess) what this file's cutoff actually is.
# ---------------------------------------------------------------------
def describe_levels(levels):
    if not levels:
        return None
    parts = []
    for column, level in levels.items():
        level = level.lower().strip()
        if level == "high":
            parts.append(f"{column} = high ({HIGH_CUTOFF} or above)")
        elif level == "low":
            parts.append(f"{column} = low ({LOW_CUTOFF} or below)")
        elif level == "moderate":
            parts.append(f"{column} = moderate (between {LOW_CUTOFF} and {HIGH_CUTOFF})")
    return "Levels used: " + "; ".join(parts) + "."


# =====================================================================
# DATA FUNCTIONS
# -----------------------------------------------------------------
# Each one below does ONE lookup job and hands back a plain dictionary
# - never a written sentence. Turning the dictionary into a sentence
# a rep can read happens later, in the PRESENTATION section further
# down this file.
# =====================================================================

# ---------------------------------------------------------------------
# Looks up ONE specific HCP by their exact NPI number. This is the
# main way this file's data joins up with the RAG/document side of the
# system - both sides use the same NPI as the shared ID.
# ---------------------------------------------------------------------
def get_row_by_npi(npi):
    try:
        npi_int = int(npi)
    except (ValueError, TypeError):
        return {"kind": "hcp_lookup", "found": False,
                "error": f"'{npi}' is not a valid NPI (not a number)."}

    row = df[df["npi"] == npi_int]
    if row.empty:
        return {"kind": "hcp_lookup", "found": False,
                "error": f"No HCP found with NPI {npi_int}."}

    return {"kind": "hcp_lookup", "found": True, **row.iloc[0].to_dict()}


# ---------------------------------------------------------------------
# Same exact lookup as get_row_by_npi (all the fields are already
# there). This name is kept as an alias only because ask_a_question.py
# still calls it under this name today - it will be removed once that
# file is cleaned up in the next pass, so there's only one real lookup
# function instead of two names for the same thing.
# ---------------------------------------------------------------------
def hcp_scripts(npi):
    return get_row_by_npi(npi)


# ---------------------------------------------------------------------
# Finds the ONE HCP in a given state with the highest monthly Rx
# volume (i.e. writes the most prescriptions right now). Zero-writers
# (HCPs who write nothing) are excluded, since "top prescriber" should
# never be able to return someone who prescribes nothing.
# ---------------------------------------------------------------------
def top_prescriber(state):
    clean_state, err = _clean_state(state)
    if err:
        return {"kind": "top_prescriber", "found": False, "state": state, "error": err}

    sub = df[(df["state"] == clean_state) & (~df["zero_writer"])]
    if sub.empty:
        return {"kind": "top_prescriber", "found": False, "state": state,
                "error": f"No active (non-zero) GLP-1 writers found in {state}."}

    top = sub.sort_values("rx_volume_monthly", ascending=False).iloc[0]
    return {"kind": "top_prescriber", "found": True, "state": state,
            "sort_reason": "Sorted by rx_volume_monthly (highest first) - "
                            "this is the single highest-volume active prescriber in this state.",
            **top.to_dict()}


# ---------------------------------------------------------------------
# GENERIC yes/no counter for ANY flag column (see FLAG_COLUMNS above) -
# zero_writer, targeted, sample_request_recent, or a new one added to
# the spreadsheet later. Always computes BOTH counts (how many are
# True, how many are False) - `asked_about` only controls which one
# gets highlighted by the presentation function later, so the two
# never get confused with each other.
#
# This exists because count_writers() used to do this same job as a
# one-off, hard-coded to zero_writer specifically - and a real test
# found that its presentation function could report the wrong count
# when a question specifically asked about the "false" side. Rather
# than patch that single case, this is the one shared, careful version
# every flag-counting function should build on, so the same mistake
# can't quietly happen again on a different column.
# ---------------------------------------------------------------------
def count_by_flag(state, column, true_label, false_label, asked_about=None):
    if column not in FLAG_COLUMNS:
        return {"kind": "flag_count", "found": False,
                "error": f"'{column}' isn't a recognized yes/no column. "
                         f"Flag columns available: {list(FLAG_COLUMNS)}"}

    clean_state, err = _clean_state(state)
    if err:
        return {"kind": "flag_count", "found": False, "state": state, "error": err}

    in_state = df[df["state"] == clean_state]
    is_true = in_state[column].fillna(False).astype(bool)
    total = len(in_state)
    n_true = int(is_true.sum())
    n_false = total - n_true

    return {"kind": "flag_count", "found": True, "state": state, "column": column,
            true_label: n_true, false_label: n_false, "total": total,
            "asked_about": asked_about or false_label}


# ---------------------------------------------------------------------
# Counts how many HCPs in a state are actively writing GLP-1
# prescriptions (rather than zero-writers), out of the total HCPs on
# file for that state.
#
# This is now a thin wrapper around count_by_flag() (see above) - kept
# under its own specific name and with its original "active"/"zero"
# keys because ask_a_question.py and agent_tools.py already call this
# function by this exact name and expect those exact keys back. Only
# the underlying computation moved to the shared, safer primitive;
# nothing calling count_writers() today needs to change.
#
# asked_about: "active" (default, matches old behaviour) or "zero" -
# which count the caller wants highlighted when this gets turned into
# a sentence by format_count_writers() below.
# ---------------------------------------------------------------------
def count_writers(state, asked_about="active"):
    result = count_by_flag(state, column="zero_writer",
                            true_label="zero", false_label="active",
                            asked_about=asked_about)
    if not result["found"]:
        return {"kind": "writer_count", "found": False, "state": state,
                "error": result["error"]}
    return {"kind": "writer_count", "found": True, "state": result["state"],
            "active": result["active"], "zero": result["zero"],
            "total": result["total"], "asked_about": result["asked_about"]}


# ---------------------------------------------------------------------
# Returns the top N HCPs nationally, ranked by propensity score,
# optionally narrowed to one tier (High/Medium/Watch). n=None means
# "no limit, give me everyone that matches."
# ---------------------------------------------------------------------
def top_n_by_propensity(n=None, tier="High"):
    err = _check_categorical("tier", tier)
    if err:
        return {"kind": "ranked_list", "found": False, "error": err}

    sub = df[df["tier"] == tier] if tier else df
    sub = sub.sort_values("propensity_score", ascending=False)
    limited = sub if n is None else sub.head(n)
    return {"kind": "ranked_list", "found": True, "n": n, "tier": tier,
            "count": len(sub),
            "sort_reason": "Sorted by propensity_score (highest first).",
            "results": limited.to_dict("records")}


# ---------------------------------------------------------------------
# Counts how many HCPs of a given tier (High by default) are in each
# state, and returns the states with the most, biggest first.
# ---------------------------------------------------------------------
def states_by_tier(n=5, tier="High"):
    err = _check_categorical("tier", tier)
    if err:
        return {"kind": "state_summary", "found": False, "error": err}

    counts = df[df["tier"] == tier]["state"].value_counts().head(n)
    return {"kind": "state_summary", "found": True, "tier": tier,
            "sort_reason": f"Sorted by count of {tier}-tier HCPs per state (highest first).",
            "results": [{"state": state, "count": int(c)} for state, c in counts.items()]}


# Old name kept as an alias so nothing calling this today breaks -
# it always meant tier="High" specifically.
def states_by_high_tier(n=5):
    return states_by_tier(n=n, tier="High")


# ---------------------------------------------------------------------
# THE flexible, do-almost-anything filter. Every parameter is
# optional - only the ones actually passed in get applied.
#
#   - Word-based columns (state, region, specialty, tier,
#     dominant_competitor, formulary_tier, dominant_payer,
#     ax_relationship): pass the real value directly.
#   - targeted / zero_writer / recent_sample_request: pass True/False
#     or 1/0.
#   - decile: pass a whole number 1-10.
#   - 0-to-1 SCORE columns (see SCORE_COLUMNS): use `levels`, e.g.
#     levels={"switching_score": "high", "pa_burden": "low"} - this
#     applies the file's one standard cutoff (0.75 / 0.25).
#   - ANY numeric column, scored or not: use `extra_filters` for an
#     exact number, e.g. extra_filters={"days_since_contact": (30, None)}
#     means "30 or more days since contact."
#   - ANY word-based column not covered by a named parameter above:
#     use `extra_categorical`, e.g. extra_categorical={"region": "South"}.
#     (This exists so a brand new text column added to the spreadsheet
#     later is still filterable immediately, with no code change here.)
#   - sort_by / ascending: works on ANY real column name, in either
#     direction - "highest first" (ascending=False, the default) or
#     "lowest first" (ascending=True).
#
# top=None means "no limit, return every match" - only cap the results
# if a specific number was actually asked for.
# ---------------------------------------------------------------------
def filter_hcps(state=None, region=None, specialty=None, tier=None, targeted=None,
                dominant_competitor=None, formulary_tier=None, recent_sample_request=None,
                dominant_payer=None, ax_relationship=None, zero_writer=None, decile=None,
                levels=None, extra_filters=None, extra_categorical=None,
                sort_by="propensity_score", ascending=False, top=None):

    # ---- Check every value is real before touching the data ----
    clean_state, state_err = _clean_state(state)
    if state_err:
        return {"kind": "filtered_list", "found": False, "error": state_err}

    for column, value in (("region", region), ("tier", tier),
                           ("dominant_competitor", dominant_competitor),
                           ("formulary_tier", formulary_tier), ("dominant_payer", dominant_payer),
                           ("ax_relationship", ax_relationship), ("specialty", specialty)):
        err = _check_categorical(column, value)
        if err:
            return {"kind": "filtered_list", "found": False, "error": err}

    if decile is not None and decile not in range(1, 11):
        return {"kind": "filtered_list", "found": False,
                "error": f"decile must be a whole number from 1 to 10, got {decile}."}

    if levels:
        for column, level in levels.items():
            err = _check_level(column, level)
            if err:
                return {"kind": "filtered_list", "found": False, "error": err}

    if extra_categorical:
        for column, value in extra_categorical.items():
            if column not in df.columns:
                return {"kind": "filtered_list", "found": False,
                        "error": f"'{column}' is not a real column in this data."}

    if sort_by not in df.columns:
        return {"kind": "filtered_list", "found": False,
                "error": f"'{sort_by}' is not a real column to sort by. "
                         f"Available columns: {df.columns.tolist()}"}

    # ---- Apply whichever filters were actually given ----
    sub = df.copy()
    if clean_state:
        sub = sub[sub["state"] == clean_state]
    if region:
        sub = sub[sub["region"] == region]
    if specialty:
        sub = sub[sub["specialty"] == specialty]
    if tier:
        sub = sub[sub["tier"] == tier]
    if targeted is not None:
        sub = sub[sub["targeted"] == targeted]
    if dominant_competitor:
        sub = sub[sub["dominant_competitor"] == dominant_competitor]
    if formulary_tier:
        sub = sub[sub["formulary_tier"] == formulary_tier]
    if recent_sample_request is not None:
        sub = sub[sub["sample_request_recent"] == recent_sample_request]
    if dominant_payer:
        sub = sub[sub["dominant_payer"] == dominant_payer]
    if ax_relationship:
        if ax_relationship == AX_RELATIONSHIP_ZERO_LABEL:
            sub = sub[sub["ax_relationship"].isna()]
        else:
            sub = sub[sub["ax_relationship"] == ax_relationship]
    if zero_writer is not None:
        sub = sub[sub["zero_writer"] == zero_writer]
    if decile is not None:
        sub = sub[sub["decile"] == decile]

    if levels:
        for column, level in levels.items():
            min_v, max_v = _level_to_range(level)
            if min_v is not None:
                sub = sub[sub[column] >= min_v]
            if max_v is not None:
                sub = sub[sub[column] <= max_v]

    if extra_filters:
        for col, (min_v, max_v) in extra_filters.items():
            if col not in sub.columns:
                continue  # not a real column - ignored rather than crashing
            if min_v is not None:
                sub = sub[sub[col] >= min_v]
            if max_v is not None:
                sub = sub[sub[col] <= max_v]

    if extra_categorical:
        for column, value in extra_categorical.items():
            if isinstance(value, (list, tuple, set)):
                sub = sub[sub[column].isin(value)]
            else:
                sub = sub[sub[column] == value]

    sub = sub.sort_values(sort_by, ascending=ascending)
    limited = sub if top is None else sub.head(top)

    return {
        "kind": "filtered_list",
        "found": True,
        "count": len(sub),
        "filters": {"state": state, "region": region, "specialty": specialty, "tier": tier,
                    "targeted": targeted, "dominant_competitor": dominant_competitor,
                    "formulary_tier": formulary_tier,
                    "recent_sample_request": recent_sample_request,
                    "dominant_payer": dominant_payer, "ax_relationship": ax_relationship,
                    "zero_writer": zero_writer, "decile": decile,
                    "levels": levels, "extra_filters": extra_filters,
                    "extra_categorical": extra_categorical,
                    "sort_by": sort_by, "ascending": ascending},
        "results": limited.to_dict("records"),
    }


# =====================================================================
# PRESENTATION FUNCTIONS
# -----------------------------------------------------------------
# These turn the plain dictionaries above into readable sentences.
# Kept in their own section, separate from the data functions, so
# nothing that reads the raw dictionaries (a future AI agent, an eval
# script, a different display) is ever forced to write a sentence
# apart to get the numbers back out.
# =====================================================================

# ---------------------------------------------------------------------
# Turns a top_prescriber() result into one readable sentence.
# ---------------------------------------------------------------------
def format_top_prescriber(data):
    if not data["found"]:
        return data["error"]
    return (
        f"{data['sort_reason']}\n"
        f"Top GLP-1 prescriber in {data['state'].title()}: NPI {data['npi']} "
        f"({data['specialty']}) - {data['rx_volume_monthly']} scripts/month, "
        f"propensity {data['propensity_score']:.2f}, tier {data['tier']}.\n"
        f"Access: dominant payer {data['dominant_payer']}, formulary status "
        f"{data['formulary_tier']}, PA burden {data['pa_burden']:.2f}.\n"
        f"Competitive: dominant competitor {data['dominant_competitor']}, "
        f"switching score {data['switching_score']:.2f}.\n"
        f"Engagement: targeted={bool(data['targeted'])}, "
        f"AX relationship {_format_cell(data['ax_relationship'])}, "
        f"days since last rep contact {data['days_since_contact']:.0f}."
    )


# ---------------------------------------------------------------------
# GENERIC presentation for any count_by_flag() result - turns it into
# one readable sentence, always leading with whichever side
# `asked_about` says the caller actually wanted, never defaulting
# silently to the other one.
# ---------------------------------------------------------------------
def format_count_by_flag(data):
    if not data["found"]:
        return data["error"]
    highlight = data.get("asked_about")
    count = data.get(highlight)
    return (f"{count} {highlight} HCPs in {data['state'].title()} "
            f"(out of {data['total']} HCPs), by {data['column']}.")


# ---------------------------------------------------------------------
# Turns a count_writers() result into one readable sentence. Kept
# separate from format_count_by_flag() (rather than replaced by it)
# because this one has specific, friendlier wording for this
# particular column ("active GLP-1 writers", "zero-writer HCPs... i.e.
# HCPs on file who currently write no GLP-1 prescriptions") that a
# fully generic version can't know to write.
# ---------------------------------------------------------------------
def format_count_writers(data):
    if not data["found"]:
        return data["error"]
    if data.get("asked_about") == "zero":
        return (f"{data['zero']} zero-writer HCPs in {data['state'].title()} "
                f"(out of {data['total']} HCPs) - i.e. HCPs on file who currently "
                f"write no GLP-1 prescriptions.")
    return (f"{data['active']} active GLP-1 writers in {data['state'].title()} "
            f"(out of {data['total']} HCPs).")


# ---------------------------------------------------------------------
# Turns a top_n_by_propensity() result into a readable numbered list.
# ---------------------------------------------------------------------
def format_top_n_by_propensity(data):
    if not data["found"]:
        return data["error"]
    count_label = f"Top {data['n']}" if data["n"] is not None else "All"
    lines = []
    if data.get("sort_reason"):
        lines.append(data["sort_reason"])
    lines.append(f"{count_label} {data['tier'] or ''}-tier HCPs by propensity:")
    for r in data["results"]:
        lines.append(f"  NPI {r['npi']} ({r['specialty']}, {r['state'].title()}) "
                      f"- propensity {r['propensity_score']:.2f}, rank {r['propensity_rank']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Turns a states_by_tier() result into a readable list.
# ---------------------------------------------------------------------
def format_states_by_tier(data):
    if not data["found"]:
        return data["error"]
    lines = [data["sort_reason"], f"States with the most {data['tier']}-tier HCPs:"]
    for r in data["results"]:
        lines.append(f"  {r['state'].title()}: {r['count']}")
    return "\n".join(lines)


# Old name kept as an alias for the same reason as states_by_high_tier above.
def format_states_by_high_tier(data):
    return format_states_by_tier(data)


# ---------------------------------------------------------------------
# Turns any single cell value into safe display text - a missing/NaN
# value shows as "n/a" (not a raw "nan", which reads like a software
# bug to a rep), a float gets rounded to 2 decimals, everything else
# shows as-is. Used everywhere a single row's values get displayed.
# ---------------------------------------------------------------------
def _format_cell(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:  # a NaN never equals itself - the standard no-import way to check
            return "n/a"
        return f"{value:.2f}"
    return str(value)


# The always-shown columns on every row of a filtered list, regardless
# of what was asked. Never duplicated by the "extra" logic below.
_BASE_ROW_COLUMNS = ("propensity_score", "switching_score", "rx_volume_monthly")

# Maps a filter_hcps() PARAMETER name to the real COLUMN it filters on
# - a few of these differ (e.g. the parameter is "recent_sample_request"
# but the real column is "sample_request_recent").
_FILTER_PARAM_TO_COLUMN = {
    "tier": "tier",
    "targeted": "targeted",
    "dominant_competitor": "dominant_competitor",
    "formulary_tier": "formulary_tier",
    "recent_sample_request": "sample_request_recent",
    "dominant_payer": "dominant_payer",
    "ax_relationship": "ax_relationship",
    "zero_writer": "zero_writer",
    "decile": "decile",
    "region": "region",
}


# ---------------------------------------------------------------------
# Works out which columns to show on each row BEYOND the fixed base
# six (npi/state/specialty/propensity/switching/rx): every column the
# question actually filtered on - a named parameter, a level, or a
# generic numeric filter - plus whatever the list is sorted by, if
# that's something different again. This is what makes sure a
# question that filtered on e.g. sample_request_recent actually SHOWS
# that column's value on every row, not just mentions it in the
# header sentence - found missing in a real test.
# ---------------------------------------------------------------------
def _extra_columns_for(filters, sort_by):
    columns = []

    def _add(column):
        if column and column not in columns and column not in _BASE_ROW_COLUMNS:
            columns.append(column)

    for param, column in _FILTER_PARAM_TO_COLUMN.items():
        if filters.get(param) is not None:
            _add(column)
    for column in (filters.get("levels") or {}):
        _add(column)
    for column in (filters.get("extra_filters") or {}):
        _add(column)
    _add(sort_by)
    return columns


# ---------------------------------------------------------------------
# Turns a filter_hcps() result into a readable header + numbered list.
#
# FIX: this now ALWAYS states which column it actually sorted by and
# in which direction, straight from the real sort_by/ascending values
# this function was given - not a separate claim written by whoever
# called it. A real test found a case where the CALLER picked the
# wrong column to sort by (a "rank by switching score" question was
# sorted by rx_volume_monthly instead) while ALSO writing a sentence
# claiming it sorted by "the specific measure this question was
# about." That mismatch happened upstream, not in this function - but
# this function silently going along with whatever sort_by it was
# handed, with no visible record of what it actually did, is what let
# the wrong claim go unnoticed. Now it can't: this line is generated
# directly from data['filters']['sort_by'], so a wrong column choice
# is immediately visible in the output rather than hidden behind a
# possibly-incorrect sentence from elsewhere.
# ---------------------------------------------------------------------
def format_filter_hcps(data):
    if not data["found"]:
        return data["error"]
    f = data["filters"]

    sort_direction = "lowest first" if f.get("ascending") else "highest first"
    sort_line = f"Sorted by {f.get('sort_by', 'propensity_score')} ({sort_direction})."

    header = (f"{data['count']} HCPs match (state={f['state']}, region={f['region']}, "
              f"specialty={f['specialty']}, tier={f['tier']}, targeted={f['targeted']}, "
              f"competitor={f['dominant_competitor']}, formulary_tier={f['formulary_tier']}, "
              f"recent_sample_request={f['recent_sample_request']}"
              f"{', levels=' + str(f['levels']) if f.get('levels') else ''}"
              f"{', extra=' + str(f['extra_filters']) if f.get('extra_filters') else ''}"
              f"{', extra_categorical=' + str(f['extra_categorical']) if f.get('extra_categorical') else ''}"
              f"). Top {len(data['results'])}:")

    lines = [sort_line]
    levels_desc = describe_levels(f.get("levels"))
    if levels_desc:
        lines.append(levels_desc)
    lines.append(header)

    extra_columns = _extra_columns_for(f, f.get("sort_by", "propensity_score"))
    for r in data["results"]:
        line = (f"  NPI {r['npi']} ({r['specialty']}, {r['state'].title()}) "
                f"- propensity {_format_cell(r['propensity_score'])}, "
                f"switching {_format_cell(r['switching_score'])}, "
                f"rx_volume_monthly={_format_cell(r['rx_volume_monthly'])}")
        for col in extra_columns:
            line += f", {col}={_format_cell(r.get(col))}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Turns a get_row_by_npi() / hcp_scripts() result into one readable
# sentence, focused on script volume.
# ---------------------------------------------------------------------
def format_hcp_scripts(data):
    if not data["found"]:
        return data["error"]
    return (f"NPI {data['npi']} ({data['specialty']}, {data['state'].title()}): "
            f"{data['rx_volume_monthly']} GLP-1 scripts last month "
            f"({data['nrx_monthly']} new starts). Tier {data['tier']}, "
            f"propensity {data['propensity_score']:.2f}.")


# =====================================================================
# Quick manual test - only runs if you execute this file directly
# (python query_spreadsheet.py). Not used when this file is imported
# by another script.
# =====================================================================
if __name__ == "__main__":
    print(f"Loaded master data from: {_DATA_PATH}")
    print("Shape:", df.shape, "(rows, columns)")
    print("\nCategorical columns covered automatically:", CATEGORICAL_COLUMNS)
    print("\n0-1 SCORE columns (support high/moderate/low):", SCORE_COLUMNS)
    print("\nOther numeric columns (exact-number filtering only):", OTHER_NUMERIC_COLUMNS)
    print("\nFlag (yes/no) columns:", FLAG_COLUMNS)
    print()

    print("-- region filter (South) --")
    r = filter_hcps(region="South", top=3)
    print(format_filter_hcps(r), "\n")

    print("-- levels: high switching score AND low PA burden --")
    r = filter_hcps(levels={"switching_score": "high", "pa_burden": "low"}, top=5)
    print(format_filter_hcps(r), "\n")

    print("-- sort ascending by years_practice (least experienced first) --")
    r = filter_hcps(sort_by="years_practice", ascending=True, top=5)
    for row in r["results"]:
        print(f"  NPI {row['npi']} - {row['years_practice']} years in practice")
    print()

    print("-- count_writers: active (default) vs zero, same state, side by side --")
    print(" ", format_count_writers(count_writers("Texas")))
    print(" ", format_count_writers(count_writers("Texas", asked_about="zero")))
    print()

    print("-- deliberately-bad requests, to show validation catching them --")
    print(" ", filter_hcps(region="Southeast")["error"])          # not a real region
    print(" ", filter_hcps(levels={"years_practice": "high"})["error"])  # not a score column
    print(" ", count_by_flag("Texas", column="not_a_real_column",
                              true_label="x", false_label="y")["error"])  # not a real flag column