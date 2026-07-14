# structured.py
# -------------------------------------------------------------------
# The STRUCTURED query engine. Answers ranking / counting / filtering
# questions by querying the master spreadsheet directly.
# Single source of truth: the most recent data/rabivy_propensity_data*.xlsx
#
# Design: every *_query function below returns plain data (a dict, or
# a list of dicts) - never a formatted sentence. Formatting lives in
# the format_* functions instead. This split matters for two things
# downstream: (1) the multi-source synthesis stage (Q20) needs to
# combine this with RAG/narrative content in one response, which is
# much harder if this layer already locked its output into prose; and
# (2) an evaluation harness can check e.g. result["npi"] == expected
# directly, instead of regexing a sentence apart.
# -------------------------------------------------------------------

import glob
import os
import pandas as pd

DATA_DIR = "data"
FILE_PATTERN = "rabivy_propensity_data*.xlsx"


def _find_data_file():
    """Pick the most recent matching file by name (dates are embedded
    as YYYY.MM.DD, which sorts correctly lexicographically) rather than
    hardcoding one dated filename that breaks the moment the file is
    refreshed."""
    candidates = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)))
    if not candidates:
        raise FileNotFoundError(
            f"No file matching '{FILE_PATTERN}' found in '{DATA_DIR}/'. "
            f"Expected the master propensity spreadsheet there."
        )
    return candidates[-1]


_DATA_PATH = _find_data_file()
df = pd.read_excel(_DATA_PATH)

# --- Load-time data-integrity guards -------------------------------
# NPI: cast to a consistent int type once, here, rather than at every
# call site. Excel can read a column with any blank cells as float,
# which silently breaks `== int(npi)` comparisons later.
_bad_npi = df["npi"].isna().sum()
df["npi"] = pd.to_numeric(df["npi"], errors="coerce")
if df["npi"].isna().any():
    _bad_npi = df["npi"].isna().sum()
    print(f"! WARNING: {_bad_npi} row(s) have a non-numeric/missing NPI - "
          f"these rows won't be reachable by NPI lookup.")
df["npi"] = df["npi"].astype("Int64")  # nullable int, survives any remaining NaNs

# State: every lookup here assumes the state column is consistently
# lowercase (filters on .lower(), displays via .title()). Fail loudly
# if that assumption is ever violated, instead of silently returning
# empty results.
_non_lower = df["state"].dropna().map(lambda s: s != s.lower())
assert not _non_lower.any(), (
    "state column contains non-lowercase values - the lower()/title() "
    "assumption in this script no longer holds. Check the master data "
    "or update this script's state-handling logic."
)


# =====================================================================
# DATA FUNCTIONS - return dicts / lists of dicts, never strings.
# =====================================================================

def get_row_by_npi(npi):
    """Look up a single HCP row by NPI. This is the join point back to
    the RAG side: chunks_tagged.json cards use the same npi value
    (card_{npi}), so this guarantees a matching format for stitching
    structured + narrative content together (Q11/Q14/Q15/Q20)."""
    try:
        npi_int = int(npi)
    except (ValueError, TypeError):
        return {"found": False, "error": f"'{npi}' is not a valid NPI (not numeric)."}

    row = df[df["npi"] == npi_int]
    if row.empty:
        return {"found": False, "error": f"No HCP found with NPI {npi_int}."}

    return {"found": True, **row.iloc[0].to_dict()}


def top_prescriber(state):
    """Highest-volume GLP-1 prescriber in a state. Returns a dict, or
    {"found": False, ...} if the state has no HCPs."""
    sub = df[df["state"] == state.lower().strip()]
    if sub.empty:
        return {"found": False, "state": state, "error": f"No HCPs found in {state}."}
    top = sub.sort_values("rx_volume_monthly", ascending=False).iloc[0]
    return {"found": True, "state": state, **top.to_dict()}


def count_writers(state):
    """How many ACTIVE GLP-1 writers (not zero-writers) are in a state."""
    in_state = df[df["state"] == state.lower().strip()]
    active = in_state[in_state["zero_writer"] == False]  # noqa: E712
    return {"state": state, "active": len(active), "total": len(in_state)}


def top_n_by_propensity(n=10, tier="High"):
    """Top N HCPs nationally by propensity score, optionally within a tier."""
    sub = df[df["tier"] == tier] if tier else df
    top = sub.sort_values("propensity_score", ascending=False).head(n)
    return {"n": n, "tier": tier, "results": top.to_dict("records")}


def states_by_high_tier(n=5):
    """Which states have the most High-tier HCPs."""
    counts = df[df["tier"] == "High"]["state"].value_counts().head(n)
    return {"results": [{"state": state, "count": int(c)} for state, c in counts.items()]}


def filter_hcps(state=None, specialty=None, tier=None, targeted=None,
                dominant_competitor=None, min_switching=None,
                sort_by="propensity_score", top=10):
    """Flexible multi-field filter over the HCP table (covers most
    targeting questions)."""
    if sort_by not in df.columns:
        return {"found": False,
                "error": f"'{sort_by}' is not a valid column to sort by. "
                         f"Available columns: {df.columns.tolist()}"}

    sub = df.copy()
    if state:
        sub = sub[sub["state"] == state.lower().strip()]
    if specialty:
        sub = sub[sub["specialty"] == specialty]
    if tier:
        sub = sub[sub["tier"] == tier]
    if targeted is not None:
        sub = sub[sub["targeted"] == targeted]
    if dominant_competitor:
        sub = sub[sub["dominant_competitor"] == dominant_competitor]
    if min_switching is not None:
        sub = sub[sub["switching_score"] >= min_switching]
    sub = sub.sort_values(sort_by, ascending=False)

    return {
        "found": True,
        "count": len(sub),
        "filters": {"state": state, "specialty": specialty, "tier": tier,
                    "targeted": targeted, "dominant_competitor": dominant_competitor,
                    "min_switching": min_switching},
        "results": sub.head(top).to_dict("records"),
    }


def hcp_scripts(npi):
    """How many GLP-1 scripts a specific NPI wrote last month."""
    return get_row_by_npi(npi)  # same lookup - all fields already included


# =====================================================================
# PRESENTATION LAYER - turns the dicts above into display strings.
# Kept separate so nothing upstream (router, synthesis stage, eval
# harness) is ever forced to parse prose back apart.
# =====================================================================

def format_top_prescriber(data):
    if not data["found"]:
        return data["error"]
    return (f"Top GLP-1 prescriber in {data['state'].title()}: NPI {data['npi']} "
            f"({data['specialty']}) - {data['rx_volume_monthly']} scripts/month, "
            f"propensity {data['propensity_score']:.2f}, tier {data['tier']}.")


def format_count_writers(data):
    return (f"{data['active']} active GLP-1 writers in {data['state'].title()} "
            f"(out of {data['total']} HCPs).")


def format_top_n_by_propensity(data):
    lines = [f"Top {data['n']} {data['tier'] or ''}-tier HCPs by propensity:"]
    for r in data["results"]:
        lines.append(f"  NPI {r['npi']} ({r['specialty']}, {r['state'].title()}) "
                      f"- propensity {r['propensity_score']:.2f}, rank {r['propensity_rank']}")
    return "\n".join(lines)


def format_states_by_high_tier(data):
    lines = ["States with the most High-tier HCPs:"]
    for r in data["results"]:
        lines.append(f"  {r['state'].title()}: {r['count']}")
    return "\n".join(lines)


def format_filter_hcps(data):
    if not data["found"]:
        return data["error"]
    f = data["filters"]
    header = (f"{data['count']} HCPs match (state={f['state']}, specialty={f['specialty']}, "
              f"tier={f['tier']}, targeted={f['targeted']}, competitor={f['dominant_competitor']}, "
              f"min_switching={f['min_switching']}). "
              f"Top {len(data['results'])}:")
    lines = [header]
    for r in data["results"]:
        lines.append(f"  NPI {r['npi']} ({r['specialty']}, {r['state'].title()}) "
                      f"- propensity {r['propensity_score']:.2f}, switching {r['switching_score']:.2f}, "
                      f"targeted={r['targeted']}")
    return "\n".join(lines)


def format_hcp_scripts(data):
    if not data["found"]:
        return data["error"]
    return (f"NPI {data['npi']} ({data['specialty']}, {data['state'].title()}): "
            f"{data['rx_volume_monthly']} GLP-1 scripts last month "
            f"({data['nrx_monthly']} new starts). Tier {data['tier']}, "
            f"propensity {data['propensity_score']:.2f}.")


if __name__ == "__main__":
    print(f"Loaded master data from: {_DATA_PATH}")
    print("Shape:", df.shape, "(rows, columns)")
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 3 rows (a few key columns):")
    print(df[["npi", "specialty", "state", "propensity_score", "tier", "rx_volume_monthly"]].head(3))
    print()

    print(format_top_prescriber(top_prescriber("New York")), "\n")
    print(format_count_writers(count_writers("Texas")), "\n")
    print(format_top_n_by_propensity(top_n_by_propensity(5)), "\n")
    print(format_states_by_high_tier(states_by_high_tier()), "\n")
    print(format_filter_hcps(filter_hcps(state="Florida", specialty="Endocrinology",
                                          tier="High", targeted=0, top=5)), "\n")
    print(format_filter_hcps(filter_hcps(state="Texas", dominant_competitor="Novo Nordisk",
                                          min_switching=0.6, sort_by="switching_score", top=5)), "\n")
    print(format_hcp_scripts(hcp_scripts(1000000001)))