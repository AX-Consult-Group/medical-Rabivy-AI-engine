# structured.py
# -------------------------------------------------------------------
# The STRUCTURED query engine. Answers ranking / counting / filtering
# questions by querying the master spreadsheet directly.
# Single source of truth: data/propensity_master.xlsx
# -------------------------------------------------------------------

import pandas as pd

# Load the master file (single source of truth for all HCP numbers).
df = pd.read_excel("data/rabivy_propensity_data - 2026.07.02.xlsx")

print("Loaded master data:", df.shape, "(rows, columns)")
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 3 rows (a few key columns):")
print(df[["npi", "specialty", "state", "propensity_score", "tier", "rx_volume_monthly"]].head(3))



def top_prescriber(state):
    """Highest-volume GLP-1 prescriber in a state."""
    sub = df[df["state"] == state.lower().strip()]
    if sub.empty:
        return f"No HCPs found in {state}."
    top = sub.sort_values("rx_volume_monthly", ascending=False).iloc[0]
    return (f"Top GLP-1 prescriber in {state.title()}: NPI {top['npi']} "
            f"({top['specialty']}) - {top['rx_volume_monthly']} scripts/month, "
            f"propensity {top['propensity_score']:.2f}, tier {top['tier']}.")
 
 
def count_writers(state):
    """How many ACTIVE GLP-1 writers (not zero-writers) are in a state."""
    in_state = df[df["state"] == state.lower().strip()]
    active = in_state[in_state["zero_writer"] == False]
    return f"{len(active)} active GLP-1 writers in {state.title()} (out of {len(in_state)} HCPs)."
 
 
def top_n_by_propensity(n=10, tier="High"):
    """Top N HCPs nationally by propensity score, optionally within a tier."""
    sub = df[df["tier"] == tier] if tier else df
    top = sub.sort_values("propensity_score", ascending=False).head(n)
    lines = [f"Top {n} {tier or ''}-tier HCPs by propensity:"]
    for _, r in top.iterrows():
        lines.append(f"  NPI {r['npi']} ({r['specialty']}, {r['state'].title()}) "
                     f"- propensity {r['propensity_score']:.2f}, rank {r['propensity_rank']}")
    return "\n".join(lines)
 
 
def states_by_high_tier(n=5):
    """Which states have the most High-tier HCPs."""
    counts = df[df["tier"] == "High"]["state"].value_counts().head(n)
    lines = ["States with the most High-tier HCPs:"]
    for state, c in counts.items():
        lines.append(f"  {state.title()}: {c}")
    return "\n".join(lines)
 
 
def filter_hcps(state=None, specialty=None, tier=None, targeted=None,
                dominant_competitor=None, min_switching=None, sort_by="propensity_score", top=10):
    """Flexible multi-field filter over the HCP table (covers most targeting questions)."""
    sub = df.copy()
    if state:               sub = sub[sub["state"] == state.lower().strip()]
    if specialty:           sub = sub[sub["specialty"] == specialty]
    if tier:                sub = sub[sub["tier"] == tier]
    if targeted is not None: sub = sub[sub["targeted"] == targeted]
    if dominant_competitor: sub = sub[sub["dominant_competitor"] == dominant_competitor]
    if min_switching is not None: sub = sub[sub["switching_score"] >= min_switching]
    sub = sub.sort_values(sort_by, ascending=False)
    header = (f"{len(sub)} HCPs match (state={state}, specialty={specialty}, tier={tier}, "
              f"targeted={targeted}, competitor={dominant_competitor}). Top {min(top, len(sub))}:")
    lines = [header]
    for _, r in sub.head(top).iterrows():
        lines.append(f"  NPI {r['npi']} ({r['specialty']}, {r['state'].title()}) "
                     f"- propensity {r['propensity_score']:.2f}, switching {r['switching_score']:.2f}, "
                     f"targeted={r['targeted']}")
    return "\n".join(lines)
 
 
def hcp_scripts(npi):
    """How many GLP-1 scripts a specific NPI wrote last month."""
    row = df[df["npi"] == int(npi)]
    if row.empty:
        return f"No HCP found with NPI {npi}."
    r = row.iloc[0]
    return (f"NPI {npi} ({r['specialty']}, {r['state'].title()}): "
            f"{r['rx_volume_monthly']} GLP-1 scripts last month "
            f"({r['nrx_monthly']} new starts). Tier {r['tier']}, propensity {r['propensity_score']:.2f}.")
 
 
 
if __name__ == "__main__":
    print("Loaded master data:", df.shape, "\n")
    print(top_prescriber("New York"), "\n")
    print(count_writers("Texas"), "\n")
    print(top_n_by_propensity(5), "\n")
    print(states_by_high_tier(), "\n")
    print(filter_hcps(state="Florida", specialty="Endocrinology", tier="High", targeted=0, top=5), "\n")
    print(filter_hcps(state="California", dominant_competitor="Novo Nordisk", min_switching=0.6,
                      sort_by="switching_score", top=5))