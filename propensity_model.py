# propensity_model.py
# -------------------------------------------------------------------
# Phase 2: the propensity scoring model, as executable code.
#
# PROVENANCE - read before trusting: the original Phase 2 scorecard was
# built outside this repository; only its OUTPUT (the scored
# spreadsheet in data/) was committed. The weights below were recovered
# from that spreadsheet by exact linear regression: on the 2026-07-14
# release, per-specialty fits reproduce logit_score with residuals at
# machine precision (max |error| < 1e-12, R^2 = 1.0), and
# propensity_score equals sigmoid(logit_score) to 2e-16. The recovered
# weights are round, literature-style values, which is consistent with
# the documented "literature-weighted scorecard" design. They should
# still be confirmed against the original Phase 2 implementation by its
# author before production use.
#
# Two jobs:
#   --verify   guard the pipeline: recompute every score in the current
#              data file and fail loudly if the stored scores disagree
#              (i.e. someone refreshed the data without re-scoring, or
#              scored it with different weights). main.py runs this as
#              stage 0 of every build.
#   --score    ingest a new batch: take a raw file with input columns
#              only and write a fully scored file (logit, propensity,
#              switching, rank, tier, decile) ready for the pipeline.
#
# Known approximation: the decile boundary rule. Ranks, tiers and all
# scores reproduce exactly; the decile matches on 99.9% of rows, with a
# handful of exact-boundary rows one bucket off depending on tie
# handling. --verify therefore treats scores/rank/tier strictly and
# allows a small documented tolerance on decile. To be settled against
# the original Phase 2 code.
# -------------------------------------------------------------------

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# ==== RECOVERED SCORECARD WEIGHTS (identical across specialties) =====
# Specialty conditioning enters through the data (e.g. specialty-driven
# volume and engagement distributions), not through separate weights -
# a finding of the recovery, to be confirmed by the model's author.

PROPENSITY_INTERCEPT = -0.4
PROPENSITY_WEIGHTS = {
    "rx_volume_z":            0.3,   # volume, z-scored across all HCPs
    "nrx_share":              2.0,   # share of volume that is new starts
    "pa_burden":             -2.5,   # prior-authorization burden (0-1)
    "rep_engagement_score":   0.8,   # rep engagement (0-1)
}
FORMULARY_WEIGHTS = {"Preferred": 0.0, "NonPreferred": -0.6,
                     "PARequired": -1.1, "NotCovered": -1.8}
AX_RELATIONSHIP_WEIGHTS = {None: 0.0, "One": 0.3, "TwoPlus": 0.6}

SWITCHING_INTERCEPT = -1.05
SWITCHING_WEIGHTS = {"rep_engagement_score": 1.2}
AX_SWITCHING_WEIGHTS = {None: 0.0, "One": 0.75, "TwoPlus": 1.5}

TIER_TOP_HIGH = 750      # rank 1..750            -> High
TIER_TOP_MEDIUM = 7500   # rank 751..7500         -> Medium; rest -> Watch

RAW_INPUT_COLUMNS = ["rx_volume_monthly", "nrx_share", "pa_burden",
                     "rep_engagement_score", "formulary_tier",
                     "ax_relationship", "zero_writer"]

SCORE_COLUMNS = ["rx_volume_z", "logit_score", "propensity_score",
                 "switching_logit", "switching_score",
                 "propensity_rank", "tier", "decile"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def score_hcps(df):
    """Compute every model column from raw inputs. Returns a copy with
    the SCORE_COLUMNS added/overwritten. Pure function of the batch."""
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required input columns: {missing}")
    out = df.copy()

    # Volume z-score across the whole batch (recovered: global, not
    # per-specialty; ddof=1 sample std matches the released file).
    v = out["rx_volume_monthly"].astype(float)
    out["rx_volume_z"] = (v - v.mean()) / v.std(ddof=1)

    ax = out["ax_relationship"].where(out["ax_relationship"].notna(), None)

    logit = (PROPENSITY_INTERCEPT
             + sum(w * out[c].astype(float) for c, w in PROPENSITY_WEIGHTS.items())
             + out["formulary_tier"].map(FORMULARY_WEIGHTS).astype(float)
             + ax.map(lambda a: AX_RELATIONSHIP_WEIGHTS.get(a, 0.0)).astype(float))
    out["logit_score"] = logit
    out["propensity_score"] = _sigmoid(logit)

    sw = (SWITCHING_INTERCEPT
          + sum(w * out[c].astype(float) for c, w in SWITCHING_WEIGHTS.items())
          + ax.map(lambda a: AX_SWITCHING_WEIGHTS.get(a, 0.0)).astype(float))
    out["switching_logit"] = sw
    out["switching_score"] = _sigmoid(sw)

    # National rank, 1 = highest propensity.
    out["propensity_rank"] = (out["propensity_score"]
                              .rank(ascending=False, method="first").astype(int))

    # Tier: fixed-size rank buckets.
    out["tier"] = np.where(out["propensity_rank"] <= TIER_TOP_HIGH, "High",
                   np.where(out["propensity_rank"] <= TIER_TOP_MEDIUM, "Medium", "Watch"))

    # Decile among ACTIVE writers only (zero-writers unranked). See the
    # boundary-rule note in the header.
    out["decile"] = np.nan
    active = ~out["zero_writer"].astype(bool)
    s = out.loc[active, "propensity_score"].values
    edges = np.percentile(s, np.arange(10, 100, 10), method="higher")
    out.loc[active, "decile"] = np.digitize(s, edges, right=True) + 1.0
    return out


def _find_data_file():
    candidates = sorted(glob.glob(os.path.join("data", "rabivy_propensity_data*.xlsx")))
    if not candidates:
        raise FileNotFoundError("No rabivy_propensity_data*.xlsx found in data/")
    return candidates[-1]


def verify(path=None, decile_tolerance=10):
    """Recompute all scores in a scored file and compare. Returns True
    when the stored scores are consistent with this model."""
    path = path or _find_data_file()
    df = pd.read_excel(path)
    rescored = score_hcps(df)
    ok = True
    print(f"Verifying stored scores in: {path}")

    for col in ["rx_volume_z", "logit_score", "propensity_score",
                "switching_logit", "switching_score"]:
        diff = np.max(np.abs(rescored[col] - df[col]))
        status = "OK " if diff < 1e-9 else "FAIL"
        if diff >= 1e-9:
            ok = False
        print(f"  [{status}] {col:18s} max |stored - recomputed| = {diff:.2e}")

    for col in ["propensity_rank", "tier"]:
        n_bad = int((rescored[col] != df[col]).sum())
        status = "OK " if n_bad == 0 else "FAIL"
        if n_bad:
            ok = False
        print(f"  [{status}] {col:18s} mismatched rows = {n_bad}")

    both = rescored["decile"].fillna(-1) != df["decile"].fillna(-1)
    n_bad = int(both.sum())
    status = "OK " if n_bad <= decile_tolerance else "FAIL"
    if n_bad > decile_tolerance:
        ok = False
    print(f"  [{status}] {'decile':18s} mismatched rows = {n_bad} "
          f"(tolerance {decile_tolerance}: boundary-rule ambiguity, see header)")

    print("VERIFY:", "PASS - stored scores are consistent with the scoring model."
          if ok else
          "FAIL - the data file's scores do NOT match the model. Either the "
          "file was refreshed without re-scoring (run --score on the raw "
          "batch) or it was scored with different weights (do not ingest "
          "until resolved).")
    return ok


def score_file(in_path, out_path):
    df = pd.read_excel(in_path)
    scored = score_hcps(df)
    scored.to_excel(out_path, index=False)
    print(f"Scored {len(scored)} HCPs -> {out_path}")
    print(scored["tier"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 2 propensity scoring model")
    ap.add_argument("--verify", action="store_true",
                    help="recompute scores in the current data file and compare")
    ap.add_argument("--file", default=None, help="data file to verify (default: latest in data/)")
    ap.add_argument("--score", metavar="RAW_XLSX",
                    help="score a raw batch (input columns only) and write a scored file")
    ap.add_argument("--out", default=None, help="output path for --score")
    args = ap.parse_args()

    if args.score:
        out = args.out or args.score.replace(".xlsx", " - scored.xlsx")
        score_file(args.score, out)
    else:
        sys.exit(0 if verify(args.file) else 1)
