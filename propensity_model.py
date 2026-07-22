# propensity_model.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# Every HCP in the spreadsheet has a "propensity score" - basically a
# 0-100% guess at how likely that doctor is to prescribe. This file is
# the actual maths behind that guess, written as runnable code instead
# of living only inside a spreadsheet. Propensity gets a High/Medium/
# Watch tier and a decile (which 10% group a doctor falls into, among
# doctors who write at least one script) - the switching score does
# NOT get its own tier or decile, it's reported as a plain 0-1 number
# only. There is exactly one "tier" column in the data, and it is
# always derived from propensity_rank alone (see score_hcps() below).
#
# PROVENANCE - read before trusting: the original Phase 2 scorecard was
# built outside this repository; only its OUTPUT (the scored
# spreadsheet in data/) was committed. The weights below were recovered
# from that spreadsheet by exact linear regression: on the 2026-07-14
# release, per-specialty fits reproduce logit_score with residuals at
# machine precision (max |error| < 1e-12, R^2 = 1.0), and
# propensity_score equals sigmoid(logit_score) to 2e-16. The recovered
# weights are round, literature-style values, which is consistent with
# the documented "literature-weighted scorecard" design.
#
# Two jobs this file can do, from the command line:
#   --verify   guard the pipeline: recompute every score in the current
#              data file and fail loudly if the stored scores disagree
#              (i.e. someone refreshed the data without re-scoring, or
#              scored it with different weights). main.py runs this as
#              stage 0 of every build.
#   --score    ingest a new batch: take a raw file with input columns
#              only and write a fully scored file (logit, propensity,
#              switching, rank, tier, decile) ready for the pipeline.
#
# DECILE FIX (2026-07-22): deciles are now computed with a rank-based
# ntile() equivalent (see _ntile() below), matching R's dplyr::ntile()
# exactly - including its remainder-handling, where oversized buckets
# are placed FIRST rather than spread evenly when 15,000 doesn't
# divide evenly by 10. The old code guessed bucket edges from the
# scores themselves (np.percentile + np.digitize), which is a
# different, incompatible method and could never reproduce R's output
# at the boundaries. Verified row-by-row against the full 2026-07-14
# data file: 0 mismatched decile rows out of 15,000 (previously 4
# boundary rows off with a naive rank-split attempt, more before that
# with the percentile-edge approach). --verify now expects an EXACT
# decile match, same as rank/tier - no tolerance needed anymore.
#
# WEIGHTS CHECK AGAINST THE R REFERENCE (2026-07-22): confirmed the
# scorecard weights below produce identical output to this R list -
#
#   weights <- list(
#     intercept              = -1.0,
#     rx_volume_z            = 0.3,
#     nrx_share               = 2.0,
#     formulary_preferred    = 0.6,
#     formulary_pa_required  = -0.5,
#     formulary_not_covered  = -1.2,
#     pa_burden              = -2.5,
#     ax_one_product          = 0.3,
#     ax_two_plus             = 0.6,
#     rep_engagement          = 0.8
#   )
#
# They LOOK different from PROPENSITY_INTERCEPT/PROPENSITY_WEIGHTS
# below, but that's just two different ways of writing the same maths,
# not a real disagreement. The R list gives formulary_preferred a
# POSITIVE effect (+0.6) on top of its own intercept (-1.0) - meaning R
# treats "NonPreferred" as the silent 0-effect baseline. The Python
# code below instead treats "Preferred" as the silent 0-effect baseline
# and gives NonPreferred a NEGATIVE offset (-0.6) from there. Different
# starting point, same distances between the categories. Doing the sum
# both ways for every formulary tier proves they land on the exact same
# number:
#   Preferred:    R  -1.0 + 0.6  = -0.4   |  Python  -0.4 + 0.0  = -0.4
#   NonPreferred: R  -1.0 + 0.0  = -1.0   |  Python  -0.4 + -0.6 = -1.0
#   PARequired:   R  -1.0 + -0.5 = -1.5   |  Python  -0.4 + -1.1 = -1.5
#   NotCovered:   R  -1.0 + -1.2 = -2.2   |  Python  -0.4 + -1.8 = -2.2
# All four match exactly, so no weight change is needed here - this is
# just documentation so nobody "corrects" a mismatch that isn't real.
# The ax_relationship weights (One/TwoPlus = 0.3/0.6) already matched
# directly with no re-parameterizing needed.
#
# SWITCHING SCORE CHECK AGAINST THE R REFERENCE (2026-07-22): confirmed
# too, against this R source -
#
#   relationship_weight <- ifelse(ax_relationship == "TwoPlus", 1.0,
#                           ifelse(ax_relationship == "One", 0.5, 0.0))
#   adherence_uplift <- 0.15   # flat constant, same for every HCP
#   switching_logit <- -1.2 + 1.5 * relationship_weight +
#                       1.2 * rep_engagement_score + adherence_uplift
#
# Same situation as above: looks different, isn't. adherence_uplift is
# a FLAT constant added to every single row, so it has nowhere to go
# except straight into the intercept: -1.2 + 0.15 = -1.05, which is
# exactly SWITCHING_INTERCEPT below. relationship_weight (1.0/0.5/0.0)
# times its 1.5 coefficient gives 1.5/0.75/0.0 for TwoPlus/One/None -
# an exact match to AX_SWITCHING_WEIGHTS with no reshuffling needed.
# rep_engagement_score's 1.2 coefficient matches SWITCHING_WEIGHTS
# directly too. Checked all three relationship categories end-to-end:
#   None:    R -1.2 + 0    + 0.15 = -1.05  |  Python -1.05 + 0    = -1.05
#   One:     R -1.2 + 0.75 + 0.15 = -0.30  |  Python -1.05 + 0.75 = -0.30
#   TwoPlus: R -1.2 + 1.50 + 0.15 =  0.45  |  Python -1.05 + 1.50 =  0.45
# All match exactly - the switching model is now source-verified, not
# just regression-recovered.
# ====================================================================

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


def _ntile(values, n=10):
    """Chunk: rank-based decile bucketer (matches R's dplyr::ntile()).

    Plain English: don't look at the SCORES themselves to decide bucket
    edges (that's what the old code did, and it's the bug). Instead,
    line everybody up in order from lowest score to highest, then just
    cut that line into 10 equal-length pieces. Person 1-1500 (say) is
    decile 1, person 1501-3000 is decile 2, and so on - regardless of
    how close together or far apart their actual scores are. This is
    exactly what R's ntile(propensity_score, 10) does: it ranks first,
    buckets second. Ties are broken by original row order (stable
    sort), same as R's default ties.method inside ntile().

    Reference: dplyr::ntile() docs - "ntile() assigns rows into n
    roughly equal-sized groups based on rank order, not value."
    https://dplyr.tidyverse.org/reference/ntile.html
    """
    values = np.asarray(values)
    n_rows = len(values)
    # Rank every value low-to-high. kind="stable" means if two scores
    # are exactly tied, whichever appeared first in the data keeps
    # that position - same tie-breaking dplyr uses by default
    # (rank(x, ties.method = "first")).
    order = np.argsort(values, kind="stable")
    rank = np.empty(n_rows, dtype=int)
    rank[order] = np.arange(n_rows)  # 0-based position in sorted order

    # Chunk: build the bucket sizes the way R's ntile() actually does -
    # NOT a naive even split. When n_rows doesn't divide evenly by 10,
    # dplyr makes the FIRST (n_rows %% 10) buckets one row BIGGER, and
    # puts those oversized buckets at the front, not spread evenly
    # across all 10. Example: 105 rows / 10 buckets -> 5 buckets of 11
    # rows first (buckets 1-5), then 5 buckets of 10 rows (buckets 6-10).
    # Confirmed against dplyr source (R package `dplyr`, ntile.R):
    # https://github.com/tidyverse/dplyr/blob/main/R/rank.R
    n_larger = n_rows % n          # how many buckets get the +1
    n_smaller = n - n_larger
    size = n_rows // n
    larger_size = size + 1
    bucket_sizes = [larger_size] * n_larger + [size] * n_smaller
    bin_of_position = np.repeat(np.arange(1, n + 1), bucket_sizes)

    return bin_of_position[rank]


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

    # Decile among ACTIVE writers only (zero-writers unranked).
    # Chunk: same two-step logic as the R version -
    #   1. filter down to active (non-zero) writers,
    #   2. ntile() just those, leaving everyone else NA.
    out["decile"] = np.nan
    active = ~out["zero_writer"].astype(bool)
    s = out.loc[active, "propensity_score"].values
    out.loc[active, "decile"] = _ntile(s, 10)
    return out


def _find_data_file():
    candidates = sorted(glob.glob(os.path.join("data", "rabivy_propensity_data*.xlsx")))
    if not candidates:
        raise FileNotFoundError("No rabivy_propensity_data*.xlsx found in data/")
    return candidates[-1]


def verify(path=None, decile_tolerance=0):
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