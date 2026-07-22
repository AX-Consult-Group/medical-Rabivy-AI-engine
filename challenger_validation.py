# challenger_validation.py
# -------------------------------------------------------------------
# CHAMPION vs CHALLENGER: does the a-priori propensity scorecard miss
# anything a data-driven model would find?
#
# The a-priori scorecard (propensity_model.py) was specified from
# literature, not learned from outcomes. Once adoption outcomes exist,
# that choice becomes testable: train an empirical CHALLENGER model on
# the outcomes using ALL available inputs - including ones the
# scorecard ignores - and compare it to the scorecard CHAMPION on
# held-out data. Where the challenger wins, and WHY it wins, is a map
# of what the scorecard is missing.
#
# Because Rabivy is pre-launch (synthetic project), real outcomes don't
# exist yet - so this ships with a SIMULATED launch that serves as a
# test fixture for the harness itself. The simulation's "true" adoption
# process deliberately includes effects the scorecard omits:
#
#   PLANTED GAP 1 - payer mix: commercially-insured panels adopt more.
#       (The scorecard gives payer-mix variables zero weight.)
#   PLANTED GAP 2 - specialty effect: obesity-medicine specialists
#       over-adopt relative to their scorecard ranking.
#       (The scorecard's weights are identical across specialties.)
#   PLANTED GAP 3 - volume saturation: the very highest-volume writers
#       adopt LESS than their linear volume term predicts (entrenched
#       habits). (The scorecard's volume term is linear.)
#
# A working harness must surface all three. That is the acceptance
# test: if the report below doesn't flag payer mix, specialty and
# volume saturation, the harness is broken - not the scorecard.
# When real launch data arrives, point --outcomes at it and the same
# report becomes a genuine model-gap audit.
#
# Usage:
#   python challenger_validation.py --simulate    # write simulated outcomes
#   python challenger_validation.py               # run the comparison + report
#   python challenger_validation.py --digest      # short markdown digest (CI)
# Outputs: data/simulated_launch_outcomes.csv, output/model_gap_report.md
# -------------------------------------------------------------------

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

OUTCOMES_PATH = os.path.join("data", "simulated_launch_outcomes.csv")
REPORT_PATH = os.path.join("output", "model_gap_report.md")
SEED = 42

# Features the CHAMPION scorecard actually uses (see propensity_model.py).
# Includes the grouped one-hot base names ("formulary", "ax") that
# _base_feature() produces, so champion features can never be
# misreported as blind spots.
CHAMPION_FEATURES = {"rx_volume_z", "nrx_share", "pa_burden",
                     "rep_engagement_score", "formulary_tier", "ax_relationship",
                     "formulary", "ax"}


def _find_data_file():
    candidates = sorted(glob.glob(os.path.join("data", "rabivy_propensity_data*.xlsx")))
    if not candidates:
        raise FileNotFoundError("No rabivy_propensity_data*.xlsx in data/")
    return candidates[-1]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =====================================================================
# SIMULATED LAUNCH (test fixture with planted gaps)
# =====================================================================

def simulate_outcomes(df, seed=SEED):
    rng = np.random.default_rng(seed)
    # Effect sizes are deliberately LARGE relative to the scorecard signal
    # (logit sd ~1.14): a planted gap the harness could plausibly miss is
    # a useless acceptance test. Real launch data replaces all of this.
    true_logit = (
        0.7 * df["logit_score"]                                   # scorecard signal is real...
        + 5.0 * df["pct_commercial"] - 2.2                        # GAP 1: payer mix (sd ~0.49)
        + 1.2 * (df["specialty"] == "Obesity Medicine")           # GAP 2: specialty effect
        - 1.5 * np.maximum(df["rx_volume_z"] - 1.5, 0.0)          # GAP 3: volume saturation
        + rng.normal(0.0, 0.3, len(df))                           # unexplainable noise
    )
    adopted = (rng.random(len(df)) < _sigmoid(true_logit)).astype(int)
    out = pd.DataFrame({"npi": df["npi"], "adopted_rabivy": adopted})
    return out


# =====================================================================
# CHAMPION vs CHALLENGER
# =====================================================================

def _challenger_matrix(df):
    """All plausibly-available inputs - deliberately broader than the
    scorecard, so the challenger CAN find what the champion ignores."""
    X = pd.DataFrame(index=df.index)
    for c in ["rx_volume_z", "nrx_share", "pct_commercial", "pct_medicare",
              "pct_medicaid", "pct_oop", "pa_burden", "rep_engagement_score",
              "days_since_contact", "sample_request_recent", "targeted",
              "years_practice", "obesity_prev", "pct_novo", "pct_lilly"]:
        X[c] = df[c].astype(float)
    X["zero_writer"] = df["zero_writer"].astype(int)
    for v in df["specialty"].dropna().unique():
        X[f"specialty={v}"] = (df["specialty"] == v).astype(int)
    for v in df["formulary_tier"].dropna().unique():
        X[f"formulary={v}"] = (df["formulary_tier"] == v).astype(int)
    for v in ["One", "TwoPlus"]:
        X[f"ax={v}"] = (df["ax_relationship"] == v).astype(int)
    for v in df["dominant_payer"].dropna().unique():
        X[f"payer={v}"] = (df["dominant_payer"] == v).astype(int)
    return X


def _base_feature(col):
    """Map a one-hot column back to its base feature for reporting."""
    return col.split("=")[0] if "=" in col else col


def run_comparison(df, outcomes):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    data = df.merge(outcomes, on="npi", how="inner")
    y = data["adopted_rabivy"].values
    X = _challenger_matrix(data)

    X_tr, X_te, y_tr, y_te, champ_tr, champ_te = train_test_split(
        X, y, data["propensity_score"].values,
        test_size=0.3, random_state=SEED, stratify=y)

    champion_auc = roc_auc_score(y_te, champ_te)

    clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=300)
    clf.fit(X_tr, y_tr)
    challenger_auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

    # Which features drive the challenger? (permutation importance on holdout)
    imp = permutation_importance(clf, X_te, y_te, n_repeats=5,
                                 random_state=SEED, scoring="roc_auc")
    importances = (pd.Series(imp.importances_mean, index=X.columns)
                   .groupby(_base_feature).sum().sort_values(ascending=False))

    # Blind-spot candidates: important to the challenger, ignored by the champion.
    blind_spots = importances[[f for f in importances.index
                               if f not in CHAMPION_FEATURES]].head(6)

    # Disagreement analysis: HCPs the challenger ranks in its top 10% that
    # the champion does not - what do they look like?
    data = data.assign(challenger_p=clf.predict_proba(X)[:, 1])
    k = max(1, len(data) // 10)
    top_challenger = set(data.nlargest(k, "challenger_p")["npi"])
    top_champion = set(data.nlargest(k, "propensity_score")["npi"])
    missed = data[data["npi"].isin(top_challenger - top_champion)]
    profile = {}
    if len(missed):
        pop = data
        profile = {
            "n": len(missed),
            "pct_commercial": (missed["pct_commercial"].mean(), pop["pct_commercial"].mean()),
            "rx_volume_z": (missed["rx_volume_z"].mean(), pop["rx_volume_z"].mean()),
            "share_obesity_med": ((missed["specialty"] == "Obesity Medicine").mean(),
                                   (pop["specialty"] == "Obesity Medicine").mean()),
        }
    # Calibration by volume band: a LINEAR volume term predicts the very
    # highest-volume writers adopt the most. If observed adoption falls
    # short of the champion's prediction specifically in the top band,
    # the volume relationship is non-linear (saturation) - a gap that
    # cannot appear in the blind-spot list because volume itself IS a
    # champion feature; only its functional form is wrong.
    bands = [("volume z < 1.5 (bulk)", data["rx_volume_z"] < 1.5),
             ("volume z >= 1.5 (top writers)", data["rx_volume_z"] >= 1.5)]
    calibration = []
    for label, mask in bands:
        seg = data[mask]
        if len(seg):
            calibration.append({"band": label, "n": len(seg),
                                "champion_predicted": float(seg["propensity_score"].mean()),
                                "observed": float(seg["adopted_rabivy"].mean())})

    return {"n": len(data), "adoption_rate": float(y.mean()),
            "champion_auc": champion_auc, "challenger_auc": challenger_auc,
            "importances": importances, "blind_spots": blind_spots,
            "missed_profile": profile, "calibration": calibration}


def write_report(r, digest=False):
    lift = r["challenger_auc"] - r["champion_auc"]
    lines = []
    lines.append("# Model Gap Report - champion (a-priori scorecard) vs challenger (learned)")
    lines.append("")
    lines.append(f"- HCPs with outcomes: **{r['n']:,}** | observed adoption rate: **{r['adoption_rate']:.1%}**")
    lines.append(f"- Champion holdout AUC (scorecard propensity): **{r['champion_auc']:.3f}**")
    lines.append(f"- Challenger holdout AUC (gradient boosting, all inputs): **{r['challenger_auc']:.3f}**")
    lines.append(f"- **Lift: {lift:+.3f} AUC** - "
                 + ("the data contains predictive signal the scorecard does not use."
                    if lift > 0.01 else "no material gap detected; the scorecard is competitive."))
    lines.append("")
    lines.append("## Candidate blind spots (predictive for the challenger, unused by the scorecard)")
    lines.append("")
    lines.append("| Feature | Importance (AUC loss when shuffled) |")
    lines.append("|---|---|")
    for f, v in r["blind_spots"].items():
        lines.append(f"| {f} | {v:.4f} |")
    if not digest:
        lines.append("")
        lines.append("## Challenger feature importances (top 10, one-hots grouped)")
        lines.append("")
        lines.append("| Feature | Importance |")
        lines.append("|---|---|")
        for f, v in r["importances"].head(10).items():
            lines.append(f"| {f} | {v:.4f} |")
        p = r["missed_profile"]
        if p:
            lines.append("")
            lines.append("## Who the champion mis-ranks")
            lines.append("")
            lines.append(f"{p['n']:,} HCPs sit in the challenger's top decile but not the champion's. Versus the population they have:")
            lines.append("")
            lines.append(f"- higher commercial payer share ({p['pct_commercial'][0]:.2f} vs {p['pct_commercial'][1]:.2f})")
            lines.append(f"- volume z-score {p['rx_volume_z'][0]:.2f} vs {p['rx_volume_z'][1]:.2f}")
            lines.append(f"- more Obesity Medicine ({p['share_obesity_med'][0]:.1%} vs {p['share_obesity_med'][1]:.1%})")
        lines.append("")
        lines.append("## Champion calibration by volume band (functional-form check)")
        lines.append("")
        lines.append("| Band | n | Champion predicts | Observed adoption |")
        lines.append("|---|---|---|---|")
        for c in r["calibration"]:
            lines.append(f"| {c['band']} | {c['n']:,} | {c['champion_predicted']:.1%} | {c['observed']:.1%} |")
        lines.append("")
        lines.append("A top band whose observed adoption falls clearly below the champion's "
                     "prediction indicates the volume term saturates - the relationship is "
                     "non-linear and the scorecard's linear term overrates the biggest writers.")
        lines.append("")
        lines.append("## Recommended action")
        lines.append("")
        lines.append("Do not replace the scorecard. Refit its weights including the "
                     "flagged features (keeping the interpretable scorecard form), "
                     "re-run this report, and iterate until the lift is immaterial. "
                     "The champion stays in production until a challenger beats it "
                     "consistently on held-out outcomes.")
        lines.append("")
        lines.append("*Outcomes are simulated (pre-launch): the adoption process contains "
                     "three deliberately planted effects the scorecard omits - payer mix, "
                     "a specialty effect, and volume saturation - so this report doubles "
                     "as the harness's own acceptance test: all three must appear above. "
                     "Point `--outcomes` at real launch data to turn this into a live audit.*")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Champion-challenger propensity validation")
    ap.add_argument("--simulate", action="store_true", help="(re)generate simulated launch outcomes")
    ap.add_argument("--outcomes", default=OUTCOMES_PATH, help="outcomes csv (npi, adopted_rabivy)")
    ap.add_argument("--digest", action="store_true", help="short markdown digest to stdout (for CI)")
    args = ap.parse_args()

    df = pd.read_excel(_find_data_file())

    if args.simulate or not os.path.exists(args.outcomes):
        out = simulate_outcomes(df)
        os.makedirs("data", exist_ok=True)
        out.to_csv(OUTCOMES_PATH, index=False)
        print(f"Simulated launch outcomes for {len(out):,} HCPs -> {OUTCOMES_PATH} "
              f"(adoption rate {out['adopted_rabivy'].mean():.1%})")
        if args.simulate:
            sys.exit(0)

    outcomes = pd.read_csv(args.outcomes)
    r = run_comparison(df, outcomes)
    report = write_report(r, digest=args.digest)
    if args.digest:
        print(report)
    else:
        os.makedirs("output", exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        print(f"\nSaved {REPORT_PATH}")
