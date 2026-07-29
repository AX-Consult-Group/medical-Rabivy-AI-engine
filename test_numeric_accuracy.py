# test_numeric_accuracy.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# Checks whether the agent's OWN COMPUTED MATH is correct across 5
# GENUINELY DIFFERENT arithmetic operations - not the same operation
# repeated with different filters. First version of this file (found
# 2026-07-28) tested percentage-of-group 3 times and average-of-column
# twice - that's real, but it's only 2 distinct operations wearing 5
# different filter combinations, not 5 different kinds of numeracy a
# rep might actually need. This version covers:
#   1. Percentage of a group      (kept - already hard-won and proven)
#   2. Average / mean             (kept - already hard-won and proven)
#   3. Sum / total                (new - no division involved at all)
#   4. Difference between 2 groups (new - two independent facts, then
#                                   subtraction; either fact or the
#                                   subtraction itself could be wrong)
#   5. Compound multi-step math    (new - a rate, then multiplied by a
#                                   given number; chains two operations,
#                                   so there are two places to go wrong)
#
# TWO SEPARATE CHECKS, DELIBERATELY DIFFERENT STRICTNESS:
#   - Any underlying COUNT (how many rows total, how many matched) is a
#     whole number with ZERO legitimate rounding ambiguity - checked
#     EXACTLY. A found real case (2026-07-28): the agent stated "226
#     targeted" when the true count was 228 - its own arithmetic FROM
#     that wrong count was internally consistent, so only checking the
#     final percentage would have hidden this.
#   - Any derived VALUE (percentage, mean, sum, difference, a compound
#     result) must EXACTLY match the true value rounded to whatever
#     decimal precision the agent itself used - no tolerance band, since
#     every one of these is fully computable with one correct answer.
#
# Ground truth is computed live from the real dataframe (see
# ground_truth.py's gt_pct_matching / gt_mean_column / gt_sum_column /
# gt_group_diff) - never hardcoded, runs correctly on new data the same
# way the rest of this project's evals do.
# =====================================================================

import json
import os
import re
import time

from agent import RabivyAgent
import ground_truth as gt

QUESTIONS = [
    {
        "op": "percentage of a group",
        "q": "What percentage of active GLP-1 writers in Texas are currently targeted?",
        "value_keys": ["percentage"],
        "ground_truth": lambda: gt.gt_pct_matching({"state": "Texas", "zero_writer": False}, "targeted", 1),
    },
    {
        "op": "average / mean",
        "q": "What's the average propensity score for endocrinologists in Florida?",
        "value_keys": ["mean"],
        "ground_truth": lambda: gt.gt_mean_column("propensity_score", specialty="Endocrinology", state="Florida"),
    },
    {
        "op": "sum / total",
        "q": "What's the total monthly script volume across all High-tier Obesity Medicine prescribers nationally?",
        "value_keys": ["sum"],
        "ground_truth": lambda: gt.gt_sum_column("rx_volume_monthly", tier="High", specialty="Obesity Medicine"),
    },
    {
        "op": "difference between two groups",
        "q": "How many more High-tier HCPs does California have than Texas?",
        "value_keys": ["difference"],
        "ground_truth": lambda: gt.gt_group_diff({"tier": "High", "state": "California"}, {"tier": "High", "state": "Texas"}),
    },
    {
        "op": "compound multi-step (rate x given number)",
        "q": "Endocrinologists in Florida average a certain number of monthly scripts each. "
             "If 5 more endocrinologists in Florida started prescribing at that same average rate, "
             "how many additional monthly scripts would that represent?",
        "value_keys": ["rate", "result"],
        "ground_truth": lambda: (
            lambda avg: {"rate": avg["mean"], "result": avg["mean"] * 5} if avg else None
        )(gt.gt_mean_column("rx_volume_monthly", specialty="Endocrinology", state="Florida")),
    },
]

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def _numbers_with_precision(text):
    """All numbers found in the answer text, EACH paired with how many
    decimal places it was written with (comma-formatting stripped).
    E.g. '39.6%' -> (39.6, 1 decimal); '40%' -> (40.0, 0 decimals);
    '39.58%' -> (39.58, 2 decimals). This is what makes exact,
    decimal-adaptive matching possible."""
    out = []
    for m in _NUMBER_RE.findall(text):
        cleaned = m.replace(",", "").strip(".")
        if not cleaned:
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        out.append((value, decimals))
    return out


def _exact_count_present(candidates, expected_count):
    return any(v == expected_count for v, _ in candidates)


def _exact_value_present(candidates, true_value):
    """PASS only if some number in the answer, at whatever decimal
    precision it was stated, EXACTLY matches the true value rounded to
    that same precision. No tolerance band - the true value is fully
    computable, so there's no legitimate reason for any divergence
    once the precision is matched fairly."""
    for value, decimals in candidates:
        if value == round(true_value, decimals):
            return value, decimals
    return None, None


_BAR = "=" * 72


def run():
    agent = RabivyAgent()
    print(f"\n{_BAR}\nNUMERIC / ARITHMETIC ACCURACY\n{_BAR}")
    print(f"Checking {len(QUESTIONS)} DIFFERENT arithmetic operations - EXACT counts,")
    print(f"EXACT values at whatever precision the agent itself used, no tolerance band.\n")

    results = []
    pass_count = 0

    for i, t in enumerate(QUESTIONS, start=1):
        print(f"\n{'=' * 8} QUESTION {i} [{t['op']}] {'=' * 8}")
        print(f"Q: {t['q']}")

        truth = t["ground_truth"]()
        if truth is None:
            print("  SKIP - ground truth computation returned nothing (empty subset)")
            results.append({"q": t["q"], "status": "SKIP"})
            continue

        fresh = RabivyAgent(llm=agent.llm)
        fresh._log = lambda *a, **k: None
        try:
            result = fresh.ask(t["q"])
        except Exception as e:
            print(f"  CRASH: {type(e).__name__}: {e}")
            results.append({"q": t["q"], "status": "CRASH"})
            continue

        candidates = _numbers_with_precision(result["answer"])

        # Every dict key containing "count" must appear EXACTLY.
        count_keys = {k: v for k, v in truth.items() if "count" in k}
        count_checks = {k: _exact_count_present(candidates, v) for k, v in count_keys.items()}
        counts_ok = all(count_checks.values())

        # Every value_key must EXACTLY match at whatever precision was used.
        value_checks = {}
        for vk in t["value_keys"]:
            matched_value, matched_decimals = _exact_value_present(candidates, truth[vk])
            value_checks[vk] = (matched_value, matched_decimals)
        values_ok = all(mv is not None for mv, _ in value_checks.values())

        ok = counts_ok and values_ok
        status = "PASS" if ok else "FAIL"
        if ok:
            pass_count += 1

        print(f"  Ground truth (full precision): {truth}")
        print(f"  Exact count(s) found in answer: {count_checks}")
        print(f"  Exact value match(es) found: {value_checks}")
        print(f"  -> {status}" + ("" if counts_ok else "  <-- a COUNT was wrong or missing")
              + ("" if values_ok else "  <-- a VALUE was wrong or missing"))
        print(f"  ANSWER: {result['answer'][:400]}")

        results.append({"q": t["q"], "op": t["op"], "status": status, "truth": truth,
                        "count_checks": count_checks, "value_checks": value_checks})

    print(f"\n{_BAR}\nSUMMARY\n{_BAR}")
    print(f"{pass_count} / {len(QUESTIONS)} passed - exact counts AND exact values at the")
    print(f"precision the agent itself used, across 5 DIFFERENT arithmetic operations.")

    os.makedirs("eval_runs", exist_ok=True)
    path = os.path.join("eval_runs", f"numeric_accuracy_eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pass_count": pass_count, "total": len(QUESTIONS), "results": results},
                  f, indent=2, default=str)
    print(f"\nSaved this run to {path}")
    return results


if __name__ == "__main__":
    run()