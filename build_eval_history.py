# build_eval_history.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# Rolls up every file in eval_runs/ into one time-series file,
# eval_runs/eval_history.json, so a future dashboard can plot a trend
# instead of only ever showing the latest run.
#
# eval_runs/ holds one JSON snapshot per eval run, named
# <series>_YYYYMMDD_HHMMSS.json (e.g. agent_eval_20260729_135517.json).
# Different eval scripts write different shapes of JSON - this file
# knows how to read each shape it has actually seen in this repo and
# turn it into one common summary row. A file in a shape none of these
# recognise is skipped and printed as a warning, never guessed at.
#
# Run:     python build_eval_history.py
# Writes:  eval_runs/eval_history.json
# =====================================================================

import json
import os
import re
from datetime import datetime

EVAL_RUNS_DIR = "eval_runs"
HISTORY_FILE = os.path.join(EVAL_RUNS_DIR, "eval_history.json")

# Matches "agent_eval_20260729_135517.json" -> series="agent_eval",
# ts="20260729_135517". Anything that doesn't end in a date+time stamp
# (including eval_history.json itself, on a re-run) is left alone.
FILENAME_RE = re.compile(r"^(?P<series>.+)_(?P<ts>\d{8}_\d{6})\.json$")


def _parse_filename(filename):
    """Pulls the series name and run date out of a results filename.
    Returns (None, None) if the filename doesn't match the pattern."""
    m = FILENAME_RE.match(filename)
    if not m:
        return None, None
    try:
        ts = datetime.strptime(m.group("ts"), "%Y%m%d_%H%M%S")
    except ValueError:
        return None, None
    return m.group("series"), ts


# ---------------------------------------------------------------------
# One function per JSON shape this script has actually seen in this
# repo's eval_runs/ folder. Each takes the parsed JSON and returns a
# plain summary dict, or None if the shape doesn't match (so the
# caller tries the next one). Add a new function here the day a new
# eval script starts writing a shape none of these recognise.
# ---------------------------------------------------------------------

def _shape_tally_records(data):
    """agent_eval_*.json, eval_*.json - the golden-set shape:
    {"tally": {"layer1": {...}, "layer2": {...}, "layer3": {...}},
     "records": [...]}
    Score = layer3 pass rate. Layer 1/2 are routing and retrieval on
    the way there; layer 3 is "did it actually answer correctly"."""
    tally = data.get("tally")
    records = data.get("records")
    if tally is None or records is None:
        return None
    l3 = tally.get("layer3", {})
    total = len(records)
    passed = l3.get("PASS", 0)
    return {
        "total": total,
        "score": round(passed / total, 4) if total else None,
        "layer1_pass": tally.get("layer1", {}).get("PASS"),
        "layer2_pass": tally.get("layer2", {}).get("PASS"),
        "layer3_pass": passed,
    }


def _shape_layer_counts_results(data):
    """article_agent_eval_*.json - the real-document RAG eval shape:
    {"layer_counts": {"L1": {...}, "L2": {...}, "L3": {...}},
     "results": [...]}
    Also counts how many answers were flagged pretraining_flag=true -
    answers that passed with zero grounded evidence. Tracked
    separately since a rising count here means the eval is getting
    lucky, not that retrieval is actually improving."""
    layer_counts = data.get("layer_counts")
    results = data.get("results")
    if layer_counts is None or results is None:
        return None
    l3 = layer_counts.get("L3", {})
    total = len(results)
    passed = l3.get("PASS", 0)
    pretraining_flags = sum(1 for r in results if r.get("pretraining_flag"))
    return {
        "total": total,
        "score": round(passed / total, 4) if total else None,
        "layer1_pass": layer_counts.get("L1", {}).get("PASS"),
        "layer2_pass": layer_counts.get("L2", {}).get("PASS"),
        "layer3_pass": passed,
        "pretraining_flag_count": pretraining_flags,
    }


def _shape_flat_tally(data):
    """eval_*.json / agent_eval_*.json from before the 3-layer
    (L1/L2/L3) redesign - a flat {"tally": {"PASS": N, "FAIL": N,
    ...}, "results": [...]} with no routing/retrieval/answer
    breakdown. These are real runs and kept in the history, but tagged
    schema="pre_layer_redesign" so a chart doesn't silently average
    pre- and post-redesign scores together as if they measured the
    same thing - the eval logic itself changed here, not just the
    file format (see memory: "Rebuild eval suite with 3-layer
    design + fix real bugs found along the way")."""
    tally = data.get("tally")
    results = data.get("results")
    if tally is None or results is None:
        return None
    if "PASS" not in tally or any(k in tally for k in ("layer1", "layer2", "layer3")):
        return None
    total = len(results)
    passed = tally.get("PASS", 0)
    return {
        "total": total,
        "score": round(passed / total, 4) if total else None,
        "schema": "pre_layer_redesign",
    }


def _shape_pass_count_total(data):
    """numeric_accuracy_eval_*.json:
    {"pass_count": N, "total": N, "results": [...]}"""
    if "pass_count" not in data or "total" not in data:
        return None
    total = data["total"]
    passed = data["pass_count"]
    return {"total": total, "score": round(passed / total, 4) if total else None}


def _shape_consistent_total(data):
    """phrasing_consistency_eval_*.json, paraphrase_eval_*.json:
    {"consistent": N, "total": N, "results": [...]}"""
    if "consistent" not in data or "total" not in data:
        return None
    total = data["total"]
    passed = data["consistent"]
    return {"total": total, "score": round(passed / total, 4) if total else None}


def _shape_rank1_flat(data):
    """retrieval_ranking_eval_*.json (flat-count variant):
    {"rank1_count": N, "present_not_rank1_count": N,
     "missing_count": N, "total": N}
    Score = how often the correct chunk actually ranked #1 - this is
    the raw material behind the tree's "Yes" vs "Yes, but" split."""
    if "rank1_count" not in data or "total" not in data:
        return None
    total = data["total"]
    passed = data["rank1_count"]
    return {"total": total, "score": round(passed / total, 4) if total else None}


def _shape_counts_dict(data):
    """article_retrieval_eval_*.json (counts-dict variant):
    {"counts": {"RANK1": N, "PARTIAL(1/3)": N, "MISSING": N, ...},
     "results": [...]}
    Score = RANK1 divided by the sum of every bucket, since this shape
    doesn't carry a top-level "total" field."""
    counts = data.get("counts")
    if counts is None:
        return None
    total = sum(counts.values())
    passed = counts.get("RANK1", 0)
    return {"total": total, "score": round(passed / total, 4) if total else None}


# Tried in this order for every file - first one that returns a real
# summary (not None) wins.
_SHAPE_PARSERS = [
    _shape_tally_records,
    _shape_layer_counts_results,
    _shape_flat_tally,
    _shape_pass_count_total,
    _shape_consistent_total,
    _shape_rank1_flat,
    _shape_counts_dict,
]


def build_history():
    """Walks every JSON file in eval_runs/, summarises the ones in a
    recognised shape, and writes eval_runs/eval_history.json - one row
    per run, oldest first, with a "series" field so a dashboard can
    filter or group on it directly."""
    rows = []
    skipped = []

    for filename in sorted(os.listdir(EVAL_RUNS_DIR)):
        if not filename.endswith(".json") or filename == "eval_history.json":
            continue

        series, run_date = _parse_filename(filename)
        if series is None:
            skipped.append((filename, "filename doesn't match <series>_YYYYMMDD_HHMMSS.json"))
            continue

        path = os.path.join(EVAL_RUNS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = None
        for parser in _SHAPE_PARSERS:
            summary = parser(data)
            if summary is not None:
                break

        if summary is None:
            skipped.append((filename, "JSON shape not recognised by any parser"))
            continue

        row = {
            "series": series,
            "date": run_date.strftime("%Y-%m-%d"),
            "datetime": run_date.isoformat(),
            "file": filename,
        }
        row.update(summary)
        rows.append(row)

    rows.sort(key=lambda r: r["datetime"])

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    series_count = len(set(r["series"] for r in rows))
    print(f"Wrote {len(rows)} run(s) across {series_count} series to {HISTORY_FILE}")
    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for filename, reason in skipped:
            print(f"  - {filename}: {reason}")


if __name__ == "__main__":
    build_history()
