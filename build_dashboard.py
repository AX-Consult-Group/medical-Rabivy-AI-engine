# build_dashboard.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# Builds dashboard.html - the Rabivy AI Engine maintenance dashboard.
# Everything - tagging each question with its decision-tree leaf,
# reading the eval runs, and rendering the page - lives in this one
# file, since the dashboard is the only thing that needs any of it.
#
# STATUS: all three tabs are built. "Golden test set" reads the latest
# test_the_agent.py and test_the_system.py runs, tags every question,
# and renders a filterable, expandable page. "Live queries" and "RLHF
# feedback" read from output/QUERY_LOG/ and output/RLHF_FEEDBACK/. All
# three also have a Trend view - a day-by-day red/amber/green count,
# built from every historical eval_runs/ file for Golden, and from the
# logs' own timestamps for Live/RLHF.
#
# Run:  python build_dashboard.py
# Writes: dashboard.html (repo root, alongside evals.html/index.html
#          so it publishes the same way via GitHub Pages)
# =====================================================================

import json
import os
import re
import string
from datetime import datetime

EVAL_RUNS_DIR = "eval_runs"
OUTPUT_PATH = "dashboard.html"

# Same pattern as build_eval_history.py - "<series>_YYYYMMDD_HHMMSS.json"
FILENAME_RE = re.compile(r"^(?P<series>.+)_(?P<ts>\d{8}_\d{6})\.json$")

# =====================================================================
# GOLDEN TEST TREE - LEAF TAGGING
# ---------------------------------------------------------------------
# Takes ONE record from test_the_agent.py's or test_the_system.py's
# saved JSON and works out which leaf of the golden test set decision
# tree it lands on, plus that leaf's colour. This now matches the
# finalised 6-leaf tree exactly (Figma diagram, 2026-08-05), root
# question generalised to cover chunks, filters, AND rule checks:
#
#   Retrieval Result - "Did it retrieve the correct chunk, apply the
#   correct filters, or satisfy the correct rule check?"
#     Yes / Yes, but (evidence used)
#       -> answer correct    : Correct answer      (green)
#       -> answer wrong      : Synthesis issue      (red)
#     No (evidence NOT used)
#       -> gave a real (non-decline) answer anyway
#            -> correct      : Correct anyway       (amber)
#            -> wrong        : Hallucination         (red)
#       -> declined / said it didn't know ("Honest no")
#            -> ground truth says no answer exists  : Correctly rejected (green)
#            -> ground truth says a real answer DID exist : Retrieval fail (amber)
#
# Exactly 2 (evidence yes) + 4 (evidence no: declined-or-not x
# correct-or-not) = 6 leaves, every combination covered - no
# "needs review"/"not applicable" catch-all any more.
#
# "Did it decline" isn't in the saved JSON as its own field, so it's
# read off the answer TEXT itself via a phrase list (_is_decline,
# below) - this is deliberately different from the old verdict-based
# split it replaces. Two reasons: (1) the tree's own question is about
# the FORM of the answer ("no answer given but says so"), which is a
# text property, not the auditor's groundedness judgement, so a text
# check is actually the more faithful match, not a downgrade; (2) the
# audit verdict field only exists on test_the_agent.py runs (the LLM
# agent) - test_the_system.py's regex router has no audit step at all,
# so a verdict-only approach could never classify the regex router's
# runs into these 6 leaves. A text check works for both.
# =====================================================================

# Phrases that signal "the system said it couldn't answer" rather than
# stating a real (possibly wrong) fact. Deliberately broad rather than
# exhaustive - tune this list if a real answer keeps landing in the
# wrong bucket.
_DECLINE_PHRASES = [
    "not found", "no record", "doesn't exist", "does not exist",
    "no data", "unable to find", "cannot find", "can't find", "couldn't find",
    "not on file", "not in our", "no matching", "no such",
    "don't have", "do not have", "i don't know", "i do not know",
    "need more information", "need the npi", "need to know which",
    "please provide", "please specify", "which specific", "which one",
    "clarify",
]


def _is_decline(answer):
    """True if the answer's own wording reads as 'I can't answer this'
    rather than a stated fact - see the module docstring above for why
    this is a text check, not a verdict lookup."""
    text = (answer or "").lower()
    return any(phrase in text for phrase in _DECLINE_PHRASES)


def classify(record):
    """Returns (leaf_name, color) for one record. Every record lands
    on one of exactly 6 leaves - see the module docstring above."""
    l2_detail = record.get("layer2_detail")
    l3_correct = record.get("layer3") == "PASS"
    is_narrative = isinstance(l2_detail, dict) and "rank" in l2_detail

    if is_narrative:
        # Rank #1 and rank 2-5 both count as "evidence used" now - the
        # tree no longer splits on rank position at the leaf level
        # (it's still visible in the Layer 2 detail panel either way).
        evidence_ok = l2_detail.get("rank") is not None
    else:
        # Structured lookup or edge-case/trick question: whatever
        # shape produced layer2's PASS/FAIL (filters, or a custom rule
        # check), a rule-check PASS counts as "evidence used" too -
        # the tree's root question explicitly includes "satisfied the
        # correct rule check" as a Yes.
        evidence_ok = record.get("layer2") == "PASS"

    if evidence_ok and l3_correct:
        return "Correct answer", "green"
    if evidence_ok and not l3_correct:
        return "Synthesis issue", "red"

    # Evidence NOT used (the tree's "No" branch) - split on whether
    # the answer was a decline, then on whether that was the right
    # call.
    if _is_decline(record.get("answer")):
        if l3_correct:
            return "Correctly rejected", "green"
        return "Retrieval fail", "amber"
    if l3_correct:
        return "Correct anyway", "amber"
    return "Hallucination", "red"


# =====================================================================
# READING THE LATEST RUNS
# =====================================================================

def _latest_run(series):
    """Newest eval_runs/<series>_<timestamp>.json for an EXACT series
    name (not a prefix match - "eval" must not also match
    "eval_history" or a future "eval_labels" series)."""
    best = None
    if not os.path.isdir(EVAL_RUNS_DIR):
        return None
    for filename in os.listdir(EVAL_RUNS_DIR):
        m = FILENAME_RE.match(filename)
        if not m or m.group("series") != series:
            continue
        if best is None or filename > best:
            best = filename
    return os.path.join(EVAL_RUNS_DIR, best) if best else None


def _fmt_expected(expected):
    """layer3_detail's "expected" side can be a plain string, or a
    list mixing plain strings with OR-groups (lists of synonyms) -
    flatten all of that into one readable line for the table."""
    if expected is None:
        return ""
    if isinstance(expected, list):
        parts = []
        for item in expected:
            if isinstance(item, list):
                parts.append(" / ".join(str(x) for x in item))
            else:
                parts.append(str(item))
        return ", ".join(parts)
    return str(expected)


def _kpi(records, rows):
    total = len(records) or 1
    l1 = sum(1 for r in records if r.get("layer1") is True)
    l2 = sum(1 for r in records if r.get("layer2") == "PASS")
    l3 = sum(1 for r in records if r.get("layer3") == "PASS")
    overall = sum(1 for r in records
                  if r.get("layer1") is True and r.get("layer2") == "PASS" and r.get("layer3") == "PASS")
    pct = lambda n: round(100 * n / total)
    # Leaf-colour counts - NOT the same thing as the raw layer3 pass/fail
    # count above. Red/amber here means "needs a human look", straight
    # from the decision tree's own colour, so at a glance you can see how
    # much review work is actually sitting in this run - a 97% layer3
    # pass rate doesn't say whether the other 3% is red (block/review
    # now) or amber (lower-priority, sample it).
    red_n = sum(1 for r in rows if r["color"] == "red")
    amber_n = sum(1 for r in rows if r["color"] == "amber")
    green_n = sum(1 for r in rows if r["color"] == "green")
    return {"total": len(records), "overall_pct": pct(overall), "overall_n": overall,
            "l1_pct": pct(l1), "l1_n": l1, "l2_pct": pct(l2), "l2_n": l2,
            "l3_pct": pct(l3), "l3_n": l3,
            "red_n": red_n, "amber_n": amber_n, "green_n": green_n}


def _retrieval_detail(record):
    """Pulls out whatever layer2_detail has to show in the expanded
    row's Layer 2 section - always returns SOMETHING now, in one of
    four shapes, so every question type has a real Layer 2 to show
    (not just narrative + plain-filters ones):
      "narrative" - ranked/scored chunk list (search_documents ran)
      "filters"   - expected-vs-actual filter values (a lookup ran)
      "tag"       - just an expected tag/id, no rank data (older
                    narrative-style checks that didn't capture rank)
      "rule"      - a plain rules_check/CLARIFICATION-style check;
                    all we have is layer2's own PASS/FAIL and a note
    """
    l2 = record.get("layer2_detail")
    if isinstance(l2, dict) and "rank" in l2:
        return {"kind": "narrative", "rank": l2.get("rank"),
                "candidates": l2.get("top_results", [])}
    if isinstance(l2, dict) and "expected" in l2 and "actual" in l2:
        return {"kind": "filters", "expected": l2["expected"], "actual": l2["actual"]}
    if isinstance(l2, list):
        return {"kind": "tag", "expected": l2}
    return {"kind": "rule", "note": l2 if isinstance(l2, str) else None,
            "passed": record.get("layer2") == "PASS"}


def _question_type(record):
    """'narrative' if a real search_documents call happened (rank
    data present), else 'lookup' - covers structured spreadsheet
    lookups AND edge-case/trick questions alike, since neither goes
    through chunk retrieval."""
    l2 = record.get("layer2_detail")
    return "narrative" if isinstance(l2, dict) and "rank" in l2 else "lookup"


def _routed_to(record):
    """What Layer 1 actually routed to, for the expand panel - a tool
    name (or list of them) for test_the_agent.py, an engine string
    for test_the_system.py."""
    actual = (record.get("layer1_detail") or {}).get("actual")
    if isinstance(actual, list):
        return ", ".join(actual) if actual else "(no tool called)"
    if isinstance(actual, str):
        return actual
    return "(unknown)"


def _rows(records):
    out = []
    for n, r in enumerate(records, start=1):
        leaf, color = classify(r)
        l3d = r.get("layer3_detail")
        expected = l3d.get("expected") if isinstance(l3d, dict) else None
        out.append({
            "n": n,
            "q": r.get("q", ""),
            "qtype": _question_type(r),
            "routed_to": _routed_to(r),
            "l1": bool(r.get("layer1")),
            "l2": r.get("layer2") == "PASS",
            "l3": r.get("layer3") == "PASS",
            "leaf": leaf, "color": color,
            "expected": _fmt_expected(expected),
            "answer": r.get("answer") or "",
            "retrieval": _retrieval_detail(r),
        })
    return out


def _load_source(series, label):
    trend = _golden_trend(series)
    path = _latest_run(series)
    if path is None:
        return {"label": label, "run": None, "kpi": _kpi([], []), "rows": [], "trend": trend}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", [])
    rows = _rows(records)
    return {"label": label, "run": os.path.basename(path),
            "kpi": _kpi(records, rows), "rows": rows, "trend": trend}


def _golden_trend(series):
    """One point per historical eval_runs/<series>_<ts>.json file for
    this exact series - NOT just the latest run, unlike _load_source
    above. Reuses the same classify()/_rows() logic a live KPI uses,
    so a historical point means the same thing today's number does.

    Runs whose records don't have the layer1/layer2/layer3 shape
    classify() actually expects - the older, pre-3-layer-redesign
    format (see build_eval_history.py's own note on this exact
    problem) - are skipped rather than silently mis-tagged with a
    classification scheme that didn't exist yet when they ran.

    One point per CALENDAR DAY - if a series ran more than once on the
    same day, only the last run of that day is kept, matching how the
    live KPI box always reflects the latest run rather than an
    average of everything that happened that day."""
    if not os.path.isdir(EVAL_RUNS_DIR):
        return []
    by_day = {}
    for filename in sorted(os.listdir(EVAL_RUNS_DIR)):
        m = FILENAME_RE.match(filename)
        if not m or m.group("series") != series:
            continue
        try:
            ts = datetime.strptime(m.group("ts"), "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        path = os.path.join(EVAL_RUNS_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        records = data.get("records", [])
        if not records or not all(k in records[0] for k in ("layer2", "layer3")):
            continue  # pre-redesign or unrecognised shape - skip, don't mis-tag
        rows = _rows(records)
        date = ts.strftime("%Y-%m-%d")
        by_day[date] = {  # later filename on the same day overwrites - last run wins
            "date": date,
            "green": sum(1 for r in rows if r["color"] == "green"),
            "amber": sum(1 for r in rows if r["color"] == "amber"),
            "red": sum(1 for r in rows if r["color"] == "red"),
            "total": len(rows),
            "run": filename,
        }
    return sorted(by_day.values(), key=lambda d: d["date"])


# =====================================================================
# LIVE QUERIES + RLHF FEEDBACK - two DIFFERENT tabs, built from the same
# joined data but deliberately NOT merged into one:
#
#   Live Queries  - EVERY real query chat_ui.py has handled, classified
#                   ONLY by the 4 automated gates (Tool grounding /
#                   Retrieval confidence / Source citations /
#                   Hallucination audit). This is the live-query decision
#                   tree's own logic - it has no opinion on whether an
#                   answer was actually CORRECT, only whether the 4
#                   mechanical checks passed. Every query gets a row here,
#                   reviewed or not.
#   RLHF feedback - ONLY queries a rep has actually reviewed. Reuses the
#                   SAME 6-leaf vocabulary as the golden test set
#                   (Correct answer / Synthesis issue / Correct anyway /
#                   Correctly rejected / Retrieval fail / Hallucination),
#                   since a rep's review reconstructs those same leaves
#                   for a query where ground truth isn't known up front.
#
# A KPI box on the Live Queries tab shows what fraction of all queries
# have been picked up by RLHF yet, so the two tabs stay visibly linked
# without being the same table.
# =====================================================================

QUERY_LOG_PATH = os.path.join("output", "QUERY_LOG", "query_log.jsonl")
RLHF_LOG_PATH = os.path.join("output", "RLHF_FEEDBACK", "rlhf_log.jsonl")

# Each gate's OWN failure severity - these are NOT uniform, but only ONE
# of them is actually red. Reconciled 2026-08-07: "Source citations"
# used to be red here, but chat_ui.py's real quarantine rule
# (quarantined = verdict in ("fail", "audit_error")) never once
# consults this gate - only the Hallucination audit's own verdict blocks
# an answer. A missing citation is common and often legitimate on a
# follow-up question that's correctly answering from conversation
# memory rather than a fresh tool call (see agent.py's SYSTEM_PROMPT:
# follow-ups are meant to resolve from context). Coloring that red
# claimed a severity this gate was never actually enforcing, and could
# make a perfectly fine follow-up look like it needed urgent review.
_GATE_FAIL_COLOR = {
    "Tool grounding": "amber",
    "Retrieval confidence": "amber",
    "Source citations": "amber",
    "Hallucination audit": "red",
}
_COLOR_RANK = {"green": 0, "amber": 1, "red": 2}


def _gate_color(gate):
    """One gate's own colour: green if it passed, its own severity
    colour if it failed, amber if ok is None (audit not run - offline
    mode - unknown isn't a pass, but it isn't a confirmed problem
    either, so it lands as a caution rather than a red flag)."""
    ok = gate.get("ok")
    if ok is True:
        return "green"
    if ok is False:
        return _GATE_FAIL_COLOR.get(gate.get("name"), "amber")
    return "amber"  # ok is None - not checked


def _worst_color(colors):
    if not colors:
        return "green"
    return max(colors, key=lambda c: _COLOR_RANK.get(c, 1))


def _live_status(gate_rows):
    """Gate-only status text + colour for the Live Queries tab - NEVER
    consults RLHF. "Clean" if all 4 passed, otherwise names exactly
    which gate(s) failed or weren't checked, coloured by the worst one."""
    failed = [g["name"] for g in gate_rows if g["ok"] is False]
    unchecked = [g["name"] for g in gate_rows if g["ok"] is None]
    color = _worst_color([g["color"] for g in gate_rows])
    if not failed and not unchecked:
        return "All 4 gates passed", color
    bits = []
    if failed:
        bits.append("Failed: " + ", ".join(failed))
    if unchecked:
        bits.append("Not checked: " + ", ".join(unchecked))
    return "; ".join(bits), color


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


_LEGACY_RATING_COLOR = {"correct": "green", "partial": "amber",
                         "wrong": "red", "incorrect": "red", "vague": "amber"}


def _rlhf_leaf(rating, evidence_answer, is_decline):
    """Same rule chat_ui.py's _rlhf_leaf uses (kept in sync manually -
    different runtime, can't share the import) - mirrors classify()'s
    6-leaf golden-set logic from a rep's (rating, evidence_answer)
    pair. Recomputed HERE from the log's raw fields every time the
    dashboard builds, rather than trusting whatever leaf/colour
    chat_ui.py stored at submission time - so a logic fix (like the
    decline-flow fix that shipped alongside this) automatically
    corrects EXISTING logged entries too, without needing reps to
    re-rate anything."""
    if rating == "vague":
        return "Vague / non-answer", "amber"
    if evidence_answer not in ("yes", "no"):
        return f"{rating} - evidence unconfirmed, flagged for review", "amber"
    correct = rating == "correct"
    evidence_ok = evidence_answer == "yes"
    if evidence_ok and correct:
        return "Correct answer", "green"
    if evidence_ok and not correct:
        return "Synthesis issue", "red"
    if not evidence_ok and correct:
        return ("Correctly rejected", "green") if is_decline else ("Correct anyway", "amber")
    return ("Retrieval fail", "amber") if is_decline else ("Hallucination", "red")


def _live_rows():
    """One row per logged query, newest first. Every row always carries
    a gate-only status/status_color (drives the Live Queries tab) plus,
    when a rep has reviewed it, a separately-computed leaf/color (drives
    the RLHF feedback tab). Handles BOTH RLHF log formats: the rich
    post-rewrite schema (enough raw fields to recompute a real leaf via
    _rlhf_leaf) and the older bare {rating, note} entries from before
    that rewrite, which only get a plain rating-based colour."""
    queries = _read_jsonl(QUERY_LOG_PATH)
    rlhf_by_qid = {}
    for e in _read_jsonl(RLHF_LOG_PATH):
        qid = e.get("query_id")
        if qid:
            rlhf_by_qid[qid] = e  # last rating wins if rated twice

    rows = []
    for n, q in enumerate(queries, start=1):
        gates = q.get("gates") or []
        gate_rows = [{"name": g.get("name"), "ok": g.get("ok"),
                      "detail": g.get("detail"), "color": _gate_color(g)} for g in gates]
        status, status_color = _live_status(gate_rows)

        rlhf = rlhf_by_qid.get(q.get("query_id"))
        reviewed = rlhf is not None
        leaf = color = None
        rlhf_detail = None
        if reviewed:
            if "evidence_answer" in rlhf:
                # New-format entry - enough raw fields to recompute a
                # real leaf. Decline detection is recomputed from the
                # stored answer text via the SAME _is_decline used for
                # the golden set, not a trusted stored flag.
                answer_text = rlhf.get("answer") or q.get("answer") or ""
                is_decline = _is_decline(answer_text)
                leaf, color = _rlhf_leaf(rlhf.get("rating"), rlhf.get("evidence_answer"), is_decline)
            else:
                # Legacy pre-rewrite entry - only a bare rating, no
                # answer/evidence to recompute anything from.
                leaf = (rlhf.get("rating") or "reviewed").title() + " (legacy rating, no detail)"
                color = _LEGACY_RATING_COLOR.get(rlhf.get("rating"), "amber")
            rlhf_detail = {
                "rating": rlhf.get("rating"),
                "evidence_answer": rlhf.get("evidence_answer"),
                "confirmed_chunk_id": rlhf.get("confirmed_chunk_id"),
                "confirmed_chunk_rank": rlhf.get("confirmed_chunk_rank"),
                "note": rlhf.get("note"),
                "rated_at": rlhf.get("rated_at"),
            }
        rows.append({
            "n": n,
            "query_id": q.get("query_id"),
            "asked_at": q.get("asked_at"),
            "q": q.get("question", ""),
            "run_status": q.get("status"),
            "answer": q.get("answer") or "",
            "gates": gate_rows,
            "status": status, "status_color": status_color,
            "reviewed": reviewed,
            "leaf": leaf, "color": color,
            "rlhf": rlhf_detail,
            # Added 2026-08-07 alongside chat_ui.py's preceding_context
            # logging - rows written before that change simply won't have
            # this key, so .get() with a [] default rather than assuming
            # it's always present.
            "preceding_context": q.get("preceding_context") or [],
        })
    rows.reverse()  # newest query first
    for i, r in enumerate(rows, start=1):
        r["n"] = i
    return rows


def _live_kpi(rows):
    """Gate-only KPIs for the Live Queries tab - red/amber/green here
    come from status_color (the 4 gates), never from an RLHF leaf."""
    total = len(rows) or 1

    def gate_stat(name):
        vals = [g["ok"] for r in rows for g in r["gates"] if g["name"] == name]
        n_ok = sum(1 for v in vals if v is True)
        return round(100 * n_ok / total), n_ok

    clean_n = sum(1 for r in rows if r["gates"] and all(g["ok"] is True for g in r["gates"]))
    reviewed_n = sum(1 for r in rows if r["reviewed"])
    tool_pct, tool_n = gate_stat("Tool grounding")
    retr_pct, retr_n = gate_stat("Retrieval confidence")
    cite_pct, cite_n = gate_stat("Source citations")
    audit_pct, audit_n = gate_stat("Hallucination audit")
    return {
        "total": len(rows),
        "clean_pct": round(100 * clean_n / total), "clean_n": clean_n,
        "tool_pct": tool_pct, "tool_n": tool_n,
        "retr_pct": retr_pct, "retr_n": retr_n,
        "cite_pct": cite_pct, "cite_n": cite_n,
        "audit_pct": audit_pct, "audit_n": audit_n,
        "reviewed_pct": round(100 * reviewed_n / total), "reviewed_n": reviewed_n,
        "red_n": sum(1 for r in rows if r["status_color"] == "red"),
        "amber_n": sum(1 for r in rows if r["status_color"] == "amber"),
        "green_n": sum(1 for r in rows if r["status_color"] == "green"),
    }


def _rlhf_kpi(rows):
    """Leaf-colour KPIs for the RLHF feedback tab - reviewed queries
    only. red/amber/green here come from the rep-reconstructed 6-leaf
    classification, never from the raw gates."""
    reviewed_rows = [r for r in rows if r["reviewed"]]
    total = len(reviewed_rows) or 1
    return {
        "total": len(reviewed_rows),
        "red_n": sum(1 for r in reviewed_rows if r["color"] == "red"),
        "amber_n": sum(1 for r in reviewed_rows if r["color"] == "amber"),
        "green_n": sum(1 for r in reviewed_rows if r["color"] == "green"),
    }


def _day_bucket(by_day, date):
    return by_day.setdefault(date, {"date": date, "green": 0, "amber": 0, "red": 0, "total": 0})


def _live_trend(rows):
    """Day-by-day green/amber/red counts for the Live Queries tab,
    grouped by when each question was actually ASKED (asked_at) -
    every logged query counts here, reviewed or not, same as the tab
    itself. Colour comes from status_color (the 4 gates), never an
    RLHF leaf."""
    by_day = {}
    for r in rows:
        date = (r.get("asked_at") or "")[:10]
        if not date:
            continue
        _day_bucket(by_day, date)[r["status_color"]] += 1
        _day_bucket(by_day, date)["total"] += 1
    return sorted(by_day.values(), key=lambda d: d["date"])


def _rlhf_trend(rows):
    """Day-by-day green/amber/red counts for the RLHF feedback tab,
    grouped by when the REVIEW happened (rated_at) - not when the
    question was originally asked, since a rep can review a question
    days after it was logged. Reviewed queries only, colour from the
    rep-reconstructed leaf, same as the tab itself."""
    by_day = {}
    for r in rows:
        if not r["reviewed"]:
            continue
        date = ((r.get("rlhf") or {}).get("rated_at") or "")[:10]
        if not date:
            continue
        _day_bucket(by_day, date)[r["color"]] += 1
        _day_bucket(by_day, date)["total"] += 1
    return sorted(by_day.values(), key=lambda d: d["date"])


def _load_live():
    rows = _live_rows()
    return {"label": "Live queries", "kpi": _live_kpi(rows),
            "rlhf_kpi": _rlhf_kpi(rows), "rows": rows,
            "live_trend": _live_trend(rows), "rlhf_trend": _rlhf_trend(rows)}


# =====================================================================
# DECISION TREE DIAGRAM - built from the SAME leaf names classify()
# actually returns, so the "Decision tree logic" tab in the dashboard
# can't quietly drift out of sync with what the code does, the way a
# hand-maintained Figma image could. Titles/subtitles for the 6 real
# leaves are copied word-for-word from DIAG_DESC in the JS below -
# keep those two in sync if either one changes.
# =====================================================================

_TREE_BOXES = {
    # "leaf": True marks the 6 real classify() outputs - drawn with a
    # bolder border so it's visually obvious which boxes are actual
    # endpoints versus intermediate yes/no gates.
    "root":     dict(x=420, y=20,  w=240, h=110, kind="neutral", leaf=False, title="Retrieval Result",
                      sub="Did it retrieve the correct chunk, apply the correct filters, "
                          "or satisfy the correct rule check?"),
    "yes":      dict(x=60,  y=210, w=240, h=140, kind="green", leaf=False, title="Yes",
                      sub="Correct chunk ranked #1, correct filters applied, or rule check passed"),
    "yesbut":   dict(x=340, y=210, w=240, h=140, kind="amber", leaf=False, title="Yes, but",
                      sub="Correct chunk not #1 but in top 5"),
    "no":       dict(x=780, y=210, w=240, h=140, kind="red", leaf=False, title="No",
                      sub="Correct chunk/filters/rule check not satisfied"),
    "correct":  dict(x=40,  y=430, w=220, h=110, kind="green", leaf=True, title="Correct answer",
                      sub="The correct chunk/filters/rule check was used, and the answer was correct."),
    "synth":    dict(x=290, y=430, w=220, h=110, kind="red", leaf=True, title="Synthesis issue",
                      sub="The correct chunk/filters/rule check was used, but it still surfaced a wrong answer."),
    "anyway":   dict(x=540, y=430, w=220, h=110, kind="amber", leaf=True, title="Correct anyway",
                      sub="Got the answer right even without the correct chunk/filters."),
    "honestno": dict(x=790, y=430, w=220, h=110, kind="amber", leaf=False, title="Honest no",
                      sub="No answer given, but the system said so"),
    "halluc":   dict(x=1040, y=430, w=220, h=110, kind="red", leaf=True, title="Hallucination",
                      sub="Evidence was not surfaced, and the system gave an incorrect answer anyway."),
    "rejected": dict(x=685, y=590, w=200, h=110, kind="green", leaf=True, title="Correctly rejected",
                      sub="No answer exists, and the system correctly said so."),
    "retfail":  dict(x=915, y=590, w=200, h=110, kind="amber", leaf=True, title="Retrieval fail",
                      sub="An answer exists, but the system said it couldn't find it."),
}

# Each entry is one "bus" connector: every source box's bottom feeds a
# shared horizontal line, which then drops to every target box's top -
# matches the merged-arrow look of the Figma diagram (e.g. both "Yes"
# and "Yes, but" feed the SAME two leaves below them).
_TREE_CONNECTORS = [
    {"sources": ["root"], "targets": ["yes", "yesbut", "no"]},
    {"sources": ["yes", "yesbut"], "targets": ["correct", "synth"]},
    {"sources": ["no"], "targets": ["anyway", "honestno", "halluc"]},
    {"sources": ["honestno"], "targets": ["rejected", "retfail"]},
]

_TREE_KIND_FILL = {
    "green": "var(--green-bg)", "amber": "var(--amber-bg)",
    "red": "var(--red-bg)", "neutral": "var(--card-2)",
}
# Leaf (endpoint) boxes get a noticeably stronger tint than the
# intermediate yes/no gate boxes - the bolder border alone wasn't
# obvious enough on its own.
_TREE_KIND_FILL_LEAF = {
    "green": "var(--green-bg-strong)", "amber": "var(--amber-bg-strong)",
    "red": "var(--red-bg-strong)", "neutral": "var(--card-2)",
}
_TREE_KIND_STROKE = {
    "green": "var(--green)", "amber": "var(--amber)",
    "red": "var(--red)", "neutral": "var(--text-faint)",
}

_TREE_LEGEND_ROWS = [
    ("green", "System working. No action needed."),
    ("amber", "System partially working. Potential review/batch review. Track prevalence over time."),
    ("red", "System not working. Review all occurrences. Track prevalence over time."),
]

# How far short of the box edge a connector line stops, so the
# arrowhead sits cleanly in the gap instead of poking into the box.
_TREE_ARROW_GAP = 7


# Generic renderer - takes a boxes dict + connectors list + legend rows
# as PARAMETERS rather than reading the golden tree's module-level
# constants directly, so the exact same code draws both the golden
# 6-leaf tree and the live-query 4-gate tree below (two data shapes,
# one renderer - keeps the two diagrams visually identical in style).

def _tree_box_point(boxes, key, edge):
    """Centre-x plus the y of whichever edge ('top'/'bottom'/'mid') is
    asked for - used to anchor connector lines to a box."""
    b = boxes[key]
    cx = b["x"] + b["w"] / 2
    y = {"top": b["y"], "bottom": b["y"] + b["h"], "mid": b["y"] + b["h"] / 2}[edge]
    return cx, y


def _tree_box_svg(boxes, key):
    b = boxes[key]
    fill = (_TREE_KIND_FILL_LEAF if b["leaf"] else _TREE_KIND_FILL)[b["kind"]]
    stroke = _TREE_KIND_STROKE[b["kind"]]
    stroke_width = 3 if b["leaf"] else 1.5
    return (
        f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" height="{b["h"]}" rx="12" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        f'<foreignObject x="{b["x"] + 10}" y="{b["y"] + 8}" width="{b["w"] - 20}" height="{b["h"] - 16}">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" '
        f'style="display:flex;flex-direction:column;justify-content:center;height:100%">'
        f'<p class="tree-box-title">{b["title"]}</p>'
        f'<p class="tree-box-sub">{b["sub"]}</p>'
        f'</div></foreignObject>'
    )


def _tree_connector_svg(boxes, conn, bus_gap=20):
    src_pts = [_tree_box_point(boxes, k, "bottom") for k in conn["sources"]]
    # Targets stop _TREE_ARROW_GAP short of the box top, so the
    # arrowhead (which itself has zero overshoot past the line's end -
    # see the marker's refX below) lands in a clean gap rather than
    # overlapping the box border.
    tgt_pts = [(x, y - _TREE_ARROW_GAP) for x, y in
               (_tree_box_point(boxes, k, "top") for k in conn["targets"])]
    bus_y = src_pts[0][1] + bus_gap
    xs = [p[0] for p in src_pts] + [p[0] for p in tgt_pts]
    bus_x0, bus_x1 = min(xs), max(xs)
    lines = [f'<line x1="{x}" y1="{y}" x2="{x}" y2="{bus_y}" class="tree-line"/>' for x, y in src_pts]
    lines.append(f'<line x1="{bus_x0}" y1="{bus_y}" x2="{bus_x1}" y2="{bus_y}" class="tree-line"/>')
    lines += [f'<line x1="{x}" y1="{bus_y}" x2="{x}" y2="{y}" class="tree-line" '
              f'marker-end="url(#tree-arrow)"/>' for x, y in tgt_pts]
    return "\n".join(lines)


def _tree_svg(boxes, connectors, legend_rows):
    boxes_svg = "\n".join(_tree_box_svg(boxes, k) for k in boxes)
    connectors_svg = "\n".join(_tree_connector_svg(boxes, c) for c in connectors)
    # Legend as a centred vertical stack under the tree (green, then
    # amber, then red) - spreading 3 items evenly across the full
    # width looked odd because "Green" is a much shorter line than the
    # other two, so the gaps between them read as uneven.
    legend_items = "".join(
        f'<div class="tree-legend-row"><span class="tree-legend-dot" '
        f'style="background:var(--{color})"></span><p class="tree-legend-text">{text}</p></div>'
        for color, text in legend_rows
    )
    tree_bottom = max(b["y"] + b["h"] for b in boxes.values())
    legend_y = tree_bottom + 30
    legend_h = 190
    view_w = max(b["x"] + b["w"] for b in boxes.values()) + 20
    view_h = legend_y + legend_h + 10
    legend = (
        f'<foreignObject x="0" y="{legend_y}" width="{view_w}" height="{legend_h}">'
        '<div xmlns="http://www.w3.org/1999/xhtml" '
        'style="display:flex;flex-direction:column;align-items:center;gap:8px">'
        f'{legend_items}</div></foreignObject>'
    )
    return f'''<svg viewBox="0 0 {view_w} {view_h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block">
  <defs>
    <marker id="tree-arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="var(--text-faint)"/>
    </marker>
  </defs>
  {connectors_svg}
  {boxes_svg}
  {legend}
</svg>'''


def golden_tree_svg():
    return _tree_svg(_TREE_BOXES, _TREE_CONNECTORS, _TREE_LEGEND_ROWS)


# =====================================================================
# LIVE QUERY DECISION TREE DIAGRAM - the 4 automated gates, each drawn
# as its own small root->outcomes tree (they're INDEPENDENT checks, not
# a single sequential branch like the golden tree, so each gate gets
# its own pair/triple of leaves directly underneath it rather than
# sharing a bus with the others). Same visual language as the golden
# tree (colours, arrow gap, bold leaf borders) via the same _tree_svg().
# =====================================================================

# Box sizes here are deliberately more generous than the tight 120x80
# used in an earlier draft - that draft clipped the sub-text (e.g.
# "A low-confidence retrieval was flagged" didn't fit in an 80px-tall
# box and got cut off, invisibly, since foreignObject content
# overflowing its box isn't reliably shown). Leaves are now 190x110,
# gates 220x110 - same proportions as the golden tree's own boxes,
# which never had this problem.
_LIVE_TREE_BOXES = {
    "root": dict(x=785, y=20, w=340, h=100, kind="neutral", leaf=False,
                 title="Live query answered",
                 sub="Every real question gets checked against 4 independent automated gates"),

    "tool":  dict(x=88,   y=180, w=220, h=110, kind="neutral", leaf=False, title="Tool grounding",
                  sub="Was a tool called to ground the answer?"),
    "retr":  dict(x=524,  y=180, w=220, h=110, kind="neutral", leaf=False, title="Retrieval confidence",
                  sub="Were all retrievals confident - none flagged low?"),
    "cite":  dict(x=960,  y=180, w=220, h=110, kind="neutral", leaf=False, title="Source citations",
                  sub="Did the answer cite at least one source?"),
    "audit": dict(x=1499, y=180, w=220, h=110, kind="neutral", leaf=False, title="Hallucination audit",
                  sub="Did the answer pass the groundedness check? (LLM mode only)"),

    "tool_pass": dict(x=0,   y=340, w=190, h=110, kind="green", leaf=True, title="Called",
                       sub="Tool grounding passed"),
    "tool_fail": dict(x=206, y=340, w=190, h=110, kind="amber", leaf=True, title="Not called",
                       sub="No tool was called - answer isn't evidence-based"),

    "retr_pass": dict(x=436, y=340, w=190, h=110, kind="green", leaf=True, title="Confident",
                       sub="All retrievals confident"),
    "retr_fail": dict(x=642, y=340, w=190, h=110, kind="amber", leaf=True, title="Low confidence",
                       sub="A low-confidence retrieval was flagged"),

    "cite_pass": dict(x=872,  y=340, w=190, h=110, kind="green", leaf=True, title="Cited",
                       sub="Answer includes at least one source citation"),
    # Amber, not red - matches _GATE_FAIL_COLOR (2026-08-07): missing a
    # citation doesn't block the answer, and is often a legitimate
    # follow-up answering from conversation memory rather than a fresh
    # tool call, not a confirmed problem the way a hallucination is.
    "cite_fail": dict(x=1078, y=340, w=190, h=110, kind="amber", leaf=True, title="Not cited",
                       sub="No inline citation found in the answer"),

    "audit_pass":      dict(x=1308, y=340, w=190, h=110, kind="green", leaf=True, title="Passed",
                             sub="Groundedness check passed"),
    "audit_fail":      dict(x=1514, y=340, w=190, h=110, kind="red", leaf=True, title="Failed",
                             sub="Groundedness check flagged a hallucination"),
    "audit_unchecked": dict(x=1720, y=340, w=190, h=110, kind="amber", leaf=True, title="Not checked",
                             sub="No auditor available - offline mode"),
}

_LIVE_TREE_CONNECTORS = [
    {"sources": ["root"], "targets": ["tool", "retr", "cite", "audit"]},
    {"sources": ["tool"], "targets": ["tool_pass", "tool_fail"]},
    {"sources": ["retr"], "targets": ["retr_pass", "retr_fail"]},
    {"sources": ["cite"], "targets": ["cite_pass", "cite_fail"]},
    {"sources": ["audit"], "targets": ["audit_pass", "audit_fail", "audit_unchecked"]},
]

_LIVE_TREE_LEGEND_ROWS = [
    ("green", "Gate passed. No action needed."),
    ("amber", "No tool called, low retrieval confidence, no citation, or not checked at all "
              "(offline mode) - worth a look, not yet a confirmed problem."),
    ("red", "Confirmed hallucination - the answer is held back from the rep until a human "
            "reviews it. Review now."),
]


def live_tree_svg():
    return _tree_svg(_LIVE_TREE_BOXES, _LIVE_TREE_CONNECTORS, _LIVE_TREE_LEGEND_ROWS)


# =====================================================================
# RLHF PROMPT-FLOW DIAGRAM - not classify() output this time, but the
# actual sequence of prompts chat_ui.py's rlhfHtml()/wireRlhf() shows a
# rep, approved as a mockup before being built in here. Matches the
# real code exactly: the root gate ("did it surface an answer or
# decline") is auto-detected from the answer text, never shown to the
# rep - every other box IS a real prompt or button. "leaf" here means
# "nothing more is asked after this" (matches _rlhf_leaf's terminal
# cases), not "this is one of the 6 golden-tree names" - Correct/
# Partial/Incorrect keep going to the evidence question, so they're
# drawn as intermediate (thin border) even though they're colour-coded.
# =====================================================================

_RLHF_TREE_BOXES = {
    "root":   dict(x=678, y=20,  w=280, h=90,  kind="neutral", leaf=False,
                    title="Did the engine surface an answer?",
                    sub="Auto-detected from the answer's own wording - not asked"),
    "yes":    dict(x=300, y=150, w=220, h=90,  kind="neutral", leaf=False, title="Yes",
                    sub="Did the engine simply surface an answer (either correct or incorrect)"),
    "no":     dict(x=1115, y=150, w=220, h=90, kind="neutral", leaf=False, title="No",
                    sub="Did the engine simply decline answering"),
    "ratingq":   dict(x=300, y=280, w=220, h=90, kind="neutral", leaf=False,
                       title="Was the answer....", sub=""),
    "declineq":  dict(x=1115, y=280, w=220, h=90, kind="neutral", leaf=False,
                       title="Was declining the right call?", sub=""),

    "correct":   dict(x=0,   y=410, w=190, h=100, kind="green", leaf=False, title="Correct",
                       sub="Answer presented was correct"),
    "partial":   dict(x=210, y=410, w=190, h=100, kind="amber", leaf=False, title="Partially correct",
                       sub="Answer presented was partially correct"),
    "incorrect": dict(x=420, y=410, w=190, h=100, kind="red", leaf=False, title="Incorrect",
                       sub="Answer presented was incorrect"),
    "vague":     dict(x=630, y=410, w=190, h=100, kind="neutral", leaf=True, title="Vague / didn't really answer",
                       sub="Vague answer"),

    "rejected":  dict(x=920,  y=410, w=190, h=100, kind="green", leaf=True, title="Correctly rejected",
                       sub="No answer exists"),
    "retfail":   dict(x=1130, y=410, w=190, h=100, kind="amber", leaf=True, title="Retrieval fail",
                       sub="An answer exists but wasn't retrieved"),
    "notsure_d": dict(x=1340, y=410, w=190, h=100, kind="neutral", leaf=True, title="Not sure",
                       sub="If the rep is unsure"),

    "evidenceq": dict(x=300, y=550, w=220, h=90, kind="neutral", leaf=False,
                       title="Was the answer in the evidence?", sub=""),

    "ev_yes":    dict(x=105, y=680, w=190, h=100, kind="green", leaf=False, title="Yes",
                       sub="Answer was in evidence"),
    "ev_no":     dict(x=315, y=680, w=190, h=100, kind="red", leaf=True, title="No",
                       sub="Answer was not in evidence"),
    "ev_idk":    dict(x=525, y=680, w=190, h=100, kind="neutral", leaf=True, title="I don't know",
                       sub="Option in case the rep doesn't know"),

    "chunkq":    dict(x=90, y=820, w=220, h=90, kind="neutral", leaf=False,
                       title="Which chunk had the correct answer?", sub=""),

    "chunk_ns":  dict(x=10,  y=950, w=180, h=90, kind="neutral", leaf=True, title="Not sure", sub=""),
    "chunk_pk":  dict(x=210, y=950, w=180, h=90, kind="neutral", leaf=True, title="Choose from chunks surfaced", sub=""),
}

_RLHF_TREE_CONNECTORS = [
    {"sources": ["root"], "targets": ["yes", "no"]},
    {"sources": ["yes"], "targets": ["ratingq"]},
    {"sources": ["no"], "targets": ["declineq"]},
    {"sources": ["ratingq"], "targets": ["correct", "partial", "incorrect", "vague"]},
    {"sources": ["declineq"], "targets": ["rejected", "retfail", "notsure_d"]},
    {"sources": ["correct", "partial", "incorrect"], "targets": ["evidenceq"]},
    {"sources": ["evidenceq"], "targets": ["ev_yes", "ev_no", "ev_idk"]},
    {"sources": ["ev_yes"], "targets": ["chunkq"]},
    {"sources": ["chunkq"], "targets": ["chunk_ns", "chunk_pk"]},
]

_RLHF_TREE_LEGEND_ROWS = [
    ("green", "System working. No action needed."),
    ("amber", "System partially working, or a lower-severity miss. Worth a look."),
    ("red", "System not working. Needs review."),
    ("text-faint", "Rep wasn't sure - flagged for a second look, not a confirmed outcome either way."),
]


def rlhf_tree_svg():
    return _tree_svg(_RLHF_TREE_BOXES, _RLHF_TREE_CONNECTORS, _RLHF_TREE_LEGEND_ROWS)


# =====================================================================
# HTML TEMPLATE
# =====================================================================

_TEMPLATE = string.Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rabivy AI Engine maintenance dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --bg:#0d0f16; --card:#1d2233; --card-2:#262c40; --border:rgba(255,255,255,.14);
  --text:#e9eaf0; --text-dim:#9295a8; --text-faint:#666980;
  --purple:#8b7ff0; --purple-bg:rgba(139,127,240,.15);
  --green:#7ed67a; --green-bg:rgba(111,207,106,.15); --green-bg-strong:rgba(111,207,106,.32);
  --amber:#f5b552; --amber-bg:rgba(240,169,64,.15); --amber-bg-strong:rgba(240,169,64,.32);
  --red:#f28783; --red-bg:rgba(240,100,95,.15); --red-bg-strong:rgba(240,100,95,.32);
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:2rem;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:22px;font-weight:500;margin:0 0 4px}
.sub{color:var(--text-dim);font-size:13px;margin:0 0 1.5rem}
.tabs{display:flex;gap:8px;border-bottom:1px solid var(--border);padding-bottom:10px;margin-bottom:1.25rem}
button{font-family:inherit;font-size:13px;padding:8px 14px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--text);cursor:pointer}
button.active{background:var(--purple);border-color:var(--purple);color:#12101f}
button:disabled{opacity:.4;cursor:default}
.controls{display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;flex-wrap:wrap;gap:10px}
.controls .src{display:flex;gap:8px}
.runinfo{font-size:12px;color:var(--text-faint)}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px}
.kpi{background:var(--card);border-radius:10px;padding:1rem;border-left:3px solid transparent}
.kpi .label{font-size:12px;color:var(--text-dim);margin:0 0 2px}
.kpi .what{font-size:11px;color:var(--text-faint);margin:0 0 8px}
.kpi .value{font-size:26px;font-weight:500;margin:0}
.kpi .sub{font-size:12px;color:var(--text-faint);margin:4px 0 0}
.kpi.overall .value{color:var(--purple)}
.kpis-review{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:1.5rem}
.kpis-live{grid-template-columns:repeat(5,1fr)}
.kpis-review4{grid-template-columns:repeat(4,1fr)}
.kpi.c-red{background:var(--red-bg);border-left-color:var(--red)}
.kpi.c-amber{background:var(--amber-bg);border-left-color:var(--amber)}
.kpi.c-green{background:var(--green-bg);border-left-color:var(--green)}
.kpi.c-purple{background:var(--purple-bg);border-left-color:var(--purple)}
.kpi.c-red .value{color:var(--red)}
.kpi.c-amber .value{color:var(--amber)}
.kpi.c-green .value{color:var(--green)}
.kpi.c-purple .value{color:var(--purple)}
.legend{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.legend-pill{display:inline-flex;align-items:center;gap:6px;font-size:12px;padding:5px 10px;border-radius:20px;background:var(--card-2);border:1px solid var(--border);cursor:pointer;color:var(--text-dim);user-select:none}
.legend-pill.off{opacity:.4}
.legend-pill.on{color:var(--text)}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.tablecard{border:1px solid var(--border);border-radius:12px;overflow:hidden}
.thead{display:flex;padding:10px 16px;border-bottom:1px solid var(--border);font-size:12px;color:var(--text-dim)}
.thead .lc{cursor:help}
.row{padding:11px 16px;border-bottom:1px solid var(--border);cursor:pointer;font-size:13px}
.row:last-child{border-bottom:none}
.rowline{display:flex;align-items:center}
.num{flex:.25;color:var(--text-faint);font-variant-numeric:tabular-nums}
.q{flex:2.1;padding-right:8px}
.lc{flex:.5;text-align:center}
.diag{flex:2}
.typepill{font-size:11px;color:var(--text-faint);border:1px solid var(--border);border-radius:4px;padding:1px 6px;margin-left:6px}
.chip{font-size:11.5px;padding:3px 10px;border-radius:6px;display:inline-block;cursor:help}
.chip.green{background:var(--green-bg);color:var(--green)}
.chip.amber{background:var(--amber-bg);color:var(--amber)}
.chip.red{background:var(--red-bg);color:var(--red)}
.chip.na{background:transparent;color:var(--text-faint)}
.ok{color:var(--green)} .no{color:var(--red)} .warn{color:var(--amber)}
.detail{display:none;margin-top:10px;padding:14px 16px;background:var(--card-2);border-radius:8px;font-size:13px}
.dsection{margin-bottom:12px}
.dsection:last-child{margin-bottom:0}
.dlabel{font-size:11px;color:var(--text-faint);text-transform:uppercase;letter-spacing:.03em;margin:0 0 4px}
.dtext{margin:0;color:var(--text)}
.quote{background:var(--card);border-left:2px solid var(--border);border-radius:0 6px 6px 0;padding:8px 10px;margin:0;color:var(--text)}
.candidates{margin:4px 0 0;padding:0;list-style:none}
.candidates li{display:flex;justify-content:space-between;gap:10px;padding:3px 0;color:var(--text-dim)}
.candidates li.hit{color:var(--green)}
.kv{display:flex;gap:16px}
.kv .box{flex:1;background:var(--card);border-radius:6px;padding:8px 10px}
.footnote{font-size:12px;color:var(--text-faint);margin-top:14px}
.linkpill{font:inherit;font-size:11px;color:var(--purple);border:1px solid var(--purple);border-radius:4px;padding:1px 6px;margin-left:6px;background:none;cursor:pointer}
.linkpill:hover{background:var(--purple-bg)}
.linkbtn{background:none;border:none;color:var(--purple);text-decoration:underline;cursor:pointer;font:inherit;font-size:12px;padding:0}
.tree-card{background:var(--card);border-radius:12px;padding:24px;border:1px solid var(--border);display:none}
.tree-box-title{font-family:inherit;font-weight:600;font-size:15px;color:var(--text);margin:0 0 4px;text-align:center}
.tree-box-sub{font-family:inherit;font-size:11.5px;color:var(--text-dim);text-align:center;line-height:1.35;margin:0}
.tree-legend-title{font-size:12px;color:var(--text);font-weight:500;margin:0 0 10px}
.tree-legend-row{display:flex;gap:8px;align-items:flex-start;max-width:460px}
.tree-legend-dot{width:12px;height:12px;border-radius:50%;flex:0 0 auto;margin-top:2px}
.tree-legend-text{font-size:11.5px;color:var(--text-dim);line-height:1.35;margin:0}
.tree-line{stroke:var(--text-faint);stroke-width:2;fill:none}
</style>
</head>
<body>
<div class="wrap">
  <h1>Rabivy AI Engine maintenance dashboard</h1>
  <p class="sub">Golden test set, live queries, and RLHF feedback all live here</p>

  <div class="tabs">
    <button id="toptab-golden" class="active" onclick="setTopTab('golden')">Golden test set</button>
    <button id="toptab-live" onclick="setTopTab('live')">Live queries</button>
    <button id="toptab-rlhf" onclick="setTopTab('rlhf')">RLHF feedback</button>
  </div>

  <div id="golden-tab">
    <div class="controls">
      <div class="src">
        <button id="btn-tree" onclick="setSource('tree')">Decision tree logic</button>
        <button id="btn-agent" class="active" onclick="setSource('agent')">Test the agent (LLM)</button>
        <button id="btn-system" onclick="setSource('system')">Test the system (regex)</button>
        <button id="btn-trend" onclick="setSource('trend')">Trend</button>
      </div>
      <span class="runinfo" id="runinfo"></span>
    </div>

    <div class="kpis" id="kpis"></div>
    <div class="kpis-review" id="kpis-review"></div>

    <div class="legend" id="legend"></div>
    <div class="legend" id="typefilter"></div>

    <div class="tablecard" id="table-view">
      <div class="thead">
        <div class="num">#</div>
        <div class="q">Question</div>
        <div class="lc" title="Level 1 - routing: did it call the right tool?">L1</div>
        <div class="lc" title="Level 2 - retrieval: did it use the right filters, chunk, or rule?">L2</div>
        <div class="lc" title="Level 3 - answer: was the final answer correct?">L3</div>
        <div class="diag">Diagnosis (click a chip or row for what it means)</div>
      </div>
      <div id="rows"></div>
    </div>
    <p class="footnote" id="footnote"></p>

    <div class="tree-card" id="tree-view">
      $GOLDEN_TREE_SVG
    </div>

    <div class="tree-card" id="golden-trend-view"></div>
  </div>

  <div id="live-tab" style="display:none">
    <p class="sub" style="margin-bottom:1rem">Every real question chat_ui.py has answered, classified ONLY by the 4 automated gates - no human opinion involved. It can't tell you whether an answer was actually right, only whether the mechanical checks passed. For a human read on correctness, see the RLHF feedback tab.</p>

    <div class="controls">
      <div class="src">
        <button id="btn-live-tree" onclick="setLiveSource('tree')">Decision tree logic</button>
        <button id="btn-live-queries" class="active" onclick="setLiveSource('queries')">All queries</button>
        <button id="btn-live-trend" onclick="setLiveSource('trend')">Trend</button>
      </div>
    </div>

    <div id="live-queries-view">
      <div class="kpis kpis-live" id="live-kpis"></div>
      <div class="kpis-review kpis-review4" id="live-kpis-review"></div>

      <div class="legend" id="live-legend"></div>

      <div class="tablecard" id="live-table-view">
        <div class="thead">
          <div class="num">#</div>
          <div class="q">Question</div>
          <div class="lc" title="Tool grounding - was a tool called to ground the answer?">Tool</div>
          <div class="lc" title="Retrieval confidence - no low-confidence retrievals flagged?">Retr</div>
          <div class="lc" title="Source citations - did the answer cite at least one source?">Cite</div>
          <div class="lc" title="Hallucination audit - did the answer pass the groundedness check?">Audit</div>
          <div class="diag">Status (click a chip or row for detail)</div>
        </div>
        <div id="live-rows"></div>
      </div>
      <p class="footnote" id="live-footnote"></p>
    </div>

    <div class="tree-card" id="live-tree-view">
      $LIVE_TREE_SVG
    </div>

    <div class="tree-card" id="live-trend-view"></div>
  </div>

  <div id="rlhf-tab" style="display:none">
    <p class="sub" style="margin-bottom:1rem">Only queries a rep has actually reviewed - reconstructs the SAME 6-leaf classification the golden test set uses, from what a rep observed on a real answer. See the Live Queries tab for every query, reviewed or not.</p>

    <div class="controls">
      <div class="src">
        <button id="btn-rlhf-tree" onclick="setRlhfSource('tree')">Decision tree logic</button>
        <button id="btn-rlhf-reviewed" class="active" onclick="setRlhfSource('reviewed')">Reviewed queries</button>
        <button id="btn-rlhf-trend" onclick="setRlhfSource('trend')">Trend</button>
      </div>
    </div>

    <div id="rlhf-reviewed-view">
      <div class="kpis-review" id="rlhf-kpis-review"></div>

      <div class="legend" id="rlhf-legend"></div>

      <div class="tablecard" id="rlhf-table-view">
        <div class="thead">
          <div class="num">#</div>
          <div class="q">Question</div>
          <div class="diag">Diagnosis (click a chip or row for what it means)</div>
        </div>
        <div id="rlhf-rows"></div>
      </div>
      <p class="footnote" id="rlhf-footnote"></p>
    </div>

    <div class="tree-card" id="rlhf-tree-view">
      $RLHF_TREE_SVG
    </div>

    <div class="tree-card" id="rlhf-trend-view"></div>
  </div>
</div>

<script>
const DATA = $DATA_JSON;
let source = 'agent';
let activeColors = new Set(['green','amber','red']);
let activeType = 'all';

const DIAG_DESC = {
  'Correct answer': 'The correct chunk/filters/rule check was used, and the answer was correct.',
  'Synthesis issue': 'The correct chunk/filters/rule check was used, but it still surfaced a wrong answer.',
  'Correct anyway': 'Got the answer right even without the correct chunk/filters.',
  'Correctly rejected': "No answer exists, and the system correctly said so.",
  'Retrieval fail': "An answer exists, but the system said it couldn't find it.",
  'Hallucination': 'Evidence was not surfaced, and the system gave an incorrect answer anyway.'
};
function diagDesc(leaf){
  return DIAG_DESC[leaf] || leaf;
}

let topTab = 'golden';
function setTopTab(t){
  topTab = t;
  document.getElementById('toptab-golden').className = t==='golden' ? 'active' : '';
  document.getElementById('toptab-live').className = t==='live' ? 'active' : '';
  document.getElementById('toptab-rlhf').className = t==='rlhf' ? 'active' : '';
  document.getElementById('golden-tab').style.display = t==='golden' ? '' : 'none';
  document.getElementById('live-tab').style.display = t==='live' ? '' : 'none';
  document.getElementById('rlhf-tab').style.display = t==='rlhf' ? '' : 'none';
  if(t==='live') renderLive();
  if(t==='rlhf') renderRlhf();
}

let liveSource = 'queries';
function setLiveSource(s){
  liveSource = s;
  document.getElementById('btn-live-tree').className = s==='tree' ? 'active' : '';
  document.getElementById('btn-live-queries').className = s==='queries' ? 'active' : '';
  document.getElementById('btn-live-trend').className = s==='trend' ? 'active' : '';
  document.getElementById('live-queries-view').style.display = s==='queries' ? '' : 'none';
  document.getElementById('live-tree-view').style.display = s==='tree' ? 'block' : 'none';
  document.getElementById('live-trend-view').style.display = s==='trend' ? 'block' : 'none';
  if(s==='trend') renderLiveTrend();
}

let rlhfSource = 'reviewed';
function setRlhfSource(s){
  rlhfSource = s;
  document.getElementById('btn-rlhf-tree').className = s==='tree' ? 'active' : '';
  document.getElementById('btn-rlhf-reviewed').className = s==='reviewed' ? 'active' : '';
  document.getElementById('btn-rlhf-trend').className = s==='trend' ? 'active' : '';
  document.getElementById('rlhf-reviewed-view').style.display = s==='reviewed' ? '' : 'none';
  document.getElementById('rlhf-tree-view').style.display = s==='tree' ? 'block' : 'none';
  document.getElementById('rlhf-trend-view').style.display = s==='trend' ? 'block' : 'none';
  if(s==='trend') renderRlhfTrend();
}

function setSource(s){
  source = s;
  document.getElementById('btn-agent').className = s==='agent' ? 'active' : '';
  document.getElementById('btn-system').className = s==='system' ? 'active' : '';
  document.getElementById('btn-tree').className = s==='tree' ? 'active' : '';
  document.getElementById('btn-trend').className = s==='trend' ? 'active' : '';

  // The tree and trend views both replace the KPIs/legend/table
  // entirely - the tree is a static diagram of the classification
  // logic, the trend is a chart over past runs, neither is the
  // per-question table for the CURRENT run.
  var isData = (s === 'agent' || s === 'system');
  ['kpis','kpis-review','legend','typefilter','table-view','footnote'].forEach(function(id){
    document.getElementById(id).style.display = isData ? '' : 'none';
  });
  document.getElementById('tree-view').style.display = s==='tree' ? 'block' : 'none';
  document.getElementById('golden-trend-view').style.display = s==='trend' ? 'block' : 'none';
  document.getElementById('runinfo').textContent = isData ? document.getElementById('runinfo').textContent : '';

  if(isData) render();
  if(s==='trend') renderGoldenTrend();
}

function colorVar(c){
  return c==='green' ? 'var(--green)' : c==='amber' ? 'var(--amber)' : c==='red' ? 'var(--red)' : 'var(--text-faint)';
}

// ---------------------------------------------------------------------
// TREND CHARTS - one small stacked-bar SVG per tab, hand-rolled the
// same way the decision trees above are (no charting library, so the
// dashboard stays one self-contained file). Each bar is one calendar
// day; green/amber/red counts stack bottom-to-top. Data comes
// straight from DATA.agent.trend / DATA.system.trend /
// DATA.live.live_trend / DATA.live.rlhf_trend - all computed in
// build_dashboard.py, this just draws it.
// ---------------------------------------------------------------------
function trendLegend(){
  return '<div style="display:flex;gap:16px;margin:0 0 10px;font-size:12px;color:var(--text-dim)">'
    + ['green','amber','red'].map(function(c){
        return '<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
          +'background:'+colorVar(c)+';margin-right:6px;vertical-align:middle"></span>'
          +c.charAt(0).toUpperCase()+c.slice(1)+'</span>';
      }).join('')
    +'</div>';
}
function niceStep(rough){
  // Rounds a rough "gap between gridlines" up to a tidy 1/2/5x10^n
  // value, the same rule most charting libraries use, so the y-axis
  // reads 0/5/10/15 instead of 0/4.3/8.6/12.9.
  if(rough<=0) return 1;
  var mag = Math.pow(10, Math.floor(Math.log10(rough)));
  var norm = rough/mag;
  var step = norm<=1 ? 1 : norm<=2 ? 2 : norm<=5 ? 5 : 10;
  return step*mag;
}
function trendSvg(daily){
  if(!daily || !daily.length){
    return '<svg viewBox="0 0 400 90" width="100%" height="90" xmlns="http://www.w3.org/2000/svg">'
      +'<text x="200" y="48" text-anchor="middle" fill="var(--text-faint)" font-size="13">'
      +'Not enough data logged yet - check back once more days have gone by.</text></svg>';
  }
  // Three lines - one per colour - each following that colour's own
  // count day to day, rather than a stacked total. Drawn inside a
  // bordered plot box so it's clear where the data starts and ends,
  // with full gridlines (not just a top/bottom line) and real axis
  // labels - closer to a normal chart than the first pass at this.
  var leftPad=54, rightPad=20, topPad=16, bottomPad=48, colW=50, height=240;
  var plotH = height - topPad - bottomPad;
  var plotW = Math.max(300, Math.max(1, daily.length-1)*colW);
  var chartW = leftPad + plotW + rightPad;

  var maxVal = 1;
  daily.forEach(function(d){ ['green','amber','red'].forEach(function(c){ maxVal = Math.max(maxVal, d[c]||0); }); });
  var step = niceStep(maxVal/4 || 1);
  var yTop = step * Math.max(1, Math.ceil((maxVal*1.05)/step));
  var yTicks = [];
  for(var v=0; v<=yTop+1e-6; v+=step) yTicks.push(Math.round(v));

  function xFor(i){ return leftPad + (daily.length>1 ? i*(plotW/(daily.length-1)) : plotW/2); }
  function yFor(v){ return topPad + plotH - (v/yTop)*plotH; }

  var parts = [];
  // horizontal gridlines + y-axis value labels
  yTicks.forEach(function(v){
    var y = yFor(v);
    parts.push('<line x1="'+leftPad+'" y1="'+y+'" x2="'+(leftPad+plotW)+'" y2="'+y+'" stroke="var(--border)" stroke-width="1" stroke-dasharray="2,3"/>');
    parts.push('<text x="'+(leftPad-8)+'" y="'+(y+4)+'" text-anchor="end" fill="var(--text-faint)" font-size="10">'+v+'</text>');
  });
  // vertical gridlines + x-axis day labels
  daily.forEach(function(d,i){
    var x = xFor(i);
    parts.push('<line x1="'+x+'" y1="'+topPad+'" x2="'+x+'" y2="'+(topPad+plotH)+'" stroke="var(--border)" stroke-width="1" stroke-dasharray="2,3"/>');
    parts.push('<text x="'+x+'" y="'+(topPad+plotH+18)+'" text-anchor="middle" fill="var(--text-faint)" font-size="10">'+esc(d.date.slice(5))+'</text>');
  });
  // the lines themselves, on top of the gridlines
  ['green','amber','red'].forEach(function(c){
    var pts = daily.map(function(d,i){ return xFor(i)+','+yFor(d[c]||0); }).join(' ');
    parts.push('<polyline points="'+pts+'" fill="none" stroke="'+colorVar(c)+'" stroke-width="1.75"/>');
    daily.forEach(function(d,i){
      parts.push('<circle cx="'+xFor(i)+'" cy="'+yFor(d[c]||0)+'" r="3" fill="'+colorVar(c)+'"/>');
    });
  });
  // bounding box drawn last, on top, so the plot area's edge is crisp
  parts.push('<rect x="'+leftPad+'" y="'+topPad+'" width="'+plotW+'" height="'+plotH+'" fill="none" stroke="var(--border)" stroke-width="1.25"/>');
  // axis titles
  parts.push('<text x="'+(leftPad+plotW/2)+'" y="'+(height-6)+'" text-anchor="middle" fill="var(--text-dim)" font-size="11">Date</text>');
  parts.push('<text x="14" y="'+(topPad+plotH/2)+'" text-anchor="middle" fill="var(--text-dim)" font-size="11" '
    +'transform="rotate(-90 14 '+(topPad+plotH/2)+')">Count</text>');

  return '<div style="overflow-x:auto"><svg viewBox="0 0 '+chartW+' '+height+'" width="'+chartW+'" height="'+height+'" '
    +'xmlns="http://www.w3.org/2000/svg">'+parts.join('')+'</svg></div>';
}
function trendCard(daily){
  return trendLegend() + trendSvg(daily);
}
function renderGoldenTrend(){
  document.getElementById('golden-trend-view').innerHTML =
    '<h3 style="margin-top:0">Test the agent (LLM)</h3>' + trendCard(DATA.agent.trend)
    + '<h3>Test the system (regex)</h3>' + trendCard(DATA.system.trend);
}
function renderLiveTrend(){
  document.getElementById('live-trend-view').innerHTML = trendCard(DATA.live.live_trend);
}
function renderRlhfTrend(){
  document.getElementById('rlhf-trend-view').innerHTML = trendCard(DATA.live.rlhf_trend);
}
function toggleColor(c){
  if(activeColors.has(c)) activeColors.delete(c); else activeColors.add(c);
  render();
}
function setType(t){
  activeType = t;
  render();
}
function renderLegend(){
  const items = [
    ['green','Green - system working'],
    ['amber','Amber - worth a look'],
    ['red','Red - needs review']
  ];
  var legendEl = document.getElementById('legend');
  legendEl.innerHTML = items.map(function(pair){
    var c = pair[0], label = pair[1];
    var cls = activeColors.has(c) ? 'on' : 'off';
    return '<span class="legend-pill '+cls+'" data-color="'+c+'">'+
      '<i class="dot" style="background:'+colorVar(c)+'"></i>'+label+'</span>';
  }).join('');
  Array.prototype.forEach.call(legendEl.children, function(el){
    el.addEventListener('click', function(){ toggleColor(el.getAttribute('data-color')); });
  });
}
function renderTypeFilter(){
  const items = [
    ['all','All question types'],
    ['narrative','Narrative (document search)'],
    ['lookup','Spreadsheet lookup / edge case']
  ];
  var el = document.getElementById('typefilter');
  el.innerHTML = items.map(function(pair){
    var t = pair[0], label = pair[1];
    var cls = activeType === t ? 'on' : 'off';
    return '<span class="legend-pill '+cls+'" data-type="'+t+'">'+label+'</span>';
  }).join('');
  Array.prototype.forEach.call(el.children, function(child){
    child.addEventListener('click', function(){ setType(child.getAttribute('data-type')); });
  });
}

function chip(leaf, color){
  const cls = color || 'na';
  return '<span class="chip '+cls+'" title="'+esc(diagDesc(leaf))+'">'+esc(leaf)+'</span>';
}
function dot(ok){ return ok ? '<span class="ok">&#10003;</span>' : '<span class="no">&#10007;</span>'; }
function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }

// Light markdown rendering for answer text - the agent writes **bold**,
// blank-line paragraph breaks, and "- " bullet lists (see agent.py's
// SYSTEM_PROMPT), which used to show up on the dashboard as literal
// asterisks and one unbroken block of text. Escapes first (same esc()
// used everywhere else) so this stays just as safe, THEN turns the
// escaped-but-still-plain markup into real HTML. Deliberately small -
// bold, paragraphs, bullets - not a full markdown parser.
function mdToHtml(s){
  if (!s) return '';
  // NOTE: this whole page is a Python string.Template, which treats
  // any dollar sign in this text as the start of ITS OWN placeholder
  // syntax and throws "Invalid placeholder" on substitute() otherwise.
  // A doubled dollar sign is Template's own escape for one literal
  // dollar sign - needed below for JS's regex capture-group reference,
  // and again further down for a regex end-of-line anchor.
  var t = esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$$1</strong>');
  var blocks = t.split(/\\n\s*\\n/);
  return blocks.map(function(block){
    // Line-by-line within the block, not "the whole block is a list or
    // it isn't" - a real answer often has a label line immediately
    // followed by bullets with no blank line between them ("Key
    // takeaways:" then "- ..." lines, per SYSTEM_PROMPT), and that
    // label line needs to stay its own line, not get folded into the list.
    var lines = block.split('\\n').filter(function(l){ return l.trim() !== ''; });
    var html = '', listBuf = [], textBuf = [];
    function flushList(){
      if (listBuf.length) { html += '<ul>' + listBuf.map(function(x){ return '<li>'+x+'</li>'; }).join('') + '</ul>'; listBuf = []; }
    }
    function flushText(){
      if (textBuf.length) { html += '<p>' + textBuf.join('<br>') + '</p>'; textBuf = []; }
    }
    lines.forEach(function(l){
      var m = /^\s*-\s+(.*)$$/.exec(l);
      if (m) { flushText(); listBuf.push(m[1]); }
      else { flushList(); textBuf.push(l); }
    });
    flushList(); flushText();
    return html;
  }).join('');
}

// Shared accordion state for all three tabs' item-detail panels (golden
// toggle(), live toggleLive(), rlhf toggleRlhf()) - only one item open
// at a time, and clicking anywhere outside the open panel (and outside
// any row, so a click that opens a DIFFERENT item isn't immediately
// undone) closes it.
let openDetailEl = null;
function _toggleDetail(el){
  if (openDetailEl && openDetailEl !== el) {
    openDetailEl.style.display = 'none';
  }
  const show = el.style.display !== 'block';
  el.style.display = show ? 'block' : 'none';
  openDetailEl = show ? el : null;
}
document.addEventListener('click', function(e){
  if (!openDetailEl) return;
  if (openDetailEl.contains(e.target)) return;
  if (e.target.closest && e.target.closest('.row')) return;
  openDetailEl.style.display = 'none';
  openDetailEl = null;
});

function retrievalHTML(ret){
  // Layer 2 body, in whichever of the four shapes _retrieval_detail()
  // produced - every question type gets something here now.
  if(!ret) return '<p class="dtext">-</p>';
  if(ret.kind === 'narrative'){
    const items = (ret.candidates||[]).map(c =>
      '<li class="'+(c.is_expected?'hit':'')+'">'+esc(c.chunk_id)+(c.is_expected?' (correct chunk)':'')+'<span>'+(c.similarity!=null?c.similarity.toFixed(3):'-')+'</span></li>'
    ).join('');
    const rankText = ret.rank ? ('Ranked #'+ret.rank+' of '+(ret.candidates||[]).length+' chunks pulled, by cosine similarity') : 'Correct chunk not found among the retrieved chunks';
    return '<p class="dtext">'+rankText+'</p><ul class="candidates">'+items+'</ul>';
  }
  if(ret.kind === 'filters'){
    return '<div class="kv">'+
      '<div class="box"><p class="dlabel">Expected</p><p class="dtext">'+esc(JSON.stringify(ret.expected))+'</p></div>'+
      '<div class="box"><p class="dlabel">Actual</p><p class="dtext">'+esc(JSON.stringify(ret.actual))+'</p></div></div>';
  }
  if(ret.kind === 'tag'){
    return '<p class="dtext">Expected tag: '+esc(JSON.stringify(ret.expected))+'</p>';
  }
  // 'rule' - a custom rules_check (fake NPI, fake region, unresolved
  // reference, etc). We don't store the tool's raw result, only
  // whether the check passed - the answer given (Layer 3, below)
  // shows what the system actually said.
  return '<p class="dtext">Custom rule check: '+(ret.passed ? '<span class="ok">passed</span>' : '<span class="no">failed</span>')+
    (ret.note ? ' ('+esc(ret.note)+')' : '')+'</p>';
}

function detailHTML(r,i){
  return '<div class="detail" id="d'+i+'">'+
    '<div class="dsection"><p class="dlabel">Diagnosis</p><p class="dtext">'+chip(r.leaf,r.color)+' &mdash; '+esc(diagDesc(r.leaf))+'</p></div>'+
    '<div class="dsection"><p class="dlabel">Layer 1 &mdash; routed to</p><p class="dtext">'+esc(r.routed_to)+'</p></div>'+
    '<div class="dsection"><p class="dlabel">Layer 2 &mdash; '+(r.qtype==='narrative' ? 'chunks retrieved' : 'filters / rule applied')+'</p>'+retrievalHTML(r.retrieval)+'</div>'+
    '<div class="dsection"><p class="dlabel">Layer 3 &mdash; expected answer</p><div class="quote">'+mdToHtml(r.expected || '(no single checkable value for this question)')+'</div></div>'+
    '<div class="dsection"><p class="dlabel">Layer 3 &mdash; actual answer</p><div class="quote">'+mdToHtml(r.answer)+'</div></div>'+
  '</div>';
}

// =====================================================================
// LIVE QUERIES TAB - gate-only. EVERY logged query gets a row, coloured
// purely by the 4 automated gates (r.status / r.status_color) - never
// by an RLHF leaf, even if one exists. The "Reviewed" KPI box and the
// small badge on a row are just a pointer over to the RLHF feedback
// tab, not a colour input here.
// =====================================================================
let liveActiveColors = new Set(['green','amber','red']);

function gateDot(g){
  const symbol = g.ok === true ? '&#10003;' : g.ok === false ? '&#10007;' : '?';
  const cls = g.color === 'green' ? 'ok' : g.color === 'red' ? 'no' : 'warn';
  return '<span class="'+cls+'" title="'+esc(g.detail || g.name || '')+'">'+symbol+'</span>';
}
function toggleLiveColor(c){
  if(liveActiveColors.has(c)) liveActiveColors.delete(c); else liveActiveColors.add(c);
  renderLive();
}
function renderLiveLegend(){
  const items = [
    ['green','Green - all gates checked out'],
    ['amber','Amber - worth a look'],
    ['red','Red - needs review']
  ];
  var el = document.getElementById('live-legend');
  el.innerHTML = items.map(function(pair){
    var c = pair[0], label = pair[1];
    var cls = liveActiveColors.has(c) ? 'on' : 'off';
    return '<span class="legend-pill '+cls+'" data-color="'+c+'"><i class="dot" style="background:'+colorVar(c)+'"></i>'+label+'</span>';
  }).join('');
  Array.prototype.forEach.call(el.children, function(child){
    child.addEventListener('click', function(){ toggleLiveColor(child.getAttribute('data-color')); });
  });
}

function liveDetailHTML(r,i){
  var gateRows = r.gates.map(function(g){
    return '<div class="dsection"><p class="dlabel">'+esc(g.name)+'</p><p class="dtext">'+gateDot(g)+' '+esc(g.detail || '(no detail)')+'</p></div>';
  }).join('');
  var reviewNote = r.reviewed
    ? '<div class="dsection"><p class="dlabel">RLHF</p><p class="dtext">A rep has reviewed this one. '
      +'<button class="linkbtn" onclick="event.stopPropagation();jumpToRlhf(\\''+esc(r.query_id)+'\\')">View in RLHF feedback tab &rarr;</button></p></div>'
    : '<div class="dsection"><p class="dlabel">RLHF</p><p class="dtext">Not yet reviewed by a rep.</p></div>';
  // Added 2026-08-07: shows exactly what the model had in memory BEFORE
  // this question - the same trimmed question/answer pairs chat_ui.py
  // snapshots at ask-time (see agent.py's _trim_history()). This is what
  // makes an Amber row from a no-new-tool-call follow-up checkable: a
  // reviewer can see whether it genuinely had nothing to go on, or
  // correctly reused real evidence from a turn or two back, instead of
  // having to guess from timestamps alone.
  var contextNote;
  if (!r.preceding_context || r.preceding_context.length === 0) {
    contextNote = '<div class="dsection"><p class="dlabel">Conversation context</p>'
      +'<p class="dtext">No prior context - this was the first question of the session '
      +'(or logged before this field existed).</p></div>';
  } else {
    // preceding_context is always strict user/assistant pairs (agent.py's
    // _trim_history() only ever stores whole Q&A turns) - step through two
    // at a time so each prior QUESTION is its own line and its ANSWER is
    // collapsed behind a click, instead of one long unbroken block of text.
    var pairs = [];
    for (var t = 0; t < r.preceding_context.length; t += 2) {
      pairs.push([r.preceding_context[t], r.preceding_context[t + 1]]);
    }
    var turns = pairs.map(function(pair, t){
      var q = pair[0] ? pair[0].content : '';
      var a = pair[1] ? pair[1].content : '(no answer recorded)';
      var cid = 'ctx' + i + '_' + t;
      return '<div class="ctxturn">'
        +'<p class="dtext ctxq">Q'+(t+1)+': '+esc(q)+'</p>'
        +'<button class="linkbtn ctxtoggle" onclick="event.stopPropagation();toggleCtxTurn(\\''+cid+'\\')">Show answer &#9656;</button>'
        +'<div class="dtext quote ctxa" id="'+cid+'" style="display:none;">'+mdToHtml(a)+'</div>'
        +'</div>';
    }).join('');
    contextNote = '<div class="dsection"><p class="dlabel">Conversation context before this question</p>'+turns+'</div>';
  }
  return '<div class="detail" id="ld'+i+'">'+
    '<div class="dsection"><p class="dlabel">Status</p><p class="dtext"><span class="chip '+(r.status_color||'na')+'">'+esc(r.status)+'</span></p></div>'+
    '<div class="dsection"><p class="dlabel">Asked</p><p class="dtext">'+esc(r.asked_at || '')+'</p></div>'+
    gateRows +
    reviewNote +
    contextNote +
    '<div class="dsection"><p class="dlabel">Answer</p><div class="quote">'+mdToHtml(r.answer)+'</div></div>'+
  '</div>';
}

function renderLive(){
  renderLiveLegend();
  const d = DATA.live;
  const k = d.kpi;
  document.getElementById('live-kpis').innerHTML =
    '<div class="kpi overall"><p class="label">Fully clean</p><p class="what">All 4 gates passed</p><p class="value">'+k.clean_pct+'%</p><p class="sub">'+k.clean_n+' / '+k.total+'</p></div>'+
    '<div class="kpi"><p class="label">Tool grounding</p><p class="what">A tool was called to ground the answer</p><p class="value">'+k.tool_pct+'%</p><p class="sub">'+k.tool_n+' / '+k.total+'</p></div>'+
    '<div class="kpi"><p class="label">Retrieval confidence</p><p class="what">No low-confidence retrieval flagged</p><p class="value">'+k.retr_pct+'%</p><p class="sub">'+k.retr_n+' / '+k.total+'</p></div>'+
    '<div class="kpi"><p class="label">Source citations included</p><p class="what">Answer cited at least one source</p><p class="value">'+k.cite_pct+'%</p><p class="sub">'+k.cite_n+' / '+k.total+'</p></div>'+
    '<div class="kpi"><p class="label">Hallucination audit</p><p class="what">Passed the groundedness check</p><p class="value">'+k.audit_pct+'%</p><p class="sub">'+k.audit_n+' / '+k.total+'</p></div>';

  document.getElementById('live-kpis-review').innerHTML =
    '<div class="kpi c-red"><p class="label">Red</p><p class="what">Needs review now</p><p class="value">'+k.red_n+'</p><p class="sub">of '+k.total+' queries</p></div>'+
    '<div class="kpi c-amber"><p class="label">Amber</p><p class="what">Worth a look</p><p class="value">'+k.amber_n+'</p><p class="sub">of '+k.total+' queries</p></div>'+
    '<div class="kpi c-green"><p class="label">Green</p><p class="what">All gates checked out</p><p class="value">'+k.green_n+'</p><p class="sub">of '+k.total+' queries</p></div>'+
    '<div class="kpi c-purple"><p class="label">Reviewed via RLHF</p><p class="what">Has a rep review - see RLHF tab</p><p class="value">'+k.reviewed_pct+'%</p><p class="sub">'+k.reviewed_n+' / '+k.total+'</p></div>';

  const rows = d.rows.filter(function(r){ return liveActiveColors.has(r.status_color || 'na'); });
  document.getElementById('live-rows').innerHTML = rows.map(function(r,i){
    var gateCells = r.gates.map(function(g){ return '<div class="lc">'+gateDot(g)+'</div>'; }).join('');
    return '<div class="row" onclick="toggleLive('+i+')">'+
      '<div class="rowline"><div class="num">'+r.n+'</div><div class="q">'+esc(r.q)
        +(r.reviewed?'<button class="linkpill" onclick="event.stopPropagation();jumpToRlhf(\\''+esc(r.query_id)+'\\')" title="View this review in the RLHF feedback tab">reviewed &rarr;</button>':'')
        +'</div>'+
      gateCells+
      '<div class="diag"><span class="chip '+(r.status_color||'na')+'">'+esc(r.status)+'</span></div></div>'+
      liveDetailHTML(r,i)+
      '</div>';
  }).join('');
  document.getElementById('live-footnote').textContent = rows.length + ' of ' + d.rows.length + ' logged queries shown.';
}

function toggleLive(i){
  _toggleDetail(document.getElementById('ld'+i));
}

// One prior turn's answer inside the "Conversation context" section -
// collapsed by default so a multi-turn history reads as a list of
// questions, not a wall of text; click a question's button to expand
// just that answer. Flips the button's own label/arrow to match.
function toggleCtxTurn(id){
  const el = document.getElementById(id);
  const show = el.style.display === 'none';
  el.style.display = show ? '' : 'none';
  const btn = el.previousElementSibling;
  if (btn) btn.innerHTML = show ? 'Hide answer &#9662;' : 'Show answer &#9656;';
}

// Jumps from a reviewed Live Queries row straight to its entry in the
// RLHF feedback tab - resets the RLHF colour filter to "show everything"
// first, so the target row can't be hidden by a filter left on from a
// previous visit, then expands and scrolls to it.
function jumpToRlhf(queryId){
  rlhfActiveColors = new Set(['green','amber','red']);
  setTopTab('rlhf');
  const reviewedRows = DATA.live.rows.filter(function(r){ return r.reviewed; });
  const idx = reviewedRows.findIndex(function(r){ return r.query_id === queryId; });
  if(idx < 0) return;
  const rowEl = document.querySelectorAll('#rlhf-rows .row')[idx];
  const detailEl = document.getElementById('rd'+idx);
  if(detailEl) detailEl.style.display = 'block';
  if(rowEl && rowEl.scrollIntoView) rowEl.scrollIntoView({behavior:'smooth', block:'center'});
}

// =====================================================================
// RLHF FEEDBACK TAB - reviewed queries ONLY. Reuses the golden set's
// own 6-leaf vocabulary/colours (r.leaf/r.color), reconstructed from a
// rep's own read of a real answer - see _rlhf_leaf() in build_dashboard.py.
// =====================================================================
let rlhfActiveColors = new Set(['green','amber','red']);

function rlhfDesc(leaf){
  if(DIAG_DESC[leaf]) return DIAG_DESC[leaf];
  if(leaf.indexOf('legacy rating') >= 0) return "Reviewed before the detailed review form existed - only a plain rating was captured, no gate/evidence detail.";
  if(leaf.indexOf('flagged for review') >= 0) return "The rep's evidence answer didn't resolve to yes/no, so this needs a second look rather than a guessed leaf.";
  return leaf;
}
function rlhfChip(row){
  const cls = row.color || 'na';
  return '<span class="chip '+cls+'" title="'+esc(rlhfDesc(row.leaf))+'">'+esc(row.leaf)+'</span>';
}
function toggleRlhfColor(c){
  if(rlhfActiveColors.has(c)) rlhfActiveColors.delete(c); else rlhfActiveColors.add(c);
  renderRlhf();
}
function renderRlhfLegend(){
  const items = [
    ['green','Green - system working'],
    ['amber','Amber - worth a look'],
    ['red','Red - needs review']
  ];
  var el = document.getElementById('rlhf-legend');
  el.innerHTML = items.map(function(pair){
    var c = pair[0], label = pair[1];
    var cls = rlhfActiveColors.has(c) ? 'on' : 'off';
    return '<span class="legend-pill '+cls+'" data-color="'+c+'"><i class="dot" style="background:'+colorVar(c)+'"></i>'+label+'</span>';
  }).join('');
  Array.prototype.forEach.call(el.children, function(child){
    child.addEventListener('click', function(){ toggleRlhfColor(child.getAttribute('data-color')); });
  });
}
function rlhfDetailHTML(r,i){
  var rl = r.rlhf || {};
  return '<div class="detail" id="rd'+i+'">'+
    '<div class="dsection"><p class="dlabel">Diagnosis</p><p class="dtext">'+rlhfChip(r)+' &mdash; '+esc(rlhfDesc(r.leaf))+'</p></div>'+
    '<div class="dsection"><p class="dlabel">Asked</p><p class="dtext">'+esc(r.asked_at || '')+'</p></div>'+
    '<div class="dsection"><p class="dlabel">Rep review</p><p class="dtext">Rating: <strong>'+esc(rl.rating || '-')+'</strong>'+
      (rl.evidence_answer ? ' &middot; Right evidence used: <strong>'+esc(rl.evidence_answer)+'</strong>' : '')+'</p>'+
      (rl.confirmed_chunk_id ? '<p class="dtext">Confirmed chunk: '+esc(rl.confirmed_chunk_id)+' (rank '+esc(String(rl.confirmed_chunk_rank))+')</p>' : '')+
      (rl.note ? '<div class="quote">'+mdToHtml(rl.note)+'</div>' : '')+
    '</div>'+
    '<div class="dsection"><p class="dlabel">Answer</p><div class="quote">'+mdToHtml(r.answer)+'</div></div>'+
  '</div>';
}
function renderRlhf(){
  renderRlhfLegend();
  const d = DATA.live;
  const k = d.rlhf_kpi;
  document.getElementById('rlhf-kpis-review').innerHTML =
    '<div class="kpi c-red"><p class="label">Red</p><p class="what">Needs review now</p><p class="value">'+k.red_n+'</p><p class="sub">of '+k.total+' reviewed queries</p></div>'+
    '<div class="kpi c-amber"><p class="label">Amber</p><p class="what">Worth a look</p><p class="value">'+k.amber_n+'</p><p class="sub">of '+k.total+' reviewed queries</p></div>'+
    '<div class="kpi c-green"><p class="label">Green</p><p class="what">System working</p><p class="value">'+k.green_n+'</p><p class="sub">of '+k.total+' reviewed queries</p></div>';

  const reviewedRows = d.rows.filter(function(r){ return r.reviewed; });
  const rows = reviewedRows.filter(function(r){ return rlhfActiveColors.has(r.color || 'na'); });
  document.getElementById('rlhf-rows').innerHTML = rows.map(function(r,i){
    return '<div class="row" onclick="toggleRlhf('+i+')">'+
      '<div class="rowline"><div class="num">'+r.n+'</div><div class="q">'+esc(r.q)+'</div>'+
      '<div class="diag">'+rlhfChip(r)+'</div></div>'+
      rlhfDetailHTML(r,i)+
      '</div>';
  }).join('');
  document.getElementById('rlhf-footnote').textContent = rows.length + ' of ' + reviewedRows.length + ' reviewed queries shown.';
}
function toggleRlhf(i){
  _toggleDetail(document.getElementById('rd'+i));
}

function render(){
  renderLegend();
  renderTypeFilter();
  const d = DATA[source];
  document.getElementById('runinfo').textContent = d.run ? ('Run: '+d.run) : 'No run found';
  const k = d.kpi;
  document.getElementById('kpis').innerHTML =
    '<div class="kpi overall"><p class="label">Overall success</p><p class="what">All 3 layers passed</p><p class="value">'+k.overall_pct+'%</p><p class="sub">'+k.overall_n+' / '+k.total+'</p></div>'+
    '<div class="kpi"><p class="label">Level 1</p><p class="what">Did it call the right tool?</p><p class="value">'+k.l1_pct+'%</p><p class="sub">'+k.l1_n+' / '+k.total+'</p></div>'+
    '<div class="kpi"><p class="label">Level 2</p><p class="what">Did it use the right filters, chunk, or rule?</p><p class="value">'+k.l2_pct+'%</p><p class="sub">'+k.l2_n+' / '+k.total+'</p></div>'+
    '<div class="kpi"><p class="label">Level 3</p><p class="what">Was the final answer correct?</p><p class="value">'+k.l3_pct+'%</p><p class="sub">'+k.l3_n+' / '+k.total+'</p></div>';

  // Leaf-colour counts, at a glance - how much of this run actually
  // needs a human, not just what fraction passed layer 3.
  document.getElementById('kpis-review').innerHTML =
    '<div class="kpi c-red"><p class="label">Red</p><p class="what">Needs review now</p><p class="value">'+k.red_n+'</p><p class="sub">of '+k.total+' questions</p></div>'+
    '<div class="kpi c-amber"><p class="label">Amber</p><p class="what">Worth a look</p><p class="value">'+k.amber_n+'</p><p class="sub">of '+k.total+' questions</p></div>'+
    '<div class="kpi c-green"><p class="label">Green</p><p class="what">System working</p><p class="value">'+k.green_n+'</p><p class="sub">of '+k.total+' questions</p></div>';

  const rows = d.rows.filter(r => activeColors.has(r.color || 'na') && (activeType === 'all' || r.qtype === activeType));
  document.getElementById('rows').innerHTML = rows.map((r,i) =>
    '<div class="row" onclick="toggle('+i+')">'+
    '<div class="rowline"><div class="num">'+r.n+'</div><div class="q">'+esc(r.q)+'<span class="typepill">'+(r.qtype==='narrative'?'narrative':'lookup')+'</span></div><div class="lc">'+dot(r.l1)+'</div><div class="lc">'+dot(r.l2)+'</div><div class="lc">'+dot(r.l3)+'</div><div class="diag">'+chip(r.leaf, r.color)+'</div></div>'+
    detailHTML(r,i)+
    '</div>'
  ).join('');
  document.getElementById('footnote').textContent = rows.length + ' of ' + d.rows.length + ' questions shown.';
}

function toggle(i){
  _toggleDetail(document.getElementById('d'+i));
}

render();
</script>
</body>
</html>
""")


def build():
    agent = _load_source("agent_eval", "Test the agent (LLM)")
    system = _load_source("eval", "Test the system (regex)")
    live = _load_live()
    data_json = json.dumps({"agent": agent, "system": system, "live": live})
    html = _TEMPLATE.substitute(DATA_JSON=data_json, GOLDEN_TREE_SVG=golden_tree_svg(),
                                 LIVE_TREE_SVG=live_tree_svg(), RLHF_TREE_SVG=rlhf_tree_svg())
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  agent  run: {agent['run']}  ({len(agent['rows'])} questions)")
    print(f"  system run: {system['run']}  ({len(system['rows'])} questions)")
    print(f"  live queries: {len(live['rows'])} logged, {live['kpi']['reviewed_n']} reviewed")


if __name__ == "__main__":
    build()
