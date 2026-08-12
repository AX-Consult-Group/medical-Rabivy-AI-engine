# human_review_ui.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# The human half of the judge-vs-human comparison, adapted from Jürgen's
# rater_ui.py on the eval-label-loop branch. Same shape - stdlib-only
# HTTPServer, one case at a time, blind, append-only JSONL votes - but
# a DIFFERENT vote schema, because the task is different:
#
#   His rater_ui.py: pick which of several candidate CHUNKS best answers
#                     the question (multiple choice + "none of these").
#   This file:        given the SAME evidence (fresh + prior conversation)
#                     and the SAME answer judge_loop.py's judge was
#                     shown, is that answer SUPPORTED or UNSUPPORTED? A
#                     binary call, same vocabulary judge_loop.py uses,
#                     so its --report is comparing like for like.
#
# WHAT THE HUMAN IS NEVER SHOWN, ON PURPOSE (same principle as his
# design): the judge's verdict, the gate colour, or whether the judge
# agreed with the gates. Seeing any of those first would anchor the
# human's read instead of giving an independent second opinion.
#
# Kept as its own file (not folded into judge_loop.py's 2026-08-07
# consolidation) because it's a genuinely different kind of thing - a
# running HTTP server, not a one-shot script - same reason Jürgen kept
# rater_ui.py separate from label_loop.py.
#
# Run AFTER `python judge_loop.py --review` (that's what produces
# output/LABEL_LOOP/judge_review.jsonl, the case pool this reads from):
#   python human_review_ui.py          then open http://localhost:8019
#
# Port 8019 - chat_ui.py uses 8017, his rater_ui.py uses 8018.
#
# Votes append to output/LABEL_LOOP/human_votes.jsonl, one JSON object
# per vote: {query_id, rater: "human", verdict, voted_at}. Re-rating a
# case (via ?redo=query_id) appends another vote rather than editing in
# place; judge_loop.py's --report keeps only the LATEST vote per
# query_id, same supersede rule his label_loop.py uses for load_votes().
# =====================================================================

import html
import json
import os
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8019
CASES_PATH = os.path.join("output", "LABEL_LOOP", "judge_review.jsonl")
VOTES_PATH = os.path.join("output", "LABEL_LOOP", "human_votes.jsonl")

_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Rabivy - Live Judge Human Review</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; background: #f4f6f8; margin: 0; }}
 .wrap {{ max-width: 860px; margin: 24px auto; padding: 0 16px; }}
 .head {{ color: #1a3550; }} .head small {{ color: #7a8a99; font-weight: normal; }}
 .progress {{ color: #55707f; margin: 4px 0 18px 0; }}
 .qcard {{ background: #fff; border: 1px solid #d7dee5; border-radius: 8px;
          padding: 16px 20px; margin-bottom: 18px; }}
 .qcard h2 {{ margin: 0; font-size: 1.15em; color: #12293e; }}
 .block {{ background: #fff; border: 1px solid #d7dee5; border-radius: 8px;
          padding: 14px 18px; margin-bottom: 12px; }}
 .label {{ font-family: Consolas, monospace; font-size: 0.8em; color: #7a8a99;
         margin-bottom: 6px; text-transform: uppercase; }}
 .snippet {{ white-space: pre-wrap; color: #24343f; font-size: 0.95em; }}
 form {{ margin: 16px 0 0 0; display: flex; gap: 12px; }}
 button {{ border: 0; border-radius: 6px; padding: 10px 20px; cursor: pointer;
           font-size: 0.95em; color: #fff; }}
 .yes button {{ background: #1f6f4a; }} .yes button:hover {{ background: #17573a; }}
 .no button {{ background: #8a2f1f; }} .no button:hover {{ background: #6f2517; }}
 .done {{ background: #eef7f1; border: 1px solid #bcd9c6; border-radius: 8px;
          padding: 22px; color: #1f5138; }}
 .note {{ color: #7a8a99; font-size: 0.85em; margin-top: 22px; }}
</style></head><body><div class="wrap">
<h1 class="head">Live judge - human review <small>- is the answer actually
supported by the evidence?</small></h1>
{body}
<p class="note">Votes append to output/LABEL_LOOP/human_votes.jsonl. The judge's
own verdict on this same case is never shown here first - that's deliberate,
same reasoning as the eval-label-loop rater page.</p>
</div></body></html>"""


# =====================================================================
# Pure helpers - kept separate from the HTTP handler so they're testable
# without spinning up a server (see test_judge_loop.py).
# =====================================================================

def load_cases():
    if not os.path.exists(CASES_PATH):
        return []
    cases = []
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_human_votes():
    """{query_id: verdict}, keeping only the LATEST vote per query_id -
    same supersede rule as label_loop.load_votes() on eval-label-loop."""
    if not os.path.exists(VOTES_PATH):
        return {}
    latest = {}
    with open(VOTES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vote = json.loads(line)
            latest[vote["query_id"]] = vote["verdict"]  # later line overwrites earlier
    return latest


def append_vote(query_id, verdict):
    os.makedirs(os.path.dirname(VOTES_PATH), exist_ok=True)
    with open(VOTES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "query_id": query_id,
            "rater": "human",
            "verdict": verdict,
            "voted_at": datetime.now(timezone.utc).isoformat(),
        }) + "\n")


def pending_cases():
    cases = load_cases()
    voted = load_human_votes()
    return cases, [c for c in cases if c["query_id"] not in voted]


def render_case(case, done_n, total_n):
    body = (
        '<p class="progress">Case ' + str(done_n + 1) + ' of ' + str(total_n) + '</p>'
        '<div class="qcard"><h2>' + html.escape(case["question"]) + '</h2></div>'
        '<div class="block"><div class="label">Evidence shown to the original answer</div>'
        '<div class="snippet">' + html.escape(case["evidence_text"]) + '</div></div>'
        '<div class="block"><div class="label">Answer given</div>'
        '<div class="snippet">' + html.escape(case["given_answer"]) + '</div></div>'
        '<form method="POST" action="/vote">'
        '<input type="hidden" name="query_id" value="' + html.escape(case["query_id"]) + '">'
        '<div class="yes"><button type="submit" name="verdict" value="supported">'
        'Supported by the evidence</button></div>'
        '<div class="no"><button type="submit" name="verdict" value="unsupported">'
        'Not supported</button></div>'
        '</form>'
    )
    return _PAGE.format(body=body)


def render_done(total_n):
    body = ('<div class="done"><b>All ' + str(total_n) + ' cases reviewed - thank you.</b><br>'
            'Next: <code>python judge_loop.py --report</code> for judge-vs-human agreement.</div>')
    return _PAGE.format(body=body)


# =====================================================================
# HTTP handler
# =====================================================================

class HumanReviewHandler(BaseHTTPRequestHandler):

    def _send(self, page, code=200):
        data = page.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        cases, pending = pending_cases()
        qs = urllib.parse.urlparse(self.path).query
        redo = urllib.parse.parse_qs(qs).get("redo", [None])[0]
        if redo:
            match = [c for c in cases if c["query_id"] == redo]
            if match:
                return self._send(render_case(match[0], len(cases) - len(pending) - 1, len(cases)))
        if not pending:
            return self._send(render_done(len(cases)))
        self._send(render_case(pending[0], len(cases) - len(pending), len(cases)))

    def do_POST(self):
        if self.path != "/vote":
            return self._send("Not found", 404)
        length = int(self.headers.get("Content-Length", 0))
        form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
        query_id = form.get("query_id", [""])[0]
        verdict = form.get("verdict", [""])[0]
        cases = load_cases()
        valid = verdict in ("supported", "unsupported") and \
            any(c["query_id"] == query_id for c in cases)
        if not valid:
            return self._send("Invalid vote", 400)
        append_vote(query_id, verdict)
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, *args):
        pass  # quiet terminal - votes land in the JSONL, not stdout


def main():
    cases, pending = pending_cases()
    if not cases:
        print(f"No cases found at {CASES_PATH} - run judge_loop.py --review first.")
        return
    print(f"Human review - {len(cases)} case(s), {len(pending)} still unrated.")
    print(f"Open http://localhost:{PORT} to rate. Ctrl+C to stop.")
    HTTPServer(("127.0.0.1", PORT), HumanReviewHandler).serve_forever()


if __name__ == "__main__":
    main()
