# rater_ui.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# The HUMAN RATER's page for the eval label loop (eval_label_loop_spec.md)
# - the human half of the two-rater design in label_loop.py. Shows one
# case at a time: the question, the shuffled candidate chunks (no
# scores, no retriever names - anti-anchoring), a button per candidate,
# and an always-present "none of these". Each click appends one vote to
# output/label_votes.jsonl and advances to the next unrated case.
#
# WHAT THE RATER IS NEVER SHOWN, ON PURPOSE:
#   - whether a case is a gold item (planted known answer) or live -
#     gold items examine the RATER, and an exam you can recognize
#     measures nothing
#   - the judge's vote, or any retrieval score/rank
#
# Run AFTER `python label_loop.py --build-cases`:
#   python rater_ui.py          then open http://localhost:8018
#
# Same stdlib-only, no-dependency pattern as chat_ui.py. Votes are
# append-only; re-rating a case (via ?redo=case_id) supersedes the
# earlier vote (label_loop.load_votes keeps the latest per rater).
# =====================================================================

import html
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import label_loop as ll

PORT = 8018

_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Rabivy - Eval Label Rater</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; background: #f4f6f8; margin: 0; }}
 .wrap {{ max-width: 860px; margin: 24px auto; padding: 0 16px; }}
 .head {{ color: #1a3550; }} .head small {{ color: #7a8a99; font-weight: normal; }}
 .progress {{ color: #55707f; margin: 4px 0 18px 0; }}
 .qcard {{ background: #fff; border: 1px solid #d7dee5; border-radius: 8px;
          padding: 16px 20px; margin-bottom: 18px; }}
 .qcard h2 {{ margin: 0; font-size: 1.15em; color: #12293e; }}
 .cand {{ background: #fff; border: 1px solid #d7dee5; border-radius: 8px;
          padding: 14px 18px; margin-bottom: 12px; }}
 .cid {{ font-family: Consolas, monospace; font-size: 0.85em; color: #7a8a99;
         margin-bottom: 6px; }}
 .snippet {{ white-space: pre-wrap; color: #24343f; font-size: 0.95em; }}
 form {{ margin: 10px 0 0 0; }}
 button {{ background: #1f6f4a; color: #fff; border: 0; border-radius: 6px;
           padding: 8px 18px; cursor: pointer; font-size: 0.95em; }}
 button:hover {{ background: #17573a; }}
 .noneopt button {{ background: #8a5a1f; }} .noneopt button:hover {{ background: #6f4715; }}
 .done {{ background: #eef7f1; border: 1px solid #bcd9c6; border-radius: 8px;
          padding: 22px; color: #1f5138; }}
 .note {{ color: #7a8a99; font-size: 0.85em; margin-top: 22px; }}
</style></head><body><div class="wrap">
<h1 class="head">Eval label rater <small>- choose the chunk that best answers the question</small></h1>
{body}
<p class="note">Votes append to output/label_votes.jsonl. Your choice is never shown
the judge's vote first, and candidates are in randomized order with no scores -
that's deliberate. If no candidate genuinely answers the question, say so:
"none of these" is a real, valuable answer, not a failure.</p>
</div></body></html>"""


def _pending_cases():
    cases = ll.load_cases()
    votes = ll.load_votes()
    return cases, [c for c in cases if (c["case_id"], "human") not in votes]


def _render_case(case, done_n, total_n):
    cand_html = ""
    for c in case["candidates"]:
        cand_html += (
            '<div class="cand"><div class="cid">' + html.escape(c["chunk_id"]) + '</div>'
            '<div class="snippet">' + html.escape(c["snippet"]) + '</div>'
            '<form method="POST" action="/vote">'
            '<input type="hidden" name="case_id" value="' + html.escape(case["case_id"]) + '">'
            '<input type="hidden" name="choice" value="' + html.escape(c["chunk_id"]) + '">'
            '<button type="submit">This one answers it best</button></form></div>')
    none_html = (
        '<div class="cand noneopt"><div class="cid">none of these</div>'
        '<div class="snippet">No candidate above genuinely answers the question.</div>'
        '<form method="POST" action="/vote">'
        '<input type="hidden" name="case_id" value="' + html.escape(case["case_id"]) + '">'
        '<input type="hidden" name="choice" value="' + ll.NONE_OF_THESE + '">'
        '<button type="submit">None of these</button></form></div>')
    body = ('<p class="progress">Case ' + str(done_n + 1) + ' of ' + str(total_n) + '</p>'
            '<div class="qcard"><h2>' + html.escape(case["question"]) + '</h2></div>'
            + cand_html + none_html)
    return _PAGE.format(body=body)


def _render_done(total_n):
    body = ('<div class="done"><b>All ' + str(total_n) + ' cases rated - thank you.</b><br>'
            'Next: <code>python label_loop.py --judge</code> (if not yet run), then '
            '<code>python label_loop.py --report</code> for kappa, gold accuracy and '
            'the adjudication queue.</div>')
    return _PAGE.format(body=body)


class RaterHandler(BaseHTTPRequestHandler):

    def _send(self, page, code=200):
        data = page.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        cases, pending = _pending_cases()
        qs = urllib.parse.urlparse(self.path).query
        redo = urllib.parse.parse_qs(qs).get("redo", [None])[0]
        if redo:
            match = [c for c in cases if c["case_id"] == redo]
            if match:
                return self._send(_render_case(match[0], len(cases) - len(pending) - 1,
                                               len(cases)))
        if not pending:
            return self._send(_render_done(len(cases)))
        self._send(_render_case(pending[0], len(cases) - len(pending), len(cases)))

    def do_POST(self):
        if self.path != "/vote":
            return self._send("Not found", 404)
        length = int(self.headers.get("Content-Length", 0))
        form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
        case_id = form.get("case_id", [""])[0]
        choice = form.get("choice", [""])[0]
        cases = ll.load_cases()
        case = next((c for c in cases if c["case_id"] == case_id), None)
        valid = case and (choice == ll.NONE_OF_THESE or
                          choice in {c["chunk_id"] for c in case["candidates"]})
        if not valid:
            return self._send("Invalid vote", 400)
        ll.append_vote(case_id, "human", choice)
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, *args):
        pass  # keep the terminal quiet - votes land in the JSONL, not stdout


def main():
    cases, pending = _pending_cases()
    print(f"Eval label rater - {len(cases)} case(s), {len(pending)} still unrated.")
    print(f"Open http://localhost:{PORT} to rate. Ctrl+C to stop.")
    HTTPServer(("127.0.0.1", PORT), RaterHandler).serve_forever()


if __name__ == "__main__":
    main()
