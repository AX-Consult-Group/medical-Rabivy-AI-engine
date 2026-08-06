# chat_ui.py
# -------------------------------------------------------------------
# Phase 6: the conversational interface - a local web console on top of
# agent.py. Not just a chat: every response ships with its full trace,
# so the pipeline is inspectable per answer:
#
#   GATES      - a per-answer checklist: tool grounding, retrieval
#                confidence, source citations, and the hallucination
#                audit (including whether a draft was auto-revised)
#   TRACE      - every tool call with its exact parameters and the
#                evidence it returned, plus the audit verdicts and,
#                when a revision fired, the rejected draft
#   QUARANTINE - human-in-the-loop exception handling: an answer whose
#                final audit is not a clean pass is WITHHELD. The
#                reviewer sees the auditor's objections and the full
#                evidence, and releases or rejects the answer
#                explicitly. Decisions are logged to
#                output/review_log.jsonl as the governance trail.
#   QUERY LOG  - every single call to /ask writes one line to
#                output/QUERY_LOG/query_log.jsonl: the question, the
#                answer, every gate result, the full audit trail, and
#                the trimmed evidence - whether the answer was
#                delivered, quarantined, or the request errored out.
#                Nothing that comes in is allowed to go un-logged.
#                Each entry gets a query_id, which is how an RLHF
#                rating (below) is tied back to the exact query it's
#                rating.
#   RLHF       - under every answer the user actually sees (delivered,
#                or a quarantined answer once a reviewer releases it):
#                first "was this correct?" (correct / partially correct
#                / incorrect / vague - didn't really answer), then one
#                follow-up - "was the right info actually in the
#                evidence shown?" (yes / no / I don't know). Combined
#                with whether the answer's OWN WORDING reads as a
#                decline (_is_decline, imported from build_dashboard.py
#                - the exact same check the golden test set tree uses),
#                this reconstructs which of the golden tree's 6 leaves
#                (Correct answer / Synthesis issue / Correct anyway /
#                Correctly rejected / Retrieval fail / Hallucination) -
#                or the two extra live-only outcomes, "Vague / non-
#                answer" and "flagged, evidence unconfirmed" - this
#                answer most likely belongs to, from a rep's own
#                judgement, without needing ground truth. See
#                _rlhf_leaf() below for the exact mapping.
#                Every rating writes ONE self-contained line to
#                output/RLHF_FEEDBACK/rlhf_log.jsonl - the full answer,
#                gates, evidence trace and audit trail are copied in
#                alongside the rating (not just a query_id pointer), so
#                one record has everything needed to understand the
#                judgement without cross-referencing the query log.
#
#   All three logs (query, RLHF, human review) live under output/ in
#   their own clearly named ALL-CAPS subfolder - output/QUERY_LOG/,
#   output/RLHF_FEEDBACK/, output/HUMAN_REVIEW/ - rather than as loose
#   files, so output/ stays scannable as it grows.
#
#   python chat_ui.py          -> open http://localhost:8017
#
# Standard library only; one RabivyAgent instance for the whole server
# session, so conversation memory works across messages. Local use only
# (binds 127.0.0.1) - a networked deployment would need auth and TLS.
# -------------------------------------------------------------------

import json
import os
import re
import threading
import time
import uuid
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

from agent import RabivyAgent
# Reused, not reimplemented - the golden test set tree's own "does this
# answer's wording read as a decline" check, so live feedback and
# golden results agree on what a decline even means. build_dashboard.py
# only imports json/os/re/string, so this doesn't pull in anything
# heavy.
from build_dashboard import _is_decline

REVIEW_LOG = os.path.join("output", "HUMAN_REVIEW", "review_log.jsonl")
QUERY_LOG = os.path.join("output", "QUERY_LOG", "query_log.jsonl")
RLHF_LOG = os.path.join("output", "RLHF_FEEDBACK", "rlhf_log.jsonl")

PORT = 8017

_agent = RabivyAgent()
_agent_lock = threading.Lock()

_CITE = re.compile(r"\[hcp_table\]|\[doc:\s*[^\]]+\]")


def _evidence_view(evidence):
    """Trim tool results for the browser: enough to inspect, not the
    full corpus. Structure is preserved; long text is clipped.

    2026-08-05: the 500-char clip on chunk text used to cut off most
    real chunks - checked against output/chunks_tagged.json directly,
    the median chunk is ~1,760 chars and the longest is 3,492, so 500
    was throwing away the majority of almost every chunk shown here.
    Raised to 4000 (comfortably above the real max) so this is
    effectively "don't truncate" in practice, with the CSS giving the
    display box a scrollable max-height instead so the page doesn't
    balloon. IMPORTANT: this ONLY affects what's shown in the browser -
    the agent already received the full chunk text when it answered
    (agent_tools.py hands the whole thing to the LLM); this function
    runs AFTER that call, purely for the human-facing trace view. So
    raising this costs nothing extra in tokens or API spend.
    """
    out = []
    for ev in evidence:
        res = ev.get("result", {})
        view = {"tool": ev["tool"], "input": ev.get("input", {})}
        if "rows" in res:
            view["summary"] = f"{res.get('total_matches')} matches in table, {res.get('returned')} returned"
            view["rows"] = res["rows"][:5]
            view["more_rows"] = max(0, res.get("returned", 0) - 5)
        elif "sections" in res:
            view["summary"] = (f"route: {res.get('retrieval_route')}"
                               + ("  |  LOW CONFIDENCE" if res.get("low_confidence") else ""))
            view["low_confidence"] = bool(res.get("low_confidence"))
            view["sections"] = [
                {"chunk_id": s["chunk_id"], "similarity": s.get("similarity"),
                 "text": (s.get("text") or "")[:4000]}
                for s in res["sections"]
            ]
        elif "row" in res:
            view["summary"] = f"NPI {res['row'].get('npi')} - table row" + (" + snapshot card" if "snapshot_card" in res else "")
            view["rows"] = [res["row"]]
            if "snapshot_card" in res:
                view["sections"] = [{"chunk_id": res["snapshot_card"]["chunk_id"],
                                     "similarity": None,
                                     "text": res["snapshot_card"]["text"][:4000]}]
        elif "error" in res:
            view["summary"] = f"ERROR: {res['error']}"
            view["error"] = True
        else:
            view["summary"] = json.dumps(res)[:200]
        out.append(view)
    return out


def _gates(result, evidence_view):
    """The per-answer security-gate checklist shown above the trace."""
    n_rows = sum(len(v.get("rows", [])) + v.get("more_rows", 0) for v in evidence_view)
    n_docs = sum(len(v.get("sections", [])) for v in evidence_view)
    low_conf = [v for v in evidence_view if v.get("low_confidence")]
    errors = [v for v in evidence_view if v.get("error")]
    citations = len(_CITE.findall(result["answer"]))
    verdict = (result.get("verification") or {}).get("verdict")

    gates = []
    gates.append({
        "name": "Tool grounding",
        "ok": bool(evidence_view),
        "detail": (f"{len(evidence_view)} tool call(s): {n_rows} table row(s), {n_docs} document section(s) retrieved"
                   if evidence_view else "No tools were called - answer is not evidence-based"),
    })
    # BUG FIX: this used to be `not low_conf and not errors`, which is a
    # vacuous truth when NO tool was called at all - both list
    # comprehensions come back empty, so "no low-confidence retrievals"
    # and "no tool errors" were both trivially true, and this gate
    # showed a clean PASS even though nothing was ever retrieved to be
    # confident about. Requiring evidence_view to be non-empty first
    # closes that gap - a query with zero tool calls now correctly
    # fails this gate too, instead of contradicting "Tool grounding"
    # right next to it.
    gates.append({
        "name": "Retrieval confidence",
        "ok": bool(evidence_view) and not low_conf and not errors,
        "detail": ("All retrievals confident, no tool errors" if evidence_view and not low_conf and not errors
                   else "No tools were called - nothing was retrieved to be confident about" if not evidence_view
                   else f"{len(low_conf)} low-confidence retrieval(s), {len(errors)} tool error(s) - see trace"),
    })
    gates.append({
        "name": "Source citations",
        "ok": citations > 0,
        "detail": f"{citations} inline citation(s) in the answer" if citations else "No inline citations found",
    })
    if verdict == "pass":
        detail = "Every claim checked against retrieved evidence: PASS"
        if result.get("revised"):
            detail = "Draft REJECTED by audit -> auto-revised -> re-audit: PASS"
        gates.append({"name": "Hallucination audit", "ok": True, "detail": detail})
    elif verdict == "fail":
        issues = (result.get("verification") or {}).get("issues", [])
        gates.append({"name": "Hallucination audit", "ok": False,
                      "detail": f"FAIL - {len(issues)} unresolved issue(s), see audit detail"})
    else:
        gates.append({"name": "Hallucination audit", "ok": None,
                      "detail": f"Not performed ({verdict}) - offline mode has no auditor"})
    return gates


def _append_jsonl(path, entry):
    """Adds one JSON object as a new line to a log file, creating
    output/ if it doesn't exist yet. Append-only, same pattern as
    REVIEW_LOG already used - a log is a record, never edited in
    place."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _rlhf_leaf(rating, evidence_answer, is_decline):
    """Maps a rep's (rating, evidence_answer) pair, plus whether the
    answer's own wording reads as a decline, onto the SAME leaves
    build_dashboard.py's classify() uses for the golden test set - so
    live feedback and golden results end up filed under one shared
    vocabulary, not two different ones.

    Question order here is the OPPOSITE of the golden tree's: the
    tree asks "was evidence used" first, then implicitly checks "was
    the answer correct". Here, correctness is asked first (the
    existing rating buttons), evidence second - but it's the same
    2x2 combination either way, so it lands on the same leaf
    regardless of which axis was asked first.

    Returns (leaf_name, color). color is None for the provisional
    "flagged for review" cases - not a real leaf, a signal this needs
    a human look rather than a guess."""
    if rating == "vague":
        return "Vague / non-answer", "amber"
    if evidence_answer not in ("yes", "no"):
        return f"{rating} - evidence unconfirmed, flagged for review", None
    correct = rating == "correct"
    evidence_ok = evidence_answer == "yes"
    if evidence_ok and correct:
        return "Correct answer", "green"
    if evidence_ok and not correct:
        return "Synthesis issue", "red"
    if not evidence_ok and correct:
        return ("Correctly rejected", "green") if is_decline else ("Correct anyway", "amber")
    return ("Retrieval fail", "amber") if is_decline else ("Hallucination", "red")


def _new_query_id():
    """A short, sortable-enough id for one /ask call. Not a full UUID
    on purpose - this shows up in the browser's network tab and in the
    RLHF POST body, so it stays readable."""
    return time.strftime("q_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]


def _log_query(query_id, question, status, result=None, gates=None,
               evidence_view=None, error=None):
    """Writes one line to output/QUERY_LOG/query_log.jsonl for every /ask call,
    no matter how it ends up: delivered straight away, held in
    quarantine, or blown up with a server error. status is one of
    "delivered", "quarantined", or "error" - the same three outcomes
    the browser already shows the user, just persisted this time
    instead of only ever living in one browser tab."""
    entry = {
        "query_id": query_id,
        "asked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "status": status,
        "answer": (result or {}).get("answer") if result else None,
        "verdict": (result.get("verification") or {}).get("verdict") if result else None,
        "revised": (result or {}).get("revised") if result else None,
        "gates": gates or [],
        "audit_trail": (result or {}).get("audit_trail", []) if result else [],
        "evidence": evidence_view or [],
        "error": error,
    }
    _append_jsonl(QUERY_LOG, entry)


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rabivy Intelligence Console</title>
<style>
  :root{--ink:#1a2332;--muted:#5b6779;--line:#dbe2ea;--brand:#7c3aed;--brand-bg:#f5f0ff;
        --ok:#047857;--ok-bg:#eefaf3;--bad:#b91c1c;--bad-bg:#fdecec;--warn:#b45309;--warn-bg:#fdf6e9;
        --user-bg:#eff4ff;--mono:ui-monospace,Consolas,monospace}
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);
       background:#f4f6fa;display:flex;flex-direction:column;height:100vh}
  header{background:#fff;border-bottom:1px solid var(--line);padding:13px 22px}
  header h1{font-size:16.5px;margin:0}
  header .sub{font-size:12px;color:var(--muted);margin-top:2px}
  #chat{flex:1;overflow-y:auto;padding:20px;max-width:980px;width:100%;margin:0 auto}
  .msg{margin-bottom:18px}
  .who{font-size:11px;color:var(--muted);margin-bottom:3px;text-transform:uppercase;letter-spacing:.05em}
  .bubble{padding:12px 16px;border-radius:12px;font-size:13.5px;line-height:1.5;background:#fff;border:1px solid var(--line)}
  .user .bubble{background:var(--user-bg);border-color:#bcd0f7;max-width:80%;margin-left:auto}
  .user .who{text-align:right}
  .bubble h3{font-size:14px;margin:10px 0 4px}
  .bubble ul{margin:6px 0;padding-left:20px}
  .bubble code{background:#eef1f6;border-radius:4px;padding:0 4px;font-size:12px}
  /* gates */
  .gates{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px;margin-top:10px}
  .gate{border:1px solid var(--line);border-radius:10px;padding:8px 11px;background:#fff;cursor:default}
  .gate .g-name{font-size:11.5px;font-weight:700;display:flex;align-items:center;gap:6px}
  .gate .g-detail{font-size:11px;color:var(--muted);margin-top:3px;line-height:1.4}
  .gate.ok{border-color:#9fd9be;background:var(--ok-bg)} .gate.ok .g-name{color:var(--ok)}
  .gate.bad{border-color:#f0b1b1;background:var(--bad-bg)} .gate.bad .g-name{color:var(--bad)}
  .gate.na{border-color:var(--line)} .gate.na .g-name{color:var(--muted)}
  /* trace */
  details{margin-top:8px;border:1px solid var(--line);border-radius:10px;background:#fff}
  details summary{cursor:pointer;padding:8px 12px;font-size:12px;font-weight:700;color:var(--brand);user-select:none}
  details[open] summary{border-bottom:1px solid var(--line)}
  .trace-body{padding:10px 14px;font-size:12px}
  .tcall{margin-bottom:12px}
  .tname{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--brand)}
  .tinput{font-family:var(--mono);font-size:11px;color:var(--muted);word-break:break-all;margin:2px 0 4px}
  .tsummary{font-size:11.5px;margin-bottom:5px}
  .lowconf{color:var(--warn);font-weight:700}
  table.rows{border-collapse:collapse;font-size:10.5px;margin:4px 0;width:100%}
  table.rows th,table.rows td{border:1px solid var(--line);padding:2px 6px;text-align:left}
  table.rows th{background:#f7f9fc}
  .chunk{border-left:3px solid var(--brand);background:var(--brand-bg);border-radius:0 8px 8px 0;
         padding:6px 10px;margin:5px 0;font-size:11.5px}
  .chunk .cid{font-family:var(--mono);font-size:10.5px;color:var(--brand);font-weight:700}
  .chunk .sim{color:var(--muted);font-size:10.5px}
  .chunk .ctext{max-height:220px;overflow-y:auto;margin-top:4px}
  .chunkpick{display:flex;flex-direction:column;gap:6px;margin:6px 0}
  .chunkpick-opt{text-align:left;border:1.5px solid var(--line);border-radius:8px;background:#fff;
                 padding:7px 10px;font-size:11.5px;cursor:pointer}
  .chunkpick-opt:hover{border-color:var(--brand)}
  .chunkpick-opt .cid{font-family:var(--mono);font-size:10.5px;color:var(--brand);font-weight:700;display:block}
  .chunkpick-opt .prev{color:var(--muted);font-size:11px;margin-top:2px}
  .audit-round{padding:6px 10px;border-radius:8px;margin:5px 0;font-size:11.5px}
  .audit-round.pass{background:var(--ok-bg)} .audit-round.fail{background:var(--bad-bg)}
  .draft{background:#fbfbfd;border:1px dashed var(--line);border-radius:8px;padding:8px 12px;
         font-size:12px;color:var(--muted);margin-top:5px;white-space:pre-wrap}
  /* quarantine */
  .qcard{border:2px solid #f0b1b1;border-radius:12px;background:var(--bad-bg);padding:14px 16px}
  .qtitle{font-size:13.5px;font-weight:800;color:var(--bad)}
  .qsub{font-size:12px;color:var(--muted);margin:4px 0 8px}
  .qissue{font-size:12px;background:#fff;border:1px solid #f0b1b1;border-radius:8px;padding:6px 10px;margin:4px 0}
  .qbtns{display:flex;gap:10px;margin-top:10px}
  .qbtn{padding:8px 16px;font-size:12.5px;font-weight:700;border-radius:8px;border:1.5px solid;cursor:pointer}
  .qapprove{background:#fff;border-color:#9fd9be;color:var(--ok)}
  .qreject{background:#fff;border-color:#f0b1b1;color:var(--bad)}
  .qdecided{font-size:12px;font-weight:700;margin-top:8px}
  /* rlhf */
  .rlhf{margin-top:10px;padding-top:10px;border-top:1px solid var(--line)}
  .rlhf-label{font-size:11px;color:var(--muted);margin-bottom:6px}
  .rlhf-btns{display:flex;gap:8px;flex-wrap:wrap}
  .rlhf-btn{padding:6px 12px;font-size:11.5px;font-weight:600;border-radius:20px;
            border:1.5px solid var(--line);background:#fff;cursor:pointer;color:var(--muted)}
  .rlhf-btn:hover{border-color:var(--brand);color:var(--brand)}
  .rlhf-step1{display:none;margin-top:10px}
  .rlhf-note{width:100%;margin-top:8px;display:none}
  .rlhf-note textarea{width:100%;font:inherit;font-size:12px;padding:8px;border:1.5px solid var(--line);
                       border-radius:8px;resize:vertical;min-height:50px;box-sizing:border-box}
  .rlhf-note button{margin-top:6px;padding:5px 14px;font-size:11.5px;font-weight:600;color:#fff;
                     background:var(--brand);border:none;border-radius:8px;cursor:pointer}
  .rlhf-done{font-size:11.5px;font-weight:700;color:var(--ok)}
  .thinking{color:var(--muted);font-size:13px}
  .dots::after{content:'';animation:d 1.2s infinite}
  @keyframes d{0%{content:'.'}33%{content:'..'}66%{content:'...'}}
  footer{background:#fff;border-top:1px solid var(--line);padding:13px 22px}
  .inputrow{max-width:980px;margin:0 auto;display:flex;gap:10px}
  #q{flex:1;padding:12px 14px;font-size:14px;border:1.5px solid var(--line);border-radius:10px;outline:none}
  #q:focus{border-color:var(--brand)}
  #send{padding:12px 22px;font-size:14px;font-weight:600;color:#fff;background:var(--brand);border:none;border-radius:10px;cursor:pointer}
  #send:disabled{opacity:.5}
</style>
</head>
<body>
<header>
  <h1>Rabivy Intelligence Console</h1>
  <div class="sub">Agentic RAG with full-trace transparency &middot; synthetic data &middot; every answer shows its evidence and the gates it cleared &middot; unverified answers are held for human review</div>
</header>
<div id="chat"></div>
<footer>
  <div class="inputrow">
    <input id="q" placeholder="e.g. Who should I target next month in New York, and what should I say to them?" autocomplete="off">
    <button id="send">Ask</button>
  </div>
</footer>
<script>
const chat=document.getElementById('chat'),q=document.getElementById('q'),send=document.getElementById('send');
function esc(t){return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function md(t){
  t=esc(t); t=t.replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
             .replace(/^#{2,3}\s*(.+)$/gm,'<h3>$1</h3>');
  const L=t.split('\n');let o=[],ul=false;
  for(const ln of L){
    if(/^\s*[-*]\s+/.test(ln)){if(!ul){o.push('<ul>');ul=true;}o.push('<li>'+ln.replace(/^\s*[-*]\s+/,'')+'</li>');}
    else{if(ul){o.push('</ul>');ul=false;}o.push(ln.trim()===''?'':'<p>'+ln+'</p>');}}
  if(ul)o.push('</ul>');return o.join('');
}
function el(html){const d=document.createElement('div');d.innerHTML=html;return d.firstElementChild;}
function addUser(text){
  chat.appendChild(el('<div class="msg user"><div class="who">You</div><div class="bubble">'+md(text)+'</div></div>'));
  chat.scrollTop=chat.scrollHeight;
}
function gateHtml(g){
  const cls=g.ok===true?'ok':(g.ok===false?'bad':'na');
  const mark=g.ok===true?'&#10003;':(g.ok===false?'&#10007;':'&#8210;');
  return '<div class="gate '+cls+'"><div class="g-name">'+mark+' '+esc(g.name)+'</div><div class="g-detail">'+esc(g.detail)+'</div></div>';
}
function rowsTable(rows){
  if(!rows||!rows.length)return '';
  const cols=['npi','specialty','state','tier','propensity_score','rx_volume_monthly','switching_score','targeted'];
  const use=cols.filter(c=>rows[0][c]!==undefined);
  let h='<table class="rows"><tr>'+use.map(c=>'<th>'+c+'</th>').join('')+'</tr>';
  for(const r of rows){h+='<tr>'+use.map(c=>'<td>'+esc(String(r[c]))+'</td>').join('')+'</tr>';}
  return h+'</table>';
}
function traceHtml(d){
  let h='';
  for(const v of d.trace){
    h+='<div class="tcall"><span class="tname">'+esc(v.tool)+'</span>';
    h+='<div class="tinput">'+esc(JSON.stringify(v.input))+'</div>';
    h+='<div class="tsummary'+(v.low_confidence?' lowconf':'')+'">'+esc(v.summary)+'</div>';
    h+=rowsTable(v.rows);
    if(v.more_rows)h+='<div class="tsummary">&hellip; and '+v.more_rows+' more row(s)</div>';
    for(const s of (v.sections||[])){
      const cut = (s.text||'').length >= 4000;
      h+='<div class="chunk"><span class="cid">'+esc(s.chunk_id)+'</span>'
        +(s.similarity!=null?' <span class="sim">similarity '+s.similarity+'</span>':'')
        +'<div class="ctext">'+esc(s.text)+(cut?'&hellip;':'')+'</div></div>';
    }
    h+='</div>';
  }
  return h||'<div class="tsummary">No tool calls (conversational turn).</div>';
}
function auditHtml(d){
  let h='';
  for(const a of (d.audit_trail||[])){
    const cls=a.verdict==='pass'?'pass':'fail';
    h+='<div class="audit-round '+cls+'"><strong>'+esc(a.stage)+' audit: '+esc(String(a.verdict).toUpperCase())+'</strong>';
    for(const i of (a.issues||[]))h+='<div>&bull; '+esc(String(i))+'</div>';
    h+='</div>';
  }
  if(d.draft_answer)h+='<div class="tsummary" style="margin-top:6px"><strong>Rejected draft (before revision):</strong></div><div class="draft">'+esc(d.draft_answer)+'</div>';
  return h||'<div class="tsummary">No audit rounds recorded.</div>';
}
function detailsHtml(d){
  return '<details><summary>Evidence trace &mdash; '+d.trace.length+' tool call(s)</summary><div class="trace-body">'+traceHtml(d)+'</div></details>'
       + '<details><summary>Audit detail</summary><div class="trace-body">'+auditHtml(d)+'</div></details>';
}
function logReview(question,d,decision){
  const last=(d.audit_trail||[]).slice(-1)[0]||{};
  fetch('/review',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:question,verdict:d.verdict,decision:decision,
                         issues:(last.issues||[]).map(String)})}).catch(()=>{});
}
// Two DIFFERENT question sets, picked by isDecline (from the server's
// /ask response - see _is_decline in build_dashboard.py). Reasoning:
// when the system declined/said "I don't know", asking "was this
// answer correct?" then "was the right info in the evidence?" is
// confusing - there's usually no evidence at all to judge, so "yes"
// could mean either "yes it was right to decline" or "yes the info
// was there" depending how the rep reads it. A real rep hit exactly
// this: rated a decline "correct" then answered the evidence question
// "yes" even though no tool/evidence was ever used, which silently
// misfiled it as "Correct answer" instead of "Correctly rejected".
// The decline branch below asks ONE direct question instead, and
// always sends evidence_answer:'no' under the hood (nothing was used,
// so there's nothing to confirm) - _rlhf_leaf's existing logic then
// lands it on "Correctly rejected" or "Retrieval fail" automatically.
function rlhfHtml(isDecline){
  if(isDecline){
    return '<div class="rlhf">'
      +'<div class="rlhf-decline"><div class="rlhf-label">The system said it couldn&rsquo;t answer this. Was that the right call?</div>'
      +'<div class="rlhf-btns">'
      +'<button class="rlhf-btn" data-decline="correct">&#10003; Yes &mdash; correct, no answer exists</button>'
      +'<button class="rlhf-btn" data-decline="incorrect">&#10007; No &mdash; an answer does exist, it should have found it</button>'
      +'<button class="rlhf-btn" data-decline="unsure">Not sure</button>'
      +'</div></div>'
      +'<div class="rlhf-note"><textarea placeholder="Anything else worth noting? (optional)"></textarea>'
      +'<button class="rlhf-submit">Submit</button></div></div>';
  }
  return '<div class="rlhf">'
    +'<div class="rlhf-step0"><div class="rlhf-label">Was this answer correct?</div>'
    +'<div class="rlhf-btns">'
    +'<button class="rlhf-btn" data-rating="correct">&#10003; Correct</button>'
    +'<button class="rlhf-btn" data-rating="partial">&#8211; Partially correct</button>'
    +'<button class="rlhf-btn" data-rating="incorrect">&#10007; Incorrect</button>'
    +'<button class="rlhf-btn" data-rating="vague">Vague &mdash; didn&rsquo;t really answer</button>'
    +'</div></div>'
    +'<div class="rlhf-step1"><div class="rlhf-label">Was the right info actually in the evidence or sources shown?</div>'
    +'<div class="rlhf-btns">'
    +'<button class="rlhf-btn" data-ev="yes">Yes</button>'
    +'<button class="rlhf-btn" data-ev="no">No</button>'
    +'<button class="rlhf-btn" data-ev="idk">I don&rsquo;t know</button>'
    +'</div></div>'
    +'<div class="rlhf-chunkpick" style="display:none"></div>'
    +'<div class="rlhf-note"><textarea placeholder="Anything else worth noting? (optional)"></textarea>'
    +'<button class="rlhf-submit">Submit</button></div></div>';
}
// Only ever shown when the rep says "Yes" AND there's at least one
// retrieved chunk to pick from (a structured-lookup-only answer has
// no chunks, so this step is skipped for those - nothing to point at).
// Recording WHICH chunk, and its rank among what was retrieved, is the
// live-traffic equivalent of the golden test set's own rank signal
// (see build_dashboard.py's classify() - rank 1 vs ranked-lower is
// exactly this same idea, just confirmed by ground truth there instead
// of a human).
function allSections(trace){
  const out=[];
  for(const v of (trace||[])) for(const s of (v.sections||[])) out.push(s);
  return out;
}
function chunkPickHtml(sections){
  let h='<div class="rlhf-label">Which chunk had the right info?</div><div class="chunkpick">';
  sections.forEach((s,i)=>{
    h+='<button class="chunkpick-opt" data-idx="'+i+'"><span class="cid">#'+(i+1)+' '+esc(s.chunk_id)+'</span>'
      +'<span class="prev">'+esc((s.text||'').slice(0,140))+'&hellip;</span></button>';
  });
  h+='<button class="chunkpick-opt" data-idx="-1" style="color:var(--muted)">Not sure which one</button></div>';
  return h;
}
// The browser already has the FULL answer (d) in memory from the
// original /ask response - passed straight through here so the RLHF
// log entry is self-contained (answer, gates, evidence, audit trail)
// rather than a bare query_id the reviewer has to go join elsewhere.
function submitRlhf(queryId,question,rating,evidenceAnswer,confirmedChunkId,confirmedChunkRank,note,box,d){
  fetch('/rlhf',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({query_id:queryId,question:question,rating:rating,evidence_answer:evidenceAnswer,
                         confirmed_chunk_id:confirmedChunkId,confirmed_chunk_rank:confirmedChunkRank,
                         note:note||'',answer:d.answer,gates:d.gates,trace:d.trace,
                         audit_trail:d.audit_trail,draft_answer:d.draft_answer,
                         verdict:d.verdict,quarantined:d.quarantined})})
    .then(r=>r.json()).then(res=>{
      box.outerHTML='<div class="rlhf"><div class="rlhf-done">&#10003; Feedback recorded &mdash; logged as: '
        +esc(res.leaf||'unclassified')+'</div></div>';
    }).catch(()=>{
      box.outerHTML='<div class="rlhf"><div class="rlhf-done">&#10003; Feedback recorded</div></div>';
    });
}
// Every query the agent answers gets a query_id from the server (see
// _new_query_id in chat_ui.py); wireRlhf is what ties a rating back to
// that exact query. Two clicks (was it correct, was the evidence
// there), then an optional note, mirrors the same two axes the golden
// test set decision tree uses - see _rlhf_leaf() in chat_ui.py for how
// the two answers get combined into one of its leaves.
function wireRlhf(node,queryId,question,d){
  const box=node.querySelector('.rlhf');
  if(!box)return;
  const noteBox=box.querySelector('.rlhf-note');
  let rating=null, evidenceAnswer=null, confirmedChunkId=null, confirmedChunkRank=null;

  const declineStep=box.querySelector('.rlhf-decline');
  if(declineStep){
    // Decline flow: one click, no evidence question at all - evidence
    // is always sent as 'no' since a decline means nothing was used,
    // so there's nothing for the rep to confirm. See rlhfHtml() above.
    const DECLINE_MAP={correct:['correct','no'],incorrect:['incorrect','no'],unsure:['partial','idk']};
    declineStep.querySelectorAll('.rlhf-btn').forEach(btn=>{
      btn.onclick=()=>{
        const pair=DECLINE_MAP[btn.dataset.decline];
        rating=pair[0]; evidenceAnswer=pair[1];
        declineStep.style.display='none';
        noteBox.style.display='block';
      };
    });
  } else {
    const step0=box.querySelector('.rlhf-step0'), step1=box.querySelector('.rlhf-step1'),
          chunkPick=box.querySelector('.rlhf-chunkpick');
    step0.querySelectorAll('.rlhf-btn').forEach(btn=>{
      btn.onclick=()=>{
        rating=btn.dataset.rating;
        step0.style.display='none';
        step1.style.display='block';
      };
    });
    step1.querySelectorAll('.rlhf-btn').forEach(btn=>{
      btn.onclick=()=>{
        evidenceAnswer=btn.dataset.ev;
        step1.style.display='none';
        const sections = evidenceAnswer==='yes' ? allSections(d.trace) : [];
        if(sections.length){
          chunkPick.innerHTML=chunkPickHtml(sections);
          chunkPick.style.display='block';
          chunkPick.querySelectorAll('.chunkpick-opt').forEach(opt=>{
            opt.onclick=()=>{
              const idx=parseInt(opt.dataset.idx,10);
              if(idx>=0){ confirmedChunkId=sections[idx].chunk_id; confirmedChunkRank=idx+1; }
              chunkPick.style.display='none';
              noteBox.style.display='block';
            };
          });
        } else {
          noteBox.style.display='block';
        }
      };
    });
  }

  noteBox.querySelector('.rlhf-submit').onclick=()=>{
    const note=noteBox.querySelector('textarea').value.trim();
    submitRlhf(queryId,question,rating,evidenceAnswer,confirmedChunkId,confirmedChunkRank,note,box,d);
  };
}
function addAnswer(d,question){
  if(d.quarantined){ addQuarantined(d,question); return; }
  let h='<div class="msg bot"><div class="who">Assistant</div><div class="bubble">'+md(d.answer);
  h+='<div class="gates">'+d.gates.map(gateHtml).join('')+'</div>';
  h+=detailsHtml(d);
  h+=rlhfHtml(d.is_decline);
  h+='</div></div>';
  const node=el(h); chat.appendChild(node); chat.scrollTop=chat.scrollHeight;
  wireRlhf(node,d.query_id,question,d);
}
function addQuarantined(d,question){
  const last=(d.audit_trail||[]).slice(-1)[0]||{};
  let h='<div class="msg bot"><div class="who">Assistant &mdash; held for review</div><div class="qcard">';
  h+='<div class="qtitle">&#9888; ANSWER HELD FOR HUMAN REVIEW</div>';
  h+='<div class="qsub">The audit could not verify this answer ('+esc(String(d.verdict))+'). It will not be released until a reviewer approves it. The auditor\'s objections:</div>';
  for(const i of (last.issues||[]).slice(0,6)) h+='<div class="qissue">'+esc(String(i))+'</div>';
  h+='<div class="gates">'+d.gates.map(gateHtml).join('')+'</div>';
  h+=detailsHtml(d);
  h+='<details><summary>Show the withheld answer</summary><div class="trace-body">'+md(d.answer)+'</div></details>';
  h+='<div class="qbtns">'
    +'<button class="qbtn qapprove">Approve &mdash; release answer</button>'
    +'<button class="qbtn qreject">Reject &mdash; discard</button></div>';
  h+='</div></div>';
  const node=el(h);
  node.querySelector('.qapprove').onclick=()=>{
    logReview(question,d,'approved');
    node.querySelector('.qcard').outerHTML=
      '<div class="bubble">'+md(d.answer)
      +'<div class="gates">'+d.gates.map(gateHtml).join('')+'</div>'
      +detailsHtml(d)
      +'<div class="qdecided" style="color:var(--ok)">&#10003; Released by reviewer (decision logged)</div>'
      +rlhfHtml(d.is_decline)+'</div>';
    // Only wire the RLHF widget once the answer is actually visible to
    // a user - a rejected quarantined answer never gets a widget at all.
    wireRlhf(node,d.query_id,question,d);
  };
  node.querySelector('.qreject').onclick=()=>{
    logReview(question,d,'rejected');
    node.querySelector('.qcard').innerHTML=
      '<div class="qtitle">Rejected by reviewer</div>'
      +'<div class="qsub">The withheld answer was discarded. Decision logged to the review trail.</div>';
  };
  chat.appendChild(node); chat.scrollTop=chat.scrollHeight;
}
async function ask(){
  const text=q.value.trim(); if(!text)return;
  q.value=''; send.disabled=true;
  addUser(text);
  const think=el('<div class="msg bot"><div class="bubble thinking dots">planning tools, querying, auditing</div></div>');
  chat.appendChild(think); chat.scrollTop=chat.scrollHeight;
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text})});
    const d=await r.json(); think.remove(); addAnswer(d,text);
  }catch(e){ think.remove(); chat.appendChild(el('<div class="msg bot"><div class="bubble">Error: '+esc(String(e))+'</div></div>')); }
  send.disabled=false; q.focus();
}
send.onclick=ask; q.addEventListener('keydown',e=>{if(e.key==='Enter')ask();}); q.focus();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/review":
            self._handle_review()
            return
        if self.path == "/rlhf":
            self._handle_rlhf()
            return
        if self.path != "/ask":
            self.send_error(404)
            return
        question = ""
        query_id = _new_query_id()
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("empty question")
            with _agent_lock:
                result = _agent.ask(question)
            ev = _evidence_view(result["evidence"])
            gates = _gates(result, ev)
            verdict = (result.get("verification") or {}).get("verdict")
            # Quarantine rule: only a clean audit pass ships directly.
            # "fail" and "audit_error" are withheld for a human.
            # "not_checked" (offline mode, no auditor) is delivered
            # but clearly labeled by its gate.
            quarantined = verdict in ("fail", "audit_error")
            _log_query(query_id, question, "quarantined" if quarantined else "delivered",
                       result=result, gates=gates, evidence_view=ev)
            # is_decline travels with the answer so the browser's RLHF
            # widget can ask a DIFFERENT, more intuitive question when the
            # system declined - see rlhfHtml()/wireRlhf() below. Computed
            # once here (same _is_decline the golden test set uses) so the
            # client never has to re-implement the phrase check itself.
            body = json.dumps({
                "query_id": query_id,
                "answer": result["answer"],
                "is_decline": _is_decline(result["answer"]),
                "gates": gates,
                "trace": ev,
                "audit_trail": result.get("audit_trail", []),
                "draft_answer": result.get("draft_answer"),
                "verdict": verdict,
                "quarantined": quarantined,
            }, default=str).encode("utf-8")
            self.send_response(200)
        except Exception as e:
            _log_query(query_id, question, "error", error=f"{type(e).__name__}: {e}")
            body = json.dumps({"query_id": query_id,
                               "answer": f"Server error: {type(e).__name__}: {e}",
                               "is_decline": False,
                               "gates": [], "trace": [], "audit_trail": [],
                               "draft_answer": None}).encode("utf-8")
            self.send_response(500)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_review(self):
        """Record a human release/reject decision on a quarantined
        answer. Appends one JSON line per decision - the audit trail a
        governance process would ask for."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            entry = {
                "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question": str(payload.get("question", ""))[:500],
                "verdict": str(payload.get("verdict", "")),
                "decision": ("approved" if payload.get("decision") == "approved"
                              else "rejected"),
                "issues": payload.get("issues", [])[:10],
            }
            os.makedirs(os.path.dirname(REVIEW_LOG), exist_ok=True)
            with open(REVIEW_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
        except Exception as e:
            body = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
            self.send_response(500)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_rlhf(self):
        """Records a human rating on an answer the user actually saw -
        correct / partially correct / incorrect / vague, plus the
        follow-up "was the right info in the evidence" answer. Every
        rating is logged, including a plain "correct" with no note -
        the point is a complete record of every judgment made, not
        just the interesting ones.

        The browser already holds the full answer/gates/trace/audit
        trail from the original /ask response (it's just sitting in
        the page's memory), so it sends all of that back here rather
        than making this handler re-read query_log.jsonl to reconstruct
        it - one self-contained record per rating, no join required to
        make sense of it later."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            rating = str(payload.get("rating", ""))
            if rating not in ("correct", "partial", "incorrect", "vague"):
                raise ValueError(f"unrecognised rating: {rating!r}")
            evidence_answer = str(payload.get("evidence_answer", ""))
            if evidence_answer not in ("yes", "no", "idk"):
                evidence_answer = "idk"
            answer_text = str(payload.get("answer", ""))
            decline = _is_decline(answer_text)
            leaf, color = _rlhf_leaf(rating, evidence_answer, decline)
            # Which chunk (if any) the rep pointed to as having the right
            # info, and its rank among what was retrieved - the live-
            # traffic equivalent of the golden test set's rank signal.
            # Only ever set when evidence_answer == "yes" AND there were
            # real chunks to choose from (see chunkPickHtml in the page's
            # JS) - None otherwise, not a guessed value.
            confirmed_chunk_id = payload.get("confirmed_chunk_id")
            confirmed_chunk_rank = payload.get("confirmed_chunk_rank")
            entry = {
                "rated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query_id": str(payload.get("query_id", ""))[:80],
                "question": str(payload.get("question", ""))[:500],
                "answer": answer_text,
                "gates": payload.get("gates", []),
                "evidence": payload.get("trace", []),
                "audit_trail": payload.get("audit_trail", []),
                "draft_answer": payload.get("draft_answer"),
                "system_verdict": payload.get("verdict"),
                "quarantined": bool(payload.get("quarantined")),
                "rating": rating,
                "evidence_answer": evidence_answer,
                "confirmed_chunk_id": (str(confirmed_chunk_id) if confirmed_chunk_id else None),
                "confirmed_chunk_rank": (int(confirmed_chunk_rank) if confirmed_chunk_rank else None),
                "auto_decline_detected": decline,
                "leaf": leaf,
                "color": color,
                "note": (str(payload.get("note", "")).strip()[:1000] or None),
            }
            _append_jsonl(RLHF_LOG, entry)
            body = json.dumps({"ok": True, "leaf": leaf, "color": color}, default=str).encode("utf-8")
            self.send_response(200)
        except Exception as e:
            body = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
            self.send_response(500)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Rabivy Intelligence Console running at {url}  (Ctrl-C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
