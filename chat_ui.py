# chat_ui.py
# -------------------------------------------------------------------
# Phase 6: the conversational interface - a local web console on top of
# agent.py. Not just a chat: every response ships with its full trace,
# so the pipeline is inspectable per answer:
#
#   GATES  - a per-answer checklist: tool grounding, retrieval
#            confidence, source citations, and the hallucination audit
#            (including whether a draft was rejected and auto-revised)
#   TRACE  - every tool call with its exact parameters and the evidence
#            it returned (table rows, document chunks with similarity
#            scores), plus the audit verdicts and, when a revision
#            fired, the rejected draft for comparison
#
#   python chat_ui.py          -> open http://localhost:8017
#
# Standard library only; one RabivyAgent instance for the whole server
# session, so conversation memory works across messages. Local use only
# (binds 127.0.0.1) - a networked deployment would need auth and TLS.
# -------------------------------------------------------------------

import json
import re
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

from agent import RabivyAgent

PORT = 8017

_agent = RabivyAgent()
_agent_lock = threading.Lock()

_CITE = re.compile(r"\[hcp_table\]|\[doc:\s*[^\]]+\]")


def _evidence_view(evidence):
    """Trim tool results for the browser: enough to inspect, not the
    full corpus. Structure is preserved; long text is clipped."""
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
                 "text": (s.get("text") or "")[:500]}
                for s in res["sections"]
            ]
        elif "row" in res:
            view["summary"] = f"NPI {res['row'].get('npi')} - table row" + (" + snapshot card" if "snapshot_card" in res else "")
            view["rows"] = [res["row"]]
            if "snapshot_card" in res:
                view["sections"] = [{"chunk_id": res["snapshot_card"]["chunk_id"],
                                     "similarity": None,
                                     "text": res["snapshot_card"]["text"][:500]}]
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
    gates.append({
        "name": "Retrieval confidence",
        "ok": not low_conf and not errors,
        "detail": ("All retrievals confident, no tool errors" if not low_conf and not errors
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
  .audit-round{padding:6px 10px;border-radius:8px;margin:5px 0;font-size:11.5px}
  .audit-round.pass{background:var(--ok-bg)} .audit-round.fail{background:var(--bad-bg)}
  .draft{background:#fbfbfd;border:1px dashed var(--line);border-radius:8px;padding:8px 12px;
         font-size:12px;color:var(--muted);margin-top:5px;white-space:pre-wrap}
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
  <div class="sub">Agentic RAG with full-trace transparency &middot; synthetic data &middot; every answer shows its evidence and the gates it cleared</div>
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
      h+='<div class="chunk"><span class="cid">'+esc(s.chunk_id)+'</span>'
        +(s.similarity!=null?' <span class="sim">similarity '+s.similarity+'</span>':'')
        +'<div>'+esc(s.text)+'&hellip;</div></div>';
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
function addAnswer(d){
  let h='<div class="msg bot"><div class="who">Assistant</div><div class="bubble">'+md(d.answer);
  h+='<div class="gates">'+d.gates.map(gateHtml).join('')+'</div>';
  h+='<details><summary>Evidence trace &mdash; '+d.trace.length+' tool call(s)</summary><div class="trace-body">'+traceHtml(d)+'</div></details>';
  h+='<details><summary>Audit detail</summary><div class="trace-body">'+auditHtml(d)+'</div></details>';
  h+='</div></div>';
  chat.appendChild(el(h)); chat.scrollTop=chat.scrollHeight;
}
async function ask(){
  const text=q.value.trim(); if(!text)return;
  q.value=''; send.disabled=true;
  addUser(text);
  const think=el('<div class="msg bot"><div class="bubble thinking dots">planning tools, querying, auditing</div></div>');
  chat.appendChild(think); chat.scrollTop=chat.scrollHeight;
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text})});
    const d=await r.json(); think.remove(); addAnswer(d);
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
        if self.path != "/ask":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("empty question")
            with _agent_lock:
                result = _agent.ask(question)
            ev = _evidence_view(result["evidence"])
            body = json.dumps({
                "answer": result["answer"],
                "gates": _gates(result, ev),
                "trace": ev,
                "audit_trail": result.get("audit_trail", []),
                "draft_answer": result.get("draft_answer"),
            }, default=str).encode("utf-8")
            self.send_response(200)
        except Exception as e:
            body = json.dumps({"answer": f"Server error: {type(e).__name__}: {e}",
                               "gates": [], "trace": [], "audit_trail": [],
                               "draft_answer": None}).encode("utf-8")
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
