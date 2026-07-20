# chat_ui.py
# -------------------------------------------------------------------
# Phase 6: the conversational interface - a local web chat on top of
# agent.py, so the system can be used (and demoed) without a terminal.
#
#   python chat_ui.py          -> open http://localhost:8017
#
# Design choices:
#   - Python standard library only (http.server) - no Flask, no npm,
#     nothing new in requirements.txt. Anyone who can run agent.py can
#     run this.
#   - ONE RabivyAgent instance lives for the whole server session, so
#     conversation memory works across messages exactly like the
#     terminal chat ("why is the second doctor not converting?").
#   - The page shows the agent's working, not just its answer: which
#     tools were called with what parameters, the verification verdict,
#     and whether the answer was auto-revised after a failed audit.
#     For a governed pharma context, showing the audit trail IS the
#     feature, not decoration.
#   - Local use only (localhost). Putting this on a network needs auth,
#     TLS and rate limiting - deliberately out of scope here.
# -------------------------------------------------------------------

import json
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

from agent import RabivyAgent

PORT = 8017

# One agent = one conversation, shared across requests (memory ON).
# A lock serialises questions - the agent's history list is not
# designed for two questions interleaving mid-loop.
_agent = RabivyAgent()
_agent_lock = threading.Lock()

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rabivy Commercial Intelligence Assistant</title>
<style>
  :root{--ink:#1a2332;--muted:#5b6779;--line:#dbe2ea;--brand:#7c3aed;--brand-bg:#f5f0ff;
        --ok:#047857;--ok-bg:#eefaf3;--warn:#b45309;--warn-bg:#fdf6e9;--user:#2563eb;--user-bg:#eff4ff}
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);
       background:#f4f6fa;display:flex;flex-direction:column;height:100vh}
  header{background:#fff;border-bottom:1px solid var(--line);padding:14px 22px}
  header h1{font-size:17px;margin:0}
  header .sub{font-size:12.5px;color:var(--muted);margin-top:2px}
  #chat{flex:1;overflow-y:auto;padding:22px;max-width:900px;width:100%;margin:0 auto}
  .msg{margin-bottom:16px;max-width:85%}
  .msg .who{font-size:11.5px;color:var(--muted);margin-bottom:3px;text-transform:uppercase;letter-spacing:.04em}
  .bubble{padding:12px 16px;border-radius:12px;font-size:14px;line-height:1.5;white-space:normal}
  .user{margin-left:auto}
  .user .bubble{background:var(--user-bg);border:1px solid #bcd0f7;border-bottom-right-radius:4px}
  .bot .bubble{background:#fff;border:1px solid var(--line);border-bottom-left-radius:4px}
  .bubble h2,.bubble h3{font-size:14.5px;margin:10px 0 4px}
  .bubble ul{margin:6px 0;padding-left:20px}
  .bubble li{margin-bottom:3px}
  .bubble code{background:#eef1f6;border-radius:4px;padding:0 4px;font-size:12.5px}
  .meta{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
  .chip{font-size:11px;border-radius:20px;padding:3px 10px;border:1px solid}
  .chip.tool{background:var(--brand-bg);border-color:#d4c3f5;color:var(--brand)}
  .chip.ok{background:var(--ok-bg);border-color:#9fd9be;color:var(--ok)}
  .chip.warn{background:var(--warn-bg);border-color:#e5c98a;color:var(--warn)}
  .thinking{color:var(--muted);font-size:13px;padding:6px 0}
  .dots::after{content:'';animation:d 1.2s infinite}
  @keyframes d{0%{content:'.'}33%{content:'..'}66%{content:'...'}}
  footer{background:#fff;border-top:1px solid var(--line);padding:14px 22px}
  .inputrow{max-width:900px;margin:0 auto;display:flex;gap:10px}
  #q{flex:1;padding:12px 14px;font-size:14px;border:1.5px solid var(--line);border-radius:10px;outline:none}
  #q:focus{border-color:var(--brand)}
  #send{padding:12px 22px;font-size:14px;font-weight:600;color:#fff;background:var(--brand);
        border:none;border-radius:10px;cursor:pointer}
  #send:disabled{opacity:.5;cursor:default}
  .hint{max-width:900px;margin:6px auto 0;font-size:11.5px;color:var(--muted)}
</style>
</head>
<body>
<header>
  <h1>Rabivy Commercial Intelligence Assistant</h1>
  <div class="sub">Agentic RAG demo &middot; synthetic data &middot; every answer is tool-grounded, cited and audited &middot; conversation memory on</div>
</header>
<div id="chat"></div>
<footer>
  <div class="inputrow">
    <input id="q" placeholder="e.g. Who should I target next month in New York, and what should I say to them?" autocomplete="off">
    <button id="send">Ask</button>
  </div>
  <div class="hint">Try a follow-up like &ldquo;and what about Texas?&rdquo; &mdash; the assistant remembers the conversation.</div>
</footer>
<script>
const chat = document.getElementById('chat');
const q = document.getElementById('q');
const send = document.getElementById('send');

// Minimal markdown -> HTML (bold, headings, bullets, code). Dependency-free
// on purpose: this page must work offline, no CDN.
function md(t){
  t = t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  t = t.replace(/`([^`]+)`/g,'<code>$1</code>');
  t = t.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  t = t.replace(/^#{2,3}\s*(.+)$/gm,'<h3>$1</h3>');
  const lines = t.split('\n'); let out=[], inList=false;
  for (const ln of lines){
    if (/^\s*[-*]\s+/.test(ln)){ if(!inList){out.push('<ul>');inList=true;} out.push('<li>'+ln.replace(/^\s*[-*]\s+/,'')+'</li>'); }
    else { if(inList){out.push('</ul>');inList=false;} out.push(ln.trim()===''?'':'<p>'+ln+'</p>'); }
  }
  if(inList) out.push('</ul>');
  return out.join('');
}

function add(who, html, metaHtml){
  const m = document.createElement('div');
  m.className = 'msg ' + (who==='You'?'user':'bot');
  m.innerHTML = '<div class="who">'+who+'</div><div class="bubble">'+html+
                (metaHtml?'<div class="meta">'+metaHtml+'</div>':'')+'</div>';
  chat.appendChild(m); chat.scrollTop = chat.scrollHeight;
  return m;
}

async function ask(){
  const text = q.value.trim(); if(!text) return;
  q.value=''; send.disabled=true;
  add('You', md(text));
  const think = add('Assistant','<span class="thinking dots">planning tools, querying, verifying</span>');
  try{
    const r = await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
                                  body:JSON.stringify({question:text})});
    const d = await r.json();
    let meta = (d.steps||[]).map(s=>'<span class="chip tool">'+s.tool+'</span>').join('');
    const v = (d.verification||{}).verdict || '-';
    if (v==='pass') meta += '<span class="chip ok">&#10003; verified against sources</span>';
    else if (v==='fail') meta += '<span class="chip warn">&#9888; audit flagged issues</span>';
    else meta += '<span class="chip warn">audit: '+v+'</span>';
    if (d.revised) meta += '<span class="chip warn">auto-revised after audit</span>';
    think.remove();
    add('Assistant', md(d.answer||'(no answer)'), meta);
  }catch(e){
    think.remove();
    add('Assistant','<em>Error: '+e+'. Is the server still running?</em>');
  }
  send.disabled=false; q.focus();
}
send.onclick = ask;
q.addEventListener('keydown', e=>{ if(e.key==='Enter') ask(); });
q.focus();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the terminal quiet
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
            body = json.dumps({
                "answer": result["answer"],
                "steps": [{"tool": s["tool"]} for s in result["steps"]],
                "verification": {"verdict": result["verification"].get("verdict"),
                                  "issues": result["verification"].get("issues", [])},
                "revised": result["revised"],
            }).encode("utf-8")
            self.send_response(200)
        except Exception as e:
            body = json.dumps({"answer": f"Server error: {type(e).__name__}: {e}",
                               "steps": [], "verification": {"verdict": "error"},
                               "revised": False}).encode("utf-8")
            self.send_response(500)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Rabivy assistant running at {url}  (Ctrl-C to stop)")
    try:
        webbrowser.open(url)  # best effort - fine if headless
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
