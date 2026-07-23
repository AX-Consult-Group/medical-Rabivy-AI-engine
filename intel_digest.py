# intel_digest.py
# -------------------------------------------------------------------
# COMPETITIVE INTELLIGENCE DIGEST: turn a batch of incoming external
# signals (conference abstracts, publications, regulatory readouts,
# payer/market news) into a ranked, email-ready briefing - and ingest
# the items into the knowledge repository so the conversational agent
# can answer questions about them afterwards.
#
# The loop:
#   1. FEED     data/ci_feed/incoming/*.md - one file per item, with a
#               small metadata header. (Simulated here; in production
#               this folder is filled by feed connectors - PubMed /
#               ClinicalTrials.gov / FDA RSS / conference scrapers.)
#   2. TRIAGE   each item is scored for noteworthiness AGAINST THE
#               BRAND STRATEGY (what threatens or supports Rabivy's
#               positioning), with a one-line "why it matters" and a
#               recommended action. Uses the LLM when a key is set;
#               falls back to a deterministic keyword scorer offline -
#               same contract as the rest of this repo.
#   3. DIGEST   output/intel_digest_<date>.md and .html (email-ready).
#               Optional SMTP send via --email (env: SMTP_HOST,
#               SMTP_PORT, SMTP_USER, SMTP_PASS, DIGEST_TO).
#   4. INGEST   processed items move to docs/competitive_feed/, which
#               the existing build pipeline (main.py) picks up
#               automatically - the next rebuild makes every item
#               retrievable by the agent ("what did Novo announce last
#               week?").
#
# Usage:
#   python intel_digest.py --simulate     # write a synthetic weekly batch
#   python intel_digest.py                # triage + digest + ingest
#   python intel_digest.py --email       # ...and send via SMTP env config
#
# All simulated content is fictional, like the rest of this project.
# -------------------------------------------------------------------

import argparse
import glob
import json
import os
import re
import shutil
import time

INCOMING_DIR = os.path.join("data", "ci_feed", "incoming")
INGESTED_DIR = os.path.join("docs", "competitive_feed")
OUTPUT_DIR = "output"

# ---- What "noteworthy" means for THIS brand: the strategy lens ------
BRAND_CONTEXT = """Rabivy (maridebart cafraglutide) is AX Pharmaceuticals'
investigational GLP-1/GIP-antagonist antibody-peptide conjugate for obesity,
positioned on: (1) monthly dosing vs weekly competitors, (2) durable weight
loss without plateau, (3) tolerability. Main competitors: Novo Nordisk
(semaglutide: Ozempic/Wegovy) and Eli Lilly (tirzepatide: Mounjaro/Zepbound).
Noteworthy = anything that shifts the competitive position: competitor
efficacy/safety readouts, dosing-convenience moves, label expansions,
regulatory decisions, payer coverage shifts, supply issues, pricing moves."""

TRIAGE_SYSTEM = f"""You are a competitive-intelligence analyst for a pharma brand team.

{BRAND_CONTEXT}

You will receive a numbered list of incoming intelligence items. For EACH item return a JSON object:
{{"id": <number>, "score": 1-5, "category": "<threat|opportunity|monitor>", "why": "<ONE sentence: why this matters for Rabivy specifically>", "action": "<ONE sentence: recommended next step for the brand team>"}}
5 = drop-everything (e.g. competitor monthly dosing data, major safety signal, label expansion). 1 = background noise.
Respond with ONLY a JSON array of these objects, nothing else."""

# Offline fallback scorer: transparent keyword heuristics.
_SIGNALS = [
    (5, r"monthly|once-monthly|4-week dosing"),
    (5, r"FDA (approv|advisory|complete response|label)"),
    (4, r"phase 3|superiority|head-to-head|cardiovascular outcomes"),
    (4, r"discontinuation|adverse|safety signal|tolerability"),
    (3, r"phase 2|weight loss|efficacy"),
    (3, r"coverage|formulary|payer|price|pricing"),
    (2, r"supply|manufacturing|capacity"),
    (2, r"real-world|adherence|persistence"),
]


def _parse_item(path):
    text = open(path, encoding="utf-8").read()
    meta = dict(re.findall(r"^(\w+):\s*(.+)$", text.split("---")[1], re.M)) \
        if text.startswith("---") else {}
    body = text.split("---", 2)[2].strip() if text.count("---") >= 2 else text
    title = next((l.lstrip("# ").strip() for l in body.splitlines()
                  if l.startswith("#")), os.path.basename(path))
    return {"path": path, "meta": meta, "title": title, "body": body}


def _triage_offline(items):
    out = []
    for i, item in enumerate(items):
        text = (item["title"] + " " + item["body"]).lower()
        score = max((s for s, pat in _SIGNALS if re.search(pat, text)), default=1)
        cat = "threat" if re.search(r"novo|lilly|semaglutide|tirzepatide", text) and score >= 4 \
            else ("opportunity" if score >= 4 else "monitor")
        out.append({"id": i, "score": score, "category": cat,
                    "why": "[offline triage - keyword score; run with an API key "
                           "for analyst-grade rationale]",
                    "action": "Review item and reassess with the full strategy lens."})
    return out


def _triage_llm(items):
    from llm_client import get_llm, MockLLM
    llm = get_llm()
    if isinstance(llm, MockLLM):
        return _triage_offline(items)
    listing = "\n\n".join(
        f"[{i}] ({it['meta'].get('type','item')} | {it['meta'].get('competitor','-')} | "
        f"{it['meta'].get('date','-')}) {it['title']}\n{it['body'][:900]}"
        for i, it in enumerate(items))
    resp = llm.complete(TRIAGE_SYSTEM, [{"role": "user", "content": listing}],
                        tools=None, max_tokens=2000, temperature=0.0)
    text = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
    m = re.search(r"\[.*\]", text, re.DOTALL)
    verdicts = json.loads(m.group(0)) if m else []
    by_id = {v["id"]: v for v in verdicts}
    return [by_id.get(i, _triage_offline([it])[0] | {"id": i})
            for i, it in enumerate(items)]


def build_digest(items, verdicts, stamp):
    ranked = sorted(zip(items, verdicts), key=lambda p: -p[1]["score"])
    lines = [f"# Competitive Intelligence Digest - {stamp}",
             "",
             f"{len(items)} new items triaged against the Rabivy strategy lens. "
             f"Top signals first.", ""]
    for it, v in ranked:
        flag = {"threat": "THREAT", "opportunity": "OPPORTUNITY", "monitor": "MONITOR"}[v["category"]]
        stars = "|" * v["score"]
        lines.append(f"## [{v['score']}/5 {flag}] {it['title']}")
        lines.append(f"*{it['meta'].get('type','item')} · {it['meta'].get('competitor','-')} · "
                     f"{it['meta'].get('source','-')} · {it['meta'].get('date','-')}*")
        lines.append("")
        lines.append(f"**Why it matters:** {v['why']}")
        lines.append(f"**Recommended action:** {v['action']}")
        lines.append("")
    lines.append("---")
    lines.append("*Items are ingested into the knowledge repository on the next build - "
                 "ask the assistant about any of them. All content in this demo is "
                 "synthetic/fictional.*")
    md = "\n".join(lines)

    rows = "".join(
        f"<div style='border-left:4px solid {('#b91c1c' if v['category']=='threat' else '#047857' if v['category']=='opportunity' else '#b45309')};"
        f"background:#fafbfd;border-radius:0 8px 8px 0;padding:10px 14px;margin:10px 0'>"
        f"<div style='font-size:11px;color:#5b6779'>{v['score']}/5 · {v['category'].upper()} · "
        f"{it['meta'].get('type','item')} · {it['meta'].get('competitor','-')} · {it['meta'].get('date','-')}</div>"
        f"<div style='font-weight:700;margin:2px 0'>{it['title']}</div>"
        f"<div style='font-size:13px'><b>Why it matters:</b> {v['why']}</div>"
        f"<div style='font-size:13px'><b>Action:</b> {v['action']}</div></div>"
        for it, v in ranked)
    html = (f"<html><body style='font-family:system-ui,sans-serif;color:#1a2332;max-width:720px;margin:auto'>"
            f"<h2>Competitive Intelligence Digest - {stamp}</h2>"
            f"<p style='color:#5b6779'>{len(items)} new items triaged against the Rabivy strategy lens.</p>"
            f"{rows}<p style='font-size:11px;color:#5b6779'>Synthetic demo content. "
            f"Items are now retrievable via the Rabivy assistant.</p></body></html>")
    return md, html


def ingest(items):
    os.makedirs(INGESTED_DIR, exist_ok=True)
    for it in items:
        shutil.move(it["path"], os.path.join(INGESTED_DIR, os.path.basename(it["path"])))


def send_email(html, stamp):
    import smtplib
    from email.mime.text import MIMEText
    host, user = os.environ.get("SMTP_HOST"), os.environ.get("SMTP_USER")
    to = os.environ.get("DIGEST_TO")
    if not (host and to):
        print("(email skipped: set SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS/DIGEST_TO)")
        return
    msg = MIMEText(html, "html")
    msg["Subject"] = f"Rabivy Competitive Intelligence Digest - {stamp}"
    msg["From"] = user or "rabivy-intel@localhost"
    msg["To"] = to
    with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", 587))) as s:
        s.starttls()
        if user:
            s.login(user, os.environ["SMTP_PASS"])
        s.send_message(msg)
    print(f"Digest emailed to {to}")


# ---------------------------------------------------------------------
SIMULATED_BATCH = [
    ("fda_readout", "Eli Lilly", "FDA.gov (simulated)",
     "FDA accepts Lilly supplemental filing for tirzepatide sleep-apnea indication",
     "The FDA has accepted Eli Lilly's supplemental application seeking a label "
     "expansion for tirzepatide (Zepbound) in obstructive sleep apnea with obesity. "
     "A decision is expected within eight months. If approved, tirzepatide would be "
     "the first GLP-1 class agent labeled for OSA, broadening prescriber rationale "
     "beyond weight loss."),
    ("conference_abstract", "Novo Nordisk", "ADA Scientific Sessions (simulated)",
     "Abstract: once-monthly semaglutide depot shows 12.9% weight loss at 48 weeks in phase 2",
     "A late-breaking abstract reports Novo Nordisk's investigational once-monthly "
     "semaglutide depot formulation achieved 12.9% mean weight loss at 48 weeks "
     "(n=384), with discontinuation for GI events of 6.1%. A phase 3 program is "
     "planned to start within the year. Monthly dosing has until now been a "
     "differentiator unique to investigational agents such as maridebart cafraglutide."),
    ("publication", "Eli Lilly", "NEJM (simulated)",
     "Publication: tirzepatide cardiovascular outcomes trial meets primary endpoint",
     "Full results of the cardiovascular outcomes trial in adults with obesity "
     "without diabetes report a 14% relative risk reduction in MACE for tirzepatide "
     "versus placebo. Outcome data of this kind materially strengthens payer and "
     "prescriber positioning for the incumbent."),
    ("payer_news", "Novo Nordisk", "Managed-care press (simulated)",
     "Large national PBM moves Wegovy to preferred tier with expanded prior-auth criteria",
     "A top-three PBM announced Wegovy moves to preferred formulary status for plan "
     "year 2027, paired with simplified prior authorization for BMI >= 35. Access "
     "friction has been a key equalizer for late entrants; simplified PA for the "
     "incumbent narrows that window."),
    ("publication", "Class-wide", "JAMA Internal Medicine (simulated)",
     "Real-world persistence study: 64% of GLP-1 initiators discontinue within 12 months",
     "A claims-based cohort (n=125,000) reports 64% of patients initiating weekly "
     "GLP-1 therapy for obesity discontinue within a year; injection burden and GI "
     "tolerability are the leading cited reasons. Persistence remains the class's "
     "largest unmet need - and the core of the monthly-dosing value story."),
    ("conference_abstract", "Eli Lilly", "Obesity Week (simulated)",
     "Abstract: oral orforglipron phase 3 shows 11.2% weight loss at 68 weeks",
     "Lilly's oral small-molecule GLP-1 orforglipron reports 11.2% mean weight loss "
     "at 68 weeks in a phase 3 readout, with once-daily oral dosing and no food "
     "restrictions. An oral option reframes the convenience conversation the class "
     "has been having around injection frequency."),
    ("market_news", "Novo Nordisk", "Financial press (simulated)",
     "Novo announces additional fill-finish capacity coming online in 2027",
     "Novo Nordisk confirmed two additional fill-finish lines will come online "
     "during 2027, easing the supply constraints that limited Wegovy starts. "
     "Supply-constrained competitor demand has been an unplanned tailwind for "
     "newer entrants' sampling programs."),
    ("regulatory_news", "Class-wide", "EMA (simulated)",
     "EMA opens class review of rare post-marketing reports of gastroparesis",
     "The EMA pharmacovigilance committee opened a routine class-wide review of "
     "rare gastroparesis reports with GLP-1 receptor agonists. No regulatory "
     "action recommended at this stage; monitoring continues."),
]


def simulate_batch(stamp):
    os.makedirs(INCOMING_DIR, exist_ok=True)
    for i, (typ, comp, src, title, body) in enumerate(SIMULATED_BATCH, 1):
        path = os.path.join(INCOMING_DIR, f"{stamp}_{i:02d}_{typ}.md")
        with open(path, "w", encoding="utf-8") as f:
            # "## " heading (not "# "): the chunker only creates chunks
            # from level-2 sections. The metadata line sits INSIDE the
            # section so recency/competitor/type words are retrievable.
            f.write(f"---\ntype: {typ}\ncompetitor: {comp}\nsource: {src}\n"
                    f"date: {stamp}\n---\n\n## {title}\n\n"
                    f"*Competitive intelligence feed - {typ} announced by {comp} "
                    f"on {stamp} (source: {src}). Recent news update.*\n\n{body}\n\n"
                    f"*Simulated item for demonstration - fictional content.*\n")
    print(f"Simulated {len(SIMULATED_BATCH)} incoming intel items -> {INCOMING_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Competitive intelligence digest")
    ap.add_argument("--simulate", action="store_true", help="write a synthetic incoming batch")
    ap.add_argument("--email", action="store_true", help="send the digest via SMTP env config")
    ap.add_argument("--no-ingest", action="store_true", help="leave items in incoming/")
    args = ap.parse_args()

    stamp = time.strftime("%Y-%m-%d")
    if args.simulate:
        simulate_batch(stamp)
        raise SystemExit(0)

    paths = sorted(glob.glob(os.path.join(INCOMING_DIR, "*.md")))
    if not paths:
        print(f"No incoming items in {INCOMING_DIR}/ - run --simulate first "
              f"(or wire a real feed connector).")
        raise SystemExit(0)

    items = [_parse_item(p) for p in paths]
    print(f"Triaging {len(items)} incoming items against the brand strategy lens...")
    verdicts = _triage_llm(items)

    md, html = build_digest(items, verdicts, stamp)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    md_path = os.path.join(OUTPUT_DIR, f"intel_digest_{stamp}.md")
    html_path = os.path.join(OUTPUT_DIR, f"intel_digest_{stamp}.html")
    open(md_path, "w", encoding="utf-8").write(md)
    open(html_path, "w", encoding="utf-8").write(html)
    print(f"Digest written: {md_path} and .html")

    if args.email:
        send_email(html, stamp)

    if not args.no_ingest:
        ingest(items)
        print(f"{len(items)} items ingested -> {INGESTED_DIR}/ "
              f"(run 'python main.py' to make them retrievable by the agent)")
