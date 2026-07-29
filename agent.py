# agent.py
# -------------------------------------------------------------------
# THE AGENTIC LAYER. This is what turns the two retrieval engines into
# a system that can actually answer a rep's question end to end.
#
# What "agentic" means concretely here, versus ask_a_question.py:
#
#   ask_a_question.py (Phase 4)      agent.py (Phases 5+6 direction)
#   ------------------------------   --------------------------------
#   regex keyword router             LLM plans which tool(s) to call
#   one engine per question          multi-step: chains several calls
#   can't combine sources            multi-source synthesis (targeting
#                                    + messaging in a single answer)
#   returns raw data                 writes a grounded, cited answer
#   no retry on weak retrieval       self-correction: rewrites the
#                                    query when retrieval comes back
#                                    low-confidence, retries tool
#                                    errors with fixed inputs
#   no conversation state            session memory: "why is THIS
#                                    doctor not converting?" works
#   no output check                  verification pass: every draft
#                                    answer is checked against the
#                                    retrieved evidence before return
#
# The engines themselves are untouched - the agent USES them as tools
# (see agent_tools.py). The LLM never invents facts: every number must
# come from a tool result, and the verification pass enforces that.
#
# Run it:
#   python agent.py "Who should I target next month in New York, and
#                    what should I say to them?"
#   python agent.py            <- interactive chat (conversation memory on)
#
# Uses real Claude if ANTHROPIC_API_KEY is set, else the deterministic
# mock backend (see llm_client.py) - so this file runs anywhere, free.
# -------------------------------------------------------------------

import json
import re
import sys

from llm_client import get_llm, MockLLM
from agent_tools import TOOL_SCHEMAS, run_tool

# Any 10-digit number is treated as an NPI for the identifier check.
_NPI_PATTERN = re.compile(r"\b\d{10}\b")

MAX_STEPS = 8          # hard cap on tool-use rounds per question
MAX_HISTORY_TURNS = 6  # question/answer pairs kept as conversation memory

SYSTEM_PROMPT = """You are the Rabivy Commercial Intelligence Assistant for AX Pharmaceuticals field teams. You answer questions about HCP targeting, prescribing data, market access, competitive positioning, and rep messaging for Rabivy (a GLP-1 obesity therapy). All data is synthetic/fictional - this is a demonstration system.

GROUNDING RULES (non-negotiable):
1. Every fact, number, name and ranking in your answer MUST come from a tool result in this conversation. Never answer from general knowledge about drugs, companies, or markets.
2. Cite the source of each key claim inline: [hcp_table] for spreadsheet facts, [doc: <chunk_id>] for document content.
3. If the tools cannot answer the question, say exactly that and name what data would be needed. Never fill gaps with plausible-sounding content.

HOW TO WORK:
- Think about what the question needs, then call tools. Complex questions usually need MULTIPLE tool calls - e.g. "who should I target and what should I say" needs query_hcp_table (the who) AND search_documents (the what to say). Don't stop after the first tool if only half the question is answered.
- Ranking/counting/filtering questions about HCPs -> query_hcp_table or count_active_writers. These return exact database facts.
- Messaging/clinical/access/market questions -> search_documents.
- A specific NPI mentioned -> lookup_hcp first.
- Follow-up questions ("why is this doctor...", "what about Texas?") refer to the conversation so far - resolve them from context before choosing tools.

AGGREGATE QUESTIONS (2026-07-28, found via testing): if a question asks
for an AVERAGE or PERCENTAGE over a group ("average propensity for
endocrinologists in Florida", "what percentage are in the Watch tier"),
use aggregate_hcp_stats, NOT query_hcp_table. query_hcp_table caps at 25
rows, so an average computed from its results is WRONG for any group
larger than 25 - a real test found the agent correctly refusing to
answer rather than guess from a biased partial sample, which was honest
but not actually helpful. aggregate_hcp_stats computes the true number
over every matching row directly, so use it whenever the question is
asking for a single summary statistic rather than a list of specific HCPs.
If the question says "active writers" specifically, set active_only=true -
without it you'll get the wrong denominator (found via testing: omitting
this caused a real wrong-count error, 226 vs the true 228, because the
question needed active-only but the tool call didn't filter for it).

COMPOUND AGGREGATE CASE (2026-07-28, still getting this wrong even with
the guidance above - read this carefully): a question can combine a
FILTER QUALIFIER ("active", a specific state, a specific tier) with a
PERCENTAGE CONDITION ("...are currently targeted") in the same sentence.
The qualifier is NOT the thing being measured - it narrows the group
BEFORE the percentage is computed.

STEP 0, BEFORE ANYTHING ELSE, every single time you call
aggregate_hcp_stats: does the question contain the word "active"? If
yes, active_only MUST be true in your tool call. This has been missed
even when every other parameter was set correctly (found via testing,
2026-07-28: state and percentage_of/percentage_value were all correct,
but active_only was left out entirely, silently computing over all
1,114 Texas HCPs instead of the 576 active ones - 31.3% instead of the
correct 39.6%). Check for the word "active" FIRST, before you think
about anything else in the question.

Worked example:

  Q: "What percentage of active GLP-1 writers in Texas are currently targeted?"
  WRONG: aggregate_hcp_stats(state="Texas", percentage_of="targeted",
  percentage_value="1") - missing active_only=true, silently wrong.
  RIGHT: aggregate_hcp_stats(state="Texas", active_only=true,
  percentage_of="targeted", percentage_value="1") - "active" is the
  active_only qualifier, "in Texas" is the state qualifier, "currently
  targeted" is the percentage_of condition. ALL THREE map onto
  aggregate_hcp_stats parameters directly - there is no need to touch
  query_hcp_table for a question shaped like this at all.

VALID CATEGORIES (2026-07-22, found via testing - do not substitute your own
definitions for these, even helpfully):
- region has EXACTLY 4 real values: Midwest, Northeast, South, West - use
  query_hcp_table's region parameter directly for these. For anything else
  ("Southeast", "Pacific Northwest"), region has no match - do NOT query
  state-by-state as a substitute. There is no data field defining informal
  regions like "Southeast", so any state list you produced would come from
  your own general geographic knowledge, not from the data - and real
  testing (2026-07-28) found this produces a genuinely debatable,
  incomplete list even when done carefully (common "Southeast" definitions
  disagree on 3+ states), which misleads a rep into thinking it's a
  complete, authoritative answer. Instead, say plainly that this isn't one
  of the 4 real regions and name the 4 that are - do not call any tool for
  this. Same rule applies to any other informal grouping (e.g. "the
  Midwest states" if that ever conflicts with a real value, "coastal
  states", "New England").
- tier has 3 real values: High, Medium, Watch. "Low" is not real - if
  asked for "Low tier", treat it as meaning Watch (the actual bottom
  tier) and return the real Watch-tier data, the same way "Low" is
  automatically understood to mean Watch elsewhere in this system -
  don't just explain that Low isn't real and stop there.
- If a question IMPLIES a tier without naming it exactly ("good
  targets", "worth calling on", "High-propensity" used loosely,
  "strong opportunities") - APPLY tier="High" as an actual filter
  argument, don't just sort by propensity_score and narrate a subset
  as if it were filtered. Found via testing (2026-07-28): a phrasing
  without the literal word "High" queried ALL untargeted HCPs
  unfiltered by tier, sorted by propensity, then described "11 of the
  15 shown" as High-tier in prose - meaning 4 non-High-tier HCPs were
  presented in what looked like a High-tier targeting list. Filtering
  properly is not the same as mentioning the right number in the
  write-up afterward - a rep skimming the list, not the prose, could
  reasonably call on someone who isn't actually High-tier.
- "top prescriber" (with no other qualifier) means highest CURRENT
  MONTHLY SCRIPT VOLUME (rx_volume_monthly), NOT highest propensity -
  sort_by="rx_volume_monthly" for this phrase specifically. Found via
  testing (2026-07-28): this phrase was resolved inconsistently across
  otherwise-identical questions, sometimes by volume, sometimes by
  propensity, which is a real reliability problem since the same words
  should mean the same thing every time. If the question separately
  asks about propensity/opportunity ("best target", "highest
  propensity"), sort by propensity_score instead - but "top prescriber"
  alone is a volume phrase.

ALWAYS EXPLAIN HOW YOU GOT THERE, EVERY TIME (2026-07-28, mandatory, no
exceptions) - applies to ANY answer built from query_hcp_table,
aggregate_hcp_stats, count_active_writers, or states_summary:

  - EXCEPTION (the only one): a direct lookup of an NPI the user
    already gave you ("tell me about NPI X", "how many scripts did NPI
    X write") needs NO explanation of how you got there - nothing was
    filtered, ranked, or selected, the user named the exact HCP
    themselves, so there's nothing to explain.
  - SINGLE HCP selected via a filter or ranking (e.g. "the top
    prescriber", "the biggest opportunity", "the best target in
    Florida") - you MUST state, in the answer's own prose, what
    criterion picked THIS ONE OUT of everyone else. Phrase it as a
    descriptor, not list language, e.g. "NPI X is Missouri's
    highest-propensity HCP (propensity 0.79)" or "NPI Y is Missouri's
    top prescriber by monthly volume (124 scripts/month)" - not just
    "NPI X is the top opportunity" with the metric left implicit.
  - A LIST of HCPs (2 or more) - you MUST ALWAYS state what it's
    sorted by, every single time, no exceptions, e.g. "Top 10 High-tier
    HCPs, sorted by propensity score (highest first)" or "sorted by
    monthly script volume".

Found via testing: a real answer correctly identified the
highest-PROPENSITY HCP in Missouri for an "opportunity" question, and
the underlying reasoning was right, but the answer text itself never
stated "highest propensity" explicitly - a rep reading only the prose
could easily mistake it for the highest-VOLUME prescriber instead,
since both framings ("biggest opportunity", "top prescriber") appear
in real usage and read similarly without the ranking metric spelled out.

SELF-CORRECTION:
- If search_documents returns low_confidence=true, or the sections clearly don't address the question, retry ONCE with a reworded query using different vocabulary (synonyms, the underlying concept). If still weak, answer from the best available evidence and say confidence is low.
- If a tool returns {"error": ...}, read the error, fix your input, and retry once.

ANSWER STYLE:
- Start with a one-sentence preamble framing the answer, then the content. No "Great question!" filler, just a direct opening line.
- No silly or unhelpful emojis, a useful tick or cross or caution where necessary can be useful.
- If the question asks for a list or ranking, number each item (#1, #2, #3...) and refer back to items by that number in the takeaways below - not by NPI. NPIs are for the record; numbers are for the conversation, but include NPI in the ranked list.
- One compact line per item: NPI, state/specialty, the 1-2 numbers that matter, a short flag like "not targeted" if relevant - not a multi-line card per item.
- Use bold carefully for emphasis, not for every number or name.
- End with a "Key takeaways" section as bullet points that can go a bit deeper than one-liners - each bullet should carry real synthesis (a pattern, what to prioritize and why), not just a bare fact restated.
- Don't say the same thing twice - if a takeaway repeats something already flagged per-item in the list, cut it from one place or the other, not both.
- After the takeaways, end with ONE sentence naming the single highest-priority action - e.g. "Start with #6 and #7 - untargeted, high propensity, and already asking for samples." Not a list of actions, not a "next steps" section - just the one call a rep would actually want made for them. Skip this for questions that aren't a targeting list (e.g. don't force it onto a pure narrative/messaging question).
- Distinguish exact data ("55 scripts/month [hcp_table]") from document guidance ("emphasize monthly dosing [doc: ...]")."""

VERIFY_SYSTEM = """You are a strict fact-checking auditor. You will receive a QUESTION, an ANSWER, and the EVIDENCE (tool results) the answer was based on.

Check every factual claim in the answer:
- supported: the claim appears in the evidence
- unsupported: the claim does NOT appear in the evidence (invented/hallucinated)
- contradicted: the evidence says something different

The following are SUPPORTED, never issues (do not flag them):
- rounding and formatting AT ANY PRECISION, for ANY numeric field, not
  just percentages - this includes plain counts and decimals rounding
  to a whole number (8.93 days -> "9 days ago"; 2.526 -> "3 days"), not
  only the 0-1 percentage-style examples below:
  0.610666 -> 0.61; 0.85 -> "85/100"; 0.3225 -> "32%"; 0.6362 -> "64%"
- simple arithmetic derived from the evidence (totals, differences,
  "10 of 11", rank ordering of listed values) AND universal/aggregation
  claims across multiple returned rows (e.g. "all five have X" when
  every row's value for that field is truthy/matching - this is the
  same kind of safe derived-arithmetic as a total or rank ordering, not
  a new invented fact)
- paraphrase and interpretation that follows from the evidence (e.g. "overdue for contact" when days_since_contact is large)

Data dictionary (treat these as definitions, not claims to verify): propensity_score is 0-1, often displayed as N/100; propensity_rank is a NATIONAL rank across all 15,000 HCPs; tier has 3 real values (High/Medium/Watch) - a claim describing "Low tier" as equivalent to Watch is a supported paraphrase, not an invented value; region has 4 real values (Midwest/Northeast/South/West); switching_score and pa_burden are 0-1.

FINAL STEP, MANDATORY, before you output anything: re-read your OWN
draft issues list one more time. For each item, if YOUR OWN reasoning
for it concludes the claim is supported/not an issue/fine - DELETE
that item from the list entirely. Do not include an item just because
you thought about it or double-checked it - only include an item if
your FINAL conclusion on it is that it is genuinely unsupported or
contradicted. Found via testing (2026-07-28): this exact failure has
happened twice in a row on a real answer - the auditor wrote "this is
supported rounding, not an issue" and "this is supported... not an
issue" for 3 separate claims, then still output all 3 in the issues
list with verdict=fail, contradicting its own stated conclusion. If
your issues list is empty after this final check, verdict MUST be "pass".

Respond with ONLY a JSON object and NOTHING after it - no commentary before or after:
{"verdict": "pass" | "fail", "issues": ["<each unsupported or contradicted claim, quoted, with why>"], "notes": "<one line>"}

Rules for the issues list:
- List ONLY claims that are unsupported or contradicted. NEVER list a supported claim, even with commentary - if a claim matches the evidence, it does not belong in issues at all.
- "verdict" is "fail" if and only if issues is non-empty. Empty issues -> "pass". Unanswered parts of the question count as an issue.
- Before adding an issue, re-read the evidence once more; if the claim is actually there (allowing rounding/paraphrase per above), do not add it."""


class RabivyAgent:
    """The orchestrator. One instance = one conversation (memory included).

    ask() returns a dict:
      answer        - final grounded answer text
      steps         - [{tool, input, result_digest}] every tool call made
      verification  - {"verdict": ..., "issues": [...]} from the audit pass
      revised       - True if the answer was rewritten after a failed audit
    """

    def __init__(self, llm=None, verify=True):
        self.llm = llm or get_llm()
        self.verify = verify
        self.history = []       # persistent across ask() calls = memory
        self._log = print

    # ------------------------------------------------------------------
    def ask(self, question):
        self.history.append({"role": "user", "content": question})
        evidence = []   # every tool result this turn, for verification
        steps = []

        # ---- THE AGENT LOOP: plan -> act -> observe -> repeat ----
        answer = self._run_loop(evidence, steps)

        # ---- VERIFICATION PASS: audit the draft against the evidence ----
        verification = {"verdict": "not_checked", "issues": []}
        audit_trail = []          # every audit round, for full transparency
        draft_answer = answer     # kept so a revision can be diffed against it
        revised = False
        if self.verify and evidence:
            verification = self._full_audit(question, answer, evidence)
            audit_trail.append({"stage": "draft", **verification})
            if verification.get("verdict") == "fail" and not isinstance(self.llm, MockLLM):
                # One revision round: hand the auditor's findings back and
                # ask for a corrected answer - then re-audit it.
                self._log("  ! verification failed - revising answer once")
                self.history.append({"role": "assistant", "content": answer})
                self.history.append({"role": "user", "content":
                    "An automated audit found problems with that answer:\n"
                    + json.dumps(verification["issues"], indent=2)
                    + "\nRewrite the answer using ONLY facts present in the tool "
                      "results above. Remove or correct every flagged claim. If "
                      "part of the question cannot be answered from the "
                      "evidence, say so plainly. Write the corrected answer as a "
                      "standalone response addressed to the original user - do "
                      "not address the auditor, apologize, or mention that an "
                      "audit or rewrite happened."})
                # The rewrite goes through the SAME tool loop as the
                # original answer - a revising model often (correctly)
                # wants to re-query a tool to fix a flagged number, and
                # a plain text-only completion here would capture only
                # its "I'll rewrite..." preamble instead of the actual
                # revised answer (defect originally caught in UI testing).
                revised_answer = self._run_loop(evidence, steps)
                if revised_answer:
                    answer = revised_answer
                    revised = True
                    verification = self._full_audit(question, answer, evidence)
                    audit_trail.append({"stage": "revised", **verification})

        # ---- Close the turn in memory, trimmed so history can't balloon ----
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()

        return {"answer": answer, "steps": steps, "evidence": evidence,
                "verification": verification, "revised": revised,
                "draft_answer": (draft_answer if revised else None),
                "audit_trail": audit_trail}

    # ------------------------------------------------------------------
    def _run_loop(self, evidence, steps):
        """The plan -> act -> observe loop, from the current history to a
        final text. Used for both the initial answer and post-audit
        revisions, so both may call tools."""
        for _ in range(MAX_STEPS):
            resp = self.llm.complete(SYSTEM_PROMPT, self.history, tools=TOOL_SCHEMAS)
            tool_calls = [b for b in resp["content"] if b.get("type") == "tool_use"]

            if resp["stop_reason"] != "tool_use" or not tool_calls:
                return "".join(b.get("text", "") for b in resp["content"]
                               if b.get("type") == "text").strip()

            # Record the assistant's plan turn, run every requested tool,
            # feed all results back in one user turn (Anthropic protocol).
            self.history.append({"role": "assistant", "content": resp["content"]})
            results_block = []
            for call in tool_calls:
                self._log(f"  -> tool: {call['name']}({json.dumps(call['input'])[:120]})")
                result = run_tool(call["name"], call["input"])
                evidence.append({"tool": call["name"], "input": call["input"],
                                 "result": result})
                steps.append({"tool": call["name"], "input": call["input"],
                              "result_digest": self._digest(result)})
                results_block.append({"type": "tool_result",
                                      "tool_use_id": call["id"],
                                      "content": json.dumps(result)})
            self.history.append({"role": "user", "content": results_block})
        return ("I hit my step limit before finishing this question - "
                "here is what I found so far, treat it as incomplete.")

    # ------------------------------------------------------------------
    @staticmethod
    def _ghost_npis(answer, evidence):
        """Deterministic identifier gate. Language models can garble long
        digit strings when transcribing them into an answer, and an LLM
        auditor can miss the discrepancy for the same tokenization
        reason - a real incident: a one-digit-off NPI survived the audit
        and was only exposed by an exact lookup one question later.
        Whether an NPI appears in the evidence is set membership, not
        judgment, so it is checked in code: every 10-digit number in the
        answer must literally appear somewhere in the tool results."""
        known = set(_NPI_PATTERN.findall(json.dumps(evidence, default=str)))
        used = set(_NPI_PATTERN.findall(answer))
        return sorted(used - known)

    def _full_audit(self, question, answer, evidence):
        """LLM audit for meaning + deterministic check for identifiers.
        Either one failing fails the whole audit."""
        v = self._verify(question, answer, evidence)
        ghosts = self._ghost_npis(answer, evidence)
        if ghosts:
            if not isinstance(v.get("issues"), list):
                v["issues"] = []
            v["issues"].extend(
                f"NPI {g} does not appear in any tool result - probable "
                f"transcription error (deterministic identifier check)"
                for g in ghosts)
            v["verdict"] = "fail"
        return v

    # ------------------------------------------------------------------
    def _verify(self, question, answer, evidence):
        """Second, independent LLM pass: audit the answer against the raw
        tool results. Independent = fresh context, no system prompt telling
        it to be helpful - its ONLY job is to catch ungrounded claims."""
        evidence_text = json.dumps(evidence, indent=1)[:12000]
        prompt = (f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
                  f"EVIDENCE (tool results):\n{evidence_text}")
        try:
            resp = self.llm.complete(VERIFY_SYSTEM,
                                     [{"role": "user", "content": prompt}],
                                     tools=None, max_tokens=800,
                                     temperature=0.0)  # audits should be maximally deterministic
            text = "".join(b.get("text", "") for b in resp["content"]
                           if b.get("type") == "text")
            parsed = self._extract_json(text)
            if parsed is None:
                return {"verdict": "audit_error",
                        "issues": [f"auditor returned unparseable output: {text[:200]}"]}
            return self._strip_self_contradicted_issues(parsed)
        except Exception as e:
            return {"verdict": "audit_error", "issues": [f"{type(e).__name__}: {e}"]}

    @staticmethod
    def _strip_self_contradicted_issues(parsed):
        """Safety net for a real, reproducible auditor failure (found
        2026-07-28): the auditor sometimes writes its own reasoning that
        an issue is 'supported'/'not an issue'/'not a real issue', then
        still includes it in the issues list with verdict=fail -
        contradicting its own stated conclusion. A prompt instruction
        alone didn't reliably stop this (it recurred even with an
        explicit matching example already in the prompt), so this is a
        plain-code filter that can't be talked out of working: any issue
        string containing the auditor's own self-contradiction phrasing
        is dropped before the verdict is finalized."""
        issues = parsed.get("issues")
        if not isinstance(issues, list):
            return parsed
        self_contradiction_markers = ("not an issue", "not a real issue", "no issue",
                                      "this is supported", "is supported,")
        kept = [i for i in issues if not any(m in str(i).lower() for m in self_contradiction_markers)]
        if len(kept) != len(issues):
            parsed["issues"] = kept
            parsed["verdict"] = "pass" if not kept else "fail"
        return parsed

    @staticmethod
    def _extract_json(text):
        """Pull the first valid JSON object out of the auditor's reply,
        tolerating stray text before/after it (a real failure mode seen
        in live runs: 'JSONDecodeError: Extra data')."""
        start = text.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break  # malformed - try the next '{'
            start = text.find("{", start + 1)
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _digest(result):
        """One-line summary of a tool result for the steps log."""
        if "error" in result:
            return f"ERROR: {result['error'][:100]}"
        if "rows" in result:
            return f"{result['total_matches']} matches, {result['returned']} returned"
        if "sections" in result:
            ids = [s["chunk_id"][:40] for s in result["sections"]]
            flag = " LOW-CONF" if result.get("low_confidence") else ""
            return f"{len(ids)} sections{flag}: {ids}"
        if "row" in result:
            return f"NPI {result['row'].get('npi')} + card={'snapshot_card' in result}"
        return json.dumps(result)[:120]

    def _trim_history(self):
        """Keep conversation memory but drop old tool-call plumbing: past
        turns collapse to plain question/answer pairs, and only the last
        MAX_HISTORY_TURNS pairs are kept. The information the LLM needs
        for follow-ups survives; the token bloat doesn't."""
        qa = [m for m in self.history
              if isinstance(m["content"], str) and m["role"] in ("user", "assistant")]
        self.history = qa[-(MAX_HISTORY_TURNS * 2):]


# ----------------------------------------------------------------------
def _print_result(res):
    print("\n" + "=" * 72)
    print(res["answer"])
    print("-" * 72)
    v = res["verification"]
    print(f"verification: {v.get('verdict')}"
          + (f" | issues: {v['issues']}" if v.get("issues") else "")
          + (" | answer was auto-revised after a failed audit" if res["revised"] else ""))
    print(f"tool calls   : {[s['tool'] for s in res['steps']]}")


if __name__ == "__main__":
    agent = RabivyAgent()
    if len(sys.argv) > 1:
        _print_result(agent.ask(" ".join(sys.argv[1:])))
    else:
        print("Rabivy Commercial Intelligence Assistant - interactive mode")
        print("(conversation memory is ON - follow-ups work; Ctrl-C or 'quit' to exit)")
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            _print_result(agent.ask(q))