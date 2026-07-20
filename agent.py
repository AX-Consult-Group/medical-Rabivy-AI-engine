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
import sys

from llm_client import get_llm, MockLLM
from agent_tools import TOOL_SCHEMAS, run_tool

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

SELF-CORRECTION:
- If search_documents returns low_confidence=true, or the sections clearly don't address the question, retry ONCE with a reworded query using different vocabulary (synonyms, the underlying concept). If still weak, answer from the best available evidence and say confidence is low.
- If a tool returns {"error": ...}, read the error, fix your input, and retry once.

ANSWER STYLE:
- Lead with the direct answer, then supporting detail. Short, scannable, rep-friendly.
- Distinguish exact data ("55 scripts/month [hcp_table]") from document guidance ("emphasize monthly dosing [doc: ...]").
- For targeting questions include NPI, specialty, state, and the score that justifies the recommendation."""

VERIFY_SYSTEM = """You are a strict fact-checking auditor. You will receive a QUESTION, an ANSWER, and the EVIDENCE (tool results) the answer was based on.

Check every factual claim in the answer:
- supported: the claim appears in the evidence
- unsupported: the claim does NOT appear in the evidence (invented/hallucinated)
- contradicted: the evidence says something different

The following are SUPPORTED, never issues (do not flag them):
- rounding and formatting AT ANY PRECISION, including to whole percentages (0.610666 -> 0.61; 0.85 -> "85/100"; 0.3225 -> "32%"; 0.6362 -> "64%")
- simple arithmetic derived from the evidence (totals, differences, "10 of 11", rank ordering of listed values)
- paraphrase and interpretation that follows from the evidence (e.g. "overdue for contact" when days_since_contact is large)

Data dictionary (treat these as definitions, not claims to verify): propensity_score is 0-1, often displayed as N/100; propensity_rank is a NATIONAL rank across all 15,000 HCPs; tier values are High/Medium/Low/Watch; switching_score and pa_burden are 0-1.

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
        revised = False
        if self.verify and evidence:
            verification = self._verify(question, answer, evidence)
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
                      "evidence, say so plainly."})
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
                    verification = self._verify(question, answer, evidence)

        # ---- Close the turn in memory, trimmed so history can't balloon ----
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()

        return {"answer": answer, "steps": steps, "evidence": evidence,
                "verification": verification, "revised": revised}

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
            return parsed
        except Exception as e:
            return {"verdict": "audit_error", "issues": [f"{type(e).__name__}: {e}"]}

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
