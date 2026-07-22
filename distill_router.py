# distill_router.py
# -------------------------------------------------------------------
# ROUTER DISTILLATION (future-work demo): can the agent write its own
# regex fast-path, so the deterministic rules don't have to be
# hand-maintained?
#
# The economics: the LLM router understands anything but costs tokens
# and latency on every question. A regex fast-path is free but
# laborious to hand-tune (see ask_a_question.py's 600 lines of
# carefully-debugged patterns). This script closes the loop:
#
#   1. MINE     - read the saved agent eval runs (eval_runs/agent_eval_*.json):
#                 every (question -> first tool the LLM chose) pair is a
#                 labelled routing decision, produced by the smart router.
#   2. PROPOSE  - ask the LLM to write candidate regex rules that
#                 reproduce those decisions (or use built-in seed
#                 candidates in mock mode - including a deliberately
#                 broken one, to show the gate catching it).
#   3. VALIDATE - the crucial step: every candidate is tested against
#                 the full observed question set. A rule is promoted
#                 ONLY if every question it matches was actually routed
#                 to the tool it predicts (100% precision). This is the
#                 same discipline that caught the human-written
#                 '"top " matches inside "stop taking"' bug - applied
#                 automatically to machine-written rules.
#   4. REPORT   - promoted rules go to output/distilled_rules.json with
#                 coverage stats: "X% of observed traffic could take the
#                 free fast path; the rest stays with the LLM router."
#
# The labour doesn't disappear - it moves up a level, from writing
# regexes to curating the eval set that judges them. That's the point.
#
#   python distill_router.py            (uses Claude if ANTHROPIC_API_KEY set)
#   AGENT_LLM=mock python distill_router.py   (free, seed-candidate demo)
# -------------------------------------------------------------------

import glob
import json
import os
import re
import sys

from llm_client import get_llm, MockLLM

# Deliberate mix for the mock/seed demo: two good rules, one rule that
# LOOKS reasonable but is too greedy (matches narrative questions), and
# the classic substring bug. The validation gate should promote the
# first two and reject the last two.
SEED_CANDIDATES = [
    {"pattern": r"\bhow many\b.*\b(writers?|prescribers?|hcps?)\b", "tool": "count_active_writers"},
    {"pattern": r"\bwhich states\b|\bstates (?:have|with) the most\b", "tool": "states_summary"},
    {"pattern": r"\bhigh\b", "tool": "query_hcp_table"},          # too greedy - should be rejected
    {"pattern": r"top ", "tool": "query_hcp_table"},              # the classic 'stop taking' bug - should be rejected
]

PROPOSE_SYSTEM = """You write routing rules. Given observed (question -> tool) decisions made by an LLM router, propose regex rules (Python re syntax, case-insensitive) that reproduce SOME of those decisions deterministically.

Rules will be auto-rejected unless EVERY observed question they match was routed to the tool they predict - so prefer precise, word-boundary patterns over broad ones. It is fine to cover only the clearest cases; uncovered questions simply stay with the LLM router.

Respond with ONLY a JSON array: [{"pattern": "...", "tool": "..."}, ...] (max 10 rules)."""


def mine_observations():
    """(question, first_tool) pairs from every saved agent eval run."""
    pairs = {}
    for path in sorted(glob.glob(os.path.join("eval_runs", "agent_eval_*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("results", []):
            tools = r.get("tools_used") or []
            if r.get("q") and tools:
                pairs[r["q"]] = tools[0]   # latest run wins for a repeated question
    return sorted(pairs.items())


def propose(llm, observations):
    if isinstance(llm, MockLLM):
        print("(mock mode: using built-in seed candidates - two good rules and "
              "two known-bad ones, so you can watch the gate work)")
        return SEED_CANDIDATES
    obs_text = "\n".join(f"- {q!r} -> {t}" for q, t in observations)
    resp = llm.complete(PROPOSE_SYSTEM,
                        [{"role": "user", "content": f"Observed decisions:\n{obs_text}"}],
                        tools=None, max_tokens=1200, temperature=0.0)
    text = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
    m = re.search(r"\[.*\]", text, re.DOTALL)
    return json.loads(m.group(0)) if m else []


def validate(candidates, observations):
    """Promote a rule only if every observed question it matches was in
    fact routed to the tool the rule predicts."""
    promoted, rejected = [], []
    for cand in candidates:
        try:
            rx = re.compile(cand["pattern"], re.IGNORECASE)
        except re.error as e:
            rejected.append({**cand, "why": f"invalid regex: {e}", "matched": 0})
            continue
        hits = [(q, t) for q, t in observations if rx.search(q)]
        wrong = [(q, t) for q, t in hits if t != cand["tool"]]
        if not hits:
            rejected.append({**cand, "why": "matches nothing observed", "matched": 0})
        elif wrong:
            example = wrong[0]
            rejected.append({**cand, "matched": len(hits),
                             "why": (f"{len(wrong)}/{len(hits)} matches routed elsewhere - "
                                     f"e.g. {example[0]!r} actually went to {example[1]}")})
        else:
            promoted.append({**cand, "matched": len(hits)})
    return promoted, rejected


if __name__ == "__main__":
    observations = mine_observations()
    if not observations:
        print("No eval runs found in eval_runs/ - run test_the_agent.py first.")
        sys.exit(1)
    print(f"Mined {len(observations)} observed routing decisions from eval_runs/.\n")

    llm = get_llm()
    candidates = propose(llm, observations)
    print(f"Candidate rules proposed: {len(candidates)}\n")

    promoted, rejected = validate(candidates, observations)

    print("PROMOTED (100% precision on observed traffic):")
    for r in promoted:
        print(f"  /{r['pattern']}/ -> {r['tool']}   (matched {r['matched']} questions)")
    print("\nREJECTED by the validation gate:")
    for r in rejected:
        print(f"  /{r['pattern']}/ -> {r['tool']}   ({r['why']})")

    covered = set()
    for r in promoted:
        rx = re.compile(r["pattern"], re.IGNORECASE)
        covered |= {q for q, _ in observations if rx.search(q)}
    pct = 100.0 * len(covered) / len(observations)
    print(f"\nDistilled fast-path coverage: {len(covered)}/{len(observations)} "
          f"observed questions ({pct:.0f}%) could skip the LLM router with zero "
          f"routing risk. Everything else stays with the LLM.")

    os.makedirs("output", exist_ok=True)
    out = {"promoted": promoted, "rejected": rejected,
           "coverage_pct": round(pct, 1), "n_observations": len(observations)}
    with open(os.path.join("output", "distilled_rules.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved output/distilled_rules.json")
