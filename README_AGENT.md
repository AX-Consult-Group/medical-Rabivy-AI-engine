# Agentic Layer

This layer completes Phases 5 and 6 of the project scope: an LLM agent that
plans and executes multi-step tool use over the two retrieval engines, an
evaluation harness with CI integration, and a conversational interface.

The retrieval engines themselves are unchanged. The agent consumes them as
tools. The original keyword router (`ask_a_question.py`) remains in the repo
as a no-LLM baseline.

All data is synthetic. Rabivy and AX Pharmaceuticals are fictional (see the
disclaimer in the main README).

## Design

The core rule is a strict separation between facts and language. Exact
numbers come from the structured query engine (a pandas layer over the
15,000-record prescriber table). Narrative content comes from semantic
retrieval over the embedded document corpus. The language model plans tool
calls, combines results, and writes the response. It is not the source of
any fact, and a verification step enforces that.

Request lifecycle:

1. The agent receives a question, plus the conversation history.
2. The model selects one or more tools and parameters. Results are fed
   back, and the model may issue further calls (up to 8 rounds). A
   retrieval flagged low-confidence is retried once with reformulated
   terms; a failed tool call is retried once with corrected input.
3. The model writes an answer with inline source citations
   (`[hcp_table]`, `[doc: <chunk_id>]`).
4. A separate audit call (fresh context, temperature 0) classifies each
   claim in the draft as supported, unsupported, or contradicted by the
   accumulated tool results. On failure, the agent revises once, with
   tool access, and re-audits.
5. The answer, tool trace, and audit verdict are returned. History is
   retained, so follow-up questions resolve against prior turns.

## Components

| File | Role |
|---|---|
| `agent.py` | Orchestrator: tool loop, synthesis, audit and revision, session memory. CLI and interactive REPL. |
| `agent_tools.py` | The two engines exposed as five typed tools (JSON Schema). Results are field-selected and rounded at the boundary. |
| `llm_client.py` | Single LLM interface: Anthropic API when `ANTHROPIC_API_KEY` is set, otherwise a deterministic offline stub for zero-cost testing. |
| `chat_ui.py` | Local web interface (standard library only). Renders the tool trace and audit verdict alongside each response. |
| `test_the_agent.py` | Agent-level evaluation: 22 scenarios plus a conversation-memory scenario. Nonzero exit on failure. |
| `.github/workflows/eval.yml` | Runs the build and the evaluation on every push and pull request. |
| `embedding_backend.py` | Embedding abstraction: MiniLM by default, TF-IDF/LSA fallback when the model is unavailable. The store records which backend built it. |
| `distill_router.py` | Experiment: derives deterministic fast-path routing rules from observed agent decisions, promoting only rules with 100% precision against the evaluation history. |

## Evaluation

Three mechanisms, at different points in the lifecycle:

- **Golden set (offline).** `test_the_agent.py` runs the scenario bank
  end-to-end and checks tool selection, retrieval, and (with a live model)
  answer content. Runs are timestamped into `eval_runs/` for comparison
  over time. CI executes this on every change, in offline mode.
- **Per-response audit (runtime).** Every answer is checked against its
  own evidence before it is returned, independent of the test set.
- **Citations (always).** Each claim is traceable to a table query or a
  document chunk by ID.

Offline mode exercises routing and retrieval deterministically at no cost;
the live configuration additionally evaluates synthesis quality and the
audit loop. Both configurations pass the full scenario bank.

## Running

```bash
pip install -r requirements.txt
python main.py                 # build the knowledge base (chunk, tag, embed)

python test_the_agent.py       # evaluation; runs offline without an API key

export ANTHROPIC_API_KEY=...   # Windows: set ANTHROPIC_API_KEY=...
python agent.py "Who should I target next month in New York, and what should I say to them?"
python agent.py                # interactive session
python chat_ui.py              # web interface at http://localhost:8017
```

## Router distillation experiment

`distill_router.py` addresses the maintenance cost of hand-written routing
rules. It mines the evaluation history for observed question-to-tool
decisions, generates candidate regular-expression rules, and accepts a rule
only if every historical question it matches was routed to the tool it
predicts. Accepted rules could serve as a zero-cost fast path in front of
the model-based router; rejected rules never ship. In testing, the
validation gate rejected a substring pattern (`top `) that matches inside
"stop taking" — the same class of defect previously found by hand in the
baseline router — and accepted a rule set covering 82% of observed traffic.

## Attribution

Phases 1–4 (knowledge repository, propensity model, structured and semantic
retrieval engines, baseline router and its evaluation) were built by a
collaborator; see the main README and commit history. This layer, Phases
5–6, builds on that foundation without modifying the engines. Two files
were touched to introduce the embedding fallback (`3_create_embeddings.py`,
`search_documents.py`); everything else is additive.
