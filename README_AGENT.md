# The Agentic Layer (Phases 5–6)

This adds an **agentic layer** on top of the existing retrieval system —
the "AI Evaluation Layer" and "Conversational Interface" phases from the
main README, built as one coherent agent. The two retrieval engines are
untouched: the agent *uses* them as tools. What gets replaced is the
regex **router** (`ask_a_question.py`), which still works standalone and
remains the no-LLM baseline.

All data remains synthetic/fictional (Rabivy and AX Pharmaceuticals are
fictional — see the main README's disclaimer).

## What the agent adds

| Capability | Where | What it means |
|---|---|---|
| LLM tool planning | `agent.py` + `agent_tools.py` | The model decides which engine(s) to call, with what parameters — no more keyword regexes deciding routing |
| Multi-step chaining | `agent.py` loop | "Compare NPI X to a typical endocrinologist" = lookup + benchmark retrieval + synthesis, in one turn |
| Multi-source synthesis | `agent.py` | The Q20 showpiece works: "who should I target **and** what should I say" joins the propensity table and the messaging docs in one cited answer |
| Retrieval self-correction | system prompt + `search_documents` tool contract | On `low_confidence=true` the agent rewords the query and retries; tool errors are read, fixed, retried |
| Answer verification | `agent.py` `_verify()` | An independent second LLM pass audits every claim in the draft against the raw tool results; failed audits trigger one revision + re-audit |
| Conversation memory | `RabivyAgent.history` | "Why is **this** doctor not converting?" resolves from the session |
| Zero-cost testing | `llm_client.py` `MockLLM` | The full agent stack runs deterministically with no API key — behaviour (tool selection + retrieval) is testable for free |

## Files

- `llm_client.py` — one LLM interface: Anthropic Claude (when `ANTHROPIC_API_KEY` is set) or a deterministic mock (otherwise / `AGENT_LLM=mock`)
- `agent_tools.py` — the two engines exposed as 5 typed tools: `query_hcp_table`, `lookup_hcp`, `count_active_writers`, `states_summary`, `search_documents`
- `agent.py` — the orchestrator: plan → act → observe loop, synthesis, verification pass, session memory, CLI/interactive chat
- `test_the_agent.py` — agent-level eval: the same question bank as `test_the_system.py`, now expected to come back as complete answers; plus a conversation-memory scenario
- `embedding_backend.py` — pluggable embedding backend: MiniLM as before, with an offline TF-IDF+LSA fallback (recorded in `embedding_meta.json`) for machines that can't reach HuggingFace. `3_create_embeddings.py` and `search_documents.py` now import it; behaviour on a normal machine is unchanged.

## Run it

```bash
pip install -r requirements.txt
python main.py                      # rebuild the knowledge base (unchanged)

# no API key needed - free, deterministic:
python test_the_agent.py            # behavioural eval (tool selection + retrieval)

# with a real LLM:
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=sk-ant-...
python agent.py "Who should I target next month in New York, and what should I say to them?"
python agent.py                     # interactive chat, conversation memory on
python test_the_agent.py            # full eval incl. answer quality + verification
```

## Grounding principle

Exact numbers only ever come from the structured engine; narrative
content only ever comes from retrieval. The LLM plans, joins, and
phrases — and the verification pass rejects any claim that isn't in the
retrieved evidence. The model is never the source of a fact.
