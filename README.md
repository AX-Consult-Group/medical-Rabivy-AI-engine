# Rabivy AI Engine

## What this is

This is a natural-language question-answering tool for a pharmaceutical sales team. A rep can ask something like *"Who should I target next month in New York, and what should I say to them?"* and get back a real, data-grounded answer - not a guess, and not something the AI made up.

It works two ways, both included in this repo:

- **A fast, free, deterministic router** (`ask_a_question.py`) - handles a question by matching keywords with regex, then calling one of two engines directly. No AI involved in the routing itself, so it's instant and costs nothing to run.
- **An AI agent** (`agent.py`) - actually understands the question, decides which tool(s) to call (possibly several, chained together), and writes a real answer in its own words - while still only ever using real numbers pulled from the two engines, never inventing one.

Both paths share the exact same two underlying engines, so neither one can drift out of sync with what the data actually says.

**Disclaimer:** Rabivy and AX Pharmaceuticals are fictional, created solely for this demonstration/portfolio project. Not commissioned by any pharmaceutical company. For the original project plan this was scoped against, see [PROJECT_BRIEF.md](PROJECT_BRIEF.md).

---

## The full pipeline, visually

![Graphic breakdown of the full AI engine pipeline](figures/rabivy_layered_schematic_V3.svg)

Every file below is arranged in layers - each layer only ever talks to the layer directly above or below it. Read top to bottom.

---

## Layer by layer

**Layer 1 - turns raw documents into searchable pieces.** Three scripts run once, in order, to prepare the knowledge base: `1_chunk_documents.py` splits every source document into small, labeled pieces (one chunk per section or HCP card); `2_tag_chunks.py` adds extra labels to the chunks so they are easily traceable; `3_create_embeddings.py` converts each piece into a vector of numbers that captures its meaning, so the system can search by meaning, not just keywords.

**Layer 2 - the orchestrator.** `main.py` runs all three Layer 1 steps in the correct order, checking after each one that it actually worked before moving on. This is the one file to run to rebuild the knowledge base from scratch.

**Layer 3 - the two raw sources everything else reads from.** The `output/` folder is what `main.py` actually produces - a saved, ready-to-search copy of every narrative document (Layer 1). The master spreadsheet (`data/*.xlsx`) is a separate source entirely, holding the real HCP-level prescribing and propensity data.
 
In a real deployment, both of these would be built from bought and licensed data, not typed by hand:
 
- **The spreadsheet's structured fields** would typically come from vendors like IQVIA (Xponent, OneKey), Symphony Health, or Komodo Health for prescribing volume and payer mix, plus Veeva CRM for rep activity and call history - each HCP record merged together by NPI (National Provider Identifier), the one ID that's consistent across every source.
- **The narrative documents Layer 1 chunks up** would be the unstructured side: clinical trial write-ups and publications, conference proceedings and poster presentations, competitive intelligence briefs, rep talking points and objection-handling guides, and payer/drug-benefit coverage documents.

In this project, all of the above is synthetic data generated to look like the real thing, not actual licensed data.

**Layer 4 - the two engines.** `query_spreadsheet.py` is the structured query engine - it answers ranking, counting, and filtering questions by looking directly at the spreadsheet. `search_documents.py` is the document search engine - it finds the most relevant chunks for a narrative question, using whichever embedding model built the store (see `embedding_backend.py` below).

**Layer 5 - the two paths that actually ask a question.** This is where the two approaches diverge, both sitting on the exact same Layer 4 engines underneath:
- **Path A, the regex router:** `ask_a_question.py` reads a question with keyword/regex rules, decides which engine(s) it needs, calls them, and returns the raw answer - no real understanding, just pattern matching. `test_the_system.py` sends 30 real questions (golden test set) through it automatically and reports what passed or failed.
- **Path B, the agent:** `agent_tools.py` is the bridge that turns a tool call the agent picks into a real call on the two engines above. `test_the_agent.py` sends the exact same 30 questions through the agent instead, and scores tool selection, retrieval, and (with a live model) answer quality.

**Layer 6 - the agent's brain.** `llm_client.py` is the actual interface to Claude - `agent.py` calls it twice per question, once to pick a tool, once to write the final answer. Uses the real API if a key is set, otherwise a free, offline stand-in with no AI at all, for zero-cost testing. `agent.py` itself is the orchestrator: it runs the tool-selection loop, writes the answer, then runs a separate audit pass checking its own answer against the evidence before returning it, revising once if the audit fails.

**Layer 7 - what a rep actually opens.** `chat_ui.py` is a local web interface showing the answer plus the tool trace and audit verdict alongside it, so nothing is a black box.

**Standalone - real, but not part of either path above.** `propensity_model.py` encodes the scoring formula as runnable code; `main.py` runs it automatically before building anything (Stage 0), so a stale or wrong score gets caught immediately rather than silently flowing downstream. `challenger_validation.py` checks `propensity_model.py`'s scores against real (or simulated) outcomes, and separately trains its own data-driven model to flag patterns the hand-built scorecard might be missing. `distill_router.py` is an experiment: it looks at the agent's own past tool-choice decisions and tries writing free regex rules that would reproduce them, promoting a rule only if it's 100% correct on every past match - not wired into `chat_ui.py` yet.

---

## Every file, in one table

| File | Layer | What it does |
|---|---|---|
| `1_chunk_documents.py` | 1 | Splits source documents into small, labeled, searchable pieces |
| `2_tag_chunks.py` | 1 | Adds competitor/brand tags to each piece |
| `3_create_embeddings.py` | 1 | Converts each piece into a vector for meaning-based search |
| `main.py` | 2 | Runs the whole build pipeline, in order, with checks |
| `output/` folder | 3 | What `main.py` produces - the saved, searchable documents |
| `data/*.xlsx` (master spreadsheet) | 3 | The real HCP prescribing/propensity data |
| `query_spreadsheet.py` | 4 | Structured engine - ranking, counting, filtering |
| `search_documents.py` | 4 | Document engine - meaning-based search over chunks |
| `ask_a_question.py` | 5 (Path A) | Regex router - no AI, keyword matching only |
| `test_the_system.py` | 5 (Path A) | Runs 30 questions through the router, scores results |
| `agent_tools.py` | 5 (Path B) | Bridges the agent's tool calls to the two real engines |
| `test_the_agent.py` | 5 (Path B) | Runs the same 30 questions through the agent, scores results |
| `llm_client.py` | 6 | The real interface to Claude (or a free offline stand-in) |
| `agent.py` | 6 | Orchestrator - tool loop, answer synthesis, self-audit |
| `chat_ui.py` | 7 | Local web interface showing the answer plus its evidence |
| `embedding_backend.py` | standalone | Shared fallback so build and query never use mismatched embedding models |
| `propensity_model.py` | standalone | The scoring formula as code; guards `main.py` against stale scores |
| `challenger_validation.py` | standalone | Checks the scorecard against real outcomes and a learned model |
| `distill_router.py` | standalone | Experiment: turns the agent's own past decisions into free regex rules |

---

## Running it

```bash
pip install -r requirements.txt

python main.py                 # build the knowledge base (chunk, tag, embed)

python test_the_system.py      # test Path A - the regex router (free, no API key)
python test_the_agent.py       # test Path B - the agent (free in mock mode, no API key needed)

export ANTHROPIC_API_KEY=...   # Windows: set ANTHROPIC_API_KEY=...
python agent.py "Who should I target next month in New York, and what should I say to them?"
python agent.py                # interactive session, with conversation memory
python chat_ui.py               # web interface at http://localhost:8017
```

`test_the_agent.py` runs in one of two modes automatically, depending on whether a key is set:
- **Mock mode** (no key) - free, deterministic, checks tool selection and retrieval only.
- **Real mode** (key set) - additionally checks answer quality and the audit verdict, using the real model.

CI (`.github/workflows/eval.yml`) runs the full build plus both test files, in mock mode, on every push and pull request - so a change that breaks routing, retrieval, or the agent loop turns the commit red before it reaches anyone. It also runs `challenger_validation.py`'s model-gap digest.

---

## Current status / known limitations

- **Retrieval confidence on some narrative questions is modest** - a known limitation of the fast, free embedding model on short or informally-phrased questions, not a bug. A better embedding model would fix it; parked for now since this is a demonstration system.
- **`distill_router.py` is a real experiment, not live** - it isn't wired into `chat_ui.py`, so it doesn't affect what a rep actually sees today.
- Both Path A and Path B sit on the same Layer 4 engines - a change to those engines affects both, so both are kept in sync deliberately, not by accident.

---

## Attribution

Phases 1-4 (knowledge repository, propensity model, structured and semantic retrieval engines, the regex router and its evaluation) and Phases 5-6 (the agentic layer, evaluation harness, and conversational interface) were both built as part of this project, by collaborators working on their own layers of the same system - see commit history for the detailed timeline. The regex router (Path A) and the agent (Path B) were deliberately kept as two separate, comparable paths over the same underlying engines, rather than one replacing the other.