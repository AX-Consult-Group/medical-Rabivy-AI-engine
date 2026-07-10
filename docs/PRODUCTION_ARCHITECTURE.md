# Rabivy Commercial AI Engine - How It Works in Production

A one-page explainer of how the knowledge repository is fed, joined, and kept current in a real deployment. The showcase runs on synthetic data; this is the architecture the same design maps onto once licensed data is in place.

## The join key: NPI

Every US prescriber has one permanent, government-issued **National Provider Identifier (NPI)** - a public 10-digit number. It is the universal key that every commercial data source is tied to. All prescriber data joins on NPI. In our simulation the NPI is the primary identifier; the `HCP####` label is only an internal convenience.

## Where the data comes from

AX Pharmaceuticals does not receive one file. It licenses separate feeds, each keyed to NPI, and joins them internally:

| Data layer | Source | Typical refresh |
|----|----|----|
| Provider identity (name, specialty, address) | NPPES (public), IQVIA OneKey | Quarterly |
| Rx / NRx volume by product | IQVIA Xponent, Symphony Health | Weekly–monthly |
| Payer mix, formulary, PA burden | Komodo Health, Optum (claims) | Monthly |
| Rep activity, samples, targeting | Veeva CRM (internal) | Continuous |
| Propensity & switching scores | Built in-house (Phase 2 model) | On data refresh |

Real feeds arrive as **CSV / Parquet files on SFTP or cloud storage, or via API** — increasingly delivered straight into a cloud warehouse (Snowflake / Databricks). Excel is only a human-friendly stand-in; it is not the delivery format.

## How it becomes the master

```         
   NPPES ─────┐
   IQVIA ─────┤
   Komodo ────┼──► join on NPI ──►  MASTER TABLE  ──►  one row per prescriber
   Veeva ─────┤                     (the .xlsx we simulate)
   Scores ────┘
```

One row per NPI, columns drawn from every source. That joined table is the single source of truth for both the propensity model and the knowledge repository.

## How the knowledge repository stays current

The repository is not a frozen snapshot. It regenerates from the master on a schedule:

1.  **New vendor files land** (e.g. weekly IQVIA Rx drop).
2.  **A scheduled pipeline re-runs the join**, rebuilding the master table. This is the same generator script we already have, run automatically instead of by hand.
3.  **HCP cards regenerate** from the refreshed master — each card carries a `data as of` date.
4.  **Only changed cards are re-embedded** into the vector database, keyed by NPI. Unchanged cards and the static strategy/clinical documents stay put.

The practical point: the "pipeline" is our existing card generator plus a scheduler. Dynamic updating means *run it when new data arrives, re-embed only what changed.* No exotic infrastructure.

## Two retrieval paths (why both exist)

- **Structured engine** answers exact numeric and ranking questions ("top prescriber in NY", "scripts written by NPI X") by querying the master table directly — fast, exact, no hallucination.
- **RAG / vector search** answers characterisation questions ("summarise this doctor", "which endos look like switch targets") by retrieving the natural-language HCP cards and reference documents.

A query router decides which path each question takes. Numbers go to the table; narrative goes to the cards.

## Honest constraints (worth stating up front)

- Licensed prescriber data is **expensive and contractually restricted** — full US coverage requires vendor contracts, and some claims data is de-identified at the patient level. This is exactly why the showcase runs on synthetic data.
- The synthetic master is **structurally faithful**: every column maps to a real, purchasable field keyed to NPI. When real feeds are contracted, the same pipeline, schema, and repository design carry over unchanged.

## What the showcase demonstrates

We do not need the live pipeline to prove the concept. We demonstrate one refresh cycle: change rows in the master, re-run the generator, and show the HCP card — and the AI answer — update accordingly. That proves the architecture works end to end.
