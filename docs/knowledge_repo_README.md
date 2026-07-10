# Rabivy Commercial AI Engine - Knowledge Repository

Simulated document corpus for the Phase 3-4 RAG showcase. All per-HCP figures are drawn verbatim from the prescriber master dataset (n = 15,000 HCPs, reporting period June 2026); the strategic, rep-facing, and clinical documents are curated reference material.

## Structure

| Layer                 | Folder           | Docs   | Chunking               |
|-----------------------|------------------|--------|------------------------|
| Per-HCP snapshots     | `hcp_snapshots/` | 15,000 | one chunk per document |
| Reference / benchmark | `reference/`     | 2      | by section             |
| Rep-facing field      | `rep_field/`     | 3      | by section             |
| Strategic             | `strategic/`     | 2      | by section             |
| Clinical              | `clinical/`      | 3      | by section             |

Total corpus size: 26.7 MB.

## Chunking guidance

- **HCP snapshots**: embed each `.md` as a single chunk. Each file is named by NPI; NPI is the primary metadata key (with internal HCP ref as secondary).
- **Reference/rep/strategic/clinical docs**: split on `##` section headers. State market summary splits per-state.
- Suggested metadata tags per chunk: `doc_type`, `npi` (snapshots only), `state`, `specialty`, `source_type`, `date`.

## The GTM strategy PDF is ingested separately

The Phase 1 GTM whiteboard (existing PDF) is a real source document and is chunked directly at ingestion - it is not regenerated here.

## Production architecture

See `PRODUCTION_ARCHITECTURE.md` for how this repo is fed, joined on NPI, and refreshed dynamically in a real deployment.

## Provenance

Structured facts (Rx, NRx, payer, formulary, PA, propensity, tier, decile, targeting, engagement) trace to the master dataset. Market, coverage, and trial figures in the strategic/clinical docs are drawn from the GTM strategy references. Rabivy is investigational; all clinical content is Phase 2 / internal framing.
