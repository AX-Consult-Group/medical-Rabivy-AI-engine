# 4_create_bm25_index.py
# -------------------------------------------------------------------
# Builds the BM25 keyword index - the second half of hybrid search.
# Same input as 3_create_embeddings.py, same output/ folder, but a
# lexical (exact word/number match) index instead of a meaning-based
# vector index. search_documents.py fuses the two rankings together at
# query time - see that file for how.
#
# Only indexes the narrative chunks (HCP cards excluded) - see
# bm25_index.py's header comment for why that matters, not just for
# speed.
#
# Input:  output/chunks_tagged.json
# Output: output/bm25_index.pkl
# -------------------------------------------------------------------

import json
from bm25_index import build_bm25_index

# STEP 1: Load the tagged chunks (same source 3_create_embeddings.py reads).
with open("output/chunks_tagged.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks.")

# STEP 2: Restrict to the same narrative-only pool search_documents.py's
# semantic_search() already searches over. HCP cards are looked up by
# exact NPI/state/specialty, never free-text searched - so indexing all
# 15,000 near-identical cards here would add nothing but slower builds
# and (per bm25_index.py) actively distort BM25's term-rarity stats.
narrative_chunks = [c for c in chunks if c["doc_type"] != "hcp_card"]
print(f"Indexing {len(narrative_chunks)} narrative chunks for keyword search "
      f"({len(chunks) - len(narrative_chunks)} HCP cards excluded).")

chunk_ids = [c["chunk_id"] for c in narrative_chunks]
texts = [c["text"] for c in narrative_chunks]

# STEP 3: Build and persist the index. None means build_bm25_index()
# already printed its own warning and skipped (rank_bm25 not installed) -
# that's an expected, non-fatal outcome here, not this script's job to
# re-explain, just to exit 0 either way so main.py's pipeline continues.
index = build_bm25_index(chunk_ids, texts)

if index is None:
    print("\nSkipped: output/bm25_index.pkl was not created. Install "
          "rank_bm25 and re-run this step to enable keyword search.")
else:
    print(f"\nDone. Saved output/bm25_index.pkl ({len(chunk_ids)} chunks indexed).")
