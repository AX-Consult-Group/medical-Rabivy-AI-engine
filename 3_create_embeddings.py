# 3_create_embeddings.py
# -------------------------------------------------------------------
# Turns each tagged chunk's text into an EMBEDDING: a vector (list of
# numbers) that captures the chunk's meaning. Chunks with similar meaning
# get similar vectors, which is what makes "search by meaning" possible.
# Input:  output/chunks_tagged.json
# Output: output/embeddings.npy        (the vectors, aligned row-for-row)
#         output/embedding_ids.json    (chunk_id per row, for verifying
#                                        that alignment instead of trusting it)
#         output/embedding_meta.json   (which model/dim produced these vectors)
# -------------------------------------------------------------------

import json
import os
import numpy as np
from embedding_backend import get_build_backend

# STEP 1: Load the tagged chunks.
with open("output/chunks_tagged.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks.")

# STEP 2: Guard against empty text. An empty chunk still gets a vector from
# model.encode() - just a meaningless one that can pollute search results
# with no error to say why. Fail loudly here instead, since a silent skip
# would also break the row-for-row alignment with chunks_tagged.json.
empty_ids = [c["chunk_id"] for c in chunks if not c["text"].strip()]
if empty_ids:
    raise ValueError(
        f"{len(empty_ids)} chunk(s) have empty text - fix upstream in "
        f"1_chunk_documents.py before embedding. Affected chunk_ids "
        f"(first 10 shown): {empty_ids[:10]}"
    )

# STEP 3: Load the embedding backend.
# v1 choice: all-MiniLM-L6-v2 is a solid general-purpose default for
# prototyping, not a medical/pharma-tuned model. Worth revisiting (a
# domain-tuned model, a bigger one, or a hosted embedding API) once this
# moves off simulated data.
# The FIRST time you run this it downloads a ~90MB model, then caches it.
# If the model can't be downloaded at all (offline/sandboxed machine),
# embedding_backend.py falls back to an offline TF-IDF+LSA backend and
# records that choice in embedding_meta.json so search stays consistent.
model = get_build_backend()
MODEL_NAME = model.name

# STEP 4: Collect the text of every chunk into a list.
texts = [c["text"] for c in chunks]

# STEP 5: Convert every text into a vector.
print("Embedding chunks...")
embeddings = model.fit_encode(texts)

# STEP 6: Save the vectors. Row order matches chunks_tagged.json exactly.
os.makedirs("output", exist_ok=True)
np.save("output/embeddings.npy", embeddings)

# STEP 7: Save chunk_ids in the same row order, as a check on that
# alignment rather than a silent assumption. If chunks_tagged.json is ever
# regenerated in a different order, this file lets anything downstream
# verify (or re-align by ID) instead of trusting position i == position i.
embedding_ids = [c["chunk_id"] for c in chunks]
with open("output/embedding_ids.json", "w", encoding="utf-8") as f:
    json.dump(embedding_ids, f, indent=2)

# STEP 8: Save what produced these vectors. Same traceability principle as
# chunk_manifest.json - not just "which document did this come from" but
# "which model/version produced this vector," so a future model upgrade
# doesn't silently mix incompatible vector sizes into the same store.
meta = {
    "model": MODEL_NAME,
    "dim": int(embeddings.shape[1]),
    "n_chunks": int(embeddings.shape[0]),
}
with open("output/embedding_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# STEP 9: Report what we produced.
print(f"\nDone. Created {embeddings.shape[0]} vectors, each {embeddings.shape[1]} numbers long.")
print("Saved output/embeddings.npy, output/embedding_ids.json, output/embedding_meta.json")
print("\nExample - the first chunk as a vector (first 8 of its numbers):")
print(embeddings[0][:8])