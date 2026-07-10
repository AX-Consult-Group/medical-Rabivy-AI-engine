# embed.py
# -------------------------------------------------------------------
# Turns each tagged chunk's text into an EMBEDDING: a vector (list of
# numbers) that captures the chunk's meaning. Chunks with similar meaning
# get similar vectors, which is what makes "search by meaning" possible.
# Input:  output/chunks_tagged.json
# Output: output/embeddings.npy   (the vectors, aligned row-for-row)
# -------------------------------------------------------------------

import json
import numpy as np
from sentence_transformers import SentenceTransformer

# STEP 1: Load the tagged chunks.
with open("output/chunks_tagged.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks.")

# STEP 2: Load the embedding model.
# The FIRST time you run this it downloads a ~90MB model, then caches it.
print("Loading embedding model (first run downloads ~90MB, please wait)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 3: Collect the text of every chunk into a list.
texts = [c["text"] for c in chunks]

# STEP 4: Convert every text into a vector.
print("Embedding chunks...")
embeddings = model.encode(texts, show_progress_bar=True)

# STEP 5: Save the vectors. Row order matches chunks_tagged.json exactly.
np.save("output/embeddings.npy", embeddings)

# STEP 6: Report what we produced.
print(f"\nDone. Created {embeddings.shape[0]} vectors, each {embeddings.shape[1]} numbers long.")
print("Saved output/embeddings.npy")
print("\nExample - the first chunk as a vector (first 8 of its numbers):")
print(embeddings[0][:8])