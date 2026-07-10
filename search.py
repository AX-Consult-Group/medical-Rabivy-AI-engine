# search.py
# -------------------------------------------------------------------
# Ask a question in plain English; get back the chunks whose MEANING is
# closest to it. This is the "retrieval" in retrieval-augmented generation.
# Uses: output/chunks_tagged.json  +  output/embeddings.npy  (from embed.py)
# -------------------------------------------------------------------

import json
import numpy as np
from sentence_transformers import SentenceTransformer

# STEP 1: Load the chunks and their embeddings (same row order).
with open("output/chunks_tagged.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
embeddings = np.load("output/embeddings.npy")
print(f"Loaded {len(chunks)} chunks and their embeddings.\n")

# STEP 2: Load the SAME model used in embed.py (so questions and chunks
# live in the same "meaning space"). Uses the model already downloaded.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-normalise the chunk vectors once (needed for cosine similarity).
chunk_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


# STEP 3: The search function.
def search(question, top_k=3):
    # Turn the question into a vector with the same model.
    q = model.encode([question])[0]
    q = q / np.linalg.norm(q)
    # Cosine similarity: how aligned each chunk is with the question (1 = identical meaning).
    scores = chunk_norms @ q
    # Indices of the highest-scoring chunks, best first.
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


# STEP 4: Try a few questions and print the best-matching chunks.
questions = [
    "How is Rabivy different from Zepbound?",
    "A doctor says they are already happy with Ozempic - what do I say?",
    "What does the GLP-1 market look like in Missouri?",
]

for question in questions:
    print("=" * 72)
    print("Q:", question)
    for chunk, score in search(question):
        print(f"  [{score:.3f}]  {chunk['chunk_id']}")
        print(f"           doc_type={chunk['doc_type']}  state={chunk['state']!r}  competitor={chunk['competitor']!r}")
    print()

print("=" * 72)
print("Tip: change the questions above, or call search('your question') yourself.")