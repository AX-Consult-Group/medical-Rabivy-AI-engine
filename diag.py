# diag.py - one-off: where does the CORRECT chunk rank for the failing question?
import json, numpy as np
from sentence_transformers import SentenceTransformer

chunks = json.load(open("output/chunks_tagged.json"))
embeddings = np.load("output/embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
narrative_idx = [i for i, c in enumerate(chunks) if c["doc_type"] != "hcp_card"]

question = "Why do patients stop taking GLP-1s after a year?"
q = model.encode([question])[0]; q = q/np.linalg.norm(q)
scores = chunk_norms[narrative_idx] @ q
order = np.argsort(scores)[::-1]

print(f"Q: {question}\n")
print("Top 8 narrative chunks by score:")
for rank, o in enumerate(order[:8], 1):
    c = chunks[narrative_idx[o]]
    print(f"  {rank:2d}. [{scores[o]:.3f}] {c['chunk_id']}")

print("\nWhere the 'discontinue' / 'persistence' chunks actually rank:")
for rank, o in enumerate(order, 1):
    cid = chunks[narrative_idx[o]]["chunk_id"]
    if "discontinue" in cid or "persistence" in cid:
        print(f"  rank {rank}: [{scores[o]:.3f}] {cid}")