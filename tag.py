# tag.py
# -------------------------------------------------------------------
# Adds the one tag that needs reading the text: `competitor`.
# (doc_type, state, specialty and npi are already set by chunk.py.)
# Input:  output/chunks.json
# Output: output/chunks_tagged.json / .csv
# -------------------------------------------------------------------

import json
import csv
import os

# STEP 1: Load the chunks chunk.py produced.
with open("output/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks to tag.")

# STEP 2: Detect competitor mentions. Narrative docs use drug names;
# cards use the maker names (Novo Nordisk / Eli Lilly) - catch both.
def find_competitors(text):
    t = text.lower()
    hits = []
    if any(w in t for w in ["semaglutide", "ozempic", "wegovy", "novo nordisk"]):
        hits.append("semaglutide")
    if any(w in t for w in ["tirzepatide", "mounjaro", "zepbound", "eli lilly", "lilly"]):
        hits.append("tirzepatide")
    return ", ".join(hits)

# STEP 3: Add the competitor tag to every chunk.
for c in chunks:
    c["competitor"] = find_competitors(c["text"])

# STEP 4: Save to new files (originals untouched).
os.makedirs("output", exist_ok=True)
with open("output/chunks_tagged.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

columns = ["chunk_id", "source_doc", "doc_type", "npi", "state", "specialty", "competitor", "heading", "text"]
try:
    with open("output/chunks_tagged.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(chunks)
except PermissionError:
    print("\n! chunks_tagged.csv is open in Excel - close it and re-run.")

# STEP 5: Short summary.
n_comp = sum(1 for c in chunks if c["competitor"])
print(f"Added a competitor tag to {n_comp} chunks.")
print("Done. Saved output/chunks_tagged.json and output/chunks_tagged.csv")