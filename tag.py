# tag.py
# -------------------------------------------------------------------
# Reads the chunks produced by chunk.py and adds metadata TAGS to each.
# Tags let the search system filter chunks BEFORE doing a semantic search.
# Input:  output/chunks.json
# Output: output/chunks_tagged.json  and  output/chunks_tagged.csv
# -------------------------------------------------------------------

import json
import csv
import os

# STEP 1: Load the chunks that chunk.py already produced.
with open("output/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks to tag.")


# STEP 2: A small helper that spots competitor mentions in a piece of text.
def find_competitors(text):
    t = text.lower()
    hits = []
    if any(word in t for word in ["semaglutide", "ozempic", "wegovy"]):
        hits.append("semaglutide")
    if any(word in t for word in ["tirzepatide", "mounjaro", "zepbound"]):
        hits.append("tirzepatide")
    return ", ".join(hits)   # "" if none found


# STEP 3: Walk through each chunk and add the new tags.
for c in chunks:
    # state: only the state_market_summary chunks name a state; it's the heading.
    if c["source_doc"] == "state_market_summary":
        c["state"] = c["heading"]
    else:
        c["state"] = ""

    # competitor: whichever rival drugs are named in the chunk text.
    c["competitor"] = find_competitors(c["text"])


# STEP 4: Save the tagged chunks to NEW files (originals stay untouched).
os.makedirs("output", exist_ok=True)

with open("output/chunks_tagged.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

columns = ["chunk_id", "source_doc", "doc_type", "state", "competitor", "heading", "text"]
try:
    with open("output/chunks_tagged.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(chunks)
except PermissionError:
    print("\n! Could not write chunks_tagged.csv - it's probably open in Excel. Close it and re-run.")


# STEP 5: Print a short summary so you can see the tags took effect.
n_state = sum(1 for c in chunks if c["state"])
n_comp = sum(1 for c in chunks if c["competitor"])
print(f"Added a state tag to {n_state} chunks.")
print(f"Added a competitor tag to {n_comp} chunks.")
print("\nDone. Saved output/chunks_tagged.json and output/chunks_tagged.csv")