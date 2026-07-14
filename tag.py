# tag.py
# -------------------------------------------------------------------
# Adds tags that need reading the text: competitor_molecule,
# competitor_brand, competitor_company, brand (own-brand mentions).
# (doc_type, state, specialty and npi are already set by chunk.py.)
# Input:  output/chunks.json
# Output: output/chunks_tagged.json / .csv
# -------------------------------------------------------------------

import json
import csv
import os
import re

# STEP 1: Load the chunks chunk.py produced.
with open("output/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks to tag.")

# STEP 2: Config. One place to add/edit competitors - no need to touch
# the matching logic below when a third competitor drug shows up.
# "molecule" / "company" are single labels; "brands" is a list because
# a company can have more than one brand on the market (Lilly has two).
COMPETITOR_KEYWORDS = {
    "novo_nordisk": {
        "molecule": "semaglutide",
        "brands": ["ozempic", "wegovy"],
        "company": "novo nordisk",
    },
    "eli_lilly": {
        "molecule": "tirzepatide",
        "brands": ["mounjaro", "zepbound"],
        "company": "eli lilly",
    },
}

# Our own brand - separate from competitor logic, but same matching approach.
OWN_BRAND_KEYWORDS = ["rabivy"]


def _word_in_text(word, text):
    """Word-boundary match instead of substring, so 'lilly' doesn't hit
    'Lillyanne' or similar names once we're on real HCP data."""
    return re.search(r"\b" + re.escape(word) + r"\b", text) is not None


# STEP 3: Detect mentions. Narrative docs tend to use drug/brand names,
# cards tend to use maker names - so all three are checked against all
# three keyword types (a card can say "Ozempic" too).
def find_tags(text):
    t = text.lower()

    molecules, brands, companies = set(), set(), set()

    for info in COMPETITOR_KEYWORDS.values():
        if _word_in_text(info["molecule"], t):
            molecules.add(info["molecule"])
        for brand in info["brands"]:
            if _word_in_text(brand, t):
                brands.add(brand)
        if _word_in_text(info["company"], t):
            companies.add(info["company"])

    own_brand = [w for w in OWN_BRAND_KEYWORDS if _word_in_text(w, t)]

    return {
        "competitor_molecule": sorted(molecules),
        "competitor_brand": sorted(brands),
        "competitor_company": sorted(companies),
        "brand": sorted(own_brand),
    }


# STEP 4: Add tags to every chunk. Kept as lists here - only flattened
# to comma-joined strings when we write the CSV below, so anything
# downstream that filters the JSON does a list-membership check, not
# a string split.
for c in chunks:
    tags = find_tags(c["text"])
    c["competitor_molecule"] = tags["competitor_molecule"]
    c["competitor_brand"] = tags["competitor_brand"]
    c["competitor_company"] = tags["competitor_company"]
    c["brand"] = tags["brand"]

# STEP 5: Save to new files (originals untouched).
os.makedirs("output", exist_ok=True)
with open("output/chunks_tagged.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

columns = [
    "chunk_id", "source_doc", "source_path", "doc_type", "npi", "state", "specialty",
    "competitor_molecule", "competitor_brand", "competitor_company", "brand",
    "heading", "text",
]
try:
    with open("output/chunks_tagged.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for c in chunks:
            row = dict(c)
            # Flatten only for CSV - spreadsheets can't hold real lists anyway.
            for field in ("competitor_molecule", "competitor_brand", "competitor_company", "brand"):
                row[field] = ", ".join(row[field])
            writer.writerow(row)
except PermissionError:
    print("\n! chunks_tagged.csv is open in Excel - close it and re-run.")

# STEP 6: Sanity-check summary - overall + breakdown by doc_type, so you
# can eyeball whether e.g. zero cards got tagged (a sign something's off
# with how competitor names show up in card text).
def has_any_competitor_tag(c):
    return bool(c["competitor_molecule"] or c["competitor_brand"] or c["competitor_company"])

def has_brand_tag(c):
    return bool(c["brand"])

n_comp = sum(1 for c in chunks if has_any_competitor_tag(c))
n_brand = sum(1 for c in chunks if has_brand_tag(c))
print(f"Tagged {n_comp} chunks with a competitor mention (molecule/brand/company).")
print(f"Tagged {n_brand} chunks with an own-brand (Rabivy) mention.")

print("\nBreakdown by doc_type:")
doc_types = sorted(set(c["doc_type"] for c in chunks))
for dt in doc_types:
    subset = [c for c in chunks if c["doc_type"] == dt]
    comp_n = sum(1 for c in subset if has_any_competitor_tag(c))
    brand_n = sum(1 for c in subset if has_brand_tag(c))
    print(f"  {dt}: {len(subset)} chunks | {comp_n} competitor-tagged | {brand_n} brand-tagged")

print("\nDone. Saved output/chunks_tagged.json and output/chunks_tagged.csv")