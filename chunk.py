# chunk.py
# -------------------------------------------------------------------
# Turns the documents in docs/ into chunks, using TWO strategies:
#   - Narrative docs  -> split into one chunk per "## " section.
#   - HCP cards       -> ONE chunk for the whole card (never split),
#                        with NPI / state / specialty pulled out as tags.
# Repo docs (READMEs, architecture) are skipped.
# -------------------------------------------------------------------

import glob
import re
import os
import json
import csv
from collections import Counter

# Repo-documentation files to ignore (not knowledge content).
SKIP_FILENAMES = {"README.md", "knowledge_repo_README.md", "PRODUCTION_ARCHITECTURE.md"}

# A file is an HCP card if it lives in hcp_snapshots/ or is named like a card.
def is_card(path):
    name = os.path.basename(path).lower()
    return ("hcp_snapshots" in path.lower()) or name.startswith("npi_") or name.startswith("sample_card")

# Pull a "- Label: value" line out of a card's text.
def extract_field(text, label):
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith(f"- {label.lower()}:"):
            return s.split(":", 1)[1].strip()
    return ""


all_md_paths = glob.glob("docs/**/*.md", recursive=True)

all_chunks = []
n_cards = 0
n_narrative_docs = 0
n_skipped = 0

for path in all_md_paths:
    filename = os.path.basename(path)

    if filename in SKIP_FILENAMES:
        n_skipped += 1
        continue

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if is_card(path):
        # ---- CARD: one whole chunk, with metadata pulled out ----
        match = re.search(r"(\d{10})", filename)          # NPIs are 10 digits
        npi = match.group(1) if match else filename.replace(".md", "")
        location = extract_field(text, "Location")        # e.g. "Arkansas (South)"
        state = location.split("(")[0].strip()            # -> "Arkansas"
        specialty = extract_field(text, "Specialty")      # e.g. "Primary Care"

        all_chunks.append({
            "chunk_id": f"card_{npi}",
            "source_doc": filename.replace(".md", ""),
            "doc_type": "hcp_card",
            "npi": npi,
            "state": state,
            "specialty": specialty,
            "heading": f"HCP {npi}",
            "text": text.strip(),
        })
        n_cards += 1

    else:
        # ---- NARRATIVE: split into one chunk per "## " section ----
        doc_name = filename.replace(".md", "")
        doc_type = os.path.basename(os.path.dirname(path))
        n_narrative_docs += 1

        for section in re.split(r"\n(?=## )", text):
            section = section.strip()
            if not section:
                continue
            first_line = section.splitlines()[0]
            if not first_line.startswith("## "):
                continue
            heading = first_line.replace("#", "").strip()
            # For the state summary doc, the heading IS the state.
            state = heading if doc_name == "state_market_summary" else ""

            all_chunks.append({
                "chunk_id": f"{doc_name}__{heading.lower().replace(' ', '_')}",
                "source_doc": doc_name,
                "doc_type": doc_type,
                "npi": "",
                "state": state,
                "specialty": "",
                "heading": heading,
                "text": section,
            })

# ---- Summary (no giant per-chunk list, since there are thousands) ----
print(f"Narrative docs chunked : {n_narrative_docs}")
print(f"Cards added (1 each)   : {n_cards}")
print(f"Meta files skipped     : {n_skipped}")
print(f"TOTAL chunks           : {len(all_chunks)}")
print("By doc_type            :", dict(Counter(c["doc_type"] for c in all_chunks)))

# ---- Save ----
os.makedirs("output", exist_ok=True)
with open("output/chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

columns = ["chunk_id", "source_doc", "doc_type", "npi", "state", "specialty", "heading", "text"]
try:
    with open("output/chunks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_chunks)
except PermissionError:
    print("\n! chunks.csv is open in Excel - close it and re-run.")

print("\nDone. Saved output/chunks.json and output/chunks.csv")