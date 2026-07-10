# chunk.py
# Splits narrative markdown documents into chunks at each "## " heading.
# Skips repo documentation (READMEs, architecture) and HCP report cards
# (cards are handled separately because they are chunked whole, not split).

import glob
import re
import os
import json
import csv

# --- What to skip -------------------------------------------------------
# 1. Repo documentation files (describe the project, not knowledge content).
SKIP_FILENAMES = {
    "knowledge_repo_README.md",
    "PRODUCTION_ARCHITECTURE.md",
}
# 2. Report cards live in hcp_snapshots/ or are named sample_card / card_NPI.
#    These get their own whole-file handling in a later step, so skip for now.
def is_report_card(path):
    name = os.path.basename(path).lower()
    return ("hcp_snapshots" in path) or name.startswith("sample_card") or name.startswith("card_npi")
# -----------------------------------------------------------------------


# STEP 1: Find every .md file inside the "docs" folder AND its subfolders.
all_md_paths = glob.glob("docs/**/*.md", recursive=True)

# Keep only the narrative content docs (drop meta files and cards).
doc_paths = []
skipped = []
for path in all_md_paths:
    filename = os.path.basename(path)
    if filename in SKIP_FILENAMES or is_report_card(path):
        skipped.append(filename)
    else:
        doc_paths.append(path)

print(f"Found {len(all_md_paths)} .md files. Chunking {len(doc_paths)}, skipping {len(skipped)}.")
if skipped:
    preview = ", ".join(skipped[:5])
    more = f" ... and {len(skipped) - 5} more" if len(skipped) > 5 else ""
    print(f"Skipped (meta docs / cards): {preview}{more}")
print()


# STEP 2: Go through each content document and cut it into chunks.
all_chunks = []
for path in doc_paths:
    doc_name = os.path.basename(path).replace(".md", "")
    doc_type = os.path.basename(os.path.dirname(path))  # the subfolder, e.g. "clinical"

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = re.split(r"\n(?=## )", text)  # split wherever a line starts with "## "

    for section in sections:
        section = section.strip()
        if not section:
            continue
        first_line = section.splitlines()[0]
        if not first_line.startswith("## "):  # keep only real "## " sections
            continue                          # (skips the doc's title/intro)
        heading = first_line.replace("#", "").strip()

        chunk = {
            "chunk_id": f"{doc_name}__{heading.lower().replace(' ', '_')}",
            "source_doc": doc_name,
            "heading": heading,
            "text": section,
            "doc_type": doc_type,
        }
        all_chunks.append(chunk)

print(f"Created {len(all_chunks)} chunks in total.\n")


# STEP 3: Print each chunk so you can see the result.
for c in all_chunks:
    word_count = len(c["text"].split())
    print(f"  - {c['chunk_id']}  ({word_count} words)")


# STEP 4: Save the chunks to an "output" folder in two formats.
os.makedirs("output", exist_ok=True)

with open("output/chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

with open("output/chunks.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["chunk_id", "source_doc", "doc_type", "heading", "text"])
    writer.writeheader()
    writer.writerows(all_chunks)

print("\nDone. Saved to output/chunks.json and output/chunks.csv")