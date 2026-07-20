# 1_chunk_documents.py
# -------------------------------------------------------------------
# Turns the documents in docs/ folder into chunks, using TWO strategies:
#   - Narrative docs  -> split into one chunk per "## " section. Since the 
#                        documents were simualetd for this showcase, the 
#                        sections are already fairly self-contained. In a 
#                        real-world scenario, you might want to split by 
#                        paragraphs or sentences instead.
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

# ---- CHUNK: sanitize_for_id ----
# Turn a heading like "Coverage by Payer Type > Medicaid (state/federal, low-income)"
# into a safe chunk_id fragment: lowercase, spaces -> underscores, and strip out
# anything that isn't a letter/digit/underscore/hyphen (so ">", "(", ")", "/" etc.
# don't end up inside an id).
def sanitize_for_id(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text


# Make chunk_id unique; if it collides, append _2, _3, ... and warn.
def make_unique_id(base_id, seen_counter):
    seen_counter[base_id] += 1
    n = seen_counter[base_id]
    if n == 1:
        return base_id
    print(f"! WARNING: duplicate chunk_id base '{base_id}' seen {n} times - "
          f"suffixing as '{base_id}_{n}'")
    return f"{base_id}_{n}"


all_md_paths = glob.glob("docs/**/*.md", recursive=True)

all_chunks = []
n_cards = 0
n_narrative_docs = 0
n_skipped = 0
n_bad_npi = 0
n_h3_splits = 0
id_counter = Counter()

for path in all_md_paths:
    filename = os.path.basename(path)
    rel_path = os.path.relpath(path)  # full relative path, kept for traceability

    if filename in SKIP_FILENAMES:
        n_skipped += 1
        continue

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if is_card(path):
        # ---- CARD: one whole chunk, with metadata pulled out ----
        match = re.search(r"(\d{10})", filename)          # NPIs are 10 digits
        npi = match.group(1) if match else filename.replace(".md", "")

        # Safety net: NPI must be exactly 10 digits. If it isn't, this card
        # can't reliably join back to the master HCP table later - flag it
        # instead of silently shipping a bad key.
        if not re.fullmatch(r"\d{10}", npi):
            n_bad_npi += 1
            print(f"! WARNING: could not extract a valid 10-digit NPI from "
                  f"'{filename}' (got '{npi}') - check this file/naming.")

        location = extract_field(text, "Location")        # e.g. "Arkansas (South)"
        state = location.split("(")[0].strip()            # -> "Arkansas"
        specialty = extract_field(text, "Specialty")      # e.g. "Primary Care"

        base_id = f"card_{npi}"
        chunk_id = make_unique_id(base_id, id_counter)

        all_chunks.append({
            "chunk_id": chunk_id,
            "source_doc": filename.replace(".md", ""),
            "source_path": rel_path,
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
            parent_heading = first_line.replace("#", "").strip()

            # ---- CHUNK: check_for_h3_subsections ----
            # Does this ## section contain any "### " lines? If not, this
            # section behaves EXACTLY as it did before the fix: one chunk.
            has_h3 = re.search(r"\n### ", "\n" + section) is not None

            if not has_h3:
                heading = parent_heading
                # For the state summary doc, the heading IS the state.
                state = heading if doc_name == "state_market_summary" else ""

                base_id = f"{doc_name}__{sanitize_for_id(heading)}"
                chunk_id = make_unique_id(base_id, id_counter)

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "source_doc": doc_name,
                    "source_path": rel_path,
                    "doc_type": doc_type,
                    "npi": "",
                    "state": state,
                    "specialty": "",
                    "heading": heading,
                    "text": section,
                })
                continue

            # ---- CHUNK: split_into_h3_subsections ----
            # Pull out the intro text (prose sitting between the "## " heading
            # and the first "### " subheading) - this gets copied into EVERY
            # resulting sub-chunk, so each one is self-contained on its own.
            h3_pieces = re.split(r"\n(?=### )", section)
            intro_piece = h3_pieces[0]
            intro_text = "\n".join(intro_piece.splitlines()[1:]).strip()

            for piece in h3_pieces[1:]:
                piece = piece.strip()
                sub_first_line = piece.splitlines()[0]
                sub_heading = sub_first_line.replace("#", "").strip()
                sub_body = "\n".join(piece.splitlines()[1:]).strip()

                heading = f"{parent_heading} > {sub_heading}"
                if intro_text:
                    chunk_text = f"## {heading}\n\n{intro_text}\n\n{sub_body}"
                else:
                    chunk_text = f"## {heading}\n\n{sub_body}"

                # Use parent_heading (not the combined heading) for state,
                # in case a state-summary-style doc ever gains ### subsections.
                state = parent_heading if doc_name == "state_market_summary" else ""

                base_id = f"{doc_name}__{sanitize_for_id(heading)}"
                chunk_id = make_unique_id(base_id, id_counter)

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "source_doc": doc_name,
                    "source_path": rel_path,
                    "doc_type": doc_type,
                    "npi": "",
                    "state": state,
                    "specialty": "",
                    "heading": heading,
                    "text": chunk_text,
                })
                n_h3_splits += 1

# ---- Summary (no giant per-chunk list, since there are thousands) ----
print(f"Narrative docs chunked : {n_narrative_docs}")
print(f"Cards added (1 each)   : {n_cards}")
print(f"Meta files skipped     : {n_skipped}")
print(f"Cards with bad NPI     : {n_bad_npi}")
print(f"### sub-chunks created : {n_h3_splits}")
print(f"TOTAL chunks           : {len(all_chunks)}")
print("By doc_type            :", dict(Counter(c["doc_type"] for c in all_chunks)))

# ---- Save ----
os.makedirs("output", exist_ok=True)
with open("output/chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

columns = ["chunk_id", "source_doc", "source_path", "doc_type", "npi", "state", "specialty", "heading", "text"]
try:
    with open("output/chunks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_chunks)
except PermissionError:
    print("\n! chunks.csv is open in Excel - close it and re-run.")

# ---- Manifest: cheap chunk_id -> source_path lookup, separate from the ----
# ---- full chunk content, for quick "where did this come from" checks.  ----
manifest = {c["chunk_id"]: c["source_path"] for c in all_chunks}
with open("output/chunk_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nDone. Saved output/chunks.json, output/chunks.csv, output/chunk_manifest.json")