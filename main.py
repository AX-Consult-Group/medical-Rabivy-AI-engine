# main.py
# -------------------------------------------------------------------
# Orchestrator: runs the pipeline stages in order, each feeding the next.
#   1_chunk_documents.py -> 2_tag_chunks.py -> 3_create_embeddings.py -> (query_spreadsheet.py sanity check)
# Run this one file to rebuild the whole knowledge base from scratch.
# (search_documents.py, ask_a_question.py, and test_the_system.py are
# run separately, once the base is built.)
#
# Scope note: this rebuilds the RAG side (chunk/tag/embed). It does not
# regenerate the propensity master spreadsheet or the ground-truth eval
# question set - "rebuild everything" here means "rebuild the knowledge
# base", not "rebuild the whole system end to end."
# ------------------------------------------------------------------- 

import os
import subprocess
import sys
import time

# Each stage: the script to run, and how to sanity-check its output
# actually landed (not just that the process exited 0 - a stage can
# "succeed" while silently producing empty/malformed output, e.g. if
# docs/ matched nothing).
STAGES = [
    {
        "script": "1_chunk_documents.py",
        "check_path": "output/chunks.json",
        "min_size": 100,
    },
    {
        "script": "2_tag_chunks.py",
        "check_path": "output/chunks_tagged.json",
        "min_size": 100,
    },
    {
        "script": "3_create_embeddings.py",
        "check_path": "output/embeddings.npy",
        "min_size": 100,
    },
]


def _check_output(stage):
    path = stage["check_path"]
    if not os.path.exists(path):
        print(f"!!! {stage['script']} exited 0 but expected output "
              f"'{path}' does not exist. Pipeline stopped.")
        sys.exit(1)
    size = os.path.getsize(path)
    if size < stage["min_size"]:
        print(f"!!! {stage['script']} exited 0 but '{path}' is suspiciously "
              f"small ({size} bytes) - looks empty/malformed. Pipeline stopped.")
        sys.exit(1)


print("=" * 60)
print("Rebuilding the RAG knowledge base")
print("=" * 60)

# ---- Stage 0: score-consistency guard (propensity_model.py) --------
# Before building anything on top of the master spreadsheet, confirm
# its stored scores are what the Phase 2 scoring model would produce
# for its raw inputs. Catches the silent-drift failure mode: a data
# refresh whose propensity scores were never recomputed would otherwise
# flow into the knowledge base looking perfectly healthy. If this
# fails: re-score the raw batch (python propensity_model.py --score
# <raw.xlsx>) or resolve the weight mismatch before ingesting.
print("\n>>> Stage 0: verifying propensity scores against the scoring model ...")
guard = subprocess.run([sys.executable, "propensity_model.py", "--verify"])
if guard.returncode != 0:
    print("\n!!! Score verification failed - the spreadsheet's scores do not "
          "match the scoring model. Pipeline stopped before ingestion.")
    sys.exit(1)

for stage in STAGES:
    script = stage["script"]
    print(f"\n>>> Running {script} ...")
    start = time.time()
    # Run the stage as its own script; stop everything if one fails.
    # subprocess.run (not import) is deliberate: 1_chunk_documents.py /
    # 2_tag_chunks.py / 3_create_embeddings.py are top-level scripts,
    # not wrapped in functions, so importing them would run their code
    # anyway while tangling namespaces together and making a failure in
    # one script harder to isolate from the next.
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n!!! {script} failed (exit code {result.returncode}). Pipeline stopped.")
        sys.exit(1)

    _check_output(stage)
    print(f">>> {script} done in {time.time() - start:.1f}s - "
          f"{stage['check_path']} looks present and non-trivial.")

# ---- Final sanity check: does query_spreadsheet.py's data path still load? ----
# query_spreadsheet.py isn't part of this rebuild (it's a separate data
# source, the propensity spreadsheet, not chunked/tagged/embedded
# content) - but since it loads a dated Excel filename via glob, it's
# cheap to confirm here that it still resolves and loads cleanly,
# rather than discovering a broken path later when someone runs
# ask_a_question.py by hand.
print("\n>>> Checking query_spreadsheet.py's data path ...")
start = time.time()
check = subprocess.run(
    [sys.executable, "-c", "import query_spreadsheet; print('OK:', query_spreadsheet._DATA_PATH)"],
    capture_output=True, text=True,
)
if check.returncode != 0:
    print("!!! query_spreadsheet.py failed to load its data file:")
    print(check.stderr)
    sys.exit(1)
print(f">>> {check.stdout.strip()} ({time.time() - start:.1f}s)")

print("\n" + "=" * 60)
print("Pipeline complete. Knowledge base is ready.")
print("Run 'python search_documents.py' or 'python ask_a_question.py' to query it, "
      "or 'python test_the_system.py' to test it.")
print("=" * 60)