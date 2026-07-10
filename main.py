# main.py
# -------------------------------------------------------------------
# Orchestrator: runs the pipeline stages in order, each feeding the next.
#   chunk.py  -> tag.py  -> embed.py
# Run this one file to rebuild the whole knowledge base from scratch.
# (search.py and eval.py are run separately, once the base is built.)
# -------------------------------------------------------------------

import subprocess
import sys
import time

# The stages, in the order they must run.
stages = ["chunk.py", "tag.py", "embed.py"]

print("=" * 60)
print("Rebuilding the RAG knowledge base")
print("=" * 60)

for stage in stages:
    print(f"\n>>> Running {stage} ...")
    start = time.time()
    # Run the stage as its own script; stop everything if one fails.
    result = subprocess.run([sys.executable, stage])
    if result.returncode != 0:
        print(f"\n!!! {stage} failed. Pipeline stopped.")
        sys.exit(1)
    print(f">>> {stage} done in {time.time() - start:.1f}s")

print("\n" + "=" * 60)
print("Pipeline complete. Knowledge base is ready.")
print("Run 'python search.py' to query it, or 'python eval.py' to test it.")
print("=" * 60)