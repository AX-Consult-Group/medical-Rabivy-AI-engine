# ask.py
# -------------------------------------------------------------------
# The single entry point. Looks at a question and decides which engine
# should answer it:
#   - STRUCTURED engine  -> ranking / counting / filtering (the spreadsheet)
#   - RAG (search.py)    -> narrative / card / characterisation questions
# This is the top-level router that ties both engines into one system.
# -------------------------------------------------------------------

import re
import structured
import search   # your existing RAG router (NPI lookup / state / semantic)

# Words that signal a STRUCTURED (table-math) question.
RANK_WORDS  = ["top ", "highest", "most", "rank", "leading", "biggest"]
COUNT_WORDS = ["how many", "number of", "count"]
LIST_WORDS  = ["list", "which ", "show me", "find ", "who are", "give me"]

STATES = sorted({s for s in structured.df["state"].unique()})

def _find_state(ql):
    for s in STATES:
        if s in ql:
            return s
    return None

def ask(question):
    ql = question.lower()

    # 1. "how many ... writers in X"  -> count
    if any(w in ql for w in COUNT_WORDS) and ("writer" in ql or "prescriber" in ql or "hcp" in ql):
        st = _find_state(ql)
        if st:
            return "STRUCTURED / count", structured.count_writers(st)

    # 2. "how many scripts did NPI ... write" -> single-NPI script lookup
    m = re.search(r"\b(\d{10})\b", question)
    if m and ("script" in ql or "rx" in ql or "write" in ql or "wrote" in ql):
        return "STRUCTURED / hcp scripts", structured.hcp_scripts(m.group(1))

    # 3. "top prescriber in X" -> ranking within a state
    if any(w in ql for w in RANK_WORDS) and "propensity" not in ql:
        st = _find_state(ql)
        if st and ("prescriber" in ql or "writer" in ql or "volume" in ql):
            return "STRUCTURED / top prescriber", structured.top_prescriber(st)

    # 4. "top N by propensity" -> national ranking
    if any(w in ql for w in RANK_WORDS) and "propensity" in ql:
        n = 10
        mn = re.search(r"top\s+(\d+)", ql)
        if mn: n = int(mn.group(1))
        return "STRUCTURED / top by propensity", structured.top_n_by_propensity(n)

    # 5. "which states have the most high-tier" -> group/count
    if "state" in ql and "high" in ql and any(w in ql for w in RANK_WORDS + LIST_WORDS):
        return "STRUCTURED / states by high tier", structured.states_by_high_tier()

    # 6. Otherwise -> hand it to the RAG (narrative / cards / semantic)
    route, results = search.search(question)
    out = [f"(RAG route: {route})"]
    for chunk, score in results:
        tag = f"[{score:.3f}]" if score is not None else "[card]"
        out.append(f"  {tag} {chunk['chunk_id']}")
    return "RAG", "\n".join(out)


if __name__ == "__main__":
    questions = [
        "Who is the top GLP-1 prescriber in New York?",       # structured
        "How many active GLP-1 writers are in Texas?",         # structured
        "List the top 10 High-tier prescribers by propensity", # structured
        "How is Rabivy different from Zepbound?",              # RAG
    ]
    for q in questions:
        engine, answer = ask(q)
        print("=" * 72)
        print(f"Q: {q}")
        print(f"   -> {engine}")
        print(answer)
        print()