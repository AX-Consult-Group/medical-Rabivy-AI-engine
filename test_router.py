# test_router.py
# =====================================================================
# WHAT THIS DOES (plain English)
# =====================================================================
# Runs a batch of real questions straight through ask_a_question.py -
# NO large language model involved anywhere in this file. It calls
# ask() to pick a route and pull real data, then format_answer() to
# turn that into a sentence - both of those are plain code, not AI, so
# this is a pure test of the REGEX and the two engines underneath it.
#
# This is the right tool for exactly what you asked: "test the RAG
# rigorously, no LLM interpreter, test the regex." write_the_answer.py
# (the Groq step) is a separate, later layer - this file exists to
# check everything BELOW that layer is correct on its own first.
#
# HOW TO USE:
#   Run it as-is to test the built-in question list below, or import
#   run_question() to test one question interactively, or add your
#   own questions to QUESTIONS at the bottom.
#
#   python test_router.py
#
# WHAT TO LOOK FOR IN THE OUTPUT:
#   - Does the ROUTE make sense for the question? (STRUCTURED vs RAG,
#     and which specific path within that)
#   - Does the answer's sort/level explanation match what was actually
#     asked?
#   - For a question you know should have zero matches or a bad NPI,
#     does it clearly say "Answer not found"?
# =====================================================================

import ask_a_question as aq



# ---------------------------------------------------------------------
# Runs ONE question through the router and prints route + answer.
# Nothing here is AI-generated - ask() is regex + spreadsheet/document
# lookups, format_answer() is plain string-building.
# ---------------------------------------------------------------------
def run_question(question):
    print("=" * 88)
    print(f"Q: {question}")
    try:
        route, data = aq.ask(question)
    except Exception as e:
        print(f"   !! ask() raised an exception: {type(e).__name__}: {e}")
        return
    print(f"   -> {route}")
    try:
        answer = aq.format_answer(route, data)
    except Exception as e:
        print(f"   !! format_answer() raised an exception: {type(e).__name__}: {e}")
        print(f"      raw data was: {data}")
        return
    print(answer)
    print()



# =====================================================================
# THE TEST QUESTIONS
# -----------------------------------------------------------------
# Grouped by what they're meant to check. The first block is close to
# your original Possible_Questions.docx set, so you can compare this
# output against the numbers you already validated by hand. The later
# blocks specifically stress-test the things that changed in this
# cleanup pass: region, levels, explicit sorting, and "not found."
# =====================================================================
QUESTIONS = [
    # ---- Structured: exact lookup & ranking (should match your original test doc) ----
    "Who is the top GLP-1 prescriber in New York?",
    "How many GLP-1 scripts did NPI 1344001929 write last month?",
    "List the top 10 High-tier prescribers nationally by propensity score.",
    "Which states have the most High-tier HCPs?",
    "How many active GLP-1 writers are in Texas?",

    # ---- Structured: filtered targeting ----
    "Show me High-propensity endocrinologists in Florida who are not currently targeted.",
    "Which Novo-heavy prescribers in California have a high switching score?",
    "Find high-volume writers in Illinois with a preferred formulary status.",
    "Which untargeted HCPs have a recent sample request and a High tier?",

    # ---- RAG: card / narrative / strategic ----
    "Give me a summary of NPI 1344001929.",
    "What's the access situation for the top prescriber in Missouri?",
    "Compare NPI 1344001929 to a typical endocrinologist.",
    "What's our recommended messaging for competitive switchers?",
    "What is Rabivy's key differentiator versus Zepbound?",
    "How should a rep handle 'I'm already happy with Ozempic'?",
    "What's the Medicaid coverage outlook for GLP-1 obesity drugs?",

    # ---- NEW in this cleanup: region, levels, explicit sort/level statements ----
    "Which prescribers in the South region have a high switching score?",
    "Which prescribers in South Carolina have a high switching score?",  # state should win over region
    "Which HCPs have a high percent Novo and low rep engagement?",       # two-clause level test
    "Show me the lowest PA burden prescribers in Ohio.",

    # ---- "Answer not found" checks - these SHOULD come back empty/not-found ----
    "Tell me about NPI 0000000000.",                                    # fake NPI
    "Which HCPs are in the Southeast region?",                          # not a real region
    "Which HCPs have a Low tier?",                                      # "Low" isn't real - should map to Watch, not error
]


if __name__ == "__main__":
    for q in QUESTIONS:
        run_question(q)

    print("=" * 88)
    print(f"Ran {len(QUESTIONS)} questions through the router - no LLM involved anywhere above.")
    print("Check each ROUTE and answer against what you'd expect by hand.")