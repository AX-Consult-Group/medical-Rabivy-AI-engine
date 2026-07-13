# eval_full.py
# ===================================================================
# FULL-SYSTEM EVALUATION
# -------------------------------------------------------------------
# This tests the WHOLE system through ask.py (the top-level router),
# not just one engine. For each "concept" (a question we want to answer)
# we try SEVERAL PHRASINGS of the same question, and check two things:
#
#   1. ROUTING  - did ask() send it to the correct engine?
#                 (STRUCTURED for rankings/counts, RAG for narrative)
#   2. ANSWER   - does the returned answer contain the fact we expect?
#                 (a known NPI, a known count, or the right RAG chunk)
#
# Testing multiple phrasings shows how ROBUST the system is to wording.
# Where a phrasing fails, it usually means the keyword router didn't
# recognise it - which is exactly the gap the LLM layer will close.
# ===================================================================

import ask   # the single entry point; importing it loads both engines

# -------------------------------------------------------------------
# THE TEST SET
# Each concept lists:
#   expect_engine          - substring we expect in the engine label
#   expect_answer_contains - a fact that MUST appear in a correct answer
#   phrasings              - different ways a user might ask the same thing
# -------------------------------------------------------------------
TESTS = [
    {
        # Structured ranking within a state. Correct answer = NY's top NPI.
        "concept": "Top prescriber in a state",
        "expect_engine": "STRUCTURED",
        "expect_answer_contains": "1658907316",
        "phrasings": [
            "Who is the top GLP-1 prescriber in New York?",
            "Who is the highest-volume GLP-1 writer in New York?",
            "Which HCP writes the most GLP-1 prescriptions in New York?",  # stresses wording
            "Who's the biggest prescriber up in NY?",                       # stresses abbreviation
        ],
    },
    {
        # Structured count of active writers. Correct answer = 576 for Texas.
        "concept": "Count active writers in a state",
        "expect_engine": "STRUCTURED",
        "expect_answer_contains": "576",
        "phrasings": [
            "How many active GLP-1 writers are in Texas?",
            "What's the number of GLP-1 prescribers in Texas?",
            "Count the GLP-1 writers in Texas",
        ],
    },
    {
        # National ranking by propensity. Correct answer includes rank-1 NPI.
        "concept": "Top N by propensity",
        "expect_engine": "STRUCTURED",
        "expect_answer_contains": "1184547828",
        "phrasings": [
            "List the top 10 High-tier prescribers by propensity",
            "Show me the top 10 HCPs by propensity score",
            "Rank the highest propensity HCPs",
        ],
    },
    {
        # Single-NPI script lookup. NPI 1344001929 wrote 55 scripts.
        "concept": "Scripts for a specific NPI",
        "expect_engine": "STRUCTURED",
        "expect_answer_contains": "55",
        "phrasings": [
            "How many GLP-1 scripts did NPI 1344001929 write last month?",
            "What was NPI 1344001929's script volume?",
        ],
    },
    {
        # Group/count by state. California has the most High-tier HCPs.
        "concept": "States with most High-tier HCPs",
        "expect_engine": "STRUCTURED",
        "expect_answer_contains": "California",
        "phrasings": [
            "Which states have the most High-tier HCPs?",
            "What states have the most high tier doctors?",
        ],
    },
    {
        # Narrative question -> RAG. Correct chunk mentions the differentiator.
        "concept": "Rabivy vs Zepbound (narrative)",
        "expect_engine": "RAG",
        "expect_answer_contains": "differentiator",   # appears in the chunk_id
        "phrasings": [
            "How is Rabivy different from Zepbound?",
            "What sets Rabivy apart from Zepbound?",
            "Rabivy versus Zepbound - key differences?",
        ],
    },
    {
        # Narrative objection-handling -> RAG. Correct chunk is the Ozempic one.
        "concept": "Ozempic objection (narrative)",
        "expect_engine": "RAG",
        "expect_answer_contains": "objection_handling",
        "phrasings": [
            "How should a rep handle 'I'm already happy with Ozempic'?",
            "A doctor says they're happy with Ozempic, what do I say?",
        ],
    },
]

# -------------------------------------------------------------------
# RUN THE EVAL
# For every phrasing we call ask(), then score routing and answer.
# -------------------------------------------------------------------
total = 0          # total phrasings tested
passed = 0         # phrasings that passed BOTH checks

for test in TESTS:
    print("=" * 72)
    print(f"CONCEPT: {test['concept']}")
    print(f"  expect engine ~ '{test['expect_engine']}', answer contains '{test['expect_answer_contains']}'")

    concept_pass = 0
    for phrasing in test["phrasings"]:
        total += 1
        # ask() returns (engine_label, answer_text)
        engine, answer = ask.ask(phrasing)

        # CHECK 1: routing - is the expected engine in the label?
        routed_ok = test["expect_engine"] in engine
        # CHECK 2: content - is the expected fact in the answer?
        answer_ok = test["expect_answer_contains"].lower() in answer.lower()

        ok = routed_ok and answer_ok
        passed += ok
        concept_pass += ok

        mark = "PASS" if ok else "FAIL"
        # Show WHY a failure happened: wrong engine, or right engine but wrong answer.
        why = ""
        if not routed_ok:
            why = f"(routed to '{engine}' instead)"
        elif not answer_ok:
            why = "(right engine, but expected fact missing)"
        print(f"    [{mark}] {phrasing}  {why}")

    print(f"  -> {concept_pass}/{len(test['phrasings'])} phrasings handled")

# -------------------------------------------------------------------
# HEADLINE NUMBER
# -------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"OVERALL: {passed}/{total} phrasings passed (routing + correct answer).")
print("Failures are almost always wording the keyword-router didn't catch -")
print("the LLM layer is what makes routing robust to any phrasing.")
