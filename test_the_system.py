# test_the_system.py
# ===================================================================
# FULL-SYSTEM EVALUATION - covers the entire question list.
# -------------------------------------------------------------------
# Every question is sent through ask_a_question.py (the top-level
# router). For each, we report one of FOUR honest outcomes:
#
#   PASS        - routed to the right engine AND returned the correct fact
#                 (fully working today, no LLM needed).
#   ROUTES->LLM - routed correctly AND retrieved the right data/card, but
#                 turning that into a written answer needs the LLM layer.
#   NEEDS-LLM   - routed to the RIGHT engine (checked, not assumed), but full
#                 correctness needs the LLM for parsing/chaining/synthesis.
#   FAIL        - routed to the wrong engine, or the expected fact/chunk
#                 wasn't found where it should have been. This now applies
#                 to "llm"-mode tests too - a wrong route is a real bug
#                 regardless of what happens after routing.
#   CRASH       - the question blew up ask_a_question.ask() itself.
#                 Recorded, not fatal to the rest of the run.
#
# NOTE: ask_a_question.ask() returns (route: str, data: dict) - not a
# formatted string. Correctness checks below search the raw data dict
# directly (via _deep_contains) rather than a pre-formatted sentence,
# so a phrasing/formatting change in query_spreadsheet.py's or
# ask_a_question.py's format_* layer can't silently break this eval
# for the wrong reason.
#
# ALSO PRINTS THE ACTUAL ANSWER (2026-07-22): this file used to only
# show the scored PASS/FAIL, and a separate file (test_router.py) ran
# the same questions again just to print the real sentence a rep would
# see. That meant one shared question set living in two files, which is
# exactly how a stale test slipped through unnoticed once (see the
# CLARIFICATION fix below). Now every run prints the score AND the
# actual formatted answer together - test_router.py has been retired,
# this file replaces it entirely.
#
# GROUPED BY STYLE, NOT BY WHEN IT WAS ADDED (2026-07-22): questions
# used to just be appended to the end of the list as new gaps were
# found, so the not-found/edge-case checks ended up looking like an
# afterthought bolted on after everything else, rather than their own
# clear category. Every question now has a "group" key, tests are
# ordered by group, and the run() output prints a banner between
# groups so the output reads as clear sections, not one long scroll.
# Output format per question was also redone: QUESTION / STATUS+ROUTE /
# NOTE / ANSWER are now visually separated blocks instead of one
# cramped run-on paragraph.
# ===================================================================

import json
import os
import time
import ask_a_question   # single entry point; importing loads both engines

# Each test: the question, its group (for display), category, a "mode",
# and what to check.
#   mode "full"      -> check expect_engine + expect_contains  (PASS/FAIL)
#   mode "retrieval" -> check expect_engine + expect_retrieves (ROUTES->LLM/FAIL)
#   mode "llm"       -> check expect_engine only               (NEEDS-LLM/FAIL)
TESTS = [
    # =================================================================
    # GROUP 1: STRUCTURED - exact lookup, ranking, counting
    # (fully working today - no LLM needed for any of these)
    # =================================================================
    {"group": "STRUCTURED - exact lookup, ranking & counting",
     "q": "Who is the top GLP-1 prescriber in New York?",
     "cat": "structured lookup", "mode": "full",
     "expect_engine": "STRUCTURED", "expect_contains": "1658907316"},
    {"group": "STRUCTURED - exact lookup, ranking & counting",
     "q": "How many GLP-1 scripts did NPI 1344001929 write last month?",
     "cat": "structured lookup", "mode": "full",
     "expect_engine": "STRUCTURED", "expect_contains": "55"},
    {"group": "STRUCTURED - exact lookup, ranking & counting",
     "q": "List the top 10 High-tier prescribers nationally by propensity score.",
     "cat": "structured ranking", "mode": "full",
     "expect_engine": "STRUCTURED", "expect_contains": "1184547828"},
    {"group": "STRUCTURED - exact lookup, ranking & counting",
     "q": "Which states have the most High-tier HCPs?",
     "cat": "structured ranking", "mode": "full",
     "expect_engine": "STRUCTURED", "expect_contains": "California"},
    {"group": "STRUCTURED - exact lookup, ranking & counting",
     "q": "How many active GLP-1 writers are in Texas?",
     "cat": "structured count", "mode": "full",
     "expect_engine": "STRUCTURED", "expect_contains": "576"},

    # =================================================================
    # GROUP 2: STRUCTURED - filtered targeting (multi-field)
    # filter_hcps() answers these; parsing several conditions out of
    # free text (specialty + state + tier + targeted) is the LLM's job.
    # =================================================================
    {"group": "STRUCTURED - filtered targeting (multi-field)",
     "q": "Show me High-propensity endocrinologists in Florida who are not currently targeted.",
     "cat": "filtered targeting", "mode": "llm",
     "expect_engine": "STRUCTURED",
     "note": "filter_hcps() ready; multi-field parsing needs LLM"},
    {"group": "STRUCTURED - filtered targeting (multi-field)",
     "q": "Which Novo-heavy prescribers in California have a high switching score?",
     "cat": "filtered targeting", "mode": "llm",
     "expect_engine": "STRUCTURED",
     "note": "'Novo-heavy' correctly maps to the Novo Nordisk competitor "
             "filter (short form 'novo' was missing from the keyword list - "
             "'lilly' already worked as a short form, 'novo' didn't). Both "
             "the competitor and switching-score filters genuinely apply. "
             "Multi-field parsing into the filter still needs the LLM."},
    {"group": "STRUCTURED - filtered targeting (multi-field)",
     "q": "Find high-volume writers in Illinois with a preferred formulary status.",
     "cat": "filtered targeting", "mode": "llm",
     "expect_engine": "STRUCTURED",
     "note": "the formulary_tier column exists and is wired in - routes "
             "correctly, sorted by rx_volume_monthly since the question said "
             "'high-volume'. Multi-field parsing into that filter still "
             "needs the LLM."},
    {"group": "STRUCTURED - filtered targeting (multi-field)",
     "q": "Which untargeted HCPs have a recent sample request and a High tier?",
     "cat": "filtered targeting", "mode": "llm",
     "expect_engine": "STRUCTURED",
     "note": "routes via 'untargeted' + tier + recent-sample-request "
             "detection - the sample_request_recent column exists and "
             "is wired in too. Multi-field parsing into the filter still "
             "needs the LLM."},

    # =================================================================
    # GROUP 3: RAG - HCP cards & comparisons
    # =================================================================
    {"group": "RAG - HCP cards & comparisons",
     "q": "Give me a summary of NPI 1344001929.",
     "cat": "card summary", "mode": "retrieval",
     "expect_engine": "RAG", "expect_retrieves": "card_1344001929"},
    {"group": "RAG - HCP cards & comparisons",
     "q": "What's the access situation for the top prescriber in Missouri?",
     "cat": "card (chained)", "mode": "llm",
     "expect_engine": "STRUCTURED",
     "note": "chained: this only proves step 1 (find top prescriber) routes "
             "correctly; step 2 (read their card) and synthesis aren't wired"},
    {"group": "RAG - HCP cards & comparisons",
     "q": "Why is this High-tier doctor not converting?",
     "cat": "card read", "mode": "llm",
     "expect_engine": "CLARIFICATION",
     "note": "no NPI given and no prior HCP in context - same unresolved-"
             "referent case as 'Is this doctor...' below. Updated "
             "2026-07-22: this used to fall through to a weak RAG guess "
             "(top match ~0.30) because the old regex only caught 'this "
             "doctor' with nothing in between - 'this High-tier doctor' "
             "slipped past it. Fixed by allowing up to 3 words between "
             "'this/that' and the noun (see _UNRESOLVED_REFERENT_PATTERNS "
             "in ask_a_question.py)."},
    {"group": "RAG - HCP cards & comparisons",
     "q": "What's the story on GLP-1 writers in Arizona - who should I know about?",
     "cat": "card narrative", "mode": "llm",
     "expect_engine": "RAG",
     "note": "needs LLM to synthesise across several cards"},
    {"group": "RAG - HCP cards & comparisons",
     "q": "Compare NPI 1344001929 to a typical endocrinologist.",
     "cat": "comparison", "mode": "llm",
     "expect_engine": "RAG",
     "note": "retrieves the card via search_documents.py's NPI path; "
             "comparison to a benchmark needs LLM"},
    {"group": "RAG - HCP cards & comparisons",
     "q": "Is this doctor a high or low prescriber for their specialty?",
     "cat": "comparison", "mode": "llm",
     "expect_engine": "CLARIFICATION",
     "note": "no NPI given and no prior HCP in context - correctly asks "
             "for clarification instead of guessing (see "
             "_has_unresolved_referent in ask_a_question.py). Updated "
             "2026-07-22: this test predates that guard, which used to "
             "expect a silent fall-through to RAG."},

    # =================================================================
    # GROUP 4: RAG - narrative knowledge base (strategic & product content)
    # =================================================================
    {"group": "RAG - narrative knowledge base",
     "q": "What's our recommended messaging for competitive switchers?",
     "cat": "narrative", "mode": "full",
     "expect_engine": "RAG", "expect_contains": "competitive_switchers"},
    {"group": "RAG - narrative knowledge base",
     "q": "What is Rabivy's key differentiator versus Zepbound?",
     "cat": "narrative", "mode": "full",
     "expect_engine": "RAG", "expect_contains": "differentiator"},
    {"group": "RAG - narrative knowledge base",
     "q": "How should a rep handle 'I'm already happy with Ozempic'?",
     "cat": "narrative", "mode": "full",
     "expect_engine": "RAG", "expect_contains": "objection_handling"},
    {"group": "RAG - narrative knowledge base",
     "q": "What's the Medicaid coverage outlook for GLP-1 obesity drugs?",
     "cat": "narrative", "mode": "full",
     "expect_engine": "RAG", "expect_contains": "payer_access"},
    {"group": "RAG - narrative knowledge base",
     "q": "What does the GLP-1 market look like in Missouri?",
     "cat": "narrative (state filter)", "mode": "full",
     "expect_engine": "RAG", "expect_contains": "state_market_summary__missouri",
     "note": "Missouri is the ONLY test in this file that exercises "
             "search_documents.py's state-filter path specifically, "
             "rather than general narrative semantic search."},
    {"group": "RAG - narrative knowledge base",
     "q": "How is Rabivy's mechanism different from tirzepatide?",
     "cat": "narrative", "mode": "full",
     "expect_engine": "RAG",
     "expect_contains": ["mechanism_comparison", "molecule_and_mechanism",
                          "how_is_this_different_from_ti"]},
    {"group": "RAG - narrative knowledge base",
     "q": "What is Rabivy's main dosing advantage?",
     "cat": "narrative", "mode": "full",
     "expect_engine": "RAG",
     "expect_contains": ["monthly_dosing", "where_rabivy_wins", "positioning_summary"]},
    {"group": "RAG - narrative knowledge base",
     "q": "What does a typical endocrinologist look like?",
     "cat": "narrative (benchmark)", "mode": "full",
     "expect_engine": "RAG", "expect_contains": "endocrinology"},
    {"group": "RAG - narrative knowledge base",
     "q": "How is prior authorization affecting access?",
     "cat": "narrative", "mode": "full",
     "expect_engine": "RAG", "expect_contains": ["prior_auth", "access"]},
    {"group": "RAG - narrative knowledge base",
     "q": "Why do patients stop taking GLP-1s after a year?",
     "cat": "narrative", "mode": "full",
     "expect_engine": "RAG", "expect_contains": ["why_patients_discontinue", "persistence", "discontinue"]},

    # =================================================================
    # GROUP 5: MULTI-SOURCE (structured + narrative + LLM, all at once)
    # =================================================================
    {"group": "MULTI-SOURCE (structured + narrative + LLM)",
     "q": "Who should I target next month in New York, and what should I say to them?",
     "cat": "multi-source showpiece", "mode": "llm",
     "expect_engine": "RAG",
     "note": "the showpiece: joins targeting + messaging - pure LLM synthesis. "
             "'who should I target' is recognised (routes plain targeting "
             "questions to STRUCTURED), but this question ALSO asks 'what "
             "should I say to them' - a deliberate guard backs off the "
             "structured trigger when messaging language is present, so this "
             "correctly still falls to RAG instead of silently dropping the "
             "messaging half of the question"},

    # =================================================================
    # GROUP 6: EDGE CASES - not-found, nonsense input & synonym mapping
    # These SHOULD come back empty/not-found or remapped - what's being
    # checked is that the router fails LOUDLY and CORRECTLY on bad input
    # instead of silently guessing or matching the wrong thing.
    # =================================================================
    {"group": "EDGE CASES - not-found, nonsense input & synonym mapping",
     "q": "Tell me about NPI 0000000000.",
     "cat": "not-found (fake NPI)", "mode": "full",
     "expect_engine": "RAG", "expect_contains": "no hcp card found",
     "note": "fake NPI - must say clearly not found, never fall through "
             "to an unrelated semantic search"},
    {"group": "EDGE CASES - not-found, nonsense input & synonym mapping",
     "q": "Which HCPs are in the Southeast region?",
     "cat": "not-found (fake region)", "mode": "full",
     "expect_engine": "STRUCTURED", "expect_contains": "not a real region",
     "note": "'Southeast' isn't a real region in this data - must reject "
             "explicitly rather than silently matching zero rows with no "
             "explanation"},
    {"group": "EDGE CASES - not-found, nonsense input & synonym mapping",
     "q": "Which HCPs have a Low tier?",
     "cat": "tier synonym mapping", "mode": "full",
     "expect_engine": "STRUCTURED", "expect_contains": "watch",
     "note": "'Low' isn't a real tier value - should map to 'Watch' (the "
             "real bottom tier) and return actual matches, not error out "
             "or silently return zero results"},
    {"group": "EDGE CASES - not-found, nonsense input & synonym mapping",
     "q": "Which HCPs have days since contact over 90?",
     "cat": "generic numeric threshold", "mode": "llm",
     "expect_engine": "STRUCTURED",
     "note": "exercises _detect_generic_numeric_filters on "
             "days_since_contact - the generic 'above/below a real number' "
             "path, not the high/moderate/low 0-1 score path. mode is "
             "'llm' (routing-only) since the exact row count depends on "
             "live data. PHRASING MATTERS (found 2026-07-22): the column "
             "phrase must come BEFORE the operator+number ('days since "
             "contact over 90' works; 'more than 90 days since contact' "
             "does not - no wildcard for word order). Also strict about "
             "filler words - built straight from the literal column name, "
             "unlike COLUMN_SYNONYMS elsewhere which tolerates 'days since "
             "(last) contact'. Real, untracked limitation - worth a proper "
             "fix later, not blocking now."},
]


# Previously tracked here as a known gap: "Why do patients stop taking
# GLP-1s after a year?" used to return zero results, because "top " (from
# the ranking-intent check) matched inside "stop taking". That routing
# bug is now fixed - this question retrieves correctly and has been moved
# up into the main TESTS list above as a normal narrative test. Kept this
# list here, empty, in case a new real gap needs tracking the same way.
KNOWN_LIMITS = [
]


def _deep_contains(obj, needle):
    """Search a nested dict/list structure for a substring, instead of
    depending on the exact wording of a pre-formatted sentence. A
    formatting change in query_spreadsheet.py's format_* functions
    (e.g. '55 scripts' -> '55.0 scripts last month') can't silently
    break this check for reasons unrelated to actual correctness."""
    if obj is None:
        return False
    if isinstance(obj, dict):
        return any(_deep_contains(v, needle) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_deep_contains(v, needle) for v in obj)
    return needle in str(obj).lower()


def _matches_any(data, expected):
    """expected can be a single string or a list of acceptable
    alternatives (any one matching counts as a pass) - carried over from
    eval.py's ground truth, which allowed several valid chunk-id
    substrings per question since more than one chunk can legitimately
    answer the same question."""
    alternatives = expected if isinstance(expected, list) else [expected]
    return any(_deep_contains(data, alt.lower()) for alt in alternatives)


def _retrieved_chunk_ids(data):
    """For RAG results, list the chunk_ids actually retrieved - carried
    over from eval.py's 'top: [...]' printout, so a FAIL is debuggable
    at a glance instead of needing a separate run to see what matched."""
    if isinstance(data, dict) and "chunks" in data:
        return [c["chunk_id"] for c in data["chunks"]]
    return None


_BAR = "=" * 72
_THIN = "-" * 72


def _print_group_banner(group_name):
    """Printed once, the first time a question from a new group appears
    - this is what replaces the old flat list with the newest questions
    looking bolted onto the end. Each group is now a clearly labelled
    section instead of one long undifferentiated scroll."""
    print("\n" + _BAR)
    print(f"  {group_name}")
    print(_BAR)


def _print_result(n, t, status, route_detail, note, answer_text, retrieved, error_text=None):
    """One question, one fully enclosed box. The question text itself
    is front and center in the header line (not hidden under a
    'QUESTION:' label further down) - Q# / question / STATUS, then a
    separator, then ROUTE/NOTE/ANSWER, then a closing border."""
    print(f"\n{_THIN}")
    print(f"Q{n}: {t['q']} - {status}")
    print(_THIN)
    if route_detail:
        print(f"  ROUTE    : {route_detail}")
    if note:
        print(f"  NOTE     : {note}")
    if retrieved is not None:
        print(f"  RETRIEVED: {[c[:40] for c in retrieved]}")
    if error_text:
        print(f"  ERROR    : {error_text}")
    if answer_text is not None:
        print(f"  ANSWER   :")
        for line in answer_text.splitlines():
            print(f"    {line}")
    print(_THIN)


def run(tests, header):
    print("\n" + "#" * 72)
    print(f"# {header}")
    print("#" * 72)
    tally = {"PASS": 0, "FAIL": 0, "ROUTES->LLM": 0, "NEEDS-LLM": 0, "CRASH": 0}
    records = []
    last_group = None

    for i, t in enumerate(tests, start=1):
        group = t.get("group")
        if group and group != last_group:
            _print_group_banner(group)
            last_group = group

        try:
            engine, data = ask_a_question.ask(t["q"])
        except Exception as e:
            tally["CRASH"] += 1
            _print_result(i, t, "CRASH", None, None, None, None,
                          error_text=f"{type(e).__name__}: {e}")
            records.append({**t, "status": "CRASH", "error": str(e)})
            continue

        routed = t["expect_engine"] in engine

        if t["mode"] == "full":
            correct = _matches_any(data, t["expect_contains"])
            status = "PASS" if (routed and correct) else "FAIL"
            route_detail = engine if status == "PASS" else (
                f"{engine} (WRONG - expected {t['expect_engine']})" if not routed
                else f"{engine} (right engine, but expected fact missing)")

        elif t["mode"] == "retrieval":
            got = _matches_any(data, t["expect_retrieves"])
            status = "ROUTES->LLM" if (routed and got) else "FAIL"
            route_detail = f"{engine} - retrieved correctly, answer text needs LLM" if status == "ROUTES->LLM" else (
                f"{engine} (WRONG - expected {t['expect_engine']})" if not routed
                else f"{engine} (did not retrieve expected item)")

        else:  # "llm" - now actually checks routing, doesn't rubber-stamp it
            status = "NEEDS-LLM" if routed else "FAIL"
            route_detail = engine if routed else f"{engine} (WRONG - expected {t['expect_engine']})"

        tally[status] = tally.get(status, 0) + 1

        retrieved = _retrieved_chunk_ids(data)
        try:
            answer_text = ask_a_question.format_answer(engine, data)
        except Exception as e:
            answer_text = None
            note_extra = f"!! format_answer() raised {type(e).__name__}: {e}"
        else:
            note_extra = None

        note = t.get("note")
        if note_extra:
            note = f"{note} | {note_extra}" if note else note_extra

        _print_result(i, t, status, route_detail, note, answer_text, retrieved)
        records.append({**t, "status": status, "engine": engine})

    return tally, records


main_tally, main_records = run(TESTS, "YOUR QUESTION LIST")
if KNOWN_LIMITS:
    limit_tally, limit_records = run(KNOWN_LIMITS, "KNOWN LIMITATION (kept visible on purpose)")
else:
    limit_tally, limit_records = {}, []

# ------------------- HEADLINE SUMMARY -------------------
p = main_tally.get("PASS", 0)
r = main_tally.get("ROUTES->LLM", 0)
n = main_tally.get("NEEDS-LLM", 0)
f = main_tally.get("FAIL", 0)
c = main_tally.get("CRASH", 0)
total = len(TESTS)
awaiting_llm = r + n

# ---- Plain-English summary, meant to be readable by someone who has
# ---- never seen this codebase and doesn't code. This is deliberately
# ---- the LAST thing printed, since that's what's on screen when the
# ---- run finishes - the part someone is most likely to actually read.
print("\n" + "=" * 72)
print("IN PLAIN ENGLISH - READ THIS PART FIRST")
print("=" * 72)
print(f"[WORKING]      {p} of {total} questions work perfectly right now -")
print(f"               correct answer, no AI needed.")
print(f"[WAITING ON AI] {awaiting_llm} of {total} found the right information, but need")
print(f"               the AI-writing step (not built into this part of the")
print(f"               system yet) to turn it into a full written answer.")
print(f"[BROKEN]       {f} of {total} are genuinely wrong - routed to the wrong")
print(f"               place, or didn't find what they should have.")
print(f"[ERROR]        {c} of {total} caused the program to crash outright.")
print("-" * 72)
if f == 0 and c == 0:
    print("Nothing here is actually broken. Everything either works today, or")
    print("is correctly waiting on a piece that hasn't been built yet.")
else:
    print("Some of these need attention - see FAIL / CRASH blocks above for")
    print("which specific questions and why.")
print("=" * 72)

# ---- Technical detail, kept for anyone who wants the underlying labels ----
print("\nWhat each label above means, in the detailed results further up:")
print("  PASS         = worked correctly, right answer, no AI needed")
print("  ROUTES->LLM  = found the right info; needs the AI to write the sentence")
print("  NEEDS-LLM    = went to the right place; needs the AI for the harder part")
print("  FAIL         = something is actually wrong here - needs a look")
print("  CRASH        = the program broke on this question - needs a look")
print()
print("RESULTS ACROSS YOUR QUESTION LIST:")
print(f"  PASS (fully working now)          : {p}")
print(f"  ROUTES->LLM (retrieval ready)     : {r}")
print(f"  NEEDS-LLM (routes ok, awaits LLM) : {n}")
print(f"  FAIL (needs attention)            : {f}")
print(f"  CRASH (threw an exception)        : {c}")
print("-" * 72)
n_known = len(KNOWN_LIMITS)
if n_known:
    plural = "" if n_known == 1 else "s"
    print(f"+ {n_known} known limitation{plural} tracked separately (see above) - "
          f"not counted in the tallies above.")
else:
    print("No known limitations currently being tracked separately.")

# ------------------- PERSIST FOR TRACKING OVER TIME -------------------
os.makedirs("eval_runs", exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_path = os.path.join("eval_runs", f"eval_{timestamp}.json")
with open(run_path, "w", encoding="utf-8") as fh:
    json.dump({
        "timestamp": timestamp,
        "tally": main_tally,
        "known_limits_tally": limit_tally,
        "results": main_records,
        "known_limits_results": limit_records,
    }, fh, indent=2, default=str)
print(f"\nSaved this run to {run_path} - diff against earlier runs in "
      f"eval_runs/ to track routing/retrieval accuracy over time.")