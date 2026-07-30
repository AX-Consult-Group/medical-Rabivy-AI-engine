# test_the_system.py
# =====================================================================
# FULL-SYSTEM EVALUATION - the regex router (ask_a_question.py), tested
# end to end. THREE LAYERS are checked per question, SEPARATELY, so a
# result is honest about WHICH part is actually right or wrong:
#
#   LAYER 1 - ROUTING     Did the question go to the correct engine
#                         (STRUCTURED / RAG / CLARIFICATION)? Fixed
#                         forever - never changes just because the
#                         underlying data changes.
#
#   LAYER 2 - RULES       Did the system apply the correct RULES to get
#                         there - the right filters/sort/column, or (for
#                         RAG) the right chunk/tag(s)? Also fixed
#                         forever - the RULE a question needs is stable
#                         even though the VALUE that rule produces isn't.
#
#   LAYER 3 - ANSWER      Does the output match the CURRENT correct
#                         answer? Computed LIVE via ground_truth.py -
#                         never a frozen number. IMPORTANT: a question
#                         whose correct answer is "ask for clarification"
#                         or "say this is invalid" is JUST AS
#                         determinable as a numeric lookup - the correct
#                         behaviour never changes - so these are checked
#                         at Layer 3 too, never skipped just because the
#                         "value" is a fixed phrase rather than a number.
#                         Layer 3 is only genuinely skipped where there
#                         truly is no single checkable fact (e.g. a card
#                         lookup's full narrative text, or open-ended
#                         multi-source synthesis).
#
# GROUPING: edge cases (not-found, invalid input, AND the two
# unresolved-referent clarification questions) are grouped together at
# the end - they're all "the system must correctly refuse/ask, not
# guess" questions, so they belong together rather than the referent
# ones sitting inside "HCP cards & comparisons" as if they were normal
# lookups.

# =====================================================================
# KNOWN LIMITATION (found 2026-07-28): retrieval miss on the
# discontinuation question.
#
# Real run showed Q23 ("Why do patients stop taking GLP-1s after a
# year?") returning the wrong top chunk (0.705 similarity,
# "Monthly dosing doesn't matter that much") - the actual intended
# source chunk (why_patients_discontinue) wasn't in the top 5 at all.
#
# Found via test_retrieval_ranking.py, which checks whether the
# correct chunk is ranked first, not just present anywhere - this
# question came back "NOT IN TOP 5 AT ALL". That in turn revealed
# ground_truth.py's tag list for this question had been too loose
# (matched on plain English words that happened to appear in the
# wrong chunk's body text), which had been masking this exact miss as
# a false PASS. Tag list now tightened; Layer 2 correctly shows FAIL.
#
# Not fixed further - deliberately deferred, known embedding-model
# limitation, not a routing or logic bug.
# =====================================================================

import json
import os
import time

import ask_a_question
import query_spreadsheet as qs
import ground_truth as gt

ROUTER_TOP = gt.ROUTER_DEFAULT_TOP  # this file always tests the ROUTER's default (20), not the agent's (10)


# =====================================================================
# THE 30 QUESTIONS
# =====================================================================
QUESTIONS = [

    # =================================================================
    # GROUP 1: STRUCTURED - exact lookup, ranking & counting
    # =================================================================
    {
        "group": "STRUCTURED - exact lookup, ranking & counting", "cat": "structured lookup",
        "q": "Who is the top GLP-1 prescriber in New York?",
        "expect_engine": "STRUCTURED",
        "rules_check": lambda engine, data: "top prescriber" in engine and data.get("state", "").lower() == "new york",
        "ground_truth": lambda: gt.gt_top_prescriber("New York"),
    },
    {
        "group": "STRUCTURED - exact lookup, ranking & counting", "cat": "structured lookup",
        "q": "How many GLP-1 scripts did NPI 1344001929 write last month?",
        "expect_engine": "STRUCTURED",
        "rules_check": lambda engine, data: "hcp lookup" in engine,
        "ground_truth": lambda: gt.gt_scripts_for_npi("1344001929"),
    },
    {
        "group": "STRUCTURED - exact lookup, ranking & counting", "cat": "structured ranking",
        "q": "List the top 10 High-tier prescribers nationally by propensity score.",
        "expect_engine": "STRUCTURED",
        "rules_check": lambda engine, data: "top by propensity" in engine and data.get("tier") == "High" and data.get("n") == 10,
        # Was checking only the #1 NPI ([0]) even though the question
        # explicitly asks for 10 - fixed 2026-07-28 (found via a real run
        # where the agent correctly returned all 10 but the check only
        # verified one, silently hiding whether the other 9 were right).
        "ground_truth": lambda: gt.gt_top_n_national_npis(tier="High", n=10),
    },
    {
        "group": "STRUCTURED - exact lookup, ranking & counting", "cat": "structured ranking",
        "q": "Which states have the most High-tier HCPs?",
        "expect_engine": "STRUCTURED",
        "rules_check": lambda engine, data: "states by tier" in engine and data.get("tier") == "High",
        "ground_truth": lambda: gt.gt_top_state_by_tier("High"),
    },
    {
        "group": "STRUCTURED - exact lookup, ranking & counting", "cat": "structured count",
        "q": "How many active GLP-1 writers are in Texas?",
        "expect_engine": "STRUCTURED",
        "rules_check": lambda engine, data: "count" in engine and data.get("state", "").lower() == "texas",
        "ground_truth": lambda: gt.gt_count_writers("Texas", "active"),
    },

    # =================================================================
    # GROUP 2: STRUCTURED - filtered targeting (multi-field)
    # =================================================================
    {
        "group": "STRUCTURED - filtered targeting (multi-field)", "cat": "filtered targeting",
        "q": "Show me High-propensity endocrinologists in Florida who are not currently targeted.",
        "expect_engine": "STRUCTURED",
        "expect_filters": {"state": "Florida", "specialty": "Endocrinology", "tier": "High", "targeted": False},
        "ground_truth": lambda: gt.gt_filter_npis(top=ROUTER_TOP, state="Florida", specialty="Endocrinology",
                                                   tier="High", targeted=False),
    },
    {
        "group": "STRUCTURED - filtered targeting (multi-field)", "cat": "filtered targeting",
        "q": "Which Novo-heavy prescribers in California have a high switching score?",
        "expect_engine": "STRUCTURED",
        "expect_filters": {"state": "California", "dominant_competitor": "Novo Nordisk"},
        # HIGH_CUTOFF verified directly from query_spreadsheet.py = 0.75
        # (was guessed as 0.6, which FAILed against a real run - fixed
        # 2026-07-28). sort_by="switching_score" matches what the router
        # itself sorts by for this question (confirmed via a real run) -
        # without it, ground truth's default propensity-sorted top-20
        # and the router's switching-sorted top-20 are different slices
        # of the same ~104-row match set, so an "all must match" check
        # fails for a sort-order reason, not a real bug.
        "ground_truth": lambda: gt.gt_filter_npis(top=ROUTER_TOP, state="California",
                                                   dominant_competitor="Novo Nordisk",
                                                   extra_filters={"switching_score": (0.75, None)},
                                                   sort_by="switching_score"),
    },
    {
        "group": "STRUCTURED - filtered targeting (multi-field)", "cat": "filtered targeting",
        "q": "Find high-volume writers in Illinois with a preferred formulary status.",
        "expect_engine": "STRUCTURED",
        "expect_filters": {"state": "Illinois", "formulary_tier": "Preferred"},
        "ground_truth": lambda: gt.gt_filter_npis(top=ROUTER_TOP, state="Illinois", formulary_tier="Preferred",
                                                   sort_by="rx_volume_monthly"),
    },
    {
        "group": "STRUCTURED - filtered targeting (multi-field)", "cat": "filtered targeting",
        "q": "Which untargeted HCPs have a recent sample request and a High tier?",
        "expect_engine": "STRUCTURED",
        "expect_filters": {"targeted": False, "recent_sample_request": True, "tier": "High"},
        "ground_truth": lambda: gt.gt_filter_npis(top=ROUTER_TOP, targeted=False, recent_sample_request=True,
                                                   tier="High"),
    },

    # =================================================================
    # GROUP 3: RAG - HCP cards & comparisons
    # Only genuinely-answerable lookups live here now - the two
    # unresolved-referent ("this doctor") questions moved to EDGE CASES.
    # =================================================================
    {
        "group": "RAG - HCP cards & comparisons", "cat": "card summary",
        "q": "Give me a summary of NPI 1344001929.",
        "expect_engine": "RAG",
        "expect_tag": ["card_1344001929"],
        "note": "A specific-NPI lookup is ALWAYS fully determinable, whatever "
                "the data says - never skipped at Layer 3.",
        "ground_truth": lambda: gt.gt_hcp_card_facts("1344001929"),
    },
    {
        "group": "RAG - HCP cards & comparisons", "cat": "card (chained)",
        "q": "What's the access situation for the top prescriber in Missouri?",
        "expect_engine": "STRUCTURED",
        "note": "Chained question: this only proves step 1 (find top prescriber) "
                "routes and resolves correctly. Step 2 (read their card) and "
                "synthesis aren't wired yet.",
        "rules_check": lambda engine, data: "top prescriber" in engine and data.get("state", "").lower() == "missouri",
        "ground_truth": lambda: gt.gt_top_prescriber("Missouri"),
    },
    {
        "group": "RAG - HCP cards & comparisons", "cat": "card narrative",
        "q": "What's the story on GLP-1 writers in Arizona - who should I know about?",
        "expect_engine": "RAG",
        "narrative_key": "arizona_market",
        "note": "State summaries are static docs right now, but this Layer 3 "
                "check parses the real doc LIVE at test time (see "
                "gt_state_market_fact) rather than a hardcoded number - if "
                "the doc's Arizona figures change, this follows automatically.",
    },
    {
        "group": "RAG - HCP cards & comparisons", "cat": "comparison",
        "q": "Compare NPI 1344001929 to a typical endocrinologist.",
        "expect_engine": "RAG",
        "note": "TWO sources must both be present: the NPI's own card AND the "
                "endocrinology benchmark chunk. Layer 3 checks a fact unique "
                "to EACH side (the NPI's real script count, and the live "
                "specialty-wide mean propensity) so a comparison that only "
                "pulled one side would fail here.",
        "rules_check": lambda engine, data: gt.any_present(json.dumps(data, default=str).lower(), ["1344001929"])
                        and gt.any_present(json.dumps(data, default=str).lower(),
                                          ["specialty_benchmark_profiles__typical_endocrinology_profile", "endocrinology"]),
        "ground_truth": lambda: [gt.gt_scripts_for_npi("1344001929")] + gt.gt_specialty_benchmark_facts("Endocrinology"),
    },

    # =================================================================
    # GROUP 4: RAG - narrative knowledge base
    # =================================================================
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "What's our recommended messaging for competitive switchers?",
        "expect_engine": "RAG", "narrative_key": "competitive_switchers",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "What is Rabivy's key differentiator versus Zepbound?",
        "expect_engine": "RAG", "narrative_key": "zepbound_differentiator",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "How should a rep handle 'I'm already happy with Ozempic'?",
        "expect_engine": "RAG", "narrative_key": "ozempic_objection",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "What's the Medicaid coverage outlook for GLP-1 obesity drugs?",
        "expect_engine": "RAG", "narrative_key": "medicaid_access",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative (state filter)",
        "q": "What does the GLP-1 market look like in Missouri?",
        "expect_engine": "RAG", "narrative_key": "missouri_market",
        "note": "Layer 3 parses the real state_market_summary.md LIVE at test "
                "time (see gt_state_market_fact) rather than a hardcoded "
                "'283'/'136' - the only test here exercising "
                "search_documents.py's state-filter path specifically.",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "How is Rabivy's mechanism different from tirzepatide?",
        "expect_engine": "RAG", "narrative_key": "tirzepatide_mechanism",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "What is Rabivy's main dosing advantage?",
        "expect_engine": "RAG", "narrative_key": "dosing_advantage",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative (benchmark)",
        "q": "What does a typical endocrinologist look like?",
        "expect_engine": "RAG", "narrative_key": "typical_endocrinologist",
        "note": "Tag verified 2026-07-28. Layer 3 is computed LIVE from the "
                "dataframe (population count + mean propensity), not read "
                "from the doc - a real check found the doc's 'active writers' "
                "figure has already drifted from the live data.",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "How is prior authorization affecting access?",
        "expect_engine": "RAG", "narrative_key": "prior_auth_access",
        "note": "Layer 3 facts '31'/'41' are the PA initial-approval-rate "
                "range (31-41%) for anti-obesity GLP-1s, from "
                "real_world_evidence_brief.md's 'Prior Authorization Reality' "
                "section - confirmed by reading the source directly.",
    },
    {
        "group": "RAG - narrative knowledge base", "cat": "narrative",
        "q": "Why do patients stop taking GLP-1s after a year?",
        "expect_engine": "RAG", "narrative_key": "discontinuation",
    },

    # =================================================================
    # GROUP 5: MULTI-SOURCE (structured + narrative + LLM, all at once)
    # =================================================================
    {
        "group": "MULTI-SOURCE (structured + narrative + LLM)", "cat": "multi-source showpiece",
        "q": "Who should I target next month in New York, and what should I say to them?",
        "expect_engine": "RAG",
        "note": "The showpiece: joins targeting + messaging - pure LLM "
                "synthesis. THE RULE IS REAL AND CHECKABLE: a correct answer "
                "needs BOTH a structured targeting signal (top-propensity NY "
                "NPI) AND a messaging/talking-points chunk. Scored as FAIL, "
                "not SKIP - the regex router can only ever return ONE route "
                "per call, so it genuinely cannot satisfy this rule, and that "
                "is a real, honest limitation worth seeing rather than hiding "
                "behind a skip. test_the_agent.py's version of this question "
                "checks the identical rule and should PASS there.",
        "rules_check": lambda engine, data: gt.any_present(json.dumps(data, default=str).lower(),
                                                            ["rep_talking_points_by_segment"])
                        and gt.any_present(json.dumps(data, default=str).lower(),
                                          [(gt.gt_filter_npis(top=1, state="New York") or [""])[0]]),
        "ground_truth": lambda: [(gt.gt_filter_npis(top=1, state="New York") or [None])[0],
                                  "rep_talking_points_by_segment"],
    },

    # =================================================================
    # GROUP 6: EDGE CASES - not-found, nonsense input, synonym mapping,
    # and unresolved referents. ALL of these must correctly REFUSE or
    # ASK rather than guess - that correct behaviour is itself the fixed,
    # always-determinable "answer", so Layer 3 checks it directly instead
    # of skipping. Layer 2 is legitimately SKIP for most of these (there's
    # no separate filter/tag rule beyond "handle this input correctly").
    # =================================================================
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "unresolved referent",
        "q": "Why is this High-tier doctor not converting?",
        "expect_engine": "CLARIFICATION",
        "note": "No NPI given, no prior HCP in context - correctly asks for "
                "clarification instead of guessing. The correct answer never "
                "changes, so Layer 3 checks it directly (moved here from "
                "'HCP cards & comparisons', where it didn't really belong).",
        "rules_check": lambda engine, data: data.get("kind") == "clarification",
        "ground_truth": lambda: gt.CLARIFICATION_FACTS,
    },
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "unresolved referent",
        "q": "Is this doctor a high or low prescriber for their specialty?",
        "expect_engine": "CLARIFICATION",
        "rules_check": lambda engine, data: data.get("kind") == "clarification",
        "ground_truth": lambda: gt.CLARIFICATION_FACTS,
    },
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "not-found (fake NPI)",
        "q": "Tell me about NPI 0000000000.",
        "expect_engine": "RAG",
        "rules_check": lambda engine, data: "0000000000" in data.get("error", ""),
        "note": "Layer 2 confirms the SPECIFIC invalid NPI from the question "
                "was actually looked up (not a generic canned response).",
        "ground_truth": lambda: gt.FAKE_NPI_REJECTION_FACTS,
    },
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "not-found (fake region)",
        "q": "Which HCPs are in the Southeast region?",
        "expect_engine": "STRUCTURED",
        "rules_check": lambda engine, data: "southeast" in data.get("error", "").lower(),
        "note": "Layer 2 confirms the SPECIFIC invalid region from the "
                "question was actually processed (not a generic canned response).",
        "ground_truth": lambda: gt.FAKE_REGION_REJECTION_FACTS,
    },
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "tier synonym mapping",
        "q": "Which HCPs have a Low tier?",
        "expect_engine": "STRUCTURED",
        "rules_check": lambda engine, data: data.get("filters", {}).get("tier") == "Watch",
        "note": "'Low' isn't a real tier value - must map to 'Watch' (the "
                "real bottom tier) and return actual matches.",
        "ground_truth": lambda: gt.gt_filter_npis(top=ROUTER_TOP, tier="Watch"),
    },
    {
        "group": "EDGE CASES - not-found, nonsense input, synonym mapping & unresolved referents",
        "cat": "generic numeric threshold",
        "q": "Which HCPs have days since contact over 90?",
        "expect_engine": "STRUCTURED",
        "rules_check": lambda engine, data: (data.get("filters", {}).get("extra_filters") or {}).get(
            "days_since_contact", (None, None))[0] == 90.0,
        "note": "The rule is 'days_since_contact > 90, no upper bound' - only "
                "5,017 or so of 15,000 HCPs match, so the top-20 rows shown "
                "are an arbitrary slice. Layer 3 checks the TOTAL match count "
                "(computed live), not any specific NPI in that slice.",
        "ground_truth": lambda: gt.gt_filter_count(extra_filters={"days_since_contact": (90, None)}),
    },
]


# =====================================================================
# EVAL MACHINERY
# =====================================================================

def _haystack(data):
    return json.dumps(data, default=str).lower()


def _layer1(t, engine):
    return t["expect_engine"] in engine


def _layer2(t, engine, data):
    """Returns (status, detail) - status is PASS / FAIL / SKIP."""
    if "rules_check" in t:
        ok = t["rules_check"](engine, data)
        return ("PASS" if ok else "FAIL"), "rules_check"
    if "expect_filters" in t:
        filters = data.get("filters", {})
        mismatches = {k: (v, filters.get(k)) for k, v in t["expect_filters"].items()
                      if filters.get(k) != v}
        return ("PASS" if not mismatches else "FAIL"), (mismatches or "filters matched")
    if "expect_tag" in t:
        ok = gt.any_present(_haystack(data), t["expect_tag"])
        return ("PASS" if ok else "FAIL"), t["expect_tag"]
    if "narrative_key" in t:
        spec = gt.NARRATIVE_FACTS[t["narrative_key"]]
        if spec["tag"] is None:
            return "SKIP", "no verified tag yet"
        ok = gt.any_present(_haystack(data), spec["tag"])
        return ("PASS" if ok else "FAIL"), spec["tag"]
    return "SKIP", "no separate rule beyond correct routing/refusal for this question"


def _layer3(t, engine, data):
    """Returns (status, detail) - status is PASS / FAIL / SKIP."""
    haystack = _haystack(data)
    if "narrative_key" in t:
        spec = gt.NARRATIVE_FACTS[t["narrative_key"]]
        if spec["facts"] is not None:
            ok = gt.all_present(haystack, spec["facts"])
            return ("PASS" if ok else "FAIL"), spec["facts"]
        if spec.get("state"):
            fact = gt.gt_state_market_fact(spec["state"])
            if fact is None:
                return "SKIP", "could not parse the live doc"
            ok = fact in haystack
            return ("PASS" if ok else "FAIL"), f"expected {fact} (parsed live from the doc)"
        if spec.get("specialty"):
            facts = gt.gt_specialty_benchmark_facts(spec["specialty"])
            if facts is None:
                return "SKIP", "could not compute live specialty facts"
            ok = gt.all_present(haystack, facts)
            return ("PASS" if ok else "FAIL"), f"expected {facts} (computed live from the dataframe)"
    gt_fn = t.get("ground_truth")
    if gt_fn is None:
        return "SKIP", "no live ground truth for this question (see note above)"
    truth = gt_fn()
    if truth is None:
        return "SKIP", "ground truth computation returned nothing (data may be empty)"
    if isinstance(truth, list):
        # A list of facts (plain strings and/or OR-groups, e.g.
        # CLARIFICATION_FACTS or a comma/no-comma population variant) -
        # all_present handles both item types correctly.
        ok = gt.all_present(haystack, truth)
        return ("PASS" if ok else "FAIL"), f"expected all of {truth}"
    ok = str(truth).lower() in haystack
    return ("PASS" if ok else "FAIL"), f"expected {truth}"


_BAR = "=" * 72


def run():
    tally = {"layer1": {"PASS": 0, "FAIL": 0}, "layer2": {"PASS": 0, "FAIL": 0, "SKIP": 0},
             "layer3": {"PASS": 0, "FAIL": 0, "SKIP": 0}, "CRASH": 0}
    records = []
    last_group = None

    for i, t in enumerate(QUESTIONS, start=1):
        if t["group"] != last_group:
            print(f"\n{_BAR}\n  {t['group']}\n{_BAR}")
            last_group = t["group"]

        print(f"\n{'=' * 8} QUESTION {i} {'=' * 8}")
        print(f"Q: {t['q']}")
        print(f"[{t['cat']}]")
        if t.get("note"):
            print(f"NOTE: {t['note']}")

        try:
            engine, data = ask_a_question.ask(t["q"])
        except Exception as e:
            tally["CRASH"] += 1
            print(f"  CRASH: {type(e).__name__}: {e}")
            records.append({"q": t["q"], "crash": str(e)})
            continue

        l1_ok = _layer1(t, engine)
        tally["layer1"]["PASS" if l1_ok else "FAIL"] += 1
        print(f"  LAYER 1 (routing) : {'PASS' if l1_ok else 'FAIL'}  (got: {engine})")

        l2_status, l2_detail = _layer2(t, engine, data)
        tally["layer2"][l2_status] += 1
        print(f"  LAYER 2 (rules)   : {l2_status}  ({l2_detail})")

        l3_status, l3_detail = _layer3(t, engine, data)
        tally["layer3"][l3_status] += 1
        print(f"  LAYER 3 (answer)  : {l3_status}  ({l3_detail})")

        try:
            answer_text = ask_a_question.format_answer(engine, data)
            print(f"  ANSWER: {answer_text}")
        except Exception as e:
            print(f"  !! format_answer() raised {type(e).__name__}: {e}")

        records.append({"q": t["q"], "engine": engine, "layer1": l1_ok,
                        "layer2": l2_status, "layer3": l3_status})

    print(f"\n{_BAR}\nSUMMARY\n{_BAR}")
    print(f"Layer 1 (routing) : {tally['layer1']['PASS']} PASS / {tally['layer1']['FAIL']} FAIL")
    print(f"Layer 2 (rules)   : {tally['layer2']['PASS']} PASS / {tally['layer2']['FAIL']} FAIL "
          f"/ {tally['layer2']['SKIP']} SKIP")
    print(f"Layer 3 (answer)  : {tally['layer3']['PASS']} PASS / {tally['layer3']['FAIL']} FAIL "
          f"/ {tally['layer3']['SKIP']} SKIP")
    print(f"Crashes           : {tally['CRASH']}")
    print(f"{_BAR}\nA data refresh should only ever move Layer 3 numbers. If Layer 1 or\n"
          f"Layer 2 fail after a data refresh, that's a REAL bug - the routing/\n"
          f"rules logic doesn't depend on which HCP happens to be #1 today.\n{_BAR}")

    os.makedirs("eval_runs", exist_ok=True)
    path = os.path.join("eval_runs", f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tally": tally, "records": records}, f, indent=2, default=str)
    print(f"\nSaved this run to {path}")
    return tally, records


if __name__ == "__main__":
    run()