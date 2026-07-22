# agent_tools.py
# -------------------------------------------------------------------
# The bridge between the agent and the two existing engines. Each tool
# below wraps functionality that query_spreadsheet.py / search_documents.py
# already provide - NOTHING here re-implements retrieval or querying.
# The agentic layer's whole premise is that the LLM replaces the regex
# ROUTER (ask_a_question.py's keyword detectors), not the engines.
#
# Two exports:
#   TOOL_SCHEMAS - Anthropic tool-use JSON schemas the LLM plans with
#   run_tool()   - executes one call, always returns a JSON-safe dict
#                  (errors come back as {"error": ...}, never exceptions,
#                  so one bad call can't kill the agent loop)
#
# Result-size discipline: tools return trimmed, field-selected results
# (not whole DataFrame rows), because every byte returned is a token
# the LLM has to read. The full row is one lookup_hcp call away.
# -------------------------------------------------------------------

import json

import numpy as np

import query_spreadsheet
import search_documents

# Key fields shown in table results - enough to reason and cite, small
# enough to keep the context window sane at top=20.
_ROW_FIELDS = ["npi", "specialty", "state", "tier", "propensity_score",
               "propensity_rank", "rx_volume_monthly", "nrx_monthly",
               "switching_score", "targeted", "dominant_competitor",
               "formulary_tier", "pa_burden", "days_since_contact",
               "sample_request_recent", "zero_writer"]


def _trim_row(row):
    out = {}
    for f in _ROW_FIELDS:
        if f in row:
            v = row[f]
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (float, np.floating)):
                # Round EVERY float (plain Python floats included - pandas
                # to_dict() can hand back either). 3 decimals is plenty of
                # precision for scores; long float tails also made the
                # verification auditor flag honest rounding in answers as
                # mismatches against the raw evidence.
                v = round(float(v), 3)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            elif hasattr(v, "item"):  # any other numpy scalar
                v = v.item()
            out[f] = v
    return out


# =====================================================================
# TOOL SCHEMAS (what the LLM sees)
# =====================================================================

TOOL_SCHEMAS = [
    {
        "name": "query_hcp_table",
        "description": (
            "Query the master HCP propensity table (15,000 US prescribers, "
            "synthetic data). Filter by any combination of fields, sort, and "
            "return the top rows plus the total match count. Use this for "
            "every ranking, counting, filtering or targeting question about "
            "HCPs - the numbers returned are exact database facts, never "
            "estimates. States are full names like 'Texas'."),
        "input_schema": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "description": "Full state name, e.g. 'New York'"},
                "specialty": {"type": "string",
                              "enum": ["Endocrinology", "Primary Care", "Obesity Medicine"]},
                "tier": {"type": "string", "enum": ["High", "Medium", "Watch"],
                         "description": "Propensity tier"},
                "targeted": {"type": "boolean",
                             "description": "true = currently targeted by a rep, false = not targeted"},
                "dominant_competitor": {"type": "string",
                                        "enum": ["Novo Nordisk", "Eli Lilly"]},
                "min_switching": {"type": "number",
                                  "description": "Minimum switching_score (0-1)"},
                "formulary_tier": {"type": "string",
                                   "enum": ["Preferred", "NonPreferred", "PARequired", "NotCovered"]},
                "recent_sample_request": {"type": "boolean"},
                "min_pa_burden": {"type": "number"},
                "max_pa_burden": {"type": "number"},
                "extra_filters": {
                    "type": "array",
                    "description": ("Numeric range filters on any other column, e.g. "
                                    "[{'column': 'days_since_contact', 'min': 30}]. Columns: "
                                    "rx_volume_monthly, nrx_monthly, days_since_contact, "
                                    "rep_engagement_score, propensity_score, years_practice, "
                                    "pct_novo, pct_lilly, obesity_prev."),
                    "items": {"type": "object",
                              "properties": {"column": {"type": "string"},
                                             "min": {"type": "number"},
                                             "max": {"type": "number"}},
                              "required": ["column"]},
                },
                "sort_by": {"type": "string",
                            "description": ("Column to sort by. Default "
                                            "'propensity_score'. Use 'rx_volume_monthly' for "
                                            "volume questions, 'switching_score' for switching "
                                            "questions.")},
                "ascending": {"type": "boolean",
                              "description": ("true = lowest first (e.g. 'lowest PA burden'), "
                                              "false/omit = highest first (the usual case).")},
                "top": {"type": "integer",
                        "description": "Rows to return (default 10, max 25). Total count always included."},
            },
        },
    },
    {
        "name": "lookup_hcp",
        "description": (
            "Look up ONE prescriber by their 10-digit NPI. Returns their full "
            "row from the propensity table AND their narrative account "
            "snapshot card (prescribing history, payer mix, engagement notes). "
            "Use whenever a question names a specific NPI."),
        "input_schema": {
            "type": "object",
            "properties": {"npi": {"type": "string", "description": "10-digit NPI"}},
            "required": ["npi"],
        },
    },
    {
        "name": "count_active_writers",
        "description": ("Count active GLP-1 writers (excludes zero-writers) in a "
                        "state. Returns active and total counts."),
        "input_schema": {
            "type": "object",
            "properties": {"state": {"type": "string", "description": "Full state name"}},
            "required": ["state"],
        },
    },
    {
        "name": "states_summary",
        "description": "Which states have the most High-tier HCPs. Returns the top N states with counts.",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "How many states (default 5)"}},
        },
    },
    {
        "name": "search_documents",
        "description": (
            "Semantic search over the knowledge repository: clinical briefs, "
            "competitive intelligence, payer/access briefs, rep talking points, "
            "objection handling guides, specialty benchmark profiles, and "
            "state market summaries. Use for messaging, positioning, clinical, "
            "access and market-landscape questions. Returns the most relevant "
            "document sections with similarity scores and a low_confidence "
            "flag - if low_confidence is true, retry once with a differently "
            "worded query (different vocabulary, not just reordered words)."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Search query - what to search for."},
                "state": {"type": "string",
                          "description": ("Optional. Full state name, e.g. 'Missouri' - "
                                         "filters the search to that state's documents "
                                         "only. Leave unset for questions not about a "
                                         "specific state.")},
                "top_k": {"type": "integer", "description": "Sections to return (default 4, max 8)"},
            },
            "required": ["query"],
        },
    },
]


# =====================================================================
# EXECUTORS
# =====================================================================

def _query_hcp_table(inp):
    # Reconciled 2026-07-22: filter_hcps() no longer accepts
    # min_switching/min_pa_burden/max_pa_burden as direct keyword
    # arguments - those were replaced by a generic extra_filters dict
    # of {column: (min, max)} tuples, which works for ANY real column,
    # score or not. So the old "special" numeric fields fold into the
    # same extra_filters dict as the tool's own generic ones, instead
    # of being passed separately.
    extra = {}
    if inp.get("extra_filters"):
        extra.update({f["column"]: (f.get("min"), f.get("max"))
                      for f in inp["extra_filters"] if f.get("column")})
    if inp.get("min_switching") is not None:
        extra["switching_score"] = (inp["min_switching"], None)
    if inp.get("min_pa_burden") is not None or inp.get("max_pa_burden") is not None:
        extra["pa_burden"] = (inp.get("min_pa_burden"), inp.get("max_pa_burden"))
    extra = extra or None

    top = min(int(inp.get("top", 10)), 25)
    data = query_spreadsheet.filter_hcps(
        state=inp.get("state"),
        specialty=inp.get("specialty"),
        tier=inp.get("tier"),
        targeted=(None if inp.get("targeted") is None else int(inp["targeted"])),
        dominant_competitor=inp.get("dominant_competitor"),
        formulary_tier=inp.get("formulary_tier"),
        recent_sample_request=(None if inp.get("recent_sample_request") is None
                               else int(inp["recent_sample_request"])),
        extra_filters=extra,
        sort_by=inp.get("sort_by", "propensity_score"),
        ascending=bool(inp.get("ascending", False)),
        top=top,
    )
    if not data.get("found", True):
        return {"error": data.get("error", "query failed")}
    return {
        "source": "hcp_propensity_table",
        "total_matches": data["count"],
        "returned": len(data["results"]),
        "filters_applied": {k: v for k, v in data["filters"].items() if v is not None},
        "rows": [_trim_row(r) for r in data["results"]],
    }


def _lookup_hcp(inp):
    npi = str(inp.get("npi", "")).strip()
    row = query_spreadsheet.get_row_by_npi(npi)
    if not row.get("found"):
        return {"error": row.get("error", f"NPI {npi} not found")}
    row.pop("found", None)
    result = {"source": "hcp_propensity_table", "row": _trim_row(row)}
    # Join to the RAG side: the HCP's narrative snapshot card, if present.
    idx = search_documents.card_by_npi.get(npi)
    if idx is not None:
        card = search_documents.chunks[idx]
        result["snapshot_card"] = {"chunk_id": card["chunk_id"],
                                   "text": card["text"][:2500]}
    return result


def _count_active_writers(inp):
    data = query_spreadsheet.count_writers(inp["state"])
    return {"source": "hcp_propensity_table", **data}


def _states_summary(inp):
    data = query_spreadsheet.states_by_high_tier(int(inp.get("n", 5)))
    return {"source": "hcp_propensity_table", "high_tier_hcps_by_state": data["results"]}


def _search_documents(inp):
    # Reconciled 2026-07-22: search_documents.search() no longer
    # exists - replaced by semantic_search(), which returns a single
    # dict ({"kind","found","confidence","results":[{"chunk","score"}]})
    # instead of the old (route_string, [(chunk, score), ...]) tuple.
    # Also now takes state as its own explicit parameter rather than
    # relying on the state name being embedded in the free-text query -
    # see the "state" field added to this tool's input_schema above.
    top_k = min(int(inp.get("top_k", 4)), 8)
    data = search_documents.semantic_search(
        inp["query"], state=inp.get("state"), top_k=top_k)
    if not data.get("found", True):
        return {"error": data.get("error", "search failed")}
    return {
        "source": "knowledge_repository",
        "confidence": data.get("confidence"),
        "low_confidence": data.get("low_confidence", False),
        "sections": [
            {"chunk_id": r["chunk"]["chunk_id"],
             "doc": r["chunk"].get("source_doc"),
             "heading": r["chunk"].get("heading"),
             "similarity": (round(r["score"], 3) if r["score"] is not None else None),
             "text": r["chunk"]["text"][:1500]}
            for r in data["results"]
        ],
    }


_EXECUTORS = {
    "query_hcp_table": _query_hcp_table,
    "lookup_hcp": _lookup_hcp,
    "count_active_writers": _count_active_writers,
    "states_summary": _states_summary,
    "search_documents": _search_documents,
}


def run_tool(name, tool_input):
    """Execute one tool call. Always returns a dict; never raises - a
    malformed call comes back as {"error": ...} for the LLM to read,
    correct, and retry (that's self-correction working, not a crash)."""
    fn = _EXECUTORS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'. Available: {sorted(_EXECUTORS)}"}
    try:
        result = fn(tool_input or {})
        # Round-trip through JSON to guarantee serialisability (numpy
        # scalars etc.) before it goes back into the message history.
        return json.loads(json.dumps(result, default=str))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    print(json.dumps(run_tool("count_active_writers", {"state": "Texas"}), indent=2))
    print(json.dumps(run_tool("query_hcp_table",
                              {"state": "Florida", "specialty": "Endocrinology",
                               "tier": "High", "targeted": False, "top": 3}), indent=2))
    r = run_tool("lookup_hcp", {"npi": "1344001929"})
    r["snapshot_card"]["text"] = r["snapshot_card"]["text"][:200] + "..."
    print(json.dumps(r, indent=2))
    r = run_tool("search_documents", {"query": "How is Rabivy different from Zepbound?"})
    for s in r["sections"]:
        s["text"] = s["text"][:100] + "..."
    print(json.dumps(r, indent=2))