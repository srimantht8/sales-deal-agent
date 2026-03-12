"""Integration tests — verify the full 9-node LangGraph graph wires together correctly.

Runs end-to-end through the compiled graph with all external dependencies mocked
(LLM, tools, interrupt). No API keys required.

Note: Mocking `interrupt` to return a value directly bypasses LangGraph's actual
pause/resume protocol. This tests node logic flow, not the interrupt mechanism itself.
"""

import uuid
from unittest.mock import MagicMock, patch

from agent.graph import create_agent
from agent.nodes.intake import ParsedQuery
from agent.nodes.synthesis import SynthesisResult


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(parsed_query: ParsedQuery, synthesis_result: SynthesisResult, generated_text: str):
    """Build a mock LLM that handles all three invocation patterns.

    - .with_structured_output(ParsedQuery).invoke() → parsed_query
    - .with_structured_output(SynthesisResult).invoke() → synthesis_result
    - .invoke() → MagicMock(content=generated_text)  (AIMessage-like)
    """
    mock_llm = MagicMock()

    def _with_structured_output(schema):
        sub_mock = MagicMock()
        if schema is ParsedQuery:
            sub_mock.invoke.return_value = parsed_query
        elif schema is SynthesisResult:
            sub_mock.invoke.return_value = synthesis_result
        else:
            sub_mock.invoke.return_value = MagicMock()
        return sub_mock

    mock_llm.with_structured_output.side_effect = _with_structured_output
    mock_llm.invoke.return_value = MagicMock(content=generated_text)
    return mock_llm


SNOWFLAKE_CRM = {
    "company_name": "Snowflake Inc",
    "is_public_company": True,
    "deal_stage": "discovery",
    "relationship_health": "neutral",
    "deal_value": 500_000,
    "products_interested": ["Enterprise Data Platform"],
    "competitors_mentioned": ["Databricks"],
    "contacts": [{"name": "Alice", "title": "CTO", "email": "a@snow.com", "role": "champion"}],
    "interaction_history": [],
}

AURORASTACK_CRM = {
    "company_name": "AuroraStack Technologies",
    "is_public_company": False,
    "deal_stage": "renewal",
    "relationship_health": "positive",
    "deal_value": 200_000,
    "products_interested": ["Cloud Migration"],
    "competitors_mentioned": [],
    "contacts": [{"name": "Bob", "title": "VP Eng", "email": "b@aurora.io", "role": "decision_maker"}],
    "interaction_history": [],
}

WEB_RESULTS = {
    "source": "tavily",
    "results": [{"title": "Company News", "url": "https://example.com", "content": "Recent developments..."}],
}

SEC_FILINGS = {
    "company": "Snowflake Inc",
    "filing_count": 1,
    "filings": [{"filing_type": "10-K", "date": "2025-03-01", "entity": "Snowflake Inc", "url": "https://sec.gov/..."}],
}

INTERNAL_DOCS = [
    {"content": "Case study overview document", "source": "discovery/overview.md", "deal_stage": "discovery", "similarity_score": 0.85},
]

GENERATED_TEXT = "## Call Preparation Briefing\n\nThis is a generated output for testing."


def _synthesis_result(**overrides):
    defaults = {
        "company_brief": "Test company brief.",
        "talking_points": ["Point 1", "Point 2"],
        "risks": ["Risk 1"],
        "opportunities": ["Opportunity 1"],
        "confidence_score": 0.85,
        "doc_gap": None,
        "intel_gap": None,
    }
    defaults.update(overrides)
    return SynthesisResult(**defaults)


# All patches target import sites, not definition sites.
_PATCHES = {
    "intake_llm":    "agent.nodes.intake.get_llm",
    "synthesis_llm": "agent.nodes.synthesis.get_llm",
    "generate_llm":  "agent.nodes.generate.get_llm",
    "crm":           "agent.nodes.crm_lookup.crm_lookup",
    "web_research":  "agent.nodes.research.search_web",
    "sec_edgar":     "agent.nodes.research.fetch_sec_filings",
    "doc_search":    "agent.nodes.research.search_documents",
    "synth_docs":    "agent.nodes.synthesis.search_documents",
    "synth_web":     "agent.nodes.synthesis.search_web",
    "interrupt":     "agent.nodes.human_review.interrupt",
}


def _run_graph(query, crm_data, parsed_query, synthesis_result,
               interrupt_response="approve"):
    """Invoke the compiled graph with all external deps mocked."""
    mock_llm = _make_mock_llm(parsed_query, synthesis_result, GENERATED_TEXT)

    with (
        patch(_PATCHES["intake_llm"], return_value=mock_llm),
        patch(_PATCHES["synthesis_llm"], return_value=mock_llm),
        patch(_PATCHES["generate_llm"], return_value=mock_llm),
        patch(_PATCHES["crm"]) as mock_crm,
        patch(_PATCHES["web_research"]) as mock_web,
        patch(_PATCHES["sec_edgar"]) as mock_sec,
        patch(_PATCHES["doc_search"]) as mock_docs,
        patch(_PATCHES["synth_docs"]) as mock_synth_docs,
        patch(_PATCHES["synth_web"]) as mock_synth_web,
        patch(_PATCHES["interrupt"], return_value=interrupt_response),
    ):
        mock_crm.invoke.return_value = crm_data
        mock_web.invoke.return_value = WEB_RESULTS
        mock_sec.invoke.return_value = SEC_FILINGS
        mock_docs.invoke.return_value = INTERNAL_DOCS
        mock_synth_docs.invoke.return_value = INTERNAL_DOCS
        mock_synth_web.invoke.return_value = WEB_RESULTS

        agent = create_agent()
        result = agent.invoke(
            {"query": query},
            config={"configurable": {"thread_id": f"test-{uuid.uuid4().hex}"}},
        )
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPublicCompanyPipeline:
    """Snowflake (public, discovery, neutral) → 3-node fan-out, professional tone."""

    def test_full_pipeline(self):
        parsed = ParsedQuery(
            company_name="Snowflake Inc",
            query_intent="call_prep",
            urgency="normal",
        )
        state = _run_graph(
            query="Prepare me for a call with Snowflake tomorrow",
            crm_data=SNOWFLAKE_CRM,
            parsed_query=parsed,
            synthesis_result=_synthesis_result(),
        )

        # Routing
        assert state["routing_decisions"]["edgar_enabled"] is True
        assert state["tone_strategy"] == "professional"

        # SEC EDGAR ran (public company)
        assert state.get("sec_filings") is not None

        # Output generated and approved
        assert state.get("final_output") is not None
        assert state["final_output"]["status"] == "human_approved"
        assert state["final_output"]["content"] != ""


class TestPrivateCompanyPipeline:
    """AuroraStack (private, renewal, positive) → 2-node fan-out, confident tone."""

    def test_full_pipeline(self):
        parsed = ParsedQuery(
            company_name="AuroraStack Technologies",
            query_intent="research_only",
            urgency="normal",
        )
        state = _run_graph(
            query="Research AuroraStack Technologies for renewal prep",
            crm_data=AURORASTACK_CRM,
            parsed_query=parsed,
            synthesis_result=_synthesis_result(),
        )

        # Routing
        assert state["routing_decisions"]["edgar_enabled"] is False
        assert state["tone_strategy"] == "confident"

        # SEC EDGAR skipped (private company)
        assert state.get("sec_filings") is None

        # Output generated and approved
        assert state.get("final_output") is not None
        assert state["final_output"]["content"] != ""


class TestHighUrgencyAutoApproval:
    """High urgency → skip_review=True → auto_approved without interrupt."""

    def test_auto_approved(self):
        parsed = ParsedQuery(
            company_name="Snowflake Inc",
            query_intent="call_prep",
            urgency="high",
        )
        state = _run_graph(
            query="URGENT: I need a call prep for Snowflake right now",
            crm_data=SNOWFLAKE_CRM,
            parsed_query=parsed,
            synthesis_result=_synthesis_result(),
        )

        assert state.get("skip_review") is True
        assert state["final_output"]["status"] == "auto_approved"
