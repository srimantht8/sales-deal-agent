"""Deterministic unit tests for routing logic, CRM lookup, fan-out decisions,
and document loading (S3 + local).

These tests require no LLM calls — they exercise pure logic only.
"""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from agent.nodes.crm_lookup import crm_lookup_node
from agent.nodes.router import S3_QUERY_MAP, TONE_MAP, route_to_research, router_node
from tools.crm import crm_lookup
from tools.document_search import load_documents_from_s3, load_documents_local


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _router_state(**overrides) -> dict:
    """Build a minimal router-compatible state dict with sensible defaults."""
    base = {
        "is_public_company": False,
        "deal_stage": "discovery",
        "query_intent": "call_prep",
        "relationship_health": "neutral",
        "urgency": "normal",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Router node: 5 routing decisions
# ---------------------------------------------------------------------------

class TestRouterNode:
    """Test router_node() produces correct routing decisions."""

    def test_public_company_enables_edgar(self):
        result = router_node(_router_state(is_public_company=True))
        assert result["routing_decisions"]["edgar_enabled"] is True

    def test_private_company_disables_edgar(self):
        result = router_node(_router_state(is_public_company=False))
        assert result["routing_decisions"]["edgar_enabled"] is False

    @pytest.mark.parametrize("stage,expected_query", list(S3_QUERY_MAP.items()))
    def test_deal_stage_maps_to_s3_query(self, stage, expected_query):
        result = router_node(_router_state(deal_stage=stage))
        assert result["routing_decisions"]["s3_query"] == expected_query
        assert result["s3_search_query"] == expected_query

    @pytest.mark.parametrize("health,expected_tone", list(TONE_MAP.items()))
    def test_relationship_health_maps_to_tone(self, health, expected_tone):
        result = router_node(_router_state(relationship_health=health))
        assert result["routing_decisions"]["tone_strategy"] == expected_tone
        assert result["tone_strategy"] == expected_tone

    def test_high_urgency_skips_review(self):
        result = router_node(_router_state(urgency="high"))
        assert result["routing_decisions"]["skip_review"] is True
        assert result["skip_review"] is True

    def test_normal_urgency_does_not_skip_review(self):
        result = router_node(_router_state(urgency="normal"))
        assert result["routing_decisions"]["skip_review"] is False

    @pytest.mark.parametrize("intent", ["call_prep", "send_materials", "follow_up", "research_only"])
    def test_output_format_mirrors_intent(self, intent):
        result = router_node(_router_state(query_intent=intent))
        assert result["output_format"] == intent

    def test_unknown_deal_stage_falls_back(self):
        result = router_node(_router_state(deal_stage="unknown_stage"))
        assert result["routing_decisions"]["s3_query"] == "general overview"


# ---------------------------------------------------------------------------
# Parallel fan-out: Send() decisions
# ---------------------------------------------------------------------------

class TestRouteToResearch:
    """Test route_to_research() returns correct Send objects."""

    def test_public_company_fans_out_to_three_nodes(self):
        state = {"routing_decisions": {"edgar_enabled": True}}
        sends = route_to_research(state)
        node_names = [s.node for s in sends]
        assert len(sends) == 3
        assert "run_web_research" in node_names
        assert "run_s3_search" in node_names
        assert "run_sec_edgar" in node_names

    def test_private_company_fans_out_to_two_nodes(self):
        state = {"routing_decisions": {"edgar_enabled": False}}
        sends = route_to_research(state)
        node_names = [s.node for s in sends]
        assert len(sends) == 2
        assert "run_web_research" in node_names
        assert "run_s3_search" in node_names
        assert "run_sec_edgar" not in node_names

    def test_missing_routing_decisions_defaults_to_no_edgar(self):
        sends = route_to_research({})
        node_names = [s.node for s in sends]
        assert len(sends) == 2
        assert "run_web_research" in node_names
        assert "run_s3_search" in node_names


# ---------------------------------------------------------------------------
# CRM tool: lookup logic
# ---------------------------------------------------------------------------

class TestCRMLookup:
    """Test crm_lookup tool against mock data."""

    def test_exact_match(self):
        result = crm_lookup.invoke({"company_name": "NovaCrest Financial"})
        assert result["company_name"] == "NovaCrest Financial"
        assert result["deal_stage"] == "negotiation"

    def test_case_insensitive(self):
        result = crm_lookup.invoke({"company_name": "novacrest financial"})
        assert result["company_name"] == "NovaCrest Financial"

    def test_partial_match(self):
        result = crm_lookup.invoke({"company_name": "Meridian Health"})
        assert result["company_name"] == "Meridian Health Systems"

    def test_unknown_company_returns_error(self):
        result = crm_lookup.invoke({"company_name": "Nonexistent Corp"})
        assert "error" in result
        assert "available_companies" in result

    def test_all_five_companies_present(self):
        expected = [
            "Snowflake Inc", "NovaCrest Financial", "Meridian Health Systems",
            "AuroraStack Technologies", "Vertex Logistics Group",
        ]
        for name in expected:
            result = crm_lookup.invoke({"company_name": name})
            assert "error" not in result, f"CRM lookup failed for {name}"


# ---------------------------------------------------------------------------
# CRM lookup node: error handling
# ---------------------------------------------------------------------------

class TestCRMLookupNode:
    """Test crm_lookup_node() defaults on CRM miss."""

    def test_known_company_extracts_fields(self):
        result = crm_lookup_node({"company_name": "Vertex Logistics Group"})
        assert result["deal_stage"] == "evaluation"
        assert result["relationship_health"] == "at_risk"
        assert result["is_public_company"] is True

    def test_unknown_company_returns_defaults(self):
        result = crm_lookup_node({"company_name": "Nonexistent Corp"})
        assert result["deal_stage"] == "discovery"
        assert result["relationship_health"] == "neutral"
        assert result["is_public_company"] is False
        assert "error" in result["crm_data"]


# ---------------------------------------------------------------------------
# Document search: S3 loading
# ---------------------------------------------------------------------------

def _mock_s3_paginator(objects: list[dict]):
    """Build a mock boto3 S3 paginator that yields the given objects."""
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": objects}]
    client = MagicMock()
    client.get_paginator.return_value = paginator

    def mock_get_object(Bucket, Key):
        # Return markdown content based on the key
        content = f"# Doc from {Key}\n\nSample content."
        return {"Body": BytesIO(content.encode("utf-8"))}

    client.get_object.side_effect = mock_get_object
    return client


class TestDocumentSearchS3:
    """Test S3 document loading with mocked boto3."""

    @patch("boto3.client")
    def test_load_from_s3_parses_objects(self, mock_client):
        s3_objects = [
            {"Key": "documents/discovery/overview.md"},
            {"Key": "documents/negotiation/pricing.md"},
        ]
        mock_client.return_value = _mock_s3_paginator(s3_objects)

        docs = load_documents_from_s3("test-bucket")
        assert len(docs) == 2
        assert docs[0]["metadata"]["deal_stage"] == "discovery"
        assert docs[1]["metadata"]["deal_stage"] == "negotiation"
        assert "content" in docs[0]
        assert docs[0]["metadata"]["source"] == "documents/discovery/overview.md"

    @patch("boto3.client")
    def test_load_from_s3_skips_non_markdown(self, mock_client):
        s3_objects = [
            {"Key": "documents/discovery/overview.md"},
            {"Key": "documents/discovery/diagram.png"},
        ]
        mock_client.return_value = _mock_s3_paginator(s3_objects)

        docs = load_documents_from_s3("test-bucket")
        assert len(docs) == 1
        assert docs[0]["metadata"]["source"] == "documents/discovery/overview.md"

    @patch("boto3.client")
    def test_load_from_s3_empty_bucket(self, mock_client):
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        client = MagicMock()
        client.get_paginator.return_value = paginator
        mock_client.return_value = client

        docs = load_documents_from_s3("empty-bucket")
        assert docs == []


class TestDocumentSearchLocal:
    """Test local document loading against data/documents/."""

    def test_load_local_finds_all_documents(self):
        docs = load_documents_local()
        assert len(docs) == 12
        for doc in docs:
            assert "content" in doc
            assert "source" in doc["metadata"]
            assert doc["metadata"]["deal_stage"] in ("discovery", "evaluation", "negotiation", "renewal")

    def test_load_local_deal_stages_covered(self):
        docs = load_documents_local()
        stages = {doc["metadata"]["deal_stage"] for doc in docs}
        assert stages == {"discovery", "evaluation", "negotiation", "renewal"}
