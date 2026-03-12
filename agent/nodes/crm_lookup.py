"""Phase 2: CRM lookup — fetch company data from mock CRM."""

import logging

from agent.state import AgentState
from tools.crm import crm_lookup

logger = logging.getLogger(__name__)


def crm_lookup_node(state: AgentState) -> dict:
    """Look up the company in CRM and extract key fields for routing."""
    company_name = state["company_name"]
    logger.info(f"CRM lookup: {company_name}")

    result = crm_lookup.invoke({"company_name": company_name})

    if isinstance(result, dict) and "error" in result:
        logger.warning(f"CRM lookup failed: {result['error']}")
        return {
            "crm_data": result,
            "deal_stage": "discovery",
            "relationship_health": "neutral",
            "is_public_company": False,
        }

    return {
        "crm_data": result,
        "deal_stage": result.get("deal_stage", "discovery"),
        "relationship_health": result.get("relationship_health", "neutral"),
        "is_public_company": result.get("is_public_company", False),
    }
