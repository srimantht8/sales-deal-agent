"""Phase 3: Smart router — makes 5 routing decisions based on CRM data + intent."""

import logging

from langgraph.types import Send

from agent.state import AgentState

logger = logging.getLogger(__name__)

# Maps deal stage to appropriate document search queries
S3_QUERY_MAP = {
    "discovery": "case study overview introduction customer success",
    "evaluation": "technical architecture integration security compliance",
    "negotiation": "proposal pricing ROI enterprise agreement",
    "renewal": "usage expansion roadmap renewal success metrics",
}

# Maps relationship health to tone strategy
TONE_MAP = {
    "positive": "confident",
    "neutral": "professional",
    "at_risk": "empathetic",
}


def router_node(state: AgentState) -> dict:
    """Make 5 routing decisions based on CRM data and parsed intent. Pure logic, no LLM."""
    deal_stage = state.get("deal_stage", "discovery")
    intent = state.get("query_intent", "call_prep")
    health = state.get("relationship_health", "neutral")
    urgency = state.get("urgency", "normal")
    is_public = state.get("is_public_company", False)

    routing_decisions = {
        "edgar_enabled": is_public,
        "s3_query": S3_QUERY_MAP.get(deal_stage, "general overview"),
        "output_format": intent,
        "tone_strategy": TONE_MAP.get(health, "professional"),
        "skip_review": urgency == "high",
    }

    logger.info(f"Router decisions: {routing_decisions}")

    return {
        "routing_decisions": routing_decisions,
        "s3_search_query": routing_decisions["s3_query"],
        "output_format": routing_decisions["output_format"],
        "tone_strategy": routing_decisions["tone_strategy"],
        "skip_review": routing_decisions["skip_review"],
    }


def route_to_research(state: AgentState) -> list[Send]:
    """Conditional edge function: fan out to parallel research nodes via Send()."""
    sends = [
        Send("run_web_research", state),
        Send("run_s3_search", state),
    ]
    if state.get("routing_decisions", {}).get("edgar_enabled", False):
        sends.append(Send("run_sec_edgar", state))
        logger.info("Router: fan-out to web_research + s3_search + sec_edgar (public company)")
    else:
        logger.info("Router: fan-out to web_research + s3_search (private company, skipping EDGAR)")
    return sends
