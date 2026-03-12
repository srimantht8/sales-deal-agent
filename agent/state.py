"""Agent state schema for the Sales Deal Acceleration workflow."""

from typing import TypedDict


class AgentState(TypedDict):
    # Phase 1: Intake
    query: str
    company_name: str
    query_intent: str  # "call_prep" | "send_materials" | "follow_up" | "research_only"
    urgency: str  # "high" | "normal" | "low"

    # Phase 2: CRM
    crm_data: dict | None
    deal_stage: str  # "discovery" | "evaluation" | "negotiation" | "renewal"
    relationship_health: str  # "positive" | "neutral" | "at_risk"
    is_public_company: bool

    # Phase 3: Router
    routing_decisions: dict
    s3_search_query: str
    output_format: str  # "call_prep" | "send_materials" | "follow_up" | "research_only"
    tone_strategy: str  # "confident" | "professional" | "empathetic"
    skip_review: bool

    # Phase 4: Parallel research
    web_research: dict | None
    sec_filings: dict | None
    internal_docs: list[dict] | None

    # Phase 5: Synthesis
    company_brief: str
    talking_points: list[str]
    risks: list[str]
    opportunities: list[str]
    confidence_score: float
    requery_count: int  # max 2

    # Phase 6-7: Output + Human Loop
    generated_output: str
    human_feedback: str | None
    feedback_count: int  # max 2
    final_output: dict
