"""Phase 5: Agentic synthesis — merge research, identify gaps, re-query if needed."""

import json
import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from pydantic import BaseModel, Field, field_validator

from agent.state import AgentState
from config.llm import get_llm
from tools.document_search import search_documents
from tools.web_search import search_web

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are a senior sales analyst synthesizing research data for an account executive.

Your job:
1. Review all gathered research (CRM data, web research, SEC filings, internal documents)
2. Synthesize into a coherent company brief
3. Identify key talking points, risks, and opportunities
4. Assess data completeness and assign a confidence score (0.0 to 1.0)
5. Identify any critical gaps in information

For gaps:
- doc_gap: If CRM mentions a topic (e.g., security concerns, competitor comparison) but no internal documents address it, specify a search query to find relevant docs.
- intel_gap: If web research is thin on a key topic (e.g., recent company news, competitor activity), specify a search query.
- Set gaps to null if the data is sufficient.

Be specific with gap queries — use keywords that would match relevant content."""


class SynthesisResult(BaseModel):
    company_brief: str = Field(description="2-3 paragraph summary of the company and opportunity")
    talking_points: list[str] = Field(description="Key points to discuss with the customer")
    risks: list[str] = Field(description="Potential risks or concerns to be aware of")
    opportunities: list[str] = Field(description="Opportunities to leverage in the conversation")
    confidence_score: float = Field(description="Data completeness score from 0.0 to 1.0")
    doc_gap: Optional[str] = Field(default=None, description="Search query for missing internal docs, or null")
    intel_gap: Optional[str] = Field(default=None, description="Search query for missing web intel, or null")

    @field_validator("talking_points", "risks", "opportunities", mode="before")
    @classmethod
    def coerce_str_to_list(cls, v):
        """Bedrock sometimes returns list fields as JSON strings."""
        if isinstance(v, str):
            v = v.strip().rstrip(";")
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return [v]
        return v


@traceable(name="synthesis", run_type="chain")
def synthesis_node(state: AgentState) -> dict:
    """Synthesize all research data with optional re-querying for gaps."""
    logger.info("Synthesis: merging research data")

    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(SynthesisResult)

    # Gather research context
    context = {
        "company_name": state.get("company_name", ""),
        "deal_stage": state.get("deal_stage", ""),
        "query_intent": state.get("query_intent", ""),
        "relationship_health": state.get("relationship_health", ""),
        "crm_data": state.get("crm_data"),
        "web_research": state.get("web_research"),
        "sec_filings": state.get("sec_filings"),
        "internal_docs": state.get("internal_docs"),
    }

    prompt = f"""Analyze the following research data for {context['company_name']} and synthesize your findings.

Deal Stage: {context['deal_stage']}
Query Intent: {context['query_intent']}
Relationship Health: {context['relationship_health']}

CRM Data:
{json.dumps(context['crm_data'], indent=2, default=str)}

Web Research:
{json.dumps(context['web_research'], indent=2, default=str)}

SEC Filings:
{json.dumps(context['sec_filings'], indent=2, default=str) if context['sec_filings'] else 'Not applicable (private company)'}

Internal Documents:
{json.dumps(context['internal_docs'], indent=2, default=str)}"""

    result = structured_llm.invoke([
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    requery_count = state.get("requery_count", 0)

    # Re-query loop (max 2 iterations)
    while requery_count < 2 and (result.doc_gap or result.intel_gap):
        additional = {}

        if result.doc_gap and requery_count < 2:
            logger.info(f"Synthesis re-query: searching docs for '{result.doc_gap}'")
            additional["additional_docs"] = search_documents.invoke({"query": result.doc_gap, "k": 3})
            requery_count += 1

        if result.intel_gap and requery_count < 2:
            logger.info(f"Synthesis re-query: searching web for '{result.intel_gap}'")
            additional["additional_web"] = search_web.invoke({"query": result.intel_gap, "max_results": 3})
            requery_count += 1

        if not additional:
            break

        # Re-synthesize with original context + additional data
        refine_prompt = f"""Re-synthesize research for {context['company_name']} with additional data.

Original research data:
{prompt}

Previous synthesis:
{result.model_dump_json(indent=2)}

Additional research found:
{json.dumps(additional, indent=2, default=str)}

Update your synthesis incorporating all information. Adjust confidence score if data coverage improved."""

        result = structured_llm.invoke([
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=refine_prompt),
        ])

    logger.info(f"Synthesis complete: confidence={result.confidence_score}, requery_count={requery_count}")

    return {
        "company_brief": result.company_brief,
        "talking_points": result.talking_points,
        "risks": result.risks,
        "opportunities": result.opportunities,
        "confidence_score": result.confidence_score,
        "requery_count": requery_count,
    }
