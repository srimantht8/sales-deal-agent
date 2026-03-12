"""Phase 1: Query intake — parse user query with structured LLM output."""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from pydantic import BaseModel, Field

from agent.state import AgentState
from config.llm import get_llm

logger = logging.getLogger(__name__)

INTAKE_SYSTEM_PROMPT = """You are a sales assistant analyzing incoming queries from account executives.
Extract the following from the user's query:

1. company_name: The company being discussed. Extract the full company name.
2. query_intent: What the user wants to do:
   - "call_prep": Preparing for a call or meeting
   - "send_materials": Needs to send documents or materials to the customer
   - "follow_up": Writing a follow-up email after a meeting or interaction
   - "research_only": Just wants research/intel on the company, no specific action
3. urgency: How urgent is this:
   - "high": Mentions "ASAP", "urgent", "right now", "immediately", or needs something within the hour
   - "normal": Standard request without extreme time pressure, including next-day meetings ("tomorrow") or upcoming events
   - "low": Exploratory or background research with no time pressure

Be precise. If unsure about urgency, default to "normal"."""


class ParsedQuery(BaseModel):
    company_name: str = Field(description="The company name mentioned in the query")
    query_intent: Literal["call_prep", "send_materials", "follow_up", "research_only"] = Field(
        description="The user's intent: call_prep, send_materials, follow_up, or research_only"
    )
    urgency: Literal["high", "normal", "low"] = Field(
        description="How urgent the request is: high, normal, or low"
    )


@traceable(name="intake", run_type="chain")
def intake_node(state: AgentState) -> dict:
    """Parse the user query to extract company name, intent, and urgency."""
    logger.info(f"Intake: parsing query: {state['query'][:100]}...")

    llm = get_llm()
    structured_llm = llm.with_structured_output(ParsedQuery)

    result = structured_llm.invoke([
        SystemMessage(content=INTAKE_SYSTEM_PROMPT),
        HumanMessage(content=state["query"]),
    ])

    logger.info(f"Intake: company={result.company_name}, intent={result.query_intent}, urgency={result.urgency}")

    return {
        "company_name": result.company_name,
        "query_intent": result.query_intent,
        "urgency": result.urgency,
    }
