"""LangGraph StateGraph definition and compilation for the Sales Deal Agent."""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes.crm_lookup import crm_lookup_node
from agent.nodes.generate import generate_node
from agent.nodes.human_review import human_review_node
from agent.nodes.intake import intake_node
from agent.nodes.research import run_s3_search, run_sec_edgar, run_web_research
from agent.nodes.router import route_to_research, router_node
from agent.nodes.synthesis import synthesis_node
from agent.state import AgentState

logger = logging.getLogger(__name__)


def route_after_review(state: AgentState) -> str:
    """After human review: END if approved, back to generate if feedback given."""
    if state.get("final_output"):
        return END
    return "generate"


def build_graph() -> StateGraph:
    """Construct the Sales Deal Acceleration StateGraph."""
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("intake", intake_node)
    graph.add_node("crm_lookup", crm_lookup_node)
    graph.add_node("router", router_node)
    graph.add_node("run_web_research", run_web_research)
    graph.add_node("run_sec_edgar", run_sec_edgar)
    graph.add_node("run_s3_search", run_s3_search)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("generate", generate_node)
    graph.add_node("human_review", human_review_node)

    # Sequential: START → intake → crm_lookup → router
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "crm_lookup")
    graph.add_edge("crm_lookup", "router")

    # Parallel fan-out from router, converge at synthesis
    graph.add_conditional_edges("router", route_to_research)
    graph.add_edge("run_web_research", "synthesis")
    graph.add_edge("run_sec_edgar", "synthesis")
    graph.add_edge("run_s3_search", "synthesis")

    # synthesis → generate → human_review
    graph.add_edge("synthesis", "generate")
    graph.add_edge("generate", "human_review")

    # Conditional: human_review → END (approved) or → generate (feedback)
    graph.add_conditional_edges("human_review", route_after_review)

    return graph


def create_agent(checkpointer=None):
    """Build and compile the agent graph with a checkpointer for interrupt/resume."""
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = build_graph()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Agent graph compiled successfully")
    return compiled
