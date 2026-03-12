"""Streamlit UI for the Sales Deal Acceleration Agent."""

import json
import uuid
import logging

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

logging.basicConfig(level=logging.INFO)

from agent.graph import create_agent

st.set_page_config(page_title="Sales Deal Acceleration Agent", page_icon="📊", layout="wide")
st.title("📊 Sales Deal Acceleration Agent")
st.caption("LangGraph multi-system workflow orchestration for enterprise sales")

# Initialize session state
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
if "agent" not in st.session_state:
    st.session_state.agent = create_agent(st.session_state.checkpointer)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "awaiting_review" not in st.session_state:
    st.session_state.awaiting_review = False

agent = st.session_state.agent
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Sidebar
with st.sidebar:
    st.header("Configuration")
    from config.settings import settings
    st.text(f"LLM Provider: {settings.llm_provider}")
    st.text(f"Search Provider: {settings.search_provider}")
    st.text(f"Doc Source: {settings.document_source}")

    if st.button("New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.graph_state = None
        st.session_state.awaiting_review = False
        st.rerun()

    st.divider()
    st.markdown("**Sample queries:**")
    st.markdown("- Prepare me for tomorrow's call with Snowflake")
    st.markdown("- Draft a follow-up email to Meridian Health after last week's tough meeting")
    st.markdown("- What should I send to NovaCrest Financial for our pricing discussion?")
    st.markdown("- Research AuroraStack Technologies for our upcoming renewal review")


def display_state(state: dict):
    """Display agent state in expandable sections."""
    col1, col2 = st.columns(2)

    with col1:
        if state.get("company_name"):
            with st.expander("📋 Parsed Query", expanded=False):
                st.json({
                    "company_name": state.get("company_name"),
                    "query_intent": state.get("query_intent"),
                    "urgency": state.get("urgency"),
                })

        if state.get("crm_data") and "error" not in state.get("crm_data", {}):
            with st.expander("🏢 CRM Data", expanded=False):
                crm = state["crm_data"]
                st.markdown(f"**Company:** {crm.get('company_name')}")
                st.markdown(f"**Deal Stage:** {crm.get('deal_stage')}")
                st.markdown(f"**Relationship:** {crm.get('relationship_health')}")
                st.markdown(f"**Public:** {crm.get('is_public_company')}")
                st.markdown(f"**Deal Value:** ${crm.get('deal_value', 0):,}")
                if crm.get("contacts"):
                    st.markdown("**Contacts:**")
                    for c in crm["contacts"]:
                        st.markdown(f"- {c['name']} ({c['title']})")

        if state.get("routing_decisions"):
            with st.expander("🔀 Routing Decisions", expanded=False):
                st.json(state["routing_decisions"])

    with col2:
        if state.get("web_research"):
            with st.expander("🌐 Web Research", expanded=False):
                web = state["web_research"]
                if isinstance(web, dict):
                    st.markdown(f"**Source:** {web.get('source', 'unknown')}")
                    results = web.get("results", [])
                    if isinstance(results, list):
                        for r in results[:5]:
                            if isinstance(r, dict):
                                st.markdown(f"- {r.get('title', r.get('content', '')[:80])}")
                    else:
                        st.text(str(results)[:500])
                else:
                    st.text(str(web)[:500])

        if state.get("sec_filings"):
            with st.expander("📄 SEC Filings", expanded=False):
                sec = state["sec_filings"]
                if isinstance(sec, dict):
                    st.markdown(f"**Found:** {sec.get('filing_count', 0)} filings")
                    filings = sec.get("filings", [])
                    if isinstance(filings, list):
                        for f in filings[:5]:
                            if isinstance(f, dict):
                                st.markdown(f"- {f.get('filing_type', '?')} ({f.get('date', '?')})")
                else:
                    st.text(str(sec)[:500])

        if state.get("internal_docs"):
            with st.expander("📚 Internal Documents", expanded=False):
                docs = state.get("internal_docs", [])
                if isinstance(docs, list):
                    for doc in docs[:5]:
                        if isinstance(doc, dict) and "error" not in doc:
                            st.markdown(f"- **{doc.get('source', '?')}** (stage: {doc.get('deal_stage', '?')}, score: {doc.get('similarity_score', '?')})")
                else:
                    st.text(str(docs)[:500])

    # Synthesis results
    if state.get("company_brief"):
        with st.expander("🔍 Synthesis", expanded=False):
            st.markdown(state["company_brief"])
            st.markdown(f"**Confidence Score:** {state.get('confidence_score', 0):.0%}")
            st.markdown(f"**Re-queries:** {state.get('requery_count', 0)}")
            if state.get("talking_points"):
                st.markdown("**Talking Points:**")
                tps = state["talking_points"]
                if isinstance(tps, list):
                    for tp in tps:
                        st.markdown(f"- {tp}")
                else:
                    st.text(str(tps)[:500])


# Main interaction
if not st.session_state.awaiting_review:
    query = st.chat_input("Ask about a customer...")
    if query:
        with st.spinner("Running agent pipeline..."):
            try:
                agent.invoke({"query": query}, config)

                # Get full state from checkpoint (invoke result is incomplete on interrupt)
                graph_snapshot = agent.get_state(config)
                st.session_state.graph_state = dict(graph_snapshot.values)

                if graph_snapshot.next:
                    st.session_state.awaiting_review = True
                    st.rerun()

            except Exception as e:
                st.error(f"Agent error: {e}")
                import traceback
                st.code(traceback.format_exc())

# Display current state
state = st.session_state.graph_state
if state:
    if state.get("query"):
        st.markdown(f"> **Query:** {state['query']}")
        st.divider()
    display_state(state)

    # Show generated output or final output
    if state.get("final_output"):
        st.success("✅ Output approved")
        st.markdown("---")
        final = state["final_output"]
        st.markdown(final.get("content", ""))
        st.caption(f"Status: {final.get('status', '')} | Confidence: {final.get('metadata', {}).get('confidence', 'N/A')}")

    elif st.session_state.awaiting_review and state.get("generated_output"):
        st.warning("📝 Review Required")
        st.markdown("---")
        st.markdown(state["generated_output"])

        st.markdown("---")
        col_a, col_b = st.columns([1, 3])

        with col_a:
            if st.button("✅ Approve", type="primary"):
                with st.spinner("Finalizing..."):
                    agent.invoke(Command(resume="approve"), config)
                    graph_snapshot = agent.get_state(config)
                    st.session_state.graph_state = dict(graph_snapshot.values)
                    st.session_state.awaiting_review = False
                    st.rerun()

        with col_b:
            feedback = st.text_input("Provide feedback for revision:", key="feedback_input")
            if st.button("🔄 Revise") and feedback:
                with st.spinner("Regenerating with feedback..."):
                    agent.invoke(Command(resume=feedback), config)
                    graph_snapshot = agent.get_state(config)
                    st.session_state.graph_state = dict(graph_snapshot.values)
                    st.session_state.awaiting_review = bool(graph_snapshot.next)
                    st.rerun()

    elif state.get("generated_output"):
        st.markdown("---")
        st.markdown(state["generated_output"])
