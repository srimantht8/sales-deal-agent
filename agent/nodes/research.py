"""Phase 4: Parallel research nodes — web search, SEC EDGAR, document search."""

import logging

from agent.state import AgentState
from tools.document_search import search_documents
from tools.sec_edgar import fetch_sec_filings
from tools.web_search import search_web

logger = logging.getLogger(__name__)


def run_web_research(state: AgentState) -> dict:
    """Search the web for recent company news and industry intelligence."""
    company = state.get("company_name", "")
    deal_stage = state.get("deal_stage", "")
    query = f"{company} {deal_stage} recent news industry trends"

    logger.info(f"Web research: {query}")
    result = search_web.invoke({"query": query, "max_results": 5})
    return {"web_research": result}


def run_sec_edgar(state: AgentState) -> dict:
    """Fetch SEC filings for public companies."""
    company = state.get("company_name", "")
    logger.info(f"SEC EDGAR: fetching filings for {company}")

    result = fetch_sec_filings.invoke({
        "company_name": company,
        "filing_types": ["10-K", "10-Q", "8-K"],
    })
    return {"sec_filings": result}


def run_s3_search(state: AgentState) -> dict:
    """Search internal documents using FAISS vectorstore."""
    query = state.get("s3_search_query", "general overview")
    logger.info(f"Document search: {query}")

    result = search_documents.invoke({"query": query, "k": 4})
    return {"internal_docs": result}
