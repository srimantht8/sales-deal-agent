"""Web search tool — Tavily (primary) or DuckDuckGo (fallback)."""

import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to find information about a company or topic")
    max_results: int = Field(default=5, description="Maximum number of search results to return")


@tool("search_web", args_schema=WebSearchInput)
def search_web(query: str, max_results: int = 5) -> dict:
    """Search the web for current information about a company, industry trends,
    recent news, or competitive intelligence. Use specific queries for best results."""
    provider = settings.search_provider.lower()

    try:
        if provider == "tavily":
            return _search_tavily(query, max_results)
        elif provider == "duckduckgo":
            return _search_duckduckgo(query, max_results)
        else:
            logger.warning(f"Unknown search provider '{provider}', falling back to DuckDuckGo")
            return _search_duckduckgo(query, max_results)
    except Exception as e:
        logger.error(f"Search failed with {provider}: {e}")
        # Try fallback
        if provider == "tavily":
            logger.info("Falling back to DuckDuckGo")
            try:
                return _search_duckduckgo(query, max_results)
            except Exception as e2:
                return {"error": f"All search providers failed. Tavily: {e}, DuckDuckGo: {e2}", "results": []}
        return {"error": str(e), "results": []}


def _search_tavily(query: str, max_results: int) -> dict:
    from langchain_tavily import TavilySearch

    search = TavilySearch(max_results=max_results)
    results = search.invoke(query)

    # TavilySearch may return a string, a list, or a dict with nested "results"
    if isinstance(results, str):
        return {"source": "tavily", "results": [{"content": results}]}
    if isinstance(results, dict):
        return {"source": "tavily", "results": results.get("results", [results])}

    return {"source": "tavily", "results": results}


def _search_duckduckgo(query: str, max_results: int) -> dict:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        raw_results = list(ddgs.text(query, max_results=max_results))

    results = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")}
        for r in raw_results
    ]
    return {"source": "duckduckgo", "results": results}
