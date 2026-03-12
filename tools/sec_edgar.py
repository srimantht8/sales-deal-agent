"""SEC EDGAR filing search tool — queries the EFTS full-text search API."""

import logging
from datetime import datetime, timedelta

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
USER_AGENT = "SalesDealAgent/1.0 (sales-deal-agent@example.com)"

# Reusable httpx client for connection pooling
_http_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            timeout=15.0,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
    return _http_client


class SECEdgarInput(BaseModel):
    company_name: str = Field(description="The publicly traded company name to search SEC filings for")
    filing_types: list[str] = Field(
        default=["10-K", "10-Q", "8-K"],
        description="Types of SEC filings to search for (e.g., 10-K, 10-Q, 8-K)",
    )


@tool("fetch_sec_filings", args_schema=SECEdgarInput)
def fetch_sec_filings(company_name: str, filing_types: list[str] | None = None) -> dict:
    """Fetch recent SEC filings for a publicly traded company. Returns filing type,
    date, description, and link. Only works for public companies — check CRM data first."""
    if filing_types is None:
        filing_types = ["10-K", "10-Q", "8-K"]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    forms = ",".join(filing_types)
    params = {
        "q": company_name,
        "dateRange": "custom",
        "startdt": start_date.strftime("%Y-%m-%d"),
        "enddt": end_date.strftime("%Y-%m-%d"),
        "forms": forms,
    }
    try:
        client = _get_client()
        response = client.get(EDGAR_SEARCH_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"SEC EDGAR HTTP error: {e.response.status_code}")
        return {"error": f"SEC EDGAR returned {e.response.status_code}", "filings": []}
    except Exception as e:
        logger.error(f"SEC EDGAR request failed: {e}")
        return {"error": str(e), "filings": []}

    # Parse results
    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        return {
            "company": company_name,
            "filings": [],
            "message": f"No filings found for '{company_name}' in the past year.",
        }

    # Deduplicate by form type + date (EFTS returns individual exhibits)
    seen = set()
    filings = []
    for hit in hits:
        source = hit.get("_source", {})
        form = source.get("form", source.get("root_forms", ["Unknown"])[0] if source.get("root_forms") else "Unknown")
        date = source.get("file_date", "Unknown")
        entity = source.get("display_names", [""])[0] if source.get("display_names") else company_name
        dedup_key = (form, date, entity)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        adsh = source.get("adsh", "")
        filings.append({
            "filing_type": form,
            "date": date,
            "entity": entity,
            "url": f"https://www.sec.gov/Archives/edgar/data/{source.get('ciks', [''])[0]}/{adsh.replace('-', '')}/{adsh}-index.htm" if adsh else "",
        })
        if len(filings) >= 10:
            break

    return {"company": company_name, "filing_count": len(filings), "filings": filings}
