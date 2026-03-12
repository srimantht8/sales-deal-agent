"""CRM lookup tool — searches mock CRM data by company name."""

import json
import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)

# Cache CRM data at module level (static mock data, no reason to re-read)
_accounts_cache: list[dict] | None = None


def _load_accounts() -> list[dict]:
    global _accounts_cache
    if _accounts_cache is not None:
        return _accounts_cache
    crm_path = settings.project_root / settings.crm_data_path
    with open(crm_path) as f:
        _accounts_cache = json.load(f)
    return _accounts_cache


class CRMInput(BaseModel):
    company_name: str = Field(description="The company name to look up in the CRM system")


@tool("crm_lookup", args_schema=CRMInput)
def crm_lookup(company_name: str) -> dict:
    """Look up a company in the CRM system. Returns deal stage, relationship health,
    contacts, interaction history, and whether the company is publicly traded."""
    try:
        accounts = _load_accounts()
    except FileNotFoundError:
        return {"error": f"CRM data file not found at {settings.crm_data_path}"}

    # Case-insensitive partial match
    query = company_name.lower().strip()
    for account in accounts:
        name = account["company_name"].lower()
        if query in name or name in query:
            logger.info(f"CRM match found: {account['company_name']}")
            return account

    # No match
    available = [a["company_name"] for a in accounts]
    return {
        "error": f"No CRM record found for '{company_name}'",
        "available_companies": available,
    }
