"""LangChain tools for CRM, web search, SEC EDGAR, and document search."""

from tools.crm import crm_lookup
from tools.web_search import search_web
from tools.sec_edgar import fetch_sec_filings
from tools.document_search import search_documents
