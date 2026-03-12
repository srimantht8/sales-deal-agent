"""CLI demo — runs a single pre-baked query end-to-end without Streamlit."""

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main():
    import uuid

    from agent.graph import create_agent
    from config.settings import settings

    QUERY = "Prepare me for tomorrow's call with Snowflake — what's our relationship history and what should I send them?"

    print("\n" + "=" * 80)
    print("SALES DEAL ACCELERATION AGENT — CLI DEMO")
    print("=" * 80)
    print(f"  LLM Provider:     {settings.llm_provider} ({settings.bedrock_model_id})")
    print(f"  Document Source:   {settings.document_source} (set DOCUMENT_SOURCE=s3 for S3)")
    print(f"  Search Provider:   {settings.search_provider}")
    print("=" * 80)
    print(f"\nQuery: {QUERY}\n")

    agent = create_agent()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Run the agent
    print("Running agent pipeline...\n")
    result = agent.invoke({"query": QUERY}, config)

    # Check for interrupt (human review)
    snapshot = agent.get_state(config)
    if snapshot.next:
        print("--- HUMAN REVIEW ---")
        print("Auto-approving for demo...\n")
        from langgraph.types import Command
        result = agent.invoke(Command(resume="approve"), config)

    # Display results
    print("=" * 80)
    print("PHASE 1 — PARSED QUERY")
    print("=" * 80)
    print(f"  Company: {result.get('company_name')}")
    print(f"  Intent: {result.get('query_intent')}")
    print(f"  Urgency: {result.get('urgency')}")

    print("\n" + "=" * 80)
    print("PHASE 2 — CRM DATA")
    print("=" * 80)
    crm = result.get("crm_data", {})
    if crm and "error" not in crm:
        print(f"  Deal Stage: {crm.get('deal_stage')}")
        print(f"  Relationship: {crm.get('relationship_health')}")
        print(f"  Public: {crm.get('is_public_company')}")
        print(f"  Deal Value: ${crm.get('deal_value', 0):,}")
        print(f"  Contacts: {len(crm.get('contacts', []))}")
    else:
        print(f"  {crm}")

    print("\n" + "=" * 80)
    print("PHASE 3 — ROUTING DECISIONS")
    print("=" * 80)
    routing = result.get("routing_decisions", {})
    for k, v in routing.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("PHASE 4 — RESEARCH RESULTS")
    print("=" * 80)
    web = result.get("web_research", {})
    if isinstance(web, dict):
        print(f"  Web Research: {web.get('source', '?')} — {len(web.get('results', []))} results")
    sec = result.get("sec_filings")
    if sec and isinstance(sec, dict):
        print(f"  SEC Filings: {sec.get('filing_count', 0)} found")
    docs = result.get("internal_docs", [])
    print(f"  Internal Docs: {len(docs)} documents matched")

    print("\n" + "=" * 80)
    print("PHASE 5 — SYNTHESIS")
    print("=" * 80)
    print(f"  Confidence: {result.get('confidence_score', 0):.0%}")
    print(f"  Re-queries: {result.get('requery_count', 0)}")
    print(f"\n  Brief:\n  {result.get('company_brief', 'N/A')[:500]}")
    print(f"\n  Talking Points:")
    for tp in result.get("talking_points", []):
        print(f"    • {tp}")
    print(f"\n  Risks:")
    for r in result.get("risks", []):
        print(f"    ⚠ {r}")
    print(f"\n  Opportunities:")
    for o in result.get("opportunities", []):
        print(f"    ✦ {o}")

    print("\n" + "=" * 80)
    print("PHASE 6-7 — GENERATED OUTPUT")
    print("=" * 80)
    final = result.get("final_output", {})
    if final:
        print(f"  Status: {final.get('status', 'unknown')}")
        print(f"\n{final.get('content', 'No content')}")
    else:
        print(result.get("generated_output", "No output"))

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
