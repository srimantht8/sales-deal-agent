"""Multi-cloud LLM provider comparison benchmark.

Runs the same queries across all configured providers (Bedrock, OpenAI, Anthropic,
Azure OpenAI, Vertex AI) and compares latency, quality, and estimated cost.

Usage:
    python scripts/provider_comparison.py
    make compare
"""

import json
import logging
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from agent.graph import create_agent
from config.llm import clear_llm_cache
from config.settings import settings
from eval.evaluators import routing_accuracy
from eval.test_queries import TEST_QUERIES

# Use 3 representative queries (discovery, negotiation, follow-up)
BENCHMARK_QUERIES = [TEST_QUERIES[0], TEST_QUERIES[2], TEST_QUERIES[1]]

# Provider configurations: (name, settings_key_to_check, display_name)
PROVIDERS = [
    ("bedrock", "aws_access_key_id", "AWS Bedrock (Sonnet)"),
    ("openai", "openai_api_key", "OpenAI (GPT-4o)"),
    ("anthropic", "anthropic_api_key", "Anthropic (Sonnet)"),
    ("azure", "azure_openai_api_key", "Azure OpenAI (GPT-4o)"),
    ("vertex", "gcp_project_id", "GCP Vertex AI (Gemini)"),
]

# Approximate cost per query (rough estimates based on published per-token pricing
# and observed ~3500 input / ~1500 output tokens per agent run)
APPROX_COST_PER_QUERY = {
    "bedrock": 0.003,
    "openai": 0.024,
    "anthropic": 0.033,
    "azure": 0.024,
    "vertex": 0.001,
}


def _provider_available(settings_key: str) -> bool:
    return bool(getattr(settings, settings_key, ""))


def _invoke_agent(agent, query: str) -> dict:
    """Invoke the agent and auto-approve if it hits human review interrupt."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    state = agent.invoke({"query": query}, config)

    snapshot = agent.get_state(config)
    if snapshot.next:
        from langgraph.types import Command
        state = agent.invoke(Command(resume="approve"), config)

    return state


def run_benchmark():
    available = [(p, key, name) for p, key, name in PROVIDERS if _provider_available(key)]
    skipped = [(p, name) for p, key, name in PROVIDERS if not _provider_available(key)]

    if not available:
        print("No providers configured. Set API keys in .env to run the comparison.")
        sys.exit(1)

    print("\n" + "=" * 90)
    print("MULTI-CLOUD LLM PROVIDER COMPARISON")
    print("=" * 90)
    print(f"Providers available: {', '.join(name for _, _, name in available)}")
    if skipped:
        print(f"Skipped (no credentials): {', '.join(name for _, name in skipped)}")
    print(f"Queries per provider: {len(BENCHMARK_QUERIES)}")
    print()

    # Save original provider to restore after benchmark
    original_provider = settings.llm_provider

    # Create agent once — nodes resolve LLM at invocation time via get_llm()
    agent = create_agent()

    all_results = {}

    for provider_id, _, display_name in available:
        print(f"\n--- {display_name} ---")

        # Switch provider
        settings.llm_provider = provider_id
        clear_llm_cache()

        provider_results = {
            "provider": display_name,
            "queries": [],
            "total_latency": 0,
            "avg_routing_accuracy": 0,
            "total_cost_estimate": 0,
        }

        for i, test in enumerate(BENCHMARK_QUERIES):
            query = test["query"]
            expected = test["expected"]
            print(f"  Query {i + 1}: {query[:60]}...")

            start = time.time()
            try:
                state = _invoke_agent(agent, query)
                elapsed = time.time() - start

                routing = routing_accuracy(state, expected)
                cost = APPROX_COST_PER_QUERY.get(provider_id, 0)

                query_result = {
                    "query": query[:60],
                    "latency": round(elapsed, 2),
                    "routing_accuracy": routing["score"],
                    "cost_estimate": cost,
                }
                provider_results["queries"].append(query_result)
                provider_results["total_latency"] += elapsed
                provider_results["total_cost_estimate"] += cost

                print(f"    Latency: {elapsed:.1f}s | Routing: {routing['score']:.0%}")

            except Exception as e:
                elapsed = time.time() - start
                print(f"    ERROR: {e} ({elapsed:.1f}s)")
                provider_results["queries"].append({
                    "query": query[:60],
                    "error": str(e),
                    "latency": round(elapsed, 2),
                })

        # Compute averages
        successful = [q for q in provider_results["queries"] if "error" not in q]
        if successful:
            provider_results["avg_routing_accuracy"] = round(
                sum(q["routing_accuracy"] for q in successful) / len(successful), 3
            )
            provider_results["total_latency"] = round(provider_results["total_latency"], 2)
            provider_results["total_cost_estimate"] = round(provider_results["total_cost_estimate"], 6)
            provider_results["successful_count"] = len(successful)

        all_results[provider_id] = provider_results

    # Restore original provider
    settings.llm_provider = original_provider
    clear_llm_cache()

    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Provider':<28} {'Avg Latency':>12} {'Routing Acc':>12} {'Est. Cost':>12}")
    print("-" * 90)

    for provider_id, _, display_name in available:
        r = all_results.get(provider_id, {})
        count = r.get("successful_count", 0)
        if count:
            avg_lat = r["total_latency"] / count
            print(
                f"{display_name:<28} {avg_lat:>10.1f}s {r['avg_routing_accuracy']:>11.0%} "
                f"~${r['total_cost_estimate']:>9.4f}"
            )
        else:
            print(f"{display_name:<28} {'FAILED':>12}")

    print("=" * 90)
    print("Note: Cost estimates are approximate based on published per-token pricing")
    print("      and observed average token usage (~3500 input / ~1500 output per query).")

    # Save results
    output_path = Path("data/provider_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_benchmark()
