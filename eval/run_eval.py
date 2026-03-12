"""Evaluation runner — runs test queries via LangSmith evaluate() or locally.

Usage:
    python eval/run_eval.py           # LangSmith mode (pushes results to dashboard)
    python eval/run_eval.py --local   # Console-only mode (no LangSmith API needed)
"""

import logging
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from agent.graph import create_agent
from eval.evaluators import (
    confidence_calibration,
    ls_confidence_calibration,
    ls_output_quality,
    ls_routing_accuracy,
    ls_tool_selection,
    output_quality_judge,
    routing_accuracy,
    tool_selection_correctness,
)
from eval.test_queries import TEST_QUERIES

# Shared agent instance (created once, reused across evaluations)
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


def agent_target(inputs: dict) -> dict:
    """Target function for LangSmith evaluate(). Runs the full agent pipeline."""
    agent = _get_agent()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    state = agent.invoke({"query": inputs["query"]}, config)

    # Auto-approve if we hit human review interrupt
    snapshot = agent.get_state(config)
    if snapshot.next:
        from langgraph.types import Command
        state = agent.invoke(Command(resume="approve"), config)

    return state


def _ensure_dataset(client, dataset_name: str):
    """Create or reuse a LangSmith dataset from TEST_QUERIES."""
    if client.has_dataset(dataset_name=dataset_name):
        logger.info(f"Using existing dataset: {dataset_name}")
        return client.read_dataset(dataset_name=dataset_name)

    logger.info(f"Creating dataset: {dataset_name}")
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Test queries for the Sales Deal Acceleration Agent covering all routing paths.",
    )
    for test in TEST_QUERIES:
        client.create_example(
            inputs={"query": test["query"]},
            outputs={"expected": test["expected"]},
            dataset_name=dataset_name,
        )
    return dataset


def run_langsmith_evaluation():
    """Run evaluation via LangSmith evaluate() — results appear in the dashboard."""
    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()
    dataset_name = "sales-deal-agent-eval"

    _ensure_dataset(client, dataset_name)

    print("\n" + "=" * 80)
    print("SALES DEAL ACCELERATION AGENT — LANGSMITH EVALUATION")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print("Results will be pushed to LangSmith dashboard.\n")

    results = evaluate(
        agent_target,
        data=dataset_name,
        evaluators=[
            ls_routing_accuracy,
            ls_tool_selection,
            ls_output_quality,
            ls_confidence_calibration,
        ],
        experiment_prefix="sales-agent",
        description="Routing accuracy, tool selection, output quality, and confidence calibration",
        max_concurrency=1,
    )

    # Print summary from results
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("Results have been uploaded to LangSmith.")
    print(f"Experiment: {results.experiment_name}")
    print("View at: https://smith.langchain.com")
    print("=" * 80)


def run_local_evaluation():
    """Run evaluation locally (console-only, no LangSmith API required)."""
    agent = _get_agent()
    results = []

    print("\n" + "=" * 80)
    print("SALES DEAL ACCELERATION AGENT — LOCAL EVALUATION")
    print("=" * 80)

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        expected = test["expected"]
        print(f"\n--- Test {i + 1}/{len(TEST_QUERIES)} ---")
        print(f"Query: {query[:80]}...")
        print(f"Expected: {expected['company_name']} | {expected['deal_stage']} | {expected['output_format']}")

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        start_time = time.time()
        try:
            state = agent.invoke({"query": query}, config)

            # If we hit an interrupt, auto-approve
            snapshot = agent.get_state(config)
            if snapshot.next:
                from langgraph.types import Command
                state = agent.invoke(Command(resume="approve"), config)

            elapsed = time.time() - start_time

            # Run evaluators
            routing = routing_accuracy(state, expected)
            tools = tool_selection_correctness(state, expected)
            confidence = confidence_calibration(state)
            quality = output_quality_judge(state)

            result = {
                "query": query,
                "company": expected["company_name"],
                "routing_accuracy": routing,
                "tool_selection": tools,
                "confidence_calibration": confidence,
                "output_quality": quality,
                "latency_seconds": round(elapsed, 2),
                "success": True,
            }

            print(f"  ✓ Routing accuracy: {routing['score']:.0%}")
            print(f"  ✓ Tool selection: {tools['score']:.0%}")
            print(f"  ✓ Output quality: {quality['score']:.0%}")
            print(f"  ✓ Confidence calibration: {confidence['calibration']}")
            print(f"  ✓ Latency: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "query": query,
                "company": expected["company_name"],
                "error": str(e),
                "latency_seconds": round(elapsed, 2),
                "success": False,
            }
            print(f"  ✗ Error: {e}")

        results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total: {len(results)} | Passed: {len(successful)} | Failed: {len(failed)}")

    if successful:
        avg_routing = sum(r["routing_accuracy"]["score"] for r in successful) / len(successful)
        avg_tools = sum(r["tool_selection"]["score"] for r in successful) / len(successful)
        avg_latency = sum(r["latency_seconds"] for r in successful) / len(successful)
        print(f"Avg routing accuracy: {avg_routing:.0%}")
        print(f"Avg tool selection: {avg_tools:.0%}")
        print(f"Avg latency: {avg_latency:.1f}s")

    if failed:
        print("\nFailed queries:")
        for r in failed:
            print(f"  - {r['company']}: {r['error']}")

    print("=" * 80)


if __name__ == "__main__":
    if "--local" in sys.argv:
        run_local_evaluation()
    else:
        run_langsmith_evaluation()
