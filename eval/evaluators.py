"""Evaluation metrics for the Sales Deal Agent.

Provides two interfaces:
1. Direct functions: routing_accuracy(state, expected) — for local/console evaluation
2. LangSmith wrappers: ls_routing_accuracy(run, example) — for LangSmith evaluate() API
"""

import json
import logging

from pydantic import BaseModel

from config.llm import get_llm

logger = logging.getLogger(__name__)


def routing_accuracy(state: dict, expected: dict) -> dict:
    """Check if routing decisions match expected values."""
    routing = state.get("routing_decisions", {})
    checks = {
        "edgar_enabled": routing.get("edgar_enabled") == expected.get("edgar_enabled"),
        "output_format": state.get("output_format") == expected.get("output_format"),
        "tone_strategy": state.get("tone_strategy") == expected.get("tone_strategy"),
        "deal_stage_match": state.get("deal_stage") == expected.get("deal_stage"),
    }
    score = sum(checks.values()) / len(checks) if checks else 0.0
    return {"score": score, "details": checks}


def tool_selection_correctness(state: dict, expected: dict) -> dict:
    """Verify correct tools were called based on routing."""
    checks = {}

    # SEC EDGAR should only be called for public companies
    has_filings = state.get("sec_filings") is not None and state.get("sec_filings") != {}
    edgar_expected = expected.get("edgar_enabled", False)
    checks["edgar_correct"] = has_filings == edgar_expected or (not edgar_expected and not has_filings)

    # Web research should always be present
    checks["web_research_present"] = state.get("web_research") is not None

    # Internal docs should always be present
    checks["internal_docs_present"] = state.get("internal_docs") is not None

    score = sum(checks.values()) / len(checks) if checks else 0.0
    return {"score": score, "details": checks}


class JudgeResult(BaseModel):
    score: float
    feedback: str


def output_quality_judge(state: dict) -> dict:
    """Use LLM-as-judge to evaluate output quality."""
    output = state.get("generated_output", "") or state.get("final_output", {}).get("content", "")
    if not output:
        return {"score": 0.0, "feedback": "No output generated"}

    company = state.get("company_name", "Unknown")
    output_format = state.get("output_format", "unknown")

    try:
        llm = get_llm(temperature=0.0)
        judge_prompt = f"""Rate the quality of this sales output for {company} (format: {output_format}).

Score from 0.0 to 1.0 on:
1. Relevance: Does it address the customer's situation?
2. Completeness: Does it cover all expected sections?
3. Actionability: Can the sales rep use this immediately?
4. Professionalism: Is the tone appropriate?

Output to evaluate:
{output[:3000]}"""

        result = llm.with_structured_output(JudgeResult).invoke(judge_prompt)
        return {"score": result.score, "feedback": result.feedback}
    except Exception as e:
        logger.error(f"LLM judge failed: {e}")
        return {"score": 0.5, "feedback": f"Judge error: {e}"}


def confidence_calibration(state: dict) -> dict:
    """Check if confidence score is reasonable given data availability."""
    score = state.get("confidence_score", 0.0)
    has_web = bool(state.get("web_research"))
    has_docs = bool(state.get("internal_docs"))
    has_crm = bool(state.get("crm_data")) and "error" not in state.get("crm_data", {})

    data_sources = sum([has_web, has_docs, has_crm])

    # Confidence should roughly correlate with data availability
    if data_sources >= 3 and score >= 0.6:
        calibration = "well_calibrated"
    elif data_sources <= 1 and score <= 0.5:
        calibration = "well_calibrated"
    elif data_sources >= 2 and score >= 0.4:
        calibration = "acceptable"
    else:
        calibration = "miscalibrated"

    return {
        "score": 1.0 if calibration == "well_calibrated" else 0.5 if calibration == "acceptable" else 0.0,
        "confidence_score": score,
        "data_sources_available": data_sources,
        "calibration": calibration,
    }


# ---------------------------------------------------------------------------
# LangSmith-compatible evaluator wrappers
# These adapt the (state, expected) functions above to the (run, example)
# signature required by langsmith.evaluation.evaluate().
# ---------------------------------------------------------------------------

def ls_routing_accuracy(run, example) -> dict:
    """LangSmith evaluator: routing accuracy."""
    state = run.outputs or {}
    expected = (example.outputs or {}).get("expected", {})
    result = routing_accuracy(state, expected)
    return {"key": "routing_accuracy", "score": result["score"], "comment": json.dumps(result["details"])}


def ls_tool_selection(run, example) -> dict:
    """LangSmith evaluator: tool selection correctness."""
    state = run.outputs or {}
    expected = (example.outputs or {}).get("expected", {})
    result = tool_selection_correctness(state, expected)
    return {"key": "tool_selection", "score": result["score"], "comment": json.dumps(result["details"])}


def ls_output_quality(run, example) -> dict:
    """LangSmith evaluator: LLM-as-judge output quality."""
    state = run.outputs or {}
    result = output_quality_judge(state)
    return {"key": "output_quality", "score": result.get("score", 0.0), "comment": result.get("feedback", "")}


def ls_confidence_calibration(run, example) -> dict:
    """LangSmith evaluator: confidence score calibration."""
    state = run.outputs or {}
    result = confidence_calibration(state)
    return {"key": "confidence_calibration", "score": result["score"], "comment": result["calibration"]}
