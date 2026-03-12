"""Phase 7: Human-in-the-loop review — interrupt for approval or feedback."""

import logging

from langgraph.types import interrupt

from agent.state import AgentState

logger = logging.getLogger(__name__)


def _build_final_output(state: AgentState, status: str, **extra_metadata) -> dict:
    """Build a consistently-shaped final_output dict."""
    metadata = {
        "company": state.get("company_name", ""),
        "format": state.get("output_format", ""),
        "confidence": state.get("confidence_score", 0.0),
        **extra_metadata,
    }
    return {
        "final_output": {
            "content": state.get("generated_output", ""),
            "status": status,
            "metadata": metadata,
        }
    }


def human_review_node(state: AgentState) -> dict:
    """Pause for human review unless skip_review is set (high urgency)."""

    # High urgency → auto-approve
    if state.get("skip_review", False):
        logger.info("Human review: skipped (high urgency, auto-approved)")
        return _build_final_output(state, "auto_approved", urgency="high")

    # Interrupt for human review
    logger.info("Human review: pausing for human approval")
    feedback = interrupt({
        "draft": state.get("generated_output", ""),
        "message": "Please review the draft. Reply 'approve' to accept, or provide feedback for revision.",
        "company": state.get("company_name", ""),
        "confidence_score": state.get("confidence_score", 0.0),
    })

    # Handle resume
    if feedback is None or (isinstance(feedback, str) and feedback.strip().lower() == "approve"):
        logger.info("Human review: approved")
        return _build_final_output(state, "human_approved")

    # Feedback provided — check max cycles
    feedback_count = state.get("feedback_count", 0) + 1
    if feedback_count >= 2:
        logger.info(f"Human review: max feedback cycles ({feedback_count}) reached, auto-approving")
        return _build_final_output(state, "approved_after_max_feedback", feedback_cycles=feedback_count)

    # Route back to generate with feedback
    logger.info(f"Human review: feedback received (cycle {feedback_count}): {feedback[:100]}...")
    return {
        "human_feedback": feedback,
        "feedback_count": feedback_count,
    }
