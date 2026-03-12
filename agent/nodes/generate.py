"""Phase 6: Output generation — format output based on routing decisions."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from agent.state import AgentState
from config.llm import get_llm

logger = logging.getLogger(__name__)

FORMAT_PROMPTS = {
    "call_prep": """Generate a comprehensive call preparation briefing with:
1. **Executive Summary** — 2-3 sentences on the account status and call objectives
2. **Key Talking Points** — Numbered list of topics to cover, with supporting data
3. **Risk Areas** — Issues to be aware of and how to address them
4. **Recommended Questions** — 3-5 open-ended questions to ask the customer
5. **Next Steps** — Suggested action items to propose on the call
6. **Preparation Notes** — Background context and tips for the conversation""",

    "send_materials": """Generate a materials package with:
1. **Recommended Documents** — Ranked list of documents to send, with rationale for each
2. **Cover Email Draft** — Professional email to accompany the materials
   - Subject line (prepend "TIME SENSITIVE: " if urgency is high)
   - Body with personalized context and clear call-to-action
3. **Sending Strategy** — Recommended timing and follow-up plan""",

    "follow_up": """Generate a personalized follow-up email with:
1. **Subject Line** (prepend "TIME SENSITIVE: " if urgency is high)
2. **Email Body** including:
   - Reference to the previous interaction
   - Key takeaways or action items discussed
   - Value proposition reinforcement
   - Clear next steps with proposed timeline
3. **Internal Notes** — Things to track or prepare before the next interaction""",

    "research_only": """Generate a company intelligence report with:
1. **Company Overview** — Key facts, market position, and recent developments
2. **Opportunity Assessment** — How our solution fits their needs
3. **Competitive Landscape** — Known competitors and our positioning
4. **Risk Factors** — Challenges or concerns to monitor
5. **Recommendations** — Suggested strategy and next steps""",
}

TONE_INSTRUCTIONS = {
    "confident": "Use a confident, assertive tone. Emphasize our proven track record, successful deployments, and competitive advantages. Lead with strengths.",
    "professional": "Use a professional, balanced tone. Present information objectively. Let the data speak for itself. Be thorough but not aggressive.",
    "empathetic": "Use an empathetic, understanding tone. Acknowledge any challenges or concerns. Show genuine interest in solving their problems. Be patient and supportive.",
}


@traceable(name="generate", run_type="chain")
def generate_node(state: AgentState) -> dict:
    """Generate the final output document based on routing decisions."""
    output_format = state.get("output_format", "call_prep")
    tone = state.get("tone_strategy", "professional")
    company = state.get("company_name", "Unknown")

    logger.info(f"Generate: format={output_format}, tone={tone}")

    llm = get_llm(temperature=0.3)

    format_prompt = FORMAT_PROMPTS.get(output_format, FORMAT_PROMPTS["call_prep"])
    tone_instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["professional"])

    # Build context
    confidence_caveat = ""
    if state.get("confidence_score", 1.0) < 0.6:
        confidence_caveat = "\n\nIMPORTANT: Data coverage is limited. Add appropriate caveats noting that some information may be incomplete or unverified. Recommend additional research where data is thin."

    urgency_note = ""
    if state.get("skip_review", False):
        urgency_note = "\nThis is TIME SENSITIVE. For any email subject lines, prepend 'TIME SENSITIVE: ' to the subject."

    feedback_section = ""
    if state.get("human_feedback"):
        feedback_section = f"\n\nIMPORTANT REVISION INSTRUCTIONS: The user reviewed a previous draft and provided this feedback:\n\"{state['human_feedback']}\"\n\nPlease incorporate this feedback into the output."

    prompt = f"""{format_prompt}

Tone: {tone_instruction}
{confidence_caveat}{urgency_note}{feedback_section}

Company: {company}

Company Brief:
{state.get('company_brief', 'No brief available')}

Talking Points:
{json.dumps(state.get('talking_points', []), indent=2)}

Risks:
{json.dumps(state.get('risks', []), indent=2)}

Opportunities:
{json.dumps(state.get('opportunities', []), indent=2)}

CRM Data:
{json.dumps(state.get('crm_data', {}), indent=2, default=str)}

Internal Documents Found:
{json.dumps(state.get('internal_docs', []), indent=2, default=str)}"""

    response = llm.invoke([
        SystemMessage(content="You are an expert sales enablement assistant. Generate polished, actionable sales content."),
        HumanMessage(content=prompt),
    ])

    logger.info("Generate: output created")
    return {"generated_output": response.content}
