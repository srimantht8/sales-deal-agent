"""Test queries covering all 5 routing paths (see README Mock Data table)."""

TEST_QUERIES = [
    {
        "query": "Prepare me for tomorrow's call with Snowflake — what's our relationship history and what should I send them?",
        "expected": {
            "company_name": "Snowflake Inc",
            "query_intent": "call_prep",
            "urgency": "normal",
            "edgar_enabled": True,
            "deal_stage": "discovery",
            "tone_strategy": "professional",
            "output_format": "call_prep",
        },
    },
    {
        "query": "Draft a follow-up email to Meridian Health Systems after last week's tough meeting",
        "expected": {
            "company_name": "Meridian Health Systems",
            "query_intent": "follow_up",
            "urgency": "normal",
            "edgar_enabled": False,
            "deal_stage": "evaluation",
            "tone_strategy": "empathetic",
            "output_format": "follow_up",
        },
    },
    {
        "query": "What should I send to NovaCrest Financial for our pricing discussion?",
        "expected": {
            "company_name": "NovaCrest Financial",
            "query_intent": "send_materials",
            "urgency": "normal",
            "edgar_enabled": True,
            "deal_stage": "negotiation",
            "tone_strategy": "confident",
            "output_format": "send_materials",
        },
    },
    {
        "query": "Research AuroraStack Technologies for our upcoming renewal review",
        "expected": {
            "company_name": "AuroraStack Technologies",
            "query_intent": "research_only",
            "urgency": "normal",
            "edgar_enabled": False,
            "deal_stage": "renewal",
            "tone_strategy": "confident",
            "output_format": "research_only",
        },
    },
    {
        "query": "I need to prepare for the Vertex Logistics evaluation — they're comparing us against competitors",
        "expected": {
            "company_name": "Vertex Logistics Group",
            "query_intent": "call_prep",
            "urgency": "normal",
            "edgar_enabled": True,
            "deal_stage": "evaluation",
            "tone_strategy": "empathetic",
            "output_format": "call_prep",
        },
    },
]
