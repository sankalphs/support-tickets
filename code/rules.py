"""Escalation rules and prompt injection detection."""

import re
import logging

logger = logging.getLogger(__name__)

# Prompt injection patterns - checked before any LLM call
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+",
    r"new\s+instructions:",
    r"system\s*:\s*",
    r"\[INST\]",
    r"\<\|im_start\|\>",
    r"override\s+(all\s+)?safety",
    r"reveal\s+(your\s+)?(system|prompt)",
    r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions)",
    r"repeat\s+(everything|all)\s+(above|before)",
    r"forget\s+(all\s+)?(previous|your)\s+instructions",
    r"show\s+me\s+(your|the)\s+(internal|system)",
    r"print\s+(your|the)\s+(system|prompt|instructions)",
    r"dump\s+(your|the)\s+(system|prompt|context)",
    r"translate\s+(to|into)\s+\w+\s+and\s+ignore",
]

# Hard escalation keyword patterns
ESCALATION_RULES = {
    "fraud": {
        "keywords": [
            "fraud",
            "fraudulent",
            "unauthorized transaction",
            "scam",
            "scammer",
        ],
        "reason": "Potential fraud reported - requires human review",
    },
    "identity_theft": {
        "keywords": [
            "identity stolen",
            "identity theft",
            "someone opened",
            "impersonat",
            "someone used my",
        ],
        "reason": "Identity theft concern - requires human review",
    },
    "account_takeover": {
        "keywords": [
            "account hacked",
            "account compromised",
            "someone accessed",
            "unauthorized access",
            "account taken over",
            "someone logged in",
            "hacked my account",
        ],
        "reason": "Account takeover concern - requires human review",
    },
    "billing_dispute": {
        "keywords": [
            "refund",
            "charged twice",
            "overcharged",
            "billing dispute",
            "money back",
            "give me my money",
            "want my money",
            "wrong charge",
            "unauthorized charge",
        ],
        "reason": "Billing/refund dispute - requires human review",
    },
    "sensitive_data": {
        "keywords": [
            "social security",
            "ssn",
            "passport number",
            "password reset",
            "bank account number",
        ],
        "reason": "Sensitive data exposure risk - requires human review",
    },
    "social_engineering": {
        "keywords": [
            "show me your rules",
            "internal logic",
            "internal documents",
            "reveal your",
            "show me the prompt",
            "what are your instructions",
        ],
        "reason": "Social engineering attempt detected - requires escalation",
    },
}


def check_prompt_injection(text: str) -> bool:
    """Check if text contains prompt injection patterns.

    Returns True if injection detected.
    """
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning("Prompt injection detected: %s", pattern)
            return True
    return False


def check_escalation_rules(text: str) -> tuple[bool, str]:
    """Check text against hard escalation rules.

    Returns (should_escalate, reason).
    """
    text_lower = text.lower()
    for rule_name, rule in ESCALATION_RULES.items():
        for keyword in rule["keywords"]:
            if keyword in text_lower:
                logger.info(
                    "Escalation rule triggered: %s (keyword: %s)", rule_name, keyword
                )
                return True, rule["reason"]

    return False, ""
