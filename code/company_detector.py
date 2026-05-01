"""Company detection: keyword-first with LLM fallback."""

import json
import logging
import re

from openai import OpenAI

from config import COMPANY_KEYWORDS, MIMO_API_KEY, MIMO_BASE_URL, MIMO_MODEL
from prompts import COMPANY_DETECTION_PROMPT
from rules import check_prompt_injection

logger = logging.getLogger(__name__)


def detect_company_keywords(issue: str, subject: str) -> tuple[str | None, float]:
    """Fast keyword-based company detection.

    Returns (company, confidence) or (None, 0.0) if uncertain.
    """
    text = f"{subject} {issue}".lower()
    scores = {}

    for company, keywords in COMPANY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[company] = score / len(keywords)

    if not scores:
        return None, 0.0

    best_company = max(scores, key=scores.get)
    best_score = scores[best_company]

    # Accept if score >= 0.05 OR at least 1 keyword matched
    # (keyword lists vary in size, so absolute count matters too)
    if best_score >= 0.05 or (best_company and scores.get(best_company, 0) > 0):
        return best_company, best_score

    return None, 0.0


def detect_company_llm(issue: str, subject: str) -> tuple[str, float]:
    """LLM-based company detection for uncertain cases.

    Returns (company, confidence).
    """
    # Check for prompt injection first
    if check_prompt_injection(f"{subject} {issue}"):
        return "None", 0.5

    try:
        client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
        prompt = COMPANY_DETECTION_PROMPT.format(
            ticket_text=issue, subject=subject or "N/A"
        )

        response = client.chat.completions.create(
            model=MIMO_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )

        content = response.choices[0].message.content.strip()
        # Extract JSON from response
        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            result = json.loads(json_match.group())
            company = result.get("company", "None")
            confidence = float(result.get("confidence", 0.5))
            return company, confidence

    except Exception as e:
        logger.error("LLM company detection failed: %s", e)

    return "None", 0.0


def detect_company(issue: str, subject: str = "") -> tuple[str, float]:
    """Detect company using keyword-first, LLM-fallback strategy.

    Returns (company, confidence).
    """
    # Step 1: Try keyword detection
    company, confidence = detect_company_keywords(issue, subject)

    if company and confidence >= 0.3:
        logger.info("Company detected via keywords: %s (%.2f)", company, confidence)
        return company, confidence

    # Step 2: LLM fallback for uncertain cases
    logger.info("Keyword confidence low (%.2f), using LLM detection", confidence)
    company, confidence = detect_company_llm(issue, subject)
    logger.info("Company detected via LLM: %s (%.2f)", company, confidence)
    return company, confidence
