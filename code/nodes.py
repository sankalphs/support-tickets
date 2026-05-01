"""LangGraph node implementations."""

import json
import logging
import re

from openai import OpenAI

from config import MIMO_API_KEY, MIMO_BASE_URL, MIMO_MODEL
from company_detector import detect_company
from prompts import (
    CLASSIFICATION_PROMPT,
    ESCALATION_PROMPT,
    QUERY_EXPANSION_PROMPT,
    RESPONSE_GENERATION_PROMPT,
)
from retriever import HybridRetriever
from rules import check_escalation_rules, check_prompt_injection

logger = logging.getLogger(__name__)

# Global retriever instance (initialized once)
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Get or initialize the global retriever."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
        _retriever.initialize()
    return _retriever


def get_llm_client() -> OpenAI:
    """Get MiMo LLM client."""
    return OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)


def call_llm(prompt: str, max_tokens: int = 1000) -> str:
    """Call MiMo LLM and return response text."""
    client = get_llm_client()
    response = client.chat.completions.create(
        model=MIMO_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def extract_json(text: str) -> dict:
    """Extract JSON object from LLM response text."""
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try finding array
    arr_match = re.search(r"\[.*\]", text, re.DOTALL)
    if arr_match:
        try:
            return json.loads(arr_match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not extract JSON from: %s", text[:200])
    return {}


# ──────────────────────────────────────────────────────────────────
# Node: detect_company
# ──────────────────────────────────────────────────────────────────


def detect_company_node(state: dict) -> dict:
    """Detect company from ticket content."""
    issue = state.get("issue", "")
    subject = state.get("subject", "")
    csv_company = state.get("company", "None")

    # If company already provided and not None, use it
    if csv_company and csv_company != "None":
        logger.info("Company from CSV: %s", csv_company)
        return {**state, "detected_company": csv_company}

    # Detect from content
    company, confidence = detect_company(issue, subject)
    logger.info("Detected company: %s (confidence: %.2f)", company, confidence)
    return {**state, "detected_company": company}


# ──────────────────────────────────────────────────────────────────
# Node: expand_queries
# ──────────────────────────────────────────────────────────────────


def expand_queries_node(state: dict) -> dict:
    """Expand messy ticket into clean search queries."""
    issue = state.get("issue", "")
    subject = state.get("subject", "")

    # Check for prompt injection before LLM call
    if check_prompt_injection(f"{subject} {issue}"):
        logger.warning("Prompt injection detected in query expansion, using raw text")
        return {**state, "expanded_queries": [f"{subject} {issue}".strip()]}

    try:
        prompt = QUERY_EXPANSION_PROMPT.format(
            ticket_text=issue, subject=subject or "N/A"
        )
        response = call_llm(prompt, max_tokens=300)
        result = extract_json(response)
        queries = result.get("queries", [])

        if queries:
            logger.info("Expanded to %d queries: %s", len(queries), queries)
            return {**state, "expanded_queries": queries}

    except Exception as e:
        logger.error("Query expansion failed: %s", e)

    # Fallback: use raw text
    fallback = f"{subject} {issue}".strip()
    return {**state, "expanded_queries": [fallback]}


# ──────────────────────────────────────────────────────────────────
# Node: retrieve
# ──────────────────────────────────────────────────────────────────


def retrieve_node(state: dict) -> dict:
    """Retrieve relevant documents using hybrid BM25 + Vector RRF."""
    retriever = get_retriever()
    queries = state.get("expanded_queries", [])
    company = state.get("detected_company", "None")

    doc_ids, confidence = retriever.retrieve(queries, company)

    if not doc_ids:
        logger.warning("No relevant documents found (confidence: %.4f)", confidence)
        return {**state, "retrieved_docs": [], "retrieval_confidence": confidence}

    docs = retriever.get_parent_documents(doc_ids)
    logger.info("Retrieved %d documents, confidence: %.4f", len(docs), confidence)
    return {**state, "retrieved_docs": docs, "retrieval_confidence": confidence}


# ──────────────────────────────────────────────────────────────────
# Node: check_confidence
# ──────────────────────────────────────────────────────────────────


def check_confidence_node(state: dict) -> dict:
    """Check retrieval confidence, auto-escalate if too low."""
    confidence = state.get("retrieval_confidence", 0.0)
    docs = state.get("retrieved_docs", [])

    if not docs or confidence < 0.4:
        logger.warning("Low confidence (%.4f), auto-escalating", confidence)
        return {
            **state,
            "should_escalate": True,
            "escalation_reason": "No relevant support documentation found for this ticket.",
        }

    return {**state, "should_escalate": False}


# ──────────────────────────────────────────────────────────────────
# Node: check_rules
# ──────────────────────────────────────────────────────────────────


def check_rules_node(state: dict) -> dict:
    """Apply hard escalation rules and prompt injection check."""
    issue = state.get("issue", "")
    subject = state.get("subject", "")
    combined_text = f"{subject} {issue}"

    # Prompt injection check
    if check_prompt_injection(combined_text):
        logger.warning("Prompt injection detected, escalating")
        return {
            **state,
            "should_escalate": True,
            "escalation_reason": "Potential prompt injection detected in ticket content.",
        }

    # Hard escalation rules
    should_escalate, reason = check_escalation_rules(combined_text)
    if should_escalate:
        logger.info("Escalation rule triggered: %s", reason)
        return {**state, "should_escalate": True, "escalation_reason": reason}

    return {**state, "should_escalate": False}


# ──────────────────────────────────────────────────────────────────
# Node: classify
# ──────────────────────────────────────────────────────────────────


def classify_node(state: dict) -> dict:
    """Classify ticket using LLM."""
    issue = state.get("issue", "")
    subject = state.get("subject", "")
    company = state.get("detected_company", "None")
    docs = state.get("retrieved_docs", [])

    # Format retrieved articles for prompt
    articles_text = ""
    for i, doc in enumerate(docs[:5], 1):
        meta = doc.get("metadata", {})
        articles_text += f"\n--- Article {i}: {meta.get('title', 'Unknown')} ---\n"
        articles_text += f"Company: {meta.get('company', 'Unknown')}\n"
        articles_text += f"Category: {meta.get('product_area', 'Unknown')}\n"
        articles_text += f"{doc.get('content', '')[:2000]}\n"

    prompt = CLASSIFICATION_PROMPT.format(
        subject=subject or "N/A",
        issue=issue,
        company=company,
        retrieved_articles=articles_text,
    )

    try:
        response = call_llm(prompt, max_tokens=500)
        result = extract_json(response)

        classification = {
            "status": result.get("status", "escalated"),
            "product_area": result.get("product_area", "out_of_scope"),
            "request_type": result.get("request_type", "product_issue"),
        }

        logger.info("Classification: %s", classification)
        return {**state, "classification": classification}

    except Exception as e:
        logger.error("Classification failed: %s", e)
        return {
            **state,
            "classification": {
                "status": "escalated",
                "product_area": "out_of_scope",
                "request_type": "product_issue",
            },
        }


# ──────────────────────────────────────────────────────────────────
# Node: generate_response
# ──────────────────────────────────────────────────────────────────


def generate_response_node(state: dict) -> dict:
    """Generate grounded response using LLM."""
    issue = state.get("issue", "")
    subject = state.get("subject", "")
    classification = state.get("classification", {})
    docs = state.get("retrieved_docs", [])

    # Format full documents for response generation
    docs_text = ""
    for i, doc in enumerate(docs[:3], 1):
        meta = doc.get("metadata", {})
        docs_text += f"\n=== Document {i}: {meta.get('title', 'Unknown')} ===\n"
        docs_text += f"Source: {meta.get('source_url', 'N/A')}\n"
        docs_text += f"Category: {meta.get('product_area', 'Unknown')}\n"
        docs_text += f"{doc.get('content', '')[:3000]}\n"

    prompt = RESPONSE_GENERATION_PROMPT.format(
        subject=subject or "N/A",
        issue=issue,
        status=classification.get("status", "replied"),
        product_area=classification.get("product_area", "unknown"),
        request_type=classification.get("request_type", "product_issue"),
        full_parent_documents=docs_text,
    )

    try:
        response = call_llm(prompt, max_tokens=1000)
        result = extract_json(response)

        return {
            **state,
            "response": result.get("response", "Unable to generate response."),
            "justification": result.get(
                "justification", "Response generated from retrieved documentation."
            ),
        }

    except Exception as e:
        logger.error("Response generation failed: %s", e)
        return {
            **state,
            "response": "We apologize, but we were unable to process your request at this time. Please contact support directly.",
            "justification": f"Response generation failed: {e}",
        }


# ──────────────────────────────────────────────────────────────────
# Node: generate_escalation
# ──────────────────────────────────────────────────────────────────


def generate_escalation_node(state: dict) -> dict:
    """Generate escalation message."""
    issue = state.get("issue", "")
    subject = state.get("subject", "")
    escalation_reason = state.get("escalation_reason", "Requires human review")

    prompt = ESCALATION_PROMPT.format(
        ticket_text=f"Subject: {subject}\nIssue: {issue}",
        escalation_reason=escalation_reason,
    )

    try:
        response = call_llm(prompt, max_tokens=500)
        result = extract_json(response)

        classification = state.get("classification", {})
        return {
            **state,
            "response": result.get(
                "response",
                "Your case has been escalated to our specialized team for further review.",
            ),
            "justification": result.get("justification", escalation_reason),
            "classification": {
                **classification,
                "status": "escalated",
            },
        }

    except Exception as e:
        logger.error("Escalation message generation failed: %s", e)
        return {
            **state,
            "response": "Your concern has been forwarded to our specialized support team. They will review your case and get back to you.",
            "justification": escalation_reason,
            "classification": {
                **state.get("classification", {}),
                "status": "escalated",
            },
        }
