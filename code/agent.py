"""LangGraph agent graph definition."""

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from nodes import (
    check_confidence_node,
    check_rules_node,
    classify_node,
    detect_company_node,
    expand_queries_node,
    generate_escalation_node,
    generate_response_node,
    retrieve_node,
)

logger = logging.getLogger(__name__)


class TicketState(TypedDict):
    """State schema for the triage graph."""

    issue: str
    subject: str
    company: str  # from CSV
    detected_company: str  # after detection
    expanded_queries: list[str]
    retrieved_docs: list[dict]
    retrieval_confidence: float
    should_escalate: bool
    escalation_reason: str
    classification: dict  # {status, product_area, request_type}
    response: str
    justification: str


def should_continue_after_confidence(state: dict) -> str:
    """Route after confidence check."""
    if state.get("should_escalate", False):
        return "generate_escalation"
    return "check_rules"


def should_continue_after_rules(state: dict) -> str:
    """Route after rules check."""
    if state.get("should_escalate", False):
        return "generate_escalation"
    return "classify"


def should_continue_after_classify(state: dict) -> str:
    """Route after classification."""
    classification = state.get("classification", {})
    if classification.get("status") == "escalated":
        return "generate_escalation"
    return "generate_response"


def build_graph() -> StateGraph:
    """Build the LangGraph triage graph.

    Graph flow:
    detect_company -> expand_queries -> retrieve -> check_confidence
        -> (low confidence) -> generate_escalation -> END
        -> (ok) -> check_rules
            -> (rule triggered) -> generate_escalation -> END
            -> (ok) -> classify
                -> (escalated) -> generate_escalation -> END
                -> (replied) -> generate_response -> END
    """
    graph = StateGraph(TicketState)

    # Add nodes
    graph.add_node("detect_company", detect_company_node)
    graph.add_node("expand_queries", expand_queries_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_confidence", check_confidence_node)
    graph.add_node("check_rules", check_rules_node)
    graph.add_node("classify", classify_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("generate_escalation", generate_escalation_node)

    # Set entry point
    graph.set_entry_point("detect_company")

    # Add edges
    graph.add_edge("detect_company", "expand_queries")
    graph.add_edge("expand_queries", "retrieve")
    graph.add_edge("retrieve", "check_confidence")

    # Conditional: confidence check
    graph.add_conditional_edges(
        "check_confidence",
        should_continue_after_confidence,
        {
            "generate_escalation": "generate_escalation",
            "check_rules": "check_rules",
        },
    )

    # Conditional: rules check
    graph.add_conditional_edges(
        "check_rules",
        should_continue_after_rules,
        {
            "generate_escalation": "generate_escalation",
            "classify": "classify",
        },
    )

    # Conditional: classification
    graph.add_conditional_edges(
        "classify",
        should_continue_after_classify,
        {
            "generate_escalation": "generate_escalation",
            "generate_response": "generate_response",
        },
    )

    # Terminal edges
    graph.add_edge("generate_response", END)
    graph.add_edge("generate_escalation", END)

    return graph


def create_agent():
    """Create and compile the LangGraph agent."""
    graph = build_graph()
    return graph.compile()
