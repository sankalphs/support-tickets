"""Prompt templates for all LLM calls."""

SYNTHETIC_QUESTION_PROMPT = """You are simulating real support tickets. Read the following support article
and generate exactly 5 realistic support tickets, questions, or complaints
that a user would submit which this article would answer.

Vary the tone: some polite, some frustrated, some confused.
Include edge cases and multi-issue tickets.

Article: {article_content}
Company: {company}
Category: {product_area}

Return ONLY a JSON object: {{"questions": ["q1", "q2", "q3", "q4", "q5"]}}
"""

QUERY_EXPANSION_PROMPT = """You are a support search system. Rewrite the following messy support ticket
into 2-3 clean, specific search queries that would match help center articles.
Expand abbreviations, fix spelling, and extract key concepts.

Ticket: {ticket_text}
Subject: {subject}

Return ONLY a JSON object: {{"queries": ["clean query 1", "clean query 2", "clean query 3"]}}
"""

COMPANY_DETECTION_PROMPT = """Classify which company this support ticket is about.

Options: HackerRank, Claude, Visa, None

HackerRank: coding assessments, interviews, tests, recruiters, candidates,
  screen, skillup, chakra, codepair, proctor, plagiarism
Claude: AI assistant, Claude Pro/Team/Enterprise, API, conversations,
  Anthropic, artifacts, projects, sonnet, opus
Visa: credit/debit cards, transactions, merchants, disputes, fraud,
  ATM, PIN, lost/stolen card, bank, issuer
None: unrelated to all three companies

Ticket: {ticket_text}
Subject: {subject}

Return ONLY a JSON object: {{"company": "HackerRank|Claude|Visa|None", "confidence": 0.0-1.0}}
"""

CLASSIFICATION_PROMPT = """You are a support triage agent. Classify the following support ticket.

TICKET:
Subject: {subject}
Issue: {issue}
Company: {company}

RELEVANT SUPPORT ARTICLES:
{retrieved_articles}

Classify into:
- status: "replied" (if you can answer from the articles) or "escalated"
  (if sensitive, requires human action, or articles don't cover it)
- product_area: the most specific category from the articles, or "out_of_scope"
- request_type: "product_issue" | "feature_request" | "bug" | "invalid"

Rules:
- Billing disputes, fraud, account takeover -> always "escalated"
- Score disputes, refund demands -> always "escalated"
- Irrelevant/offensive content -> "invalid" + "replied"
- If articles don't contain enough info -> "escalated"
- If the ticket is about a topic completely outside HackerRank, Claude, and Visa,
  set product_area to "out_of_scope" and status to "replied"

Return ONLY a JSON object:
{{"status": "...", "product_area": "...", "request_type": "..."}}
"""

RESPONSE_GENERATION_PROMPT = """You are a helpful support agent. Generate a response to the user's ticket
using ONLY the provided support documentation.

TICKET:
Subject: {subject}
Issue: {issue}

CLASSIFICATION:
Status: {status}
Product Area: {product_area}
Request Type: {request_type}

RELEVANT DOCUMENTATION:
{full_parent_documents}

Rules:
1. Use ONLY information from the provided documentation
2. If the documentation does not contain enough information to fully answer,
   say so explicitly: "Based on our available documentation, ..."
3. Do NOT invent policies, procedures, or contact information
4. Reference the source article title when possible (e.g., "According to our
   [Article Title] guide...")
5. Be concise, professional, and empathetic
6. For out-of-scope requests, politely explain the limitation

Return ONLY a JSON object:
{{"response": "...", "justification": "Brief explanation citing source articles"}}
"""

ESCALATION_PROMPT = """Generate a brief, professional escalation message for this support ticket.

TICKET: {ticket_text}
REASON: {escalation_reason}

The message should:
1. Acknowledge the user's concern
2. Explain that their case needs specialized attention
3. Not make any promises or provide specific policies
4. Be empathetic and professional

Return ONLY a JSON object:
{{"response": "...", "justification": "Brief explanation of why escalation is needed"}}
"""
