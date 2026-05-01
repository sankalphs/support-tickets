"""Configuration constants, API keys, and allowed values."""

import os

from dotenv import load_dotenv

load_dotenv()

# API Keys
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")

# API Base URLs
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MIMO_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"

# Model IDs
MIMO_MODEL = "xiaomi/mimo-v2.5"
NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

# Rate Limits
NVIDIA_RPM = 40  # requests per minute
INDEX_CONCURRENCY = 15  # async concurrent calls for indexing

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUPPORT_DIR = os.path.join(PROJECT_ROOT, "support_tickets")
OUTPUT_CSV = os.path.join(SUPPORT_DIR, "output.csv")
INPUT_CSV = os.path.join(SUPPORT_DIR, "support_tickets.csv")
SAMPLE_CSV = os.path.join(SUPPORT_DIR, "sample_support_tickets.csv")
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".db")
CHROMA_DIR = os.path.join(DB_DIR, "chroma")
CHECKPOINT_FILE = os.path.join(DB_DIR, "index_checkpoint.json")

# Retrieval
TOP_K_VECTOR = 20
TOP_K_BM25 = 20
TOP_K_FINAL = 5
RRF_K = 60  # RRF constant
CONFIDENCE_THRESHOLD = 0.4

# Allowed Values
ALLOWED_STATUSES = ["replied", "escalated"]
ALLOWED_REQUEST_TYPES = ["product_issue", "feature_request", "bug", "invalid"]

# Output CSV Columns
OUTPUT_COLUMNS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]

# Companies
COMPANIES = ["HackerRank", "Claude", "Visa"]

# Company keywords for fast detection
COMPANY_KEYWORDS = {
    "HackerRank": [
        "assessment",
        "test",
        "recruiter",
        "candidate",
        "screen",
        "skillup",
        "chakra",
        "codepair",
        "interview",
        "hackerrank",
        "coding challenge",
        "hiring",
        "question bank",
        "proctor",
        "plagiarism",
        "time accommodation",
        "extra time",
        "test invite",
        "reinvite",
        "coding test",
        "mock interview",
        "resume",
        "certificate",
        "submission",
        "challenge",
    ],
    "Claude": [
        "claude",
        "anthropic",
        "pro plan",
        "team plan",
        "enterprise plan",
        "conversation",
        "artifact",
        "api key",
        "console",
        "max plan",
        "sonnet",
        "opus",
        "haiku",
        "lti",
        "education",
        "nonprofit",
        "government",
        "bedrock",
        "mcp",
        "connector",
        "cowork",
    ],
    "Visa": [
        "visa",
        "credit card",
        "debit card",
        "transaction",
        "merchant",
        "dispute",
        "atm",
        "pin",
        "lost card",
        "stolen card",
        "charge",
        "bank",
        "issuer",
        "card declined",
        "card stolen",
        "card lost",
        "fraud",
        "refund",
        "traveller",
        "travel",
        "cheque",
    ],
}
