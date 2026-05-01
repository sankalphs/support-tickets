# Support Triage Agent

AI-powered support ticket triage across HackerRank, Claude, and Visa ecosystems.

Built for the **HackerRank Orchestrate** 24-hour hackathon (May 1-2, 2026).

---

## Prerequisites

- Python 3.10+
- NVIDIA NIM API key (for embeddings)
- MiMo v2.5 API key (for LLM)

---

## Setup

```bash
cd code
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the **project root** (one level above `code/`):

```bash
cp ../.env.example ../.env
```

Edit `.env` with your API keys:

```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
MIMO_API_KEY=tp-xxxxxxxxxxxx
```

---

## Usage

All commands are run from the `code/` directory.

### Phase 1: Index the corpus (run once, ~90 minutes)

```bash
python main.py --index
```

This will:
1. Load all 774 support articles from `data/` (HackerRank: 438, Claude: 322, Visa: 14)
2. Generate 3-5 synthetic support questions per article via MiMo v2.5 (~3650 total)
3. Embed all questions with NVIDIA nv-embedqa-e5-v5 (rate-limited to 40 req/min)
4. Store embeddings in ChromaDB at `code/.db/chroma/`
5. Store parent documents in `code/.db/parent_store.json`

Checkpoint files are saved incrementally, so if the process is interrupted, re-running will resume from where it left off.

### Phase 2: Triage tickets (~15 minutes for 29 tickets)

```bash
python main.py --triage
```

This will:
1. Read tickets from `support_tickets/support_tickets.csv`
2. Process each ticket through the LangGraph agent pipeline
3. Write results incrementally to `support_tickets/output.csv`

### Run both phases

```bash
python main.py --index --triage
```

---

## Output Format

The output CSV (`support_tickets/output.csv`) contains:

| Column | Description |
|--------|-------------|
| `status` | `replied` or `escalated` |
| `product_area` | Most relevant support category |
| `response` | User-facing answer grounded in the corpus |
| `justification` | Concise explanation citing source articles |
| `request_type` | `product_issue`, `feature_request`, `bug`, or `invalid` |

---

## Architecture

```
Input CSV
  │
  ▼
┌─────────────────┐
│ detect_company   │  Keyword-first, LLM fallback
└────────┬────────┘
         ▼
┌─────────────────┐
│ expand_queries   │  LLM rewrites messy ticket → 2-3 clean queries
└────────┬────────┘
         ▼
┌─────────────────┐
│ retrieve         │  BM25 + Vector search with RRF fusion
└────────┬────────┘
         ▼
┌─────────────────┐
│ check_confidence │  Auto-escalate if RRF score < threshold
└────────┬────────┘
         ▼
┌─────────────────┐
│ check_rules      │  Hard rules: fraud, billing, identity theft, injection
└────────┬────────┘
    ┌────┴────┐
    ▼         ▼
 escalated  classify ──► generate_response ──► Output CSV
```

---

## File Structure

```
code/
├── main.py                # Entry point (--index / --triage)
├── config.py              # API keys, constants, thresholds
├── corpus_loader.py       # Markdown article loading with frontmatter parsing
├── indexer.py             # Async synthetic question generation + NVIDIA embedding
├── retriever.py           # Hybrid BM25 + Vector search with RRF fusion
├── agent.py               # LangGraph graph definition
├── nodes.py               # LangGraph node implementations
├── prompts.py             # All LLM prompt templates (6 prompts)
├── rules.py               # Escalation rules + prompt injection detection
├── company_detector.py    # Keyword-first + LLM fallback company detection
├── output.py              # Incremental CSV writer
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .db/                   # Created at runtime (gitignored)
    ├── chroma/            # ChromaDB persistent storage
    ├── parent_store.json  # Full article content for retrieval
    ├── index_checkpoint.json
    └── embed_checkpoint.json
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embeddings | NVIDIA nv-embedqa-e5-v5 | High-quality, 1024-dim vectors |
| LLM | MiMo v2.5 | Reasoning model with thinking tokens |
| Vector DB | ChromaDB | Local, no server, persistent |
| Retrieval | BM25 + Vector RRF | Hybrid for keyword + semantic matching |
| Agent | LangGraph | State machine with conditional branching |
| Indexing | Async parallel | 15 concurrent LLM calls for speed |
| Escalation | Rule-based + LLM | Hard rules catch obvious cases, LLM handles nuance |

---

## Dependencies

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
chromadb>=0.5.0
openai>=1.0.0
rank-bm25>=0.2.2
pandas>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0
python-dotenv>=1.0.0
aiohttp>=3.9.0
ruff>=0.4.0
```
