# Support Triage Agent

AI-powered support ticket triage across HackerRank, Claude, and Visa ecosystems.

## Setup

```bash
cd code
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required keys:
- `NVIDIA_API_KEY` - NVIDIA NIM API key for embeddings
- `MIMO_API_KEY` - MiMo v2.5 API key for LLM

## Usage

### Phase 1: Index the corpus (run once)

```bash
python main.py --index
```

This will:
1. Load all support articles from `data/`
2. Generate synthetic questions via LLM (~30-60 min)
3. Embed questions with NVIDIA nv-embedqa-e5-v5
4. Store in ChromaDB at `code/.db/chroma/`

### Phase 2: Triage tickets

```bash
python main.py --triage
```

This will:
1. Read tickets from `support_tickets/support_tickets.csv`
2. Process each through the LangGraph agent
3. Write results incrementally to `support_tickets/output.csv`

### Both phases

```bash
python main.py --index --triage
```

## Architecture

```
Ticket → Detect Company → Expand Queries → Retrieve (BM25+Vector RRF)
    → Check Confidence → Check Rules → Classify → Generate Response → CSV
```

## File Structure

- `config.py` - API keys, constants, thresholds
- `corpus_loader.py` - Markdown article loading with frontmatter parsing
- `indexer.py` - Async synthetic question generation + NVIDIA embedding
- `retriever.py` - Hybrid BM25 + Vector search with RRF fusion
- `agent.py` - LangGraph graph definition
- `nodes.py` - LangGraph node implementations
- `prompts.py` - All LLM prompt templates
- `rules.py` - Escalation rules + prompt injection detection
- `company_detector.py` - Keyword-first + LLM fallback company detection
- `output.py` - Incremental CSV writer
- `main.py` - Entry point
