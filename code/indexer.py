"""Async indexing pipeline: synthetic question generation + NVIDIA embedding."""

import asyncio
import json
import logging
import os
import re
import time

import chromadb
from openai import AsyncOpenAI
from tqdm import tqdm

from config import (
    CHECKPOINT_FILE,
    CHROMA_DIR,
    INDEX_CONCURRENCY,
    MIMO_API_KEY,
    MIMO_BASE_URL,
    MIMO_MODEL,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_EMBED_MODEL,
    NVIDIA_RPM,
)
from corpus_loader import load_full_corpus
from prompts import SYNTHETIC_QUESTION_PROMPT

logger = logging.getLogger(__name__)


def load_checkpoint() -> dict:
    """Load indexing checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_docs": [], "synthetic_questions": []}


def save_checkpoint(checkpoint: dict) -> None:
    """Save indexing checkpoint."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f)


async def generate_synthetic_questions(
    client: AsyncOpenAI, article: dict, semaphore: asyncio.Semaphore
) -> list[dict]:
    """Generate synthetic questions for a single article.

    Returns list of {question, doc_id, company, product_area, title}.
    """
    async with semaphore:
        prompt = SYNTHETIC_QUESTION_PROMPT.format(
            article_content=article["content"][:3000],  # Truncate for long articles
            company=article["company"],
            product_area=article["product_area"],
        )

        try:
            response = await client.chat.completions.create(
                model=MIMO_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()
            # Extract JSON
            json_match = re.search(r"\{[^}]+\}", content)
            if json_match:
                result = json.loads(json_match.group())
                questions = result.get("questions", [])
            else:
                # Try parsing as list directly
                questions = json.loads(content) if content.startswith("[") else []

            # Build question objects with metadata
            question_objects = []
            for q in questions[:5]:  # Max 5 questions per doc
                question_objects.append(
                    {
                        "question": q,
                        "doc_id": article["doc_id"],
                        "company": article["company"],
                        "product_area": article["product_area"],
                        "title": article["title"],
                        "source_url": article.get("source_url", ""),
                    }
                )

            return question_objects

        except Exception as e:
            logger.warning(
                "Failed to generate questions for %s: %s", article["doc_id"], e
            )
            return []


async def generate_all_synthetic_questions(
    articles: list[dict],
) -> list[dict]:
    """Generate synthetic questions for all articles asynchronously.

    Returns list of question objects.
    """
    checkpoint = load_checkpoint()
    completed_ids = set(checkpoint.get("completed_docs", []))
    all_questions = checkpoint.get("synthetic_questions", [])

    # Filter out already completed articles
    remaining = [a for a in articles if a["doc_id"] not in completed_ids]
    logger.info(
        "Generating synthetic questions: %d already done, %d remaining",
        len(completed_ids),
        len(remaining),
    )

    if not remaining:
        return all_questions

    semaphore = asyncio.Semaphore(INDEX_CONCURRENCY)
    client = AsyncOpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)

    # Process in batches for checkpointing
    batch_size = 50
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        tasks = [
            generate_synthetic_questions(client, article, semaphore)
            for article in batch
        ]

        results = []
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Synthetic questions batch {i // batch_size + 1}",
        ):
            result = await coro
            results.append(result)

        # Flatten results
        for question_list in results:
            all_questions.extend(question_list)

        # Update checkpoint
        for article in batch:
            completed_ids.add(article["doc_id"])

        checkpoint["completed_docs"] = list(completed_ids)
        checkpoint["synthetic_questions"] = all_questions
        save_checkpoint(checkpoint)
        logger.info(
            "Checkpoint saved: %d docs done, %d questions total",
            len(completed_ids),
            len(all_questions),
        )

    return all_questions


def embed_questions_batch(questions: list[dict]) -> list[dict]:
    """Embed questions using NVIDIA API with rate limiting.

    Returns list of {embedding, ...question_metadata}.
    """
    from openai import OpenAI

    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    embedded = []

    # Process in batches respecting rate limit (40/min)
    batch_size = 30  # Leave some headroom
    delay_between_batches = 60.0 / NVIDIA_RPM * batch_size + 1  # ~46 seconds

    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        texts = [q["question"] for q in batch]

        try:
            response = client.embeddings.create(
                input=texts,
                model=NVIDIA_EMBED_MODEL,
                encoding_format="float",
                extra_body={"input_type": "passage", "truncate": "NONE"},
            )

            for j, emb_data in enumerate(response.data):
                question_obj = batch[j].copy()
                question_obj["embedding"] = emb_data.embedding
                embedded.append(question_obj)

        except Exception as e:
            logger.error("Embedding batch failed at offset %d: %s", i, e)
            # Retry individual questions
            for q in batch:
                try:
                    response = client.embeddings.create(
                        input=[q["question"]],
                        model=NVIDIA_EMBED_MODEL,
                        encoding_format="float",
                        extra_body={"input_type": "passage", "truncate": "NONE"},
                    )
                    question_obj = q.copy()
                    question_obj["embedding"] = response.data[0].embedding
                    embedded.append(question_obj)
                except Exception as e2:
                    logger.error(
                        "Individual embedding failed for %s: %s", q["question"][:50], e2
                    )

        # Rate limit delay
        if i + batch_size < len(questions):
            logger.info(
                "Embedded %d/%d questions, waiting %.1fs for rate limit...",
                min(i + batch_size, len(questions)),
                len(questions),
                delay_between_batches,
            )
            time.sleep(delay_between_batches)

    return embedded


def store_in_chroma(embedded_questions: list[dict], articles: list[dict]) -> None:
    """Store embedded questions and parent documents in ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if present
    try:
        client.delete_collection("support_docs")
    except Exception:
        pass

    collection = client.create_collection(
        "support_docs",
        metadata={"hnsw:space": "cosine"},
    )

    # Store synthetic question embeddings
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, q in enumerate(embedded_questions):
        doc_id = f"sq_{q['doc_id']}_{i}"
        ids.append(doc_id)
        embeddings.append(q["embedding"])
        documents.append(q["question"])
        metadatas.append(
            {
                "doc_id": q["doc_id"],
                "company": q["company"],
                "product_area": q["product_area"],
                "title": q["title"],
                "source_url": q.get("source_url", ""),
                "is_synthetic_question": True,
            }
        )

    # Batch insert (ChromaDB has limits)
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    logger.info("Stored %d synthetic question embeddings in ChromaDB", len(ids))

    # Also store full article content for retrieval
    # Map doc_id -> article content
    article_map = {a["doc_id"]: a for a in articles}

    # Store parent documents (one per unique doc_id)
    parent_ids = []
    parent_docs = []
    parent_metas = []
    seen_doc_ids = set()

    for q in embedded_questions:
        doc_id = q["doc_id"]
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)

        article = article_map.get(doc_id)
        if article:
            parent_ids.append(doc_id)
            parent_docs.append(article["content"][:10000])  # ChromaDB has size limits
            parent_metas.append(
                {
                    "company": article["company"],
                    "product_area": article["product_area"],
                    "title": article["title"],
                    "source_url": article.get("source_url", ""),
                    "is_parent_doc": True,
                }
            )

    for i in range(0, len(parent_ids), batch_size):
        end = min(i + batch_size, len(parent_ids))
        collection.add(
            ids=parent_ids[i:end],
            documents=parent_docs[i:end],
            metadatas=parent_metas[i:end],
        )

    logger.info("Stored %d parent documents in ChromaDB", len(parent_ids))


async def run_indexing() -> None:
    """Full indexing pipeline."""
    logger.info("Starting indexing pipeline...")

    # Step 1: Load corpus
    articles = load_full_corpus()
    logger.info("Loaded %d articles", len(articles))

    # Step 2: Generate synthetic questions
    questions = await generate_all_synthetic_questions(articles)
    logger.info("Generated %d synthetic questions", len(questions))

    # Step 3: Embed questions
    logger.info("Embedding questions with NVIDIA API...")
    embedded = embed_questions_batch(questions)
    logger.info("Embedded %d questions", len(embedded))

    # Step 4: Store in ChromaDB
    logger.info("Storing in ChromaDB...")
    store_in_chroma(embedded, articles)
    logger.info("Indexing complete!")
