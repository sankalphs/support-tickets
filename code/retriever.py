"""Hybrid BM25 + Vector retrieval with RRF fusion."""

import logging

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

from config import (
    CHROMA_DIR,
    CONFIDENCE_THRESHOLD,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_EMBED_MODEL,
    RRF_K,
    TOP_K_BM25,
    TOP_K_FINAL,
    TOP_K_VECTOR,
)

logger = logging.getLogger(__name__)


class HybridRetriever:
    """BM25 + Vector search with Reciprocal Rank Fusion."""

    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.bm25 = None
        self.bm25_doc_ids = []
        self.bm25_contents = []
        self.nvidia_client = None

    def initialize(self) -> None:
        """Load ChromaDB and BM25 index."""
        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma_client.get_collection("support_docs")

        # NVIDIA client
        self.nvidia_client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

        # Build BM25 index from all documents in Chroma
        self._build_bm25_index()
        logger.info(
            "Retriever initialized: %d docs in Chroma, %d in BM25",
            self.collection.count(),
            len(self.bm25_doc_ids),
        )

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in ChromaDB."""
        # Get all documents from Chroma
        results = self.collection.get(include=["documents", "metadatas"])

        self.bm25_doc_ids = results["ids"]
        self.bm25_contents = results["documents"]

        # Tokenize for BM25
        tokenized = [doc.lower().split() for doc in self.bm25_contents]
        self.bm25 = BM25Okapi(tokenized)

    def embed_query(self, query: str) -> list[float]:
        """Embed a query using NVIDIA API."""
        response = self.nvidia_client.embeddings.create(
            input=[query],
            model=NVIDIA_EMBED_MODEL,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"},
        )
        return response.data[0].embedding

    def vector_search(
        self, query_embedding: list[float], company: str | None = None
    ) -> list[tuple[str, float]]:
        """Search ChromaDB with optional company filter.

        Returns list of (doc_id, distance).
        """
        n_results = min(TOP_K_VECTOR, self.collection.count())
        if n_results == 0:
            return []

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }

        # Visa: skip metadata filter (tiny corpus)
        # None: search all
        if company and company != "Visa" and company != "None":
            kwargs["where"] = {"company": company}

        results = self.collection.query(**kwargs)

        doc_ids = results["ids"][0]
        distances = results["distances"][0]

        return list(zip(doc_ids, distances))

    def bm25_search(self, query: str) -> list[tuple[str, float]]:
        """BM25 keyword search.

        Returns list of (doc_id, score).
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-K indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :TOP_K_BM25
        ]

        return [(self.bm25_doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]

    def rrf_fusion(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion of vector and BM25 results.

        score = sum(1 / (k + rank_i)) for each ranking list
        Lower rank = better (rank 0 is best).
        """
        rrf_scores = {}

        # Vector results (lower distance = better, so sort ascending)
        vector_sorted = sorted(vector_results, key=lambda x: x[1])
        for rank, (doc_id, _dist) in enumerate(vector_sorted):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        # BM25 results (higher score = better, so sort descending)
        bm25_sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _score) in enumerate(bm25_sorted):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def retrieve(
        self, queries: list[str], company: str | None = None
    ) -> tuple[list[str], float]:
        """Full hybrid retrieval pipeline.

        Args:
            queries: List of search queries (from query expansion)
            company: Company filter (None = search all)

        Returns:
            (list of doc_ids, confidence_score)
            If confidence < threshold, returns empty list.
        """
        all_vector_results = []
        all_bm25_results = []

        for query in queries:
            # Embed and search
            query_embedding = self.embed_query(query)
            vector_results = self.vector_search(query_embedding, company)
            bm25_results = self.bm25_search(query)
            all_vector_results.extend(vector_results)
            all_bm25_results.extend(bm25_results)

        # RRF fusion
        fused = self.rrf_fusion(all_vector_results, all_bm25_results)

        if not fused:
            logger.warning("No results found for queries: %s", queries)
            return [], 0.0

        # Top-K final results
        top_results = fused[:TOP_K_FINAL]
        doc_ids = [doc_id for doc_id, _score in top_results]
        confidence = top_results[0][1] if top_results else 0.0

        logger.info(
            "Retrieved %d docs, top confidence: %.4f (threshold: %.4f)",
            len(doc_ids),
            confidence,
            CONFIDENCE_THRESHOLD,
        )

        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(
                "Low confidence (%.4f < %.4f), signaling no relevant docs",
                confidence,
                CONFIDENCE_THRESHOLD,
            )
            return [], confidence

        return doc_ids, confidence

    def get_parent_documents(self, doc_ids: list[str]) -> list[dict]:
        """Get full document content from ChromaDB by doc_id."""
        if not doc_ids:
            return []

        results = self.collection.get(ids=doc_ids, include=["documents", "metadatas"])

        docs = []
        for i, doc_id in enumerate(results["ids"]):
            docs.append(
                {
                    "doc_id": doc_id,
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )

        return docs
