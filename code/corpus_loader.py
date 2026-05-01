"""Load and split markdown articles from the support corpus."""

import logging
import os
import re
import uuid

import yaml

from config import COMPANIES, DATA_DIR

logger = logging.getLogger(__name__)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content.

    Returns (frontmatter_dict, body_content).
    If no frontmatter found, returns ({}, original_content).
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        frontmatter = {}

    body = parts[2].strip()
    return frontmatter, body


def generate_doc_id(filepath: str) -> str:
    """Generate a deterministic UUID from filepath."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, filepath))


def infer_breadcrumbs_from_path(filepath: str, company: str) -> list[str]:
    """Infer breadcrumb hierarchy from directory path when frontmatter lacks it."""
    rel_path = os.path.relpath(filepath, os.path.join(DATA_DIR, company))
    parts = rel_path.replace(".md", "").split(os.sep)
    return parts[:-1] if len(parts) > 1 else []


def extract_product_area(breadcrumbs: list[str]) -> str:
    """Extract the most specific product area from breadcrumbs."""
    if not breadcrumbs:
        return "general"
    return breadcrumbs[-1].lower().replace(" ", "_").replace("-", "_")


def load_article(filepath: str, company: str) -> dict | None:
    """Load a single markdown article with metadata.

    Handles missing/malformed frontmatter gracefully.
    Returns None if file cannot be loaded.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        frontmatter, body = parse_frontmatter(content)

        # Infer breadcrumbs from path if missing
        breadcrumbs = frontmatter.get("breadcrumbs", [])
        if not breadcrumbs:
            breadcrumbs = infer_breadcrumbs_from_path(filepath, company)

        product_area = extract_product_area(breadcrumbs)

        return {
            "doc_id": generate_doc_id(filepath),
            "company": company,
            "title": frontmatter.get(
                "title", os.path.basename(filepath).replace(".md", "")
            ),
            "source_url": frontmatter.get(
                "source_url", frontmatter.get("final_url", "")
            ),
            "breadcrumbs": breadcrumbs,
            "product_area": product_area,
            "content": body,
            "filepath": filepath,
        }
    except Exception as e:
        logger.warning("Failed to load %s: %s", filepath, e)
        return None


def load_company_corpus(company: str) -> list[dict]:
    """Load all articles for a given company."""
    company_dir = os.path.join(DATA_DIR, company)
    if not os.path.isdir(company_dir):
        logger.error("Directory not found: %s", company_dir)
        return []

    articles = []
    for root, _dirs, files in os.walk(company_dir):
        for filename in files:
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(root, filename)
            article = load_article(filepath, company)
            if article:
                articles.append(article)

    logger.info("Loaded %d articles for %s", len(articles), company)
    return articles


def load_full_corpus() -> list[dict]:
    """Load articles from all companies.

    Returns list of article dicts with keys:
    doc_id, company, title, source_url, breadcrumbs, product_area, content, filepath
    """
    all_articles = []
    for company in COMPANIES:
        articles = load_company_corpus(company)
        all_articles.extend(articles)

    logger.info("Total articles loaded: %d", len(all_articles))
    return all_articles


def split_into_chunks(article: dict) -> list[dict]:
    """Split article content into chunks by markdown headers.

    Each chunk inherits the parent article's metadata.
    Returns list of chunk dicts.
    """
    content = article["content"]
    chunks = []

    # Split on ## headers (level 2)
    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # Extract section title from first line
        first_line = section.split("\n", 1)[0].strip()
        section_title = re.sub(r"^#+\s*", "", first_line)

        chunks.append(
            {
                "doc_id": article["doc_id"],
                "chunk_id": f"{article['doc_id']}_chunk_{i}",
                "company": article["company"],
                "title": article["title"],
                "source_url": article["source_url"],
                "breadcrumbs": article["breadcrumbs"],
                "product_area": article["product_area"],
                "section_title": section_title,
                "content": section,
                "filepath": article["filepath"],
            }
        )

    # If no ## headers found, return the whole content as one chunk
    if not chunks:
        chunks.append(
            {
                "doc_id": article["doc_id"],
                "chunk_id": f"{article['doc_id']}_chunk_0",
                "company": article["company"],
                "title": article["title"],
                "source_url": article["source_url"],
                "breadcrumbs": article["breadcrumbs"],
                "product_area": article["product_area"],
                "section_title": article["title"],
                "content": content,
                "filepath": article["filepath"],
            }
        )

    return chunks
