"""Incremental CSV output writer."""

import csv
import logging
import os

from config import OUTPUT_COLUMNS

logger = logging.getLogger(__name__)


def init_output_csv(filepath: str) -> None:
    """Create output CSV with header if it doesn't exist."""
    if os.path.exists(filepath):
        logger.info("Output CSV already exists: %s", filepath)
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
    logger.info("Created output CSV: %s", filepath)


def write_row(row: dict, filepath: str) -> None:
    """Append a single row to output CSV.

    Creates file with header if missing.
    Flushes immediately for crash safety.
    """
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if not file_exists:
            writer.writeheader()

        # Ensure all columns present, fill missing with empty string
        clean_row = {col: row.get(col, "") for col in OUTPUT_COLUMNS}
        writer.writerow(clean_row)
        f.flush()


def validate_row(row: dict) -> list[str]:
    """Validate a single output row. Returns list of errors (empty = valid)."""
    errors = []

    from config import ALLOWED_REQUEST_TYPES, ALLOWED_STATUSES

    if row.get("status") not in ALLOWED_STATUSES:
        errors.append(f"Invalid status: {row.get('status')}")

    if row.get("request_type") not in ALLOWED_REQUEST_TYPES:
        errors.append(f"Invalid request_type: {row.get('request_type')}")

    if not row.get("response", "").strip():
        errors.append("Empty response")

    if not row.get("justification", "").strip():
        errors.append("Empty justification")

    if not row.get("product_area", "").strip():
        errors.append("Empty product_area")

    return errors
