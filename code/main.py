"""Main entry point: --index to build the index, --triage to process tickets."""

import argparse
import asyncio
import logging
import os
import sys

import pandas as pd
from tqdm import tqdm

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import create_agent
from config import INPUT_CSV, OUTPUT_CSV
from output import init_output_csv, validate_row, write_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_index():
    """Run the indexing pipeline."""
    from indexer import run_indexing

    logger.info("=== Starting Indexing Pipeline ===")
    await run_indexing()
    logger.info("=== Indexing Complete ===")


def run_triage():
    """Run the triage pipeline on support tickets."""
    logger.info("=== Starting Triage Pipeline ===")

    # Load tickets
    if not os.path.exists(INPUT_CSV):
        logger.error("Input CSV not found: %s", INPUT_CSV)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    logger.info("Loaded %d tickets from %s", len(df), INPUT_CSV)

    # Initialize output
    init_output_csv(OUTPUT_CSV)

    # Create agent
    agent = create_agent()
    logger.info("Agent created successfully")

    # Process each ticket
    errors = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing tickets"):
        try:
            # Build initial state
            initial_state = {
                "issue": str(row.get("Issue", "")),
                "subject": str(row.get("Subject", "")),
                "company": str(row.get("Company", "None")),
                "detected_company": "",
                "expanded_queries": [],
                "retrieved_docs": [],
                "retrieval_confidence": 0.0,
                "should_escalate": False,
                "escalation_reason": "",
                "classification": {},
                "response": "",
                "justification": "",
            }

            # Run agent
            result = agent.invoke(initial_state)

            # Build output row
            classification = result.get("classification", {})
            output_row = {
                "issue": initial_state["issue"],
                "subject": initial_state["subject"],
                "company": initial_state["company"],
                "response": result.get("response", ""),
                "product_area": classification.get("product_area", "out_of_scope"),
                "status": classification.get("status", "escalated"),
                "request_type": classification.get("request_type", "product_issue"),
                "justification": result.get("justification", ""),
            }

            # Validate
            validation_errors = validate_row(output_row)
            if validation_errors:
                logger.warning("Row %d validation errors: %s", idx, validation_errors)

            # Write incrementally
            write_row(output_row, OUTPUT_CSV)
            logger.info(
                "Ticket %d: status=%s, product_area=%s, request_type=%s",
                idx,
                output_row["status"],
                output_row["product_area"],
                output_row["request_type"],
            )

        except Exception as e:
            logger.error("Failed to process ticket %d: %s", idx, e)
            errors += 1

            # Write error row
            error_row = {
                "issue": str(row.get("Issue", "")),
                "subject": str(row.get("Subject", "")),
                "company": str(row.get("Company", "None")),
                "response": "Error processing ticket. Escalating to human review.",
                "product_area": "out_of_scope",
                "status": "escalated",
                "request_type": "product_issue",
                "justification": f"Processing error: {e}",
            }
            write_row(error_row, OUTPUT_CSV)

    logger.info("=== Triage Complete: %d errors out of %d tickets ===", errors, len(df))
    logger.info("Output written to: %s", OUTPUT_CSV)


def main():
    parser = argparse.ArgumentParser(description="Support Triage Agent")
    parser.add_argument("--index", action="store_true", help="Run indexing pipeline")
    parser.add_argument("--triage", action="store_true", help="Run triage pipeline")
    args = parser.parse_args()

    if not args.index and not args.triage:
        parser.print_help()
        sys.exit(1)

    if args.index:
        asyncio.run(run_index())

    if args.triage:
        run_triage()


if __name__ == "__main__":
    main()
