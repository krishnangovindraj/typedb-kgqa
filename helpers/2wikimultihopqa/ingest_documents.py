#!/usr/bin/env python3
"""Ingest 2WikiMultihopQA paragraphs into TypeDB as meta-document entities."""

import argparse
import json
import sys

from typedb.driver import TypeDB, Credentials, DriverOptions, TransactionType


def build_insert_query(title: str, sentences: list[str]) -> str:
    """Build a TypeQL insert query for a single meta-document."""
    text_content = " ".join(sentences)
    # Escape quotes in text content
    escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')
    escaped_text = text_content.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'insert\n'
        f'$doc isa meta-document, has meta-page-title "{escaped_title}", has text-content "{escaped_text}";'
    )


def ingest_documents(
    address: str,
    database: str,
    username: str,
    password: str,
    dataset: list[dict],
) -> int:
    """
    Insert meta-document entities into TypeDB for all paragraphs in the dataset.

    Returns the number of documents inserted.
    """
    inserted = 0
    with TypeDB.driver(address, Credentials(username, password), DriverOptions(is_tls_enabled=False)) as driver:
        for i, example in enumerate(dataset, 1):
            question_id = example.get("_id", f"example-{i}")
            context = example["context"]
            print(f"[{i}/{len(dataset)}] Example {question_id}: {len(context)} paragraphs", file=sys.stderr)

            for title, sentences in context:
                query = build_insert_query(title, sentences)
                try:
                    with driver.transaction(database, TransactionType.WRITE) as tx:
                        tx.query(query).resolve()
                        tx.commit()
                    inserted += 1
                except Exception as e:
                    print(f"  ERROR inserting '{title}': {e}", file=sys.stderr)

    return inserted


def main():
    parser = argparse.ArgumentParser(
        description="Ingest 2WikiMultihopQA paragraphs into TypeDB as meta-document entities"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (default: all)",
    )

    # TypeDB connection options
    parser.add_argument(
        "--typedb-address",
        type=str,
        default="localhost:1729",
        help="TypeDB server address (default: localhost:1729)",
    )
    parser.add_argument(
        "--database", "-d",
        type=str,
        required=True,
        help="TypeDB database name",
    )
    parser.add_argument(
        "--typedb-username",
        type=str,
        default="admin",
        help="TypeDB username (default: admin)",
    )
    parser.add_argument(
        "--typedb-password",
        type=str,
        default="password",
        help="TypeDB password (default: password)",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from: {args.dataset}", file=sys.stderr)
    with open(args.dataset) as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[:args.limit]

    # Deduplicate paragraphs by title (same title = same page)
    seen_titles = set()
    deduped_dataset = []
    for example in dataset:
        deduped_context = []
        for title, sentences in example["context"]:
            if title not in seen_titles:
                seen_titles.add(title)
                deduped_context.append([title, sentences])
        if deduped_context:
            deduped_dataset.append({**example, "context": deduped_context})

    total_docs = sum(len(ex["context"]) for ex in deduped_dataset)
    print(f"Inserting {total_docs} unique documents (deduplicated by title) into TypeDB ({args.typedb_address}, database: {args.database})...", file=sys.stderr)

    inserted = ingest_documents(
        address=args.typedb_address,
        database=args.database,
        username=args.typedb_username,
        password=args.typedb_password,
        dataset=deduped_dataset,
    )

    print(f"Done. Inserted {inserted} meta-document entities.", file=sys.stderr)


if __name__ == "__main__":
    main()
