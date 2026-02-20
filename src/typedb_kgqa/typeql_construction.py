#!/usr/bin/env python3
"""Construct a TypeDB knowledge graph from paragraphs using an LLM.

Reads a sources file (one JSON list of page titles per line), fetches
the text-content for each title from TypeDB meta-document entities,
and generates TypeQL put statements via an LLM.
"""

import argparse
import json
import sys
from pathlib import Path

from typedb.driver import TypeDB, Credentials, DriverOptions, TransactionType

from .fetch_schema import fetch_schema
from .common import generate_query_local, generate_query_claude, extract_typeql

def fetch_document(tx, title: str) -> str:
    """Fetch text-content for a meta-document by its meta-page-title."""
    escaped = title.replace("\\", "\\\\").replace('"', '\\"')
    query = f'match $doc isa meta-document, has meta-page-title "{escaped}", has text-content $text;'
    rows = list(tx.query(query).resolve().as_concept_rows())
    if not rows:
        return None
    return rows[0].get("text").as_attribute().get_value()


def fetch_documents(tx, titles: list[str]) -> list[list]:
    """Fetch documents for a list of titles. Returns [title, text] pairs."""
    context = []
    for title in titles:
        text = fetch_document(tx, title)
        if text is None:
            print(f"  WARNING: no document found for '{title}'", file=sys.stderr)
        else:
            context.append([title, text])
    return context


def _format_paragraphs(context: list[list]) -> str:
    """Format a list of [title, text] pairs into the prompt's paragraph format."""
    parts = []
    for title, text in context:
        parts.append(f"- Title: {title}\n  Sentences: {text}")
    return "\n".join(parts)


def construct_kg(
    schema: str,
    prompt_template: str,
    context: list[list],
    use_claude: bool = False,
    model: str = None,
    url: str = "http://localhost:8080/v1",
    max_tokens: int = 4096,
) -> str:
    """
    Generate TypeQL put statements for all paragraphs in an example.

    Args:
        schema: The TypeQL schema as a string.
        prompt_template: A prompt template with {schema} and {paragraphs} placeholders.
        context: List of [title, text] pairs.
        use_claude: If True, use Claude API; otherwise use local llama-cpp server.
        model: Model name (defaults based on backend).
        url: URL of llama-cpp server (only used if use_claude=False).
        max_tokens: Maximum tokens to generate.

    Returns:
        The generated TypeQL put statements.
    """
    # TODO: Add put the source and link the source.
    paragraphs = _format_paragraphs(context)
    prompt = prompt_template.format(schema=schema, paragraphs=paragraphs)
    if use_claude:
        model = model or "claude-sonnet-4-20250514"
        text = generate_query_claude(prompt, max_tokens, model)
    else:
        model = model or "default"
        text = generate_query_local(url, prompt, max_tokens, model)

    return extract_typeql("```typeql\ninsert\n" + text)


def main():
    parser = argparse.ArgumentParser(
        description="Construct TypeDB knowledge graph from page titles using an LLM"
    )

    # Sources
    parser.add_argument(
        "--sources",
        type=str,
        required=True,
        help="Path to sources file (one JSON list of page titles per line)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of lines to process (default: all)",
    )

    # Prompt
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="src/typedb_kgqa/prompts/typeql_construction.txt",
        help="Path to prompt template file",
    )

    # TypeDB connection (required — used for both schema and document fetching)
    parser.add_argument(
        "--database", "-d",
        type=str,
        required=True,
        help="TypeDB database name",
    )
    parser.add_argument(
        "--typedb-address",
        type=str,
        default="localhost:1729",
        help="TypeDB server address (default: localhost:1729)",
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
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact schema representation",
    )

    # LLM backend options
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Use Claude API instead of local llama-cpp server (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080/v1",
        help="URL of llama-cpp server (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate per example (default: 4096)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    # Fetch schema from TypeDB
    print(f"Fetching schema from TypeDB ({args.typedb_address}, database: {args.database})...", file=sys.stderr)
    schema = fetch_schema(
        address=args.typedb_address,
        database=args.database,
        username=args.typedb_username,
        password=args.typedb_password,
        compact=args.compact,
    )

    # Load prompt template
    print(f"Loading prompt template from: {args.prompt}", file=sys.stderr)
    prompt_template = Path(args.prompt).read_text()

    # Load sources
    print(f"Loading sources from: {args.sources}", file=sys.stderr)
    with open(args.sources) as f:
        source_lines = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        source_lines = source_lines[:args.limit]

    # Show backend info
    if args.claude:
        model = args.model or "claude-sonnet-4-20250514"
        print(f"Using Claude API (model: {model})", file=sys.stderr)
    else:
        model = args.model or "default"
        print(f"Using local server: {args.url} (model: {model})", file=sys.stderr)

    # Process examples
    output_file = open(args.output, "w") if args.output else sys.stdout
    total = len(source_lines)

    with TypeDB.driver(args.typedb_address, Credentials(args.typedb_username, args.typedb_password), DriverOptions(is_tls_enabled=False)) as driver:
        try:
            for i, titles in enumerate(source_lines, 1):
                print(f"[{i}/{total}] Fetching documents for: {', '.join(titles)}", file=sys.stderr)

                with driver.transaction(args.database, TransactionType.READ) as tx:
                    context = fetch_documents(tx, titles)

                if not context:
                    print(f"  SKIP: no documents found", file=sys.stderr)
                    continue

                try:
                    result = construct_kg(
                        schema=schema,
                        prompt_template=prompt_template,
                        context=context,
                        use_claude=args.claude,
                        model=args.model,
                        url=args.url,
                        max_tokens=args.max_tokens,
                    )
                    output_file.write(f"# Sources: {json.dumps(titles)}\n")
                    for title, text in context:
                        output_file.write(f"# {title}: {text[:100]}...\n")
                    output_file.write(result)
                    output_file.write("\n\n")
                except Exception as e:
                    print(f"  ERROR: {e}", file=sys.stderr)
        finally:
            if args.output:
                output_file.close()

    print(f"Done. Processed {total} examples.", file=sys.stderr)


if __name__ == "__main__":
    main()
