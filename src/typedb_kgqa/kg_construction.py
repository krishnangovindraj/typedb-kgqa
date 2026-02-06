#!/usr/bin/env python3
"""Construct a TypeDB knowledge graph from 2WikiMultihopQA paragraphs using an LLM."""

import argparse
import json
import sys
from pathlib import Path

from .fetch_schema import fetch_schema
from .generate_query import generate_query_local, generate_query_claude


def extract_insert(response: str) -> str:
    """Extract the TypeQL insert query from the LLM response."""
    text = response.strip()
    if "```" in text:
        text = text.split("```")[0].strip()
    return "insert\n" + text


def construct_kg(
    schema: str,
    prompt_template: str,
    title: str,
    sentences: list[str],
    use_claude: bool = False,
    model: str = None,
    url: str = "http://localhost:8080/v1",
    max_tokens: int = 512,
) -> str:
    """
    Generate a TypeQL insert query for one paragraph.

    Args:
        schema: The TypeQL schema as a string.
        prompt_template: A prompt template with {schema}, {title}, and {sentences} placeholders.
        title: The paragraph title (typically the Wikipedia article title).
        sentences: List of sentences in the paragraph.
        use_claude: If True, use Claude API; otherwise use local llama-cpp server.
        model: Model name (defaults based on backend).
        url: URL of llama-cpp server (only used if use_claude=False).
        max_tokens: Maximum tokens to generate.

    Returns:
        The generated TypeQL insert query.
    """
    sentences_text = " ".join(sentences)
    prompt = prompt_template.format(schema=schema, title=title, sentences=sentences_text)

    if use_claude:
        model = model or "claude-sonnet-4-20250514"
        text = generate_query_claude(prompt, max_tokens, model)
    else:
        model = model or "default"
        text = generate_query_local(url, prompt, max_tokens, model)

    return extract_insert(text)


def main():
    parser = argparse.ArgumentParser(
        description="Construct TypeDB knowledge graph from 2WikiMultihopQA paragraphs"
    )

    # Dataset
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

    # Prompt
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="src/typedb_kgqa/prompts/kg_construction.txt",
        help="Path to prompt template file",
    )

    # Schema source: TypeDB or file
    schema_group = parser.add_mutually_exclusive_group(required=True)
    schema_group.add_argument(
        "--schema-file",
        type=str,
        help="Path to a TypeQL schema file to use as schema context",
    )
    schema_group.add_argument(
        "--database", "-d",
        type=str,
        help="TypeDB database name (fetch schema from TypeDB)",
    )

    # TypeDB connection options (only used with --database)
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
        default=512,
        help="Maximum tokens to generate per paragraph (default: 512)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    # Load schema
    if args.schema_file:
        print(f"Loading schema from file: {args.schema_file}", file=sys.stderr)
        schema = Path(args.schema_file).read_text()
    else:
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

    # Load dataset
    print(f"Loading dataset from: {args.dataset}", file=sys.stderr)
    with open(args.dataset) as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[:args.limit]

    # Show backend info
    if args.claude:
        model = args.model or "claude-sonnet-4-20250514"
        print(f"Using Claude API (model: {model})", file=sys.stderr)
    else:
        model = args.model or "default"
        print(f"Using local server: {args.url} (model: {model})", file=sys.stderr)

    # Process paragraphs
    output_file = open(args.output, "w") if args.output else sys.stdout
    total_paragraphs = sum(len(example["context"]) for example in dataset)
    processed = 0

    try:
        for i, example in enumerate(dataset):
            question_id = example.get("_id", f"example-{i}")
            for title, sentences in example["context"]:
                processed += 1
                print(f"[{processed}/{total_paragraphs}] Processing: {title}", file=sys.stderr)

                try:
                    insert_query = construct_kg(
                        schema=schema,
                        prompt_template=prompt_template,
                        title=title,
                        sentences=sentences,
                        use_claude=args.claude,
                        model=args.model,
                        url=args.url,
                        max_tokens=args.max_tokens,
                    )
                    output_file.write(f"# Source: {question_id} / {title}\n")
                    output_file.write(insert_query)
                    output_file.write("\n\n")
                except Exception as e:
                    print(f"  ERROR processing '{title}': {e}", file=sys.stderr)
    finally:
        if args.output:
            output_file.close()

    print(f"Done. Processed {processed} paragraphs.", file=sys.stderr)


if __name__ == "__main__":
    main()
