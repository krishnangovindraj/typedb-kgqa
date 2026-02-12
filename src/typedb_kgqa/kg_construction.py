#!/usr/bin/env python3
"""Construct a TypeDB knowledge graph from 2WikiMultihopQA paragraphs using an LLM."""

import argparse
import json
import sys
from pathlib import Path

from .fetch_schema import fetch_schema
from .generate_query import generate_query_local, generate_query_claude, extract_typeql


def _format_paragraphs(context: list[list]) -> str:
    """Format a list of [title, sentences] pairs into the prompt's paragraph format."""
    parts = []
    for title, sentences in context:
        sentences_text = " ".join(sentences)
        parts.append(f"- Title: {title}\n  Sentences: {sentences_text}")
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
    Generate a TypeQL insert query for all paragraphs in an example.

    Args:
        schema: The TypeQL schema as a string.
        prompt_template: A prompt template with {schema} and {paragraphs} placeholders.
        context: List of [title, sentences] pairs from one dataset example.
        use_claude: If True, use Claude API; otherwise use local llama-cpp server.
        model: Model name (defaults based on backend).
        url: URL of llama-cpp server (only used if use_claude=False).
        max_tokens: Maximum tokens to generate.

    Returns:
        The generated TypeQL insert query.
    """
    paragraphs = _format_paragraphs(context)
    prompt = prompt_template.format(schema=schema, paragraphs=paragraphs)
    # print(f"--- DEBUG: PROMPT IS ---\n{prompt}\n--- END PROMPT ---", file=sys.stderr)
    if use_claude:
        model = model or "claude-sonnet-4-20250514"
        text = generate_query_claude(prompt, max_tokens, model)
    else:
        model = model or "default"
        text = generate_query_local(url, prompt, max_tokens, model)

    return extract_typeql("```typeql\ninsert\n" + text)


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

    # Process examples
    output_file = open(args.output, "w") if args.output else sys.stdout
    total = len(dataset)

    try:
        for i, example in enumerate(dataset, 1):
            question_id = example.get("_id", f"example-{i}")
            context = example["context"]
            titles = [title for title, _ in context]
            print(f"[{i}/{total}] Processing example {question_id} ({len(context)} paragraphs: {', '.join(titles)})", file=sys.stderr)

            try:
                insert_query = construct_kg(
                    schema=schema,
                    prompt_template=prompt_template,
                    context=context,
                    use_claude=args.claude,
                    model=args.model,
                    url=args.url,
                    max_tokens=args.max_tokens,
                )
                output_file.write(f"# Example: {question_id}\n")
                for title, sentences in context:
                    output_file.write(f"# {title}: {' '.join(sentences)}\n")
                output_file.write(insert_query)
                output_file.write("\n\n")
            except Exception as e:
                print(f"  ERROR processing example {question_id}: {e}", file=sys.stderr)
    finally:
        if args.output:
            output_file.close()

    print(f"Done. Processed {total} examples.", file=sys.stderr)


if __name__ == "__main__":
    main()
