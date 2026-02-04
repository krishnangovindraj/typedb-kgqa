#!/usr/bin/env python3
"""Generate TypeQL queries from natural language questions using an LLM."""

import argparse
from pathlib import Path

from .fetch_schema import fetch_schema


def generate_query_local(
    url: str,
    prompt: str,
    max_tokens: int = 256,
    model: str = "default",
) -> str:
    """Generate using local llama-cpp server."""
    from openai import OpenAI

    client = OpenAI(base_url=url, api_key="not-needed")
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        stop=["```", "##", "Question:"],
        temperature=0.1,
    )

    # Handle both standard OpenAI format and llama-cpp server format
    if response.choices:
        return response.choices[0].text
    elif hasattr(response, "content") and response.content:
        return response.content
    else:
        raise RuntimeError(f"Could not extract text from response: {response}")


def generate_query_claude(
    prompt: str,
    max_tokens: int = 256,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Generate using Claude API."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_query(
    schema: str,
    prompt_template: str,
    question: str,
    max_tokens: int = 256,
    use_claude: bool = False,
    model: str = None,
    url: str = "http://localhost:8080/v1",
) -> str:
    """
    Generate a TypeQL query for a natural language question.

    Args:
        schema: The TypeQL schema as a string.
        prompt_template: A prompt template with {schema} and {question} placeholders.
        question: The natural language question to convert.
        max_tokens: Maximum tokens to generate.
        use_claude: If True, use Claude API; otherwise use local llama-cpp server.
        model: Model name (defaults based on backend).
        url: URL of llama-cpp server (only used if use_claude=False).

    Returns:
        The generated TypeQL query.
    """
    prompt = prompt_template.format(schema=schema, question=question)
    print(f"--- DEBUG: PROMPT IS ---\n{prompt}\n--- END PROMPT ---")
    if use_claude:
        model = model or "claude-sonnet-4-20250514"
        text = generate_query_claude(prompt, max_tokens, model)
    else:
        model = model or "default"
        text = generate_query_local(url, prompt, max_tokens, model)

    return extract_query(text)


def extract_query(response: str) -> str:
    """Extract the TypeQL query from the LLM response."""
    text = response.strip()
    if "```" in text:
        text = text.split("```")[0].strip()
    return "match\n" + text


def main():
    parser = argparse.ArgumentParser(
        description="Generate TypeQL queries from natural language questions"
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
        help="Model name (default: 'claude-sonnet-4-20250514' for Claude, 'default' for local)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
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
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact schema representation (smaller context for local LLMs)",
    )

    # Prompt and question
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Path to prompt template file (must contain {schema} and {question} placeholders)",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        required=True,
        help="The question to translate to TypeQL",
    )

    args = parser.parse_args()

    # Fetch schema from TypeDB
    print(f"Fetching schema from TypeDB ({args.typedb_address}, database: {args.database})...")
    schema = fetch_schema(
        address=args.typedb_address,
        database=args.database,
        username=args.typedb_username,
        password=args.typedb_password,
        compact=args.compact,
    )

    # Load prompt template
    print("Loading prompt template...")
    prompt_template = Path(args.prompt).read_text()

    # Show backend info
    if args.claude:
        model = args.model or "claude-sonnet-4-20250514"
        print(f"Using Claude API (model: {model})")
    else:
        model = args.model or "default"
        print(f"Using local server: {args.url} (model: {model})")

    if args.compact:
        print("Using compact schema representation")

    # Generate query
    print(f"Question: {args.question}")
    print("Generating TypeQL query...")
    print()

    query = generate_query(
        schema=schema,
        prompt_template=prompt_template,
        question=args.question,
        max_tokens=args.max_tokens,
        use_claude=args.claude,
        model=args.model,
        url=args.url,
    )

    print("Generated query:")
    print("-" * 40)
    print(query)
    print("-" * 40)


if __name__ == "__main__":
    main()
