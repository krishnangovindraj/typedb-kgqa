#!/usr/bin/env python3
"""Answer questions using RAG: embed the question, retrieve relevant documents from TypeDB, and ask an LLM."""

import argparse
import sys
from pathlib import Path

from typedb.driver import TypeDB, Credentials, DriverOptions, TransactionType

from .common import (
    get_embeddings_local,
    encode_embeddings_base64,
    generate_query_local,
    generate_query_claude,
)

MAX_DOCS_TO_RETREIVE = 5;
RAG_QUERY = """
match
  let $query-embedding = "{query_embedding}";
  let $looked-up-embedding in embeddings_by_similarity($query-embedding);
  let $neighbour-doc in graph_neigbhour_documents_by_similarity($query-embedding, $looked-up-embedding, 0.0);
  $neighbour-doc has text-content $text, has meta-page-title $title;
  select $title, $text;
  limit {max_docs};
"""


def gather_sources(tx, question_embedding_b64: str) -> list[dict]:
    """Run the RAG retrieval query and return a list of {title, text} dicts."""
    query = RAG_QUERY.format(query_embedding=question_embedding_b64, max_docs=MAX_DOCS_TO_RETREIVE)
    rows = list(tx.query(query).resolve().as_concept_rows())
    results = []
    for row in rows:
        results.append({
            "title": row.get("title").as_attribute().get_value(),
            "text": row.get("text").as_attribute().get_value(),
        })
    return results


def _format_documents(sources: list[dict]) -> str:
    """Format retrieved sources into the prompt's document format."""
    parts = []
    for src in sources:
        parts.append(f"- Title: {src['title']}\n  Content: {src['text']}")
    return "\n".join(parts)


def answer_question(
    prompt_template: str,
    sources: list[dict],
    question: str,
    use_claude: bool = False,
    model: str = None,
    url: str = "http://localhost:8080/v1",
    max_tokens: int = 256,
) -> str:
    """Feed retrieved documents and question to an LLM and return the answer."""
    documents = _format_documents(sources)
    prompt = prompt_template.format(documents=documents, question=question)

    if use_claude:
        model = model or "claude-sonnet-4-20250514"
        text = generate_query_claude(prompt, max_tokens, model)
    else:
        model = model or "default"
        text = generate_query_local(url, prompt, max_tokens, model)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Answer questions using RAG over TypeDB documents"
    )

    # Questions
    parser.add_argument(
        "--questions", "-q",
        type=str,
        required=True,
        help="Path to a file with one question per line",
    )

    # Prompt
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="src/typedb_kgqa/prompts/graphrag_answer.txt",
        help="Path to prompt template file",
    )

    # TypeDB connection
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

    # Embedding server
    parser.add_argument(
        "--embedding-url",
        type=str,
        default="http://localhost:8081/v1",
        help="URL of embedding server (default: http://localhost:8081/v1)",
    )

    # LLM backend options
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Use Claude CLI for answer generation",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080/v1",
        help="URL of llama-cpp server for answer generation (default: http://localhost:8080/v1)",
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
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    # Load prompt template
    prompt_template = Path(args.prompt).read_text()

    # Load questions
    questions = [line.strip() for line in Path(args.questions).read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} questions", file=sys.stderr)

    # Show backend info
    if args.claude:
        model = args.model or "claude-sonnet-4-20250514"
        print(f"Using Claude CLI (model: {model})", file=sys.stderr)
    else:
        model = args.model or "default"
        print(f"Using local server: {args.url} (model: {model})", file=sys.stderr)
    print(f"Embedding server: {args.embedding_url}", file=sys.stderr)

    output_file = open(args.output, "w") if args.output else sys.stdout
    total = len(questions)

    with TypeDB.driver(args.typedb_address, Credentials(args.typedb_username, args.typedb_password), DriverOptions(is_tls_enabled=False)) as driver:
        try:
            for i, question in enumerate(questions, 1):
                print(f"[{i}/{total}] Question: {question}", file=sys.stderr)

                # Embed the question
                embeddings = get_embeddings_local(args.embedding_url, [question], is_query=True)
                embedding_b64 = encode_embeddings_base64(embeddings[0])

                # Retrieve relevant documents
                with driver.transaction(args.database, TransactionType.READ) as tx:
                    sources = gather_sources(tx, embedding_b64)

                print(f"  Retrieved {len(sources)} documents", file=sys.stderr)

                # Generate answer
                answer = answer_question(
                    prompt_template=prompt_template,
                    sources=sources,
                    question=question,
                    use_claude=args.claude,
                    model=args.model,
                    url=args.url,
                    max_tokens=args.max_tokens,
                )

                output_file.write(f"{answer}\n")
        finally:
            if args.output:
                output_file.close()

    print(f"Done. Answered {total} questions.", file=sys.stderr)


if __name__ == "__main__":
    main()
