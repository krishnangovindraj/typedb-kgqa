#!/usr/bin/env python3
"""Construct a vanilla TypeDB knowledge graph from paragraphs using an LLM.

Reads a sources file (one JSON list of page titles per line), fetches
the text-content for each title from TypeDB meta-document entities,
uses an LLM to extract entities/relations/properties in simplified line format,
converts those lines to TypeQL put statements, and writes them to TypeDB.
"""

import argparse
import json
import sys
from pathlib import Path

from typedb.driver import TypeDB, Credentials, DriverOptions, TransactionType

from .common import generate_query_local, generate_query_claude, get_embeddings_local, encode_embeddings_base64
from .kg_construction import fetch_documents, _format_paragraphs


def lines_to_typeql(lines: str, embed_fn=None) -> str:
    """Convert simplified KG extraction lines to TypeQL put statements.

    Input format (one per line):
        source <page-title>
        entity <label>
        property <entity-label> <property-label> <value>
        relation <entity1-label> <relation-label> <entity2-label>

    Source lines generate meta-document puts and cause subsequent nodes
    to be linked to that source via meta-knowledge-source.

    Returns TypeQL put statements as a string.
    """
    import shlex
    import re

    def label_to_var(label: str) -> str:
        return "$" + label.replace(":", "-")

    def title_to_var(title: str) -> str:
        import unicodedata
        ascii_title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
        return "$doc-" + re.sub(r'[^a-z0-9]+', '-', ascii_title.lower()).strip('-')

    def detect_value_type(value: str) -> tuple:
        """Given a parsed value (quotes already stripped by shlex), return (attr-type, typeql-value)."""
        if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            return "date-property", value
        if value.lower() in ('true', 'false'):
            return "boolean-property", value.lower()
        try:
            float(value)
            return "numeric-property", value
        except ValueError:
            pass
        stripped_value = value.strip("\"")
        return "string-property", f'"{stripped_value}"'

    def add_knowledge_source(node_var: str):
        nonlocal current_source
        if current_source is not None:
            stmts.append(f'put $_ isa meta-knowledge-source, links (knowledge: {node_var}, source: {current_source});')

    def embed_attr(var: str, label: str) -> str:
        """Return ', has embedding "..."' fragment if embed_fn is provided."""
        if embed_fn is None:
            return ""
        return f'put {var} has embedding "{embed_fn(label)}";'

    stmts = []
    current_source = None
    done_labels = set()
    for line in lines.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if line.startswith("source "):
            title = line[len("source "):].strip("\"")
            doc_var = title_to_var(title)
            stmts.append(f'put {doc_var} isa meta-document, has meta-page-title "{title}";') # Actually always matches
            current_source = doc_var
            continue

        tokens = shlex.split(line)
        kind = tokens[0]

        if kind == "entity":
            label = tokens[1]
            if label in done_labels:
                continue
            else:
                done_labels.add(label)

            var = label_to_var(label)
            stmts.append(f'put {var} isa entity-node, has node-label "{label}";')
            stmts.append(embed_attr(var, label))
            add_knowledge_source(var)

        elif kind == "property":
            ent_label = tokens[1]
            prop_label = ent_label + "::" + tokens[2]
            raw_value = tokens[3]
            if prop_label in done_labels:
                continue
            else:
                done_labels.add(prop_label)
            ent_var = label_to_var(ent_label)
            prop_var = label_to_var(prop_label)
            attr_type, typeql_value = detect_value_type(raw_value)

            # Property node
            stmts.append(f'put {prop_var} isa property-node, links (owner: {ent_var}), has node-label "{prop_label}", has {attr_type} {typeql_value};')
            stmts.append(embed_attr(prop_var, prop_label))
            
            add_knowledge_source(prop_var)

        elif kind == "relation":
            ent1_label = tokens[1]
            rel_label = tokens[2]
            ent2_label = tokens[3]
            if rel_label in done_labels:
                continue
            else:
                done_labels.add(rel_label)

            ent1_var = label_to_var(ent1_label)
            ent2_var = label_to_var(ent2_label)
            rel_var = label_to_var("r-" + rel_label)
            stmts.append(f'put {rel_var} isa relation-node, has node-label "{rel_label}", links (related: {ent1_var}, related: {ent2_var});')
            stmts.append(embed_attr(rel_var, rel_label))
            add_knowledge_source(rel_var)
    print("\n".join(stmts))
    return "\n".join(stmts)



def extract_lines(response: str) -> str:
    """Extract the simplified lines from an LLM response (strip markdown fences)."""
    return response.strip("` ")
    # text = response.strip()
    # start_marker = "```"
    # # Find last opening fence
    # start = text.rfind(start_marker)
    # if start == -1:
    #     return text
    # start += len(start_marker)
    # # Skip language tag on same line if present
    # newline = text.find("\n", start)
    # if newline != -1:
    #     start = newline + 1
    # # Find closing fence
    # end = text.find("```", start)
    # if end == -1:
    #     end = len(text)
    # print("Extracted the indices ", start, end)
    # return text[start:end].strip()


def construct_vanilla_kg(
    prompt_template: str,
    context: list[list],
    use_claude: bool = False,
    model: str = None,
    url: str = "http://localhost:8080/v1",
    max_tokens: int = 4096,
) -> str:
    """
    Generate simplified KG lines for all paragraphs in an example.

    Returns the raw lines (entity/property/relation format).
    """
    paragraphs = _format_paragraphs(context)
    prompt = prompt_template.format(paragraphs=paragraphs)

    if use_claude:
        model = model or "claude-sonnet-4-20250514"
        text = generate_query_claude(prompt, max_tokens, model)
    else:
        model = model or "default"
        text = generate_query_local(url, prompt, max_tokens, model)
    return extract_lines(text)


def main():
    parser = argparse.ArgumentParser(
        description="Construct vanilla TypeDB knowledge graph from paragraphs using an LLM"
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
        default="src/typedb_kgqa/prompts/vanilla_kg_construction.txt",
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
        help="Use Claude CLI for generation",
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
        help="Output file for intermediate lines (default: none, only writes to TypeDB)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate TypeQL but don't write to TypeDB",
    )

    args = parser.parse_args()

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
        print(f"Using Claude CLI (model: {model})", file=sys.stderr)
    else:
        model = args.model or "default"
        print(f"Using local server: {args.url} (model: {model})", file=sys.stderr)

    output_file = open(args.output, "w") if args.output else None
    total = len(source_lines)

    with TypeDB.driver(args.typedb_address, Credentials(args.typedb_username, args.typedb_password), DriverOptions(is_tls_enabled=False)) as driver:
        try:
            for i, titles in enumerate(source_lines, 1):
                print(f"[{i}/{total}] Processing: {', '.join(titles)}", file=sys.stderr)

                # Fetch documents
                with driver.transaction(args.database, TransactionType.READ) as tx:
                    context = fetch_documents(tx, titles)

                if not context:
                    print(f"  SKIP: no documents found", file=sys.stderr)
                    continue

                try:
                    # LLM extraction
                    kg_lines = construct_vanilla_kg(
                        prompt_template=prompt_template,
                        context=context,
                        use_claude=args.claude,
                        model=args.model,
                        url=args.url,
                        max_tokens=args.max_tokens,
                    )

                    # Write intermediate lines if output file specified
                    if output_file:
                        output_file.write(kg_lines)
                        output_file.write("\n\n")

                    # Convert to TypeQL and write to TypeDB
                    typeql = lines_to_typeql(kg_lines, lambda text: encode_embeddings_base64(get_embeddings_local(args.embedding_url, [text], False)[0]))
                    print(f"  Generated {typeql.count(chr(10)) + 1} put statements", file=sys.stderr)

                    if not args.dry_run:
                        with driver.transaction(args.database, TransactionType.WRITE) as tx:
                            tx.query(typeql).resolve()
                            tx.commit()
                        print(f"  Written to TypeDB", file=sys.stderr)
                    else:
                        print(typeql)

                except Exception as e:
                    print(f"  ERROR: {e}", file=sys.stderr)
                    raise e
        finally:
            if output_file:
                output_file.close()

    print(f"Done. Processed {total} examples.", file=sys.stderr)


if __name__ == "__main__":
    main()
