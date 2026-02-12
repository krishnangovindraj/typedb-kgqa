# TypeDB Knowledge Graph for Question Answering

Using TypeDB as a knowledge graph backend for multi-hop question answering, evaluated on the [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) benchmark.

## Pipeline

1. **Define schema** -- TypeQL schema files are under `src/typedb_kgqa/typeql/`.
2. **Ingest documents** -- Load raw paragraphs from the dataset into TypeDB as `meta-document` entities.
3. **Construct KG** -- Use an LLM to extract entities and relations from the documents and generate TypeQL `put` statements.
4. **Query** -- Use an LLM to translate natural language questions into TypeQL `match` queries.

## Dataset

We use the [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) benchmark. Place dataset files (e.g. `dev.json`) under `data/2wikimultihopqa/`.

### Extracting questions and sources

`helpers/2wikimultihopqa/extract.py` extracts questions or source page titles from the dataset:

```bash
# Extract questions (one per line)
python helpers/2wikimultihopqa/extract.py data/2wikimultihopqa/dev.json questions -o questions.txt

# Extract source page titles (one JSON list per line)
python helpers/2wikimultihopqa/extract.py data/2wikimultihopqa/dev.json sources -o sources.jsonl

# Sample 10 random examples
python helpers/2wikimultihopqa/extract.py data/2wikimultihopqa/dev.json questions -n 10 --seed 42
```

## Scripts

All scripts under `src/typedb_kgqa/` are run as modules. They connect to a TypeDB instance and support `-o` for output (defaults to stdout).

### Fetch schema

Fetch and print the schema from a TypeDB database:

```bash
python -m typedb_kgqa.fetch_schema -d my_database
python -m typedb_kgqa.fetch_schema -d my_database --compact
```

### Ingest documents

Load paragraphs from a dataset JSON file into TypeDB as `meta-document` entities (with `meta-page-title` and `text-content`). Deduplicates by title.

```bash
python -m typedb_kgqa.ingest_documents --dataset data/2wikimultihopqa/dev.json -d my_database
```

### Construct knowledge graph

Read a sources file (produced by `extract.py sources`), fetch document text from TypeDB, and use an LLM to generate TypeQL `put` statements:

```bash
# Using Claude CLI
python -m typedb_kgqa.kg_construction --sources sources.jsonl -d my_database --claude -o inserts.tql

# Using local llama-cpp server
python -m typedb_kgqa.kg_construction --sources sources.jsonl -d my_database --url http://localhost:8080/v1 -o inserts.tql
```

The prompt template is at `src/typedb_kgqa/prompts/kg_construction.txt`.

### Generate queries

Read a questions file (produced by `extract.py questions`) and use an LLM to generate TypeQL `match` queries:

```bash
# Using Claude CLI
python -m typedb_kgqa.generate_query -q questions.txt -p src/typedb_kgqa/prompts/generate_query.txt -d my_database --claude -o queries.tql

# Using local llama-cpp server
python -m typedb_kgqa.generate_query -q questions.txt -p src/typedb_kgqa/prompts/generate_query.txt -d my_database -o queries.tql
```

## LLM backends

All LLM-calling scripts support two backends:

- **Claude CLI** (`--claude`): Pipes prompts to `claude -p` via stdin. Requires Claude Code CLI to be installed.
- **Local llama-cpp** (default): Calls a local OpenAI-compatible completion endpoint (`--url`, default `http://localhost:8080/v1`).
