# TypeDB as a Knowledge Graph for Question Answering
We are exploring the use of TypeDB databases for Question Answering, evaluated on the 2WikiMultihopQA benchmark (data under `/data/2wikimultihopQA`).

## Two Pipelines

### TypeQL Pipeline (`typeql_krr/`)
The LLM generates TypeQL directly. Uses a rich typed schema (`base-schema.tql` + `2wmhqa.tql`).
- `typeql_krr/construction.py` — KG construction (LLM produces TypeQL put statements)
- `typeql_krr/generate_query.py` — Question answering (LLM produces TypeQL match queries)

### GraphRAG Pipeline (`graphrag/`)
The LLM outputs a simplified line format (entity/property/relation), which is converted to TypeQL. Uses a flat schema with `node-label` attributes (`graphrag-schema.tql`).
- `graphrag/construction.py` — KG construction (LLM produces lines, converted via `lines_to_typeql`)
- `graphrag/answer.py` — Question answering via RAG (embed question, retrieve docs, LLM answers)

## Shared
- `common.py` — LLM backends (`generate_query_local`, `generate_query_claude`), embeddings, TypeQL extraction
- `fetch_schema.py` — Fetch schema from TypeDB
- Prompts are under `src/typedb_kgqa/prompts/`
- Schemas are under `src/typedb_kgqa/schema/`
