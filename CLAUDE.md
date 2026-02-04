# TypeDB as a Knowledge Graph for Question Answering
We are exploring the use of a TypeDB databases for Question Answering.
We will be focussing on the 2WikiMultihopQA benchmark (The data is under `/data/2wikimultihopQA`)
The idea is as follows:
1. We are given the schema for our domain
2. We use a local-llm on llama-cpp to construct a knowledge graph from the paragraphs in the dataset. This is in the form of TypeQL write queries.
3. We use a local-llm on llama-cpp to generate a TypeQL query for a question in the dataset.

## Status when writing this document
1. We have our generated schema under the `typeql` folder.
2. This still has to be done. 
3. We have `generate_query.py` which is meant to be used with the `prompts/generate_query.txt`
