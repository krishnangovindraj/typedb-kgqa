# TypeDB as a Knowledge Graph for Question Answering

(These are my notes where I write down my thoughts so I don't lose track. One written for you is coming soon)

The project is meant to test TypeDB as a knowledge graph for a question-answering system.
Most QA tasks for LLMs seem to have two phases:
1. Use some retrieval method (RAG, Graph-RAG) to retrieve the relevant text (documents or paragraphs)
2. Feed all of this to the LLM along with the question to get the answer.

Obviously, TypeDB can be used as a graph to do vanilla Graph-RAG. 
But a vanilla graph database can do vanilla Graph-RAG.
TypeDB is much more than a graph database. 
Its polymorphic type-system and near-natural query language TypeQL 
make it an easy choice for knowledge representation.

There are two settings for QA. The open-domain and closed-domain.
As TypeDB was built for ensuring data integrity with a well-specified schema, 
it is best suited for modelling the closed-world described by the schema.

We are playing with these ideas.

# Current work: 2WikiMultihopQA
We are currently trying to do QA on the [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) dataset. 
Though wikipedia is an open domain, the questions in this one seem to come from a closed domain.
We thus emulate a closed domain by looking at the dev set, and generating a schema (using Claude) based on these questions. We handwrote a `base-schema.tql` to use as a base for the schema generation. The generated schema is `2wmhqa.tql` These are in the `typeql` folder.

We then must construct a knowledge graph in TypeDB from the text corpus, based on the generated schema.
We assume we're only interested in the information captured by the schema.
We then generated a TypeQL query corresponding to the question and hope that it answers the question.

## LLM?
I tried using currently running a local LLM on llama-cpp for generating the TypeQL for both KG construction and question-answering.
The model I used was `TheBloke/deepseek-coder-33B-instruct-GGUF:Q4_K_M` from huggingface. 
It didn't perform nearly as well as Claude did, so I switched to using Claude via the CLI instead. 
I trust Claude isn't peeping at the other files in the dataset while generating the TypeQL.
The prompts used can be found in the `prompts` folder.
It does generate invalid TypeQL once in a while because I switched to a DSL


## Different settings: KG per example, and altogether.
As noted earlier , there's a retrieval and QA phase to this benchmark. 
In the dataset, the retrieval is already done and the relevant paragraphs are provided along with the question and expected answer in the dataset. Only the second phase (answering) is tested.
TypeQL should be able to do both phases. I would like to explore different settings.

1. Construct a fresh TypeDB database for each example, based only on the text provided with the example. 
2. Constructing a single TypeDB database from the text of all examples combined.
