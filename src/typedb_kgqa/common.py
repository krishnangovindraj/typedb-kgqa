from typing import List

def extract_typeql(response: str) -> str:
    """Extract TypeQL from an LLM response by stripping markdown fences and surrounding text."""
    from sys import stderr
    text = response.strip()
    # print(f"EXTRACT FROM:\n---\n{text}\n---\n", file=stderr)
    start_marker = "```typeql"
    end_marker = "```"
    typeql_start = text.rfind(start_marker) + len(start_marker)
    typeql_end = text.rfind(end_marker, typeql_start)
    typeql_end = typeql_end if typeql_end > -1 else len(text)
    # print(f"Get [{typeql_start}, {typeql_end}] =:\n<<\n{text[typeql_start:typeql_end]}\n>>", file=stderr)
    assert typeql_start != -1 and typeql_end != -1
    extracted = text[typeql_start:typeql_end]
    return extracted + "end;"

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
    """Generate using Claude CLI (claude -p via stdin)."""
    import subprocess

    args = ["claude", "--model", model, "-p", "\"Output only the completion and nothing else\""]
    result = subprocess.run(
        args,
        input=prompt,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {result.stderr}")
    return result.stdout

def get_embeddings_local(base_url: str, texts: List[str], is_query=False) -> List[float]:
    from openai import OpenAI

    # 1. Initialize client pointing to your local llama.cpp server
    client = OpenAI(
        base_url=base_url,
        api_key="sk-no-key-required"         # llama.cpp usually doesn't require a key
    )

    # Qwen3-8B requirement: Prefix queries for better retrieval performance
    if is_query:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
        processed_input = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    else:
        processed_input = texts

    # 2. Call the standard OpenAI embedding method
    response = client.embeddings.create(
        model="qwen3-embedding-8b", # This name is often ignored by llama.cpp but required by SDK
        input=processed_input,
    )
    
    return [item.embedding for item in response.data]

def encode_embeddings_base64(floats: List[float]) -> str: 
    import base64, struct, functools
    as_bytes = (struct.pack('<f', f) for f in floats)
    as_bytes_concatenated = functools.reduce(lambda x,y: x+y, as_bytes,b'')
    return base64.b64encode(as_bytes_concatenated)
