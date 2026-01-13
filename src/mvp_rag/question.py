from __future__ import annotations

import os
import numpy as np
from dotenv import load_dotenv
from pymilvus import connections, Collection
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

print("MILVUS_HOST =", os.getenv("MILVUS_HOST"))
print("MILVUS_PORT =", os.getenv("MILVUS_PORT"))


# -----------------------------
# Milvus Connection (LAZY)
# -----------------------------
def ensure_milvus():
    host = os.getenv("MILVUS_HOST", "milvus")
    port = os.getenv("MILVUS_PORT", "19530")

    try:
        connections.disconnect("default")
    except Exception:
        pass

    connections.connect(
        alias="default",
        host=host,
        port=port,
        timeout=30,
    )


# -----------------------------
# Embedding Model (LAZY)
# -----------------------------
def get_embedding_model() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=api_key,
    )


def normalize(v):
    v = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()


# -----------------------------
# Prompt Builder
# -----------------------------
def build_prompt(context: str, question: str) -> str:
    return f"""
🧠 Response Priority Rules

If CONTEXT is provided:
You must answer ONLY using information that appears explicitly in the CONTEXT.

if CONTEXT is not provided don't give answer instead give - context is insuffienct

Do NOT mention context in output.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""


# -----------------------------
# Core RAG Function
# -----------------------------
def answer_from_milvus(
    query: str,
    top_k: int = 5,
    model: str = "gpt-4.1",
) -> str:
    """
    Core RAG function used by FastAPI.
    """
    ensure_milvus()

    collection_name = os.getenv("MILVUS_COLLECTION1", "vlsi_docs")
    print(f"📦 Using Milvus collection: {collection_name}")
    collection = Collection(collection_name)
    collection.load()

    embedding_model = get_embedding_model()
    query_vec = [normalize(embedding_model.embed_query(query))]

    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 8}},
        limit=top_k,
        output_fields=["text"],
    )

    context = "\n\n".join(
        hit.entity.get("text", "")
        for hit in results[0]
    )

    prompt = build_prompt(context, query)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=800,
    )

    return response.output[0].content[0].text
