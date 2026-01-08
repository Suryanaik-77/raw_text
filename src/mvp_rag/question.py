from __future__ import annotations

import os
import numpy as np
from dotenv import load_dotenv
from pymilvus import connections, Collection
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Load env vars (.env or Docker)
load_dotenv()


# -----------------------------
# Milvus Connection (LAZY)
# -----------------------------
def ensure_milvus():
    if not connections.has_connection("default"):
        connections.connect(
            host=os.getenv("MILVUS_HOST", "milvus"),
            port=os.getenv("MILVUS_PORT", "19530"),
        )


# -----------------------------
# Embedding Model
# -----------------------------
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
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

If the CONTEXT does NOT provide enough information:
You must answer using your own general knowledge, but the first line must clearly be:
“Context insufficient — answering using general knowledge.”

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
    model: str = "gpt-4.1"
) -> str:
    """
    Core RAG function used by FastAPI.
    """
    # Ensure Milvus is connected
    ensure_milvus()

    # Get collection name safely
    collection_name = os.getenv("MILVUS_COLLECTION", "vlsi_docs")
    collection = Collection(collection_name)
    collection.load()

    # Embed query
    query_vec = [normalize(embedding_model.embed_query(query))]

    # Search Milvus
    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 8}},
        limit=top_k,
        output_fields=["text"]
    )

    # Build context
    context = "\n\n".join(
        hit.entity.get("text", "")
        for hit in results[0]
    )

    # Build prompt
    prompt = build_prompt(context, query)

    # LLM call
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=800
    )

    return response.output[0].content[0].text
