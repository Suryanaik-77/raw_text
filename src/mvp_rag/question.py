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
You are operating in STRICT CONTEXT-ONLY MODE.

The CONTEXT provided below is the ONLY source of truth.
Treat it as a CLOSED WORLD.

You are strictly prohibited from using any external knowledge,
training data, assumptions, or general understanding.

==============================
RESPONSE PRIORITY RULES
==============================

1. If the CONTEXT contains sufficient information:
   - Answer using ONLY information that appears explicitly
     or can be logically inferred directly from the CONTEXT.
   - You may analyze relationships BETWEEN context chunks.
   - You MUST NOT introduce any concepts, terms, or details
     that do not appear in the CONTEXT.

2. If the CONTEXT does NOT contain sufficient information
   to directly and completely answer the QUESTION:
   - Output EXACTLY:
     Context insufficient
   - Do NOT provide partial answers
   - Do NOT explain
   - Do NOT add any additional text

3. You MUST NOT copy sentences verbatim from the CONTEXT.
   - Paraphrase using your own wording only.
   - Preserve the exact technical meaning.
   - Do NOT generalize or expand beyond the CONTEXT.

4. Perform internal analysis BEFORE answering.
   - Correlate information across context chunks if needed.
   - Resolve differences ONLY using evidence from the CONTEXT.
   - NEVER mention or reveal analysis in the output.

5. If both newer and older practices appear in the CONTEXT:
   - Present the newer practice FIRST.
   - Then mention older alternatives.
   - If no temporal order is stated, do NOT infer one.

6. If the QUESTION asks for differences or comparisons:
   - The FINAL ANSWER MUST be in TABLE FORMAT.
   - Any non-table answer is INVALID.

7. Silently correct spelling or grammar mistakes in the QUESTION.
   - Do NOT mention the correction.
   - Do NOT reinterpret intent.

==============================
MANDATORY OUTPUT FORMAT
==============================

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:

if final answer is related to the command, that command must be from the provided context only other answer 'Context Insufficient'
"""


# -----------------------------
# Core RAG Function
# -----------------------------
def answer_from_milvus(
    query: str,
    top_k: int = 20,
    model: str = "gpt-4.1",
) -> str:
    """
    Core RAG function used by FastAPI.
    """
    ensure_milvus()

    collection_name = os.getenv("MILVUS_COLLECTION", "vlsi_docs")
    print(f"📦 Using Milvus collection: {collection_name}")
    collection = Collection(collection_name)
    collection.load()

    embedding_model = get_embedding_model()
    query_vec = [normalize(embedding_model.embed_query(query))]

    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 8}},
        limit=20,
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
        max_output_tokens=8000,
    )

    return response.output[0].content[0].text
