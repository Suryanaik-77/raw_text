from openai import OpenAI
import os
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pymilvus import connections, Collection

load_dotenv()

# Milvus connection
connections.connect(
    alias="default",
    host="localhost",
    port=19530
)

# OpenAI embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

def normalize(v):
    v = np.array(v, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()

def build_prompt(context, question):
    return f"""
🧠 Response Priority Rules

If CONTEXT is provided:
You must answer ONLY using information that appears explicitly in the CONTEXT.

If the CONTEXT does NOT provide enough information:
You must answer using your own general knowledge, but the first line must clearly be:
“Context insufficient — answering using general knowledge.”

Before giving the FINAL ANSWER:

Find the relationship between the CONTEXT.
Do NOT give direct sentences copied from the CONTEXT.
FIRST do analysis internally.

Verify whether the answer is outdated or newer.

If outdated → provide the newer and correct answer.

If both old and new practices/commands exist → provide final answer first and then alternatives.

If the user asks for the difference between two or more items:
The FINAL ANSWER must be in table format.

If the user makes spelling mistakes:
You must auto-correct silently and answer the intended question.

Do NOT mention context in output.

FORMAT TO FOLLOW ALWAYS:

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

def answer_from_milvus(
    query: str,
    top_k: int = 5,
    model: str = "gpt-4.1"
) -> str:
    """
    Core RAG function used by FastAPI and other modules.
    """
    client = OpenAI()

    collection = Collection(os.getenv("collection"))
    collection.load()

    query_vec = [normalize(embedding_model.embed_query(query))]

    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 8}},
        limit=top_k,
        output_fields=["text"]
    )

    context = "\n\n".join(
        hit.entity.get("text", "")
        for hit in results[0]
    )

    prompt = build_prompt(context, query)

    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=8000
    )

    return response.output[0].content[0].text

print(answer_from_milvus(
    "describes information messages related to SDC constraints.",
    3,
    "gpt-4.1"
))