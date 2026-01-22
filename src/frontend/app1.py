from __future__ import annotations

import os
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pymilvus import connections, Collection
from pymilvus.exceptions import MilvusException
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

COLLECTION_NAME = "ds"
TOP_K = 16
ANSWER_FILE = "rag_history.txt"

_COLLECTION = None
_EMBEDDINGS = None

st.set_page_config(page_title="VLSI RAG Assistant", layout="centered")
st.title("💬 VLSI RAG Assistant (Strict Context-Only RAG)")

def get_collection():
    global _COLLECTION
    if _COLLECTION is None:
        connections.connect(
            alias="default",
            host="localhost",
            port="19530",
            timeout=30,
        )
        _COLLECTION = Collection(COLLECTION_NAME)
        _COLLECTION.load()
    return _COLLECTION

def get_embedding_model():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _EMBEDDINGS

def normalize(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v.tolist() if n == 0 else (v / n).tolist()

def build_prompt(context, question):
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

2. If the CONTEXT does NOT contain sufficient information to directly and completely answer the QUESTION:
   - Output EXACTLY:
     Context insufficient
   - Do NOT provide partial answers
   - Do NOT explain
   - Do NOT add any additional text
   - if provided context is not sufficient don't answer to question

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
"""

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4.1",
        temperature=0.0,
    )

def answer_from_milvus(query, top_k=TOP_K):
    collection = get_collection()
    embedding_model = get_embedding_model()

    query_vec = [normalize(embedding_model.embed_query(query))]

    for _ in range(2):
        try:
            results = collection.search(
                data=query_vec,
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"nprobe": 8}},
                limit=top_k,
                output_fields=["text"],
            )
            break
        except MilvusException:
            time.sleep(1)
    else:
        return "Search engine temporarily unavailable. Please retry."

    if not results or not results[0]:
        return "Context insufficient"

    context = "\n\n".join(
        hit.entity.get("text", "")
        for hit in results[0]
    )

    if not context.strip():
        return "Context insufficient"

    prompt = build_prompt(context, query)
    llm = get_llm()
    response = llm.invoke(prompt)
    answer = response.content.strip()

    return answer if answer else "Context insufficient"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a VLSI / Design Compiler question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            answer = answer_from_milvus(user_input)
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.session_state.question_count += 1
    q_no = st.session_state.question_count

    with open(ANSWER_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"""{q_no}. QUESTION:
{user_input}

ANSWER:
{answer}

{'=' * 120}

"""
        )
