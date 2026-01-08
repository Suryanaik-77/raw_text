import streamlit as st
import requests
import os
from feedback_db import init_db, save_feedback

BASE_API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL = f"{BASE_API_URL}/query"

st.set_page_config(page_title="VLSI RAG Chat", layout="centered")
st.title("💬 VLSI RAG Assistant")

init_db()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a VLSI-related question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    API_URL,
                    json={"question": prompt, "top_k": 3},
                    timeout=120
                )
                res.raise_for_status()
                answer = res.json()["answer"]
            except Exception as e:
                answer = f"Error: {e}"

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.session_state.last_question = prompt
    st.session_state.last_answer = answer

# Feedback
if st.session_state.last_answer:
    st.markdown("### Was this answer helpful?")

    col1, col2, col3 = st.columns(3)

    if col1.button("✅ Correct"):
        save_feedback(
            st.session_state.last_question,
            st.session_state.last_answer,
            "correct"
        )
        st.success("Feedback saved: Correct")

    if col2.button("🟡 Partially Correct"):
        save_feedback(
            st.session_state.last_question,
            st.session_state.last_answer,
            "partially_correct"
        )
        st.warning("Feedback saved: Partially Correct")

    if col3.button("❌ Incorrect"):
        save_feedback(
            st.session_state.last_question,
            st.session_state.last_answer,
            "incorrect"
        )
        st.error("Feedback saved: Incorrect")
