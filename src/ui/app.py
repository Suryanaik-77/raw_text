import os
import requests
import streamlit as st
from pathlib import Path

API_URL = os.getenv("API_URL", "http://localhost:8000")
TOP_K = 8

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ANSWER_PATH = DATA_DIR / "answer.txt"

st.set_page_config(page_title="VLSI RAG Assistant", layout="centered")
st.title("💬 VLSI RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user" not in st.session_state:
    st.session_state.user = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

def backend_alive():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False

if not backend_alive():
    st.error("🚫 FastAPI backend is not running.")
    st.stop()

if st.session_state.user is None:
    name = st.text_input("Your name")
    if st.button("Start Chat") and name.strip():
        st.session_state.user = name.strip()
        st.rerun()
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question...")

# -------- QUERY --------
if prompt:
    payload = {"question": prompt, "top_k": TOP_K}

    response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
    response.raise_for_status()
    answer = response.json()["answer"]

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.session_state.last_question = prompt
    st.session_state.last_answer = answer

    with open(ANSWER_PATH, "a", encoding="utf-8") as f:
        f.write(f"\nQ: {prompt}\nA: {answer}\n{'='*80}\n")

    st.rerun()

# -------- FEEDBACK (OUTSIDE prompt) --------
if st.session_state.last_answer:
    st.markdown("---")
    st.subheader("📝 Feedback")

    with st.form("feedback_form", clear_on_submit=True):
        feedback = st.text_area("Was this answer helpful?")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            res = requests.post(
                f"{API_URL}/feedback",
                json={
                    "user": st.session_state.user,
                    "question": st.session_state.last_question,
                    "answer": st.session_state.last_answer,
                    "feedback": feedback
                },
                timeout=10
            )

            if res.status_code == 200:
                st.success("✅ Feedback saved")
            else:
                st.error("❌ Failed to save feedback")
