import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
TOP_K = 15

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

if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False

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

if prompt:
    payload = {"question": prompt, "top_k": TOP_K}
    try:
        response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
        response.raise_for_status()

        resp_json = response.json()
        answer = resp_json["answer"]
        chunks = resp_json.get("chunks", [])

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.session_state.last_question = prompt
        st.session_state.last_answer = answer
        st.session_state.last_chunks = chunks
        st.session_state.show_feedback = True

        st.rerun()

    except requests.exceptions.Timeout:
        st.error("⏱ Request timed out.")
    except requests.exceptions.ConnectionError:
        st.error("🚫 Backend connection lost.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

if st.session_state.last_chunks:
    st.markdown("---")
    st.subheader("📚 Retrieved Chunks")
    with st.expander("Show retrieved context"):
        for i, ch in enumerate(st.session_state.last_chunks, 1):
            st.markdown(f"**Chunk {i}** | score: `{ch['score']:.2f}`")
            st.code(ch["text"], language="text")

if st.session_state.show_feedback and st.session_state.last_answer:
    st.markdown("---")
    st.subheader("📝 Feedback")

    with st.form("feedback_form", clear_on_submit=True):
        rating = st.radio(
            "Rate this answer",
            options=[0, 1, 2, 3, 4, 5],
            format_func=lambda x: "⭐" * x if x > 0 else "No rating",
            horizontal=True
        )

        feedback = st.text_area(
            "Is answer correct/incorrect/partially correct ?",
            placeholder="Write your feedback here..."
        )

        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            res = requests.post(
                f"{API_URL}/feedback",
                json={
                    "user": st.session_state.user,
                    "question": st.session_state.last_question,
                    "answer": st.session_state.last_answer,
                    "rating":rating,
                    "feedback": feedback

                },
                timeout=10
            )

            if res.status_code == 200:
                st.success("✅ Feedback saved")
                st.session_state.show_feedback = False
            else:
                st.error("❌ Failed to save feedback")
