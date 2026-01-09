import os
import requests
import streamlit as st
import sqlite3
from pathlib import Path
from datetime import datetime

# -----------------------------
# Config
# -----------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")
TOP_K = 15

BASE_DIR = Path(__file__).resolve().parents[2]   # project root
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "feedback.db"

st.set_page_config(
    page_title="VLSI RAG Assistant",
    layout="centered"
)

st.title("💬 VLSI RAG Assistant")

# -----------------------------
# Database Setup
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            question TEXT,
            answer TEXT,
            feedback TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_feedback(user, question, answer, feedback):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO feedback (user, question, answer, feedback, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (user, question, answer, feedback, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


init_db()

# -----------------------------
# Backend Health Check
# -----------------------------
def check_backend():
    try:
        res = requests.get(f"{API_URL}/health", timeout=3)
        return res.status_code == 200
    except Exception:
        return False


if not check_backend():
    st.error("🚫 FastAPI backend is not running.\n\nPlease start the API server first.")
    st.stop()

# -----------------------------
# User Login (Name)
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.subheader("👤 Enter your name to start")
    name = st.text_input("Your name")

    if st.button("Start Chat"):
        if name.strip():
            st.session_state.user = name.strip()
            st.session_state.messages = []
            st.rerun()
        else:
            st.warning("Please enter your name.")
    st.stop()

# -----------------------------
# Chat State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display Chat History
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Chat Input
# -----------------------------
prompt = st.chat_input("Ask a question...")

# -----------------------------
# Handle New Message
# -----------------------------
if prompt:
    # Store user question
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "question": prompt,
        "top_k": TOP_K
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                answer = response.json()["answer"]

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                # -----------------------------
                # Feedback Form
                # -----------------------------
                st.markdown("---")
                st.subheader("📝 Feedback")

                with st.form(key=f"feedback_{len(st.session_state.messages)}"):
                    feedback = st.text_area(
                        "Was this answer helpful? Any comments?",
                        placeholder="Write your feedback here..."
                    )
                    submitted = st.form_submit_button("Submit Feedback")

                    if submitted:
                        save_feedback(
                            user=st.session_state.user,
                            question=prompt,
                            answer=answer,
                            feedback=feedback
                        )
                        st.success("✅ Feedback saved. Thank you!")

            except requests.exceptions.Timeout:
                st.error("⏱ Request timed out. Please try again.")

            except requests.exceptions.ConnectionError:
                st.error("🚫 Lost connection to FastAPI server.")

            except Exception as e:
                st.error(f"Unexpected error: {e}")
