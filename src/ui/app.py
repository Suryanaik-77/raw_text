import os
import requests
import streamlit as st
import sqlite3
from pathlib import Path
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
API_URL = os.getenv("API_URL", "http://localhost:8000")
TOP_K = 15

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ANSWER_PATH = DATA_DIR / "answer.txt"
DB_PATH = DATA_DIR / "feedback.db"

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="VLSI RAG Assistant",
    layout="centered"
)

st.title("💬 VLSI RAG Assistant")

# =========================================================
# SESSION STATE INIT
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user" not in st.session_state:
    st.session_state.user = None

# =========================================================
# DATABASE SETUP
# =========================================================
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

# =========================================================
# BACKEND HEALTH CHECK
# =========================================================
def backend_alive():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

if not backend_alive():
    st.error("🚫 FastAPI backend is not running.\nPlease start the API server.")
    st.stop()

# =========================================================
# USER LOGIN
# =========================================================
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

# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================================
# CHAT INPUT
# =========================================================
prompt = st.chat_input("Ask a question...")

# =========================================================
# HANDLE USER MESSAGE
# =========================================================
if prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {"question": prompt, "top_k": TOP_K}

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

                # Show answer
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                # =================================================
                # WRITE ANSWER TO FILE (GUARANTEED)
                # =================================================
                with open(ANSWER_PATH, "a", encoding="utf-8") as f:
                    f.write(f"""
USER:
{prompt}

ANSWER:
{answer}

{'='*100}
""")

                st.caption(f"📝 Answer logged to: {ANSWER_PATH}")

                # =================================================
                # FEEDBACK FORM
                # =================================================
                st.markdown("---")
                st.subheader("📝 Feedback")

                with st.form(key=f"feedback_{len(st.session_state.messages)}"):
                    feedback = st.text_area(
                        "Was this answer helpful? Any comments?"
                    )
                    submitted = st.form_submit_button("Submit Feedback")

                    if submitted:
                        save_feedback(
                            st.session_state.user,
                            prompt,
                            answer,
                            feedback
                        )
                        st.success("✅ Feedback saved. Thank you!")

            except requests.exceptions.Timeout:
                st.error("⏱ Request timed out.")

            except requests.exceptions.ConnectionError:
                st.error("🚫 Lost connection to FastAPI.")

            except Exception as e:
                st.error(f"Unexpected error: {e}")
