import sqlite3
from datetime import datetime
import os

BASE_DIR = Path(__file__).resolve().parents[2]

# Create a writable data directory
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "feedback.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            feedback TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_feedback(question, answer, feedback):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO feedback (question, answer, feedback, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        question,
        answer,
        feedback,
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()
