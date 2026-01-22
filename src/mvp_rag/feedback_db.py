import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = "/app/data/feedback.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            question TEXT,
            answer TEXT,
            rating INTEGER DEFAULT 0,
            feedback TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_feedback(user, question, answer, rating, feedback,created_at):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (user, question, answer, rating, feedback, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user,
        question,
        answer,
        rating,
        feedback,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()
