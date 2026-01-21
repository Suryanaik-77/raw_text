from fastapi import FastAPI
from pydantic import BaseModel
from src.mvp_rag.question import answer_from_milvus
from src.mvp_rag.feedback_db import init_db, save_feedback
import os
import sqlite3
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.on_event("startup")
def startup_event():
    print("🚀 FastAPI startup: initializing feedback DB")
    init_db()
    print("✅ init_db() completed")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 8

class FeedbackRequest(BaseModel):
    user: str
    question: str
    answer: str
    feedback: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryRequest):
    answer = answer_from_milvus(req.question, req.top_k)
    return {"answer": answer}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    save_feedback(
        req.user,
        req.question,
        req.answer,
        req.feedback
    )
    return {"status": "saved"}

DB_PATH = "data/feedback.db"

@app.get("/feedback/view", response_class=HTMLResponse)
def view_feedback():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user, question, answer, feedback
        FROM feedback
        ORDER BY rowid DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    html = """
    <html>
    <head>
        <title>Feedback Viewer</title>
        <style>
            body { font-family: Arial; background: #0e0e0e; color: #e5e7eb; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #374151; padding: 8px; text-align: left; }
            th { background: #1f2933; }
            tr:nth-child(even) { background: #111827; }
        </style>
    </head>
    <body>
        <h2>Feedback Records</h2>
        <table>
            <tr>
                <th>User</th>
                <th>Question</th>
                <th>Answer</th>
                <th>Feedback</th>
            </tr>
    """

    for r in rows:
        html += f"""
        <tr>
            <td>{r[0]}</td>
            <td>{r[1]}</td>
            <td>{r[2]}</td>
            <td>{r[3]}</td>
        </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    return html
