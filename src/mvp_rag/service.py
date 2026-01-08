from __future__ import annotations

import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .question import answer_from_milvus


# -----------------------------
# Pydantic Models
# -----------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    top_k: int = Field(5, ge=1, le=10, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    question: str
    answer: str


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="MVP VLSI RAG",
    version="0.1.0"
)


# -----------------------------
# Health Check
# -----------------------------
@app.get("/healthz")
def healthcheck() -> dict:
    return {"status": "ok"}


# -----------------------------
# Query Endpoint
# -----------------------------
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    try:
        answer = answer_from_milvus(
            query=req.question,
            top_k=req.top_k
        )

        return QueryResponse(
            question=req.question,
            answer=answer
        )

    except Exception as exc:
        # 🔴 THIS IS CRITICAL — shows real error in docker logs
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


__all__ = ["app"]


# -----------------------------
# Local Run (Optional)
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mvp_rag.service:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
