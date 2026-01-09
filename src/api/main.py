from fastapi import FastAPI
from src.mvp_rag.question import answer_from_milvus
from src.api.schema import QueryRequest, QueryResponse

app = FastAPI(title="MVP RAG API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    answer = answer_from_milvus(
        query=request.question,
        top_k=request.top_k
    )

    return {"answer": answer}

