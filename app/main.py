from fastapi import FastAPI
from pydantic import BaseModel
from scripts.rag_api import rag_query

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def ask(request: QueryRequest):
    return rag_query(request.question)

@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
    
