import os

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient

app = FastAPI(title="Bootstrap Business Sage API")
qc = QdrantClient(os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_KEY"))
COLL = "bbsage_core"


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(q: Query):
    hits = qc.search(
        collection_name=COLL, query_vector=qc.get_embeddings(q.question), limit=3
    )
    return {"question": q.question, "hits": hits}
