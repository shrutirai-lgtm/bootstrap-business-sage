import os
from pathlib import Path

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).resolve().parents[2] / "data" / "raw"
COLL = "bbsage_core"
EMB = "sentence-transformers/all-mpnet-base-v2"


def main() -> None:
    docs = []
    for pdf in DATA.glob("*.pdf"):
        docs += PyPDFLoader(str(pdf)).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    model = SentenceTransformer(EMB)
    vectors = [model.encode(c.page_content) for c in chunks]

    qc = QdrantClient(os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_KEY"))
    if COLL not in [c.name for c in qc.get_collections().collections]:
        qc.recreate_collection(
            COLL,
            vectors_config=models.VectorParams(
                size=768, distance=models.Distance.COSINE
            ),
        )
    qc.upload_collection(
        collection_name=COLL,
        ids=list(range(len(vectors))),
        vectors=vectors,
        payload=[c.metadata for c in chunks],
        batch_size=64,
    )
    print(f"Uploaded {len(chunks)} chunks → {COLL}")


if __name__ == "__main__":
    main()
