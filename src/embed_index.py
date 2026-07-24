import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.chunk import build_all_chunks

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index_store")
os.makedirs(INDEX_DIR, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"


def build_index(chunk_size: int = 120, overlap: int = 20):
    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    chunks = build_all_chunks(chunk_size, overlap)
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    # IndexFlatIP = exact inner-product search. Since embeddings are normalized,
    # inner product is equivalent to cosine similarity. No approximation needed
    # at this scale (thousands of chunks) -- exact search is fast enough.
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print(f"Saved index with {index.ntotal} vectors to {INDEX_DIR}/")
    return index, chunks


if __name__ == "__main__":
    build_index()
