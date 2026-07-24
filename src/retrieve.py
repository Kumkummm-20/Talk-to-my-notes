import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index_store")
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_index = None
_chunks = None


def _load():
    global _model, _index, _chunks
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        _index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    if _chunks is None:
        with open(os.path.join(INDEX_DIR, "chunks.json"), "r", encoding="utf-8") as f:
            _chunks = json.load(f)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Returns top-k chunks as:
    [{"id": ..., "source": ..., "text": ..., "score": float}, ...]
    sorted by score descending (higher = more similar).
    """
    _load()
    query_vec = _model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = _chunks[idx]
        results.append({**chunk, "score": float(score)})
    return results


if __name__ == "__main__":
    test_questions = [
        "What is encapsulation in OOPs?",
        "What is abstraction in OOPs?",
        "What is inheritance in OOPs?",
        "What is a primary key in SQL?",
        "What is a foreign key in SQL?",
        "What is a table in SQL?",
        "What does git commit do?",
        "What is the difference between git merge and git rebase?",
    ]

    for question in test_questions:
        print("=" * 60)
        print("Q:", question)
        results = retrieve(question, k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['id']}")
            print("   ", r["text"][:150].replace("\n", " "), "...")
        print()