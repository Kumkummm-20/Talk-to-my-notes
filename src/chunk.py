from src.ingest import load_all_documents


def chunk_text(text: str, source: str, chunk_size: int = 120, overlap: int = 20) -> list[dict]:
    """
    chunk_size: number of words per chunk
    overlap: number of words repeated between consecutive chunks (keeps context
             from getting cut off mid-idea)
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        chunks.append({
            "id": f"{source}::chunk_{chunk_id}",
            "source": source,
            "text": chunk_str,
        })
        chunk_id += 1
        start += chunk_size - overlap  # step forward, re-using the overlap
    return chunks


def build_all_chunks(chunk_size: int = 120, overlap: int = 20) -> list[dict]:
    documents = load_all_documents()
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc["text"], doc["source"], chunk_size, overlap))
    return all_chunks


if __name__ == "__main__":
    chunks = build_all_chunks()
    print(f"Created {len(chunks)} chunks.")
    print("\nExample chunk:")
    print(chunks[0])
