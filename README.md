# Talk to My Notes

A RAG (Retrieval-Augmented Generation) system built over my own college notes — OOPs, DBMS, SQL, Git, HTML, and Machine Learning. Beyond basic question-answering, this project focuses on two things most RAG demos skip: measuring whether retrieval is actually working, and checking whether generated answers are actually grounded in the source material.

Notes are indexed at build time (chunked, embedded, and stored in a vector index) rather than uploaded through the interface — this is a personal assistant over a fixed set of notes, not a multi-user upload tool.

**Live demo:** *[(deployed link)](https://talk-to-my-notes-rag.streamlit.app/)*
**Source code:** *[(GitHub link)](https://github.com/Kumkummm-20/Talk-to-my-notes)*

## Architecture

A query is embedded and matched against indexed note chunks using FAISS. The top matches are passed to an LLM to generate an answer, which is then checked by a separate model call for grounding before being shown to the user. The same retriever also feeds a labeled evaluation set used to measure retrieval quality independently of generation.

![Architecture](assests/architecture.jpg)

## Why this project is different from a basic RAG demo

Most RAG tutorials stop at "it retrieves something and generates an answer." Two problems with that: you have no way to know if retrieval is actually finding the right information, and you have no way to catch the model quietly answering from its own training data instead of your notes. This project addresses both:

- A retrieval evaluation harness that measures Hit Rate@k and MRR against a labeled question set
- A hallucination guard that runs a second, independent model call to verify every generated answer is supported by the retrieved context

## Evaluation results

Evaluated on a 50-question labeled set spanning all six note topics (DBMS, OOPs, Git, SQL, HTML, Machine Learning).

| Config | Hit Rate@k | MRR |
|---|---|---|
| chunk_size=120, overlap=20, k=5 | 0.933 (n=15) | 0.822 (n=15) |
| chunk_size=120, overlap=20, k=10 | 0.96 (n=50) | 0.763 (n=50) |

Two questions missed retrieval entirely out of 50.

One miss (INNER JOIN) was diagnosed: the correct chunk contains the term "INNER JOIN," but the chunk itself is dominated by unrelated sample-data tables, with the JOIN explanation only starting near the chunk boundary. This is a chunking granularity issue rather than a keyword or ranking problem — the chunk's embedding ends up representing the sample data more than the JOIN concept. A fix would be chunking on document section boundaries rather than fixed word counts.

A second miss (a DBMS abstraction-levels question) wasn't further diagnosed — the same chunking-boundary or phrasing-mismatch issue is the likely cause, but wasn't confirmed.

**Notable pattern:** SQL join-type questions (LEFT, RIGHT, FULL, CROSS, NATURAL JOIN) consistently ranked between 5-8, well below the average rank of 1-2 seen elsewhere. This wasn't a miss, but a clustering effect — the source document describes all join types in adjacent, similarly-worded sections ("returns rows that have matching values...", "returns all rows from..."), which makes their embeddings harder for the retriever to tell apart. A natural next step would be splitting join-type explanations into more distinct chunks, or testing whether a reranking step helps disambiguate near-duplicate embeddings.

## Limitations

**OCR on handwritten notes.** Several source PDFs are handwritten and had to go through OCR (Tesseract) rather than direct text extraction. Output quality varied a lot by handwriting clarity — one short file came out unusable and was manually retyped as a clean text file. A longer handwritten file was left out of the evaluation set rather than retyped by hand, and is flagged here as an open item. A vision-based transcription approach would likely handle this better than Tesseract for handwriting specifically.

**Hallucination guard is LLM-as-judge, not a trained classifier.** Using a second model call to check grounding is simple to implement and works well in practice, but it costs an extra API call per answer. A smaller, fine-tuned classifier would be cheaper to run at scale, though it would need labeled training data to build.

## Stack

| Component | Tool | Reason |
|---|---|---|
| Embeddings | sentence-transformers (MiniLM) | Small, fast, runs on CPU |
| Vector search | FAISS (IndexFlatIP) | Exact search, no approximation needed at this scale |
| Generation + grounding check | Groq API (Llama 3.3 70B) | Fast inference, free tier |
| OCR | Tesseract + pdf2image | Needed for handwritten note sources |
| Interface | Streamlit | Fast to build a usable UI without frontend code |
| Hosting | Streamlit Community Cloud | Free, deploys directly from GitHub |

## Setup

```bash
git clone <your-repo-url>
cd rag-eval-guard
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Get a Groq API key at https://console.groq.com/keys, then copy to `.env` and add it.

If you're on Windows and working with handwritten/scanned PDFs, you'll also need Tesseract OCR (https://github.com/UB-Mannheim/tesseract/wiki) and Poppler (https://github.com/oschwartz10612/poppler-windows/releases) installed and added to your system PATH.

## Pipeline

**1. Ingestion** (`src/ingest.py`) — reads every file in `data/`. For PDFs, tries direct text extraction first; if a page returns almost no text, treats it as a scanned image and runs OCR on it instead.
```bash
python -m src.ingest
```

**2. Chunking** (`src/chunk.py`) — splits documents into overlapping ~120-word chunks so ideas aren't cut off mid-sentence between chunks.
```bash
python -m src.chunk
```

**3. Embedding and indexing** (`src/embed_index.py`) — embeds every chunk with MiniLM and stores the vectors in a FAISS index.
```bash
python -m src.embed_index
```

**4. Retrieval** (`src/retrieve.py`) — embeds a query and returns the most similar chunks from the index. Used by both the app and the eval harness.
```bash
python -m src.retrieve
```

**5. Generation** (`src/generate.py`) — sends the retrieved chunks and question to Groq with an instruction to answer only from the given context.
```bash
python -m src.generate
```

**6. Evaluation** (`src/eval.py`) — runs a labeled question set through the retriever and computes Hit Rate@k and MRR.
```bash
python -m src.eval
```

**7. Hallucination guard** (`src/guard.py`) — an independent model call that checks whether the generated answer is actually supported by the retrieved context, flagging any unsupported claims.
```bash
python -m src.guard
```

**8. Application** (`app.py`) — combines all of the above into a Streamlit interface: ask a question, see the answer with a grounding indicator, inspect retrieved chunks, browse past questions from the sidebar, and re-run the evaluation harness on demand.
```bash
streamlit run app.py
```

## Deployment

1. Push this repo to GitHub (`.env` and `venv/` are already excluded via `.gitignore`)
2. On https://share.streamlit.io, connect the repo and set `app.py` as the entry point
3. Add `GROQ_API_KEY` under the app's Secrets settings
4. Deploy

## Command reference

```bash
python -m src.ingest          # load and OCR documents
python -m src.chunk           # split into chunks
python -m src.embed_index     # build the FAISS index
python -m src.retrieve        # test retrieval
python -m src.generate        # test generation
python -m src.eval            # run Hit Rate + MRR evaluation
python -m src.guard           # test the hallucination guard
streamlit run app.py          # launch the app
```
