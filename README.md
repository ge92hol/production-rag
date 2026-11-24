# Production RAG with Hybrid Search

A Retrieval-Augmented Generation pipeline over internal documents. It combines dense vector search with BM25 keyword search, reranks the results with a cross-encoder, and generates grounded answers with inline citations that are then checked against their sources.

## Features

- **Multi-format ingestion**: PDF, Markdown, HTML and plain text, normalised to text with metadata (source, section heading, page).
- **Chunking strategies**: `fixed`, `recursive` and `semantic`, with near-duplicate removal.
- **Hybrid retrieval**: ChromaDB (Gemini embeddings) plus BM25, fused with weighted Reciprocal Rank Fusion.
- **Reranking**: cross-encoder `ms-marco-MiniLM-L-6-v2`.
- **Corrective retrieval**: an LLM grades the evidence and, if it is weak, rewrites the query and retrieves once more.
- **Grounded generation**: answers carry inline citations, or the pipeline reports insufficient evidence.
- **Citation verification**: an LLM judge checks each claim against the chunk it cites.
- **Confidence scoring**: a composite of retrieval relevance, citation support and completeness.

## Architecture

```
ingest ─► chunk ─► embed ─► Chroma ─┐
                    └─► BM25 ───────┤
                                    ▼
question ─► dense + sparse ─► RRF fusion ─► rerank ─► evidence grading
                                                        │ (weak → rewrite & retry once)
                                                        ▼
                         grounded generation ─► citation verification ─► confidence
```

| Module | Role |
|---|---|
| `ingestion.py` | Load and normalise documents into `data/processed` |
| `chunking.py` | Chunking strategies |
| `vector_store.py` / `bm25_store.py` | Dense and sparse indexes |
| `hybrid_retriever.py`, `reranker.py` | Fusion and reranking (`dense`, `hybrid`, `hybrid_rerank`) |
| `corrective_retrieval.py` | Evidence grading and query rewriting |
| `generation.py`, `citation_verification.py`, `confidence.py` | Answering, verification, scoring |
| `pipeline.py` | End-to-end `answer_question` |
| `config.py` | All tunable settings (models, top-k, weights, thresholds) |
| `main.py` | CLI |

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r rag_chain/requirements.txt
```

The cross-encoder reranker model is downloaded from Hugging Face the first time it is used.

## Environment variables

Copy `.env.example` to `.env` in the project root and fill it in:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key, used for embeddings and all LLM calls |

The default rate limit (`LLM_REQUESTS_PER_MINUTE = 5` in `config.py`) suits the Gemini free tier.

## Usage

Run the commands from the `rag_chain/` directory:

```bash
cd rag_chain

# Ingest a document (pdf, md, html, txt)
python main.py ingest path/to/document.pdf --strategy semantic

# Rebuild both indexes from the processed documents
python main.py reindex --strategy recursive

# Inspect retrieval only (no LLM calls)
python main.py query

# Grounded answer with citations
python main.py ask "How do I request access to the staging environment?"
python main.py ask --mode hybrid --no-correction --json "..."
```

Indexes and processed data are written locally (`rag_chain/chroma_db/`, `rag_chain/bm25_index.pkl`, `rag_chain/data/`) and are not committed.
