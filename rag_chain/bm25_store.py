import os
import pickle
import re

from rank_bm25 import BM25Okapi
from config import BM25_PATH

# In-process cache so the BM25 index is not rebuilt on every query.
_cache = {"mtime": None, "chunks": [], "bm25": None}


def tokenize(text):
    # Used for both corpus and query. \w keeps identifiers like max_retries intact.
    return re.findall(r"\b\w+\b", text.lower())


def load_bm25_chunks():
    if not BM25_PATH.exists():
        return []

    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


def save_bm25_chunks(chunks):
    # Write atomically so a crash mid-write can't corrupt the corpus.
    tmp_path = BM25_PATH.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(chunks, f)
    os.replace(tmp_path, BM25_PATH)


def add_bm25_chunks(chunks):
    existing_chunks = load_bm25_chunks()
    existing_chunks.extend(chunks)
    save_bm25_chunks(existing_chunks)


def get_bm25_index():
    mtime = BM25_PATH.stat().st_mtime_ns if BM25_PATH.exists() else None

    if mtime != _cache["mtime"]:
        chunks = load_bm25_chunks()
        _cache["chunks"] = chunks
        _cache["bm25"] = BM25Okapi([tokenize(c.page_content) for c in chunks]) if chunks else None
        _cache["mtime"] = mtime

    return _cache["chunks"], _cache["bm25"]


def retrieve_bm25(query, k):
    chunks, bm25 = get_bm25_index()

    if bm25 is None:
        return []

    scores = bm25.get_scores(tokenize(query))

    indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    # Chunks with no matching terms (score 0) are not real sparse hits; don't let them into RRF.
    return [chunks[i] for i in indices if scores[i] > 0]
