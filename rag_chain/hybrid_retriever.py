from bm25_store import retrieve_bm25
from vector_store import retrieve_dense
from reranker import rerank
from config import (
    DENSE_TOP_K,
    SPARSE_TOP_K,
    FUSION_TOP_K,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    RRF_K,
    RETRIEVAL_MODE,
    FINAL_TOP_K
)


def chunk_key(doc):
    # chunk_id is the shared Chroma/BM25 id assigned at ingestion. (source, chunk_index)
    # alone is not unique once a file is indexed with more than one strategy.
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id:
        return chunk_id
    return (
        doc.metadata.get("source_document"),
        doc.metadata.get("chunking_strategy"),
        doc.metadata.get("chunk_index")
    )


def reciprocal_rank_fusion(dense_docs, sparse_docs, dense_weight=DENSE_WEIGHT,
                           sparse_weight=SPARSE_WEIGHT, rrf_k=RRF_K, top_k=FUSION_TOP_K):
    """Weighted RRF: score(d) = sum over lists of weight / (rrf_k + rank_in_list)."""
    scores = {}
    documents = {}

    for docs, weight in ((dense_docs, dense_weight), (sparse_docs, sparse_weight)):
        for rank, doc in enumerate(docs, start=1):
            key = chunk_key(doc)
            documents.setdefault(key, doc)
            scores[key] = scores.get(key, 0) + weight / (rrf_k + rank)

    ranked_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True
    )

    return [documents[key] for key in ranked_keys[:top_k]]


def retrieve_hybrid(query):
    dense_docs = retrieve_dense(query, DENSE_TOP_K)
    sparse_docs = retrieve_bm25(query, SPARSE_TOP_K)

    fused_docs = reciprocal_rank_fusion(
        dense_docs,
        sparse_docs
    )

    # Rerank only after fusion: up to FUSION_TOP_K candidates in, RERANK_TOP_K out.
    return rerank(query, fused_docs)


RETRIEVAL_MODES = ("dense", "hybrid", "hybrid_rerank")


def retrieve(query, mode=RETRIEVAL_MODE, top_k=FINAL_TOP_K):
    """Single entry point for the benchmarkable retrieval variants (A/B/C).

    Every mode returns at most `top_k` chunks so variants get the same context budget.
    "hybrid_rerank" is exactly retrieve_hybrid (the Phase 2 pipeline).
    """
    if mode == "dense":
        return retrieve_dense(query, top_k)
    if mode == "hybrid":
        fused = reciprocal_rank_fusion(
            retrieve_dense(query, DENSE_TOP_K),
            retrieve_bm25(query, SPARSE_TOP_K)
        )
        return fused[:top_k]
    if mode == "hybrid_rerank":
        return retrieve_hybrid(query)[:top_k]
    raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")
