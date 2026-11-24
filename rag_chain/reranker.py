from functools import lru_cache

from langchain_core.documents import Document

from config import RERANKER_MODEL, RERANK_TOP_K


@lru_cache(maxsize=1)
def get_reranker():
    # Lazy load: importing this module (e.g. during ingest) shouldn't load torch/the model.
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL)


def rerank(query, documents, top_k=RERANK_TOP_K):
    if not documents:
        return []

    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]

    scores = get_reranker().predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda pair: pair[1],
        reverse=True
    )

    # Copy before annotating: BM25 hits are shared objects from the in-process cache,
    # so writing the score onto them would leak across queries.
    return [
        Document(page_content=doc.page_content,
                 metadata={**doc.metadata, "rerank_score": float(score)})
        for doc, score in ranked[:top_k]
    ]
