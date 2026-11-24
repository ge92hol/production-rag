"""Day 6 — answer confidence as visible components plus an explicit weighted composite.

Components (each 0..1, or None when not measurable for this run):
  retrieval         mean sigmoid(cross-encoder logit) over the top RETRIEVAL_CONFIDENCE_TOP_N
                    chunks. None for retrieval modes without the reranker.
  citation_support  mean over claims of {supported: 1, partially_supported: 0.5, else: 0},
                    using the LLM-judge verdicts. None if the answer has no claims.
  completeness      mean over question parts of {answered: 1, partially_answered: 0.5,
                    not_answered: 0}. Self-reported by the generator (not judged).

composite = sum(w_i * c_i) / sum(w_i) over components that are not None,
with w_i = CONFIDENCE_WEIGHTS (config.py).
"""
import math

from config import CONFIDENCE_WEIGHTS, RETRIEVAL_CONFIDENCE_TOP_N

CLAIM_SCORES = {"supported": 1.0, "partially_supported": 0.5}
PART_SCORES = {"answered": 1.0, "partially_answered": 0.5, "not_answered": 0.0}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def retrieval_confidence(documents, top_n=RETRIEVAL_CONFIDENCE_TOP_N):
    scores = [doc.metadata.get("rerank_score") for doc in documents[:top_n]]
    if not scores or any(score is None for score in scores):
        return None
    return sum(sigmoid(score) for score in scores) / len(scores)


def citation_support(verification):
    claims = verification["claims"] if verification else []
    if not claims:
        return None
    return sum(CLAIM_SCORES.get(c["status"], 0.0) for c in claims) / len(claims)


def completeness(answer):
    if answer is None or not answer.parts:
        return None
    return sum(PART_SCORES[p.status] for p in answer.parts) / len(answer.parts)


def composite(components, weights=CONFIDENCE_WEIGHTS):
    available = {name: value for name, value in components.items() if value is not None}
    total_weight = sum(weights[name] for name in available)
    if not total_weight:
        return None
    return sum(weights[name] * value for name, value in available.items()) / total_weight


def score_confidence(documents, answer=None, verification=None):
    components = {
        "retrieval": retrieval_confidence(documents),
        "citation_support": citation_support(verification),
        "completeness": completeness(answer),
    }
    rounded = {k: None if v is None else round(v, 3) for k, v in components.items()}
    score = composite(components)
    return {
        "components": rounded,
        "composite": None if score is None else round(score, 3),
        "weights": dict(CONFIDENCE_WEIGHTS),
    }
