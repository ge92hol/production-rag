"""End-to-end question answering: retrieval (+ optional correction) -> grounded
generation -> citation verification -> confidence.

Every stage is switchable through arguments (defaults come from config.py), so the
Day 8 variants are just different calls:
  A. answer_question(q, retrieval_mode="dense",         correction=False)
  B. answer_question(q, retrieval_mode="hybrid",        correction=False)
  C. answer_question(q, retrieval_mode="hybrid_rerank", correction=False)
  D. answer_question(q, retrieval_mode="hybrid_rerank", correction=True)
Returns a JSON-serialisable dict (no API concerns here).
"""
import logging
from functools import partial

from citation_verification import verify_citations
from config import CORRECTION_ENABLED, MAX_CORRECTION_RETRIES, MIN_RETRIEVAL_CONFIDENCE, RETRIEVAL_MODE
from confidence import retrieval_confidence, score_confidence
from corrective_retrieval import corrective_retrieve
from generation import generate_answer, render_answer
from hybrid_retriever import retrieve
from llm import chunk_reference

logger = logging.getLogger(__name__)

INSUFFICIENT_ANSWER = ("I don't know: the available documents do not contain enough "
                       "evidence to answer this question.")


def run_retrieval(question, retrieval_mode, correction, max_retries):
    retriever = partial(retrieve, mode=retrieval_mode)
    if correction:
        result = corrective_retrieve(question, retriever, max_retries)
    else:
        result = {"documents": retriever(question), "grade": None,
                  "correction": {"enabled": False, "triggered": False, "original_query": question,
                                 "rewritten_query": None, "retry_count": 0}}
    info = result["correction"]
    if info["triggered"]:
        logger.info("correction triggered (%s): %r -> %r, retries=%d",
                    info["trigger_reason"], question, info["rewritten_query"], info["retry_count"])
    return result


def documents_to_check(sources, refs=None):
    """Distinct source documents worth checking by hand, in retrieval-rank order."""
    picked = [s for s in sources if refs is None or s["ref"] in refs] or sources
    seen, out = set(), []
    for source in picked:
        key = (source["source_document"], source["section_heading"], source["page_number"])
        if key not in seen:
            seen.add(key)
            out.append({k: source[k] for k in
                        ("ref", "source_document", "section_heading", "page_number", "chunk_id")})
    return out


def insufficient_response(decided_by, reason, evidence_found, not_established, sources, relevant_refs=None):
    answer = INSUFFICIENT_ANSWER
    if evidence_found:
        answer += f" Related evidence found: {evidence_found}"
    return {
        "status": "insufficient_evidence",
        "answer": answer,
        "insufficient_evidence": {
            "decided_by": decided_by,  # "retrieval" | "grader" | "generator"
            "reason": reason,
            "evidence_found": evidence_found or None,
            "not_established": not_established,
            "documents_to_check": documents_to_check(sources, relevant_refs),
        },
    }


def answer_question(question, retrieval_mode=RETRIEVAL_MODE, correction=CORRECTION_ENABLED,
                    max_retries=MAX_CORRECTION_RETRIES, verify=True,
                    min_retrieval_confidence=MIN_RETRIEVAL_CONFIDENCE):
    retrieval = run_retrieval(question, retrieval_mode, correction, max_retries)
    documents = retrieval["documents"]
    grade = retrieval["grade"]
    sources = [chunk_reference(i, doc) for i, doc in enumerate(documents, start=1)]

    result = {
        "question": question,
        "status": None,
        "answer": None,
        "evidence_status": None,
        "parts": [],
        "verification": None,
        "confidence": None,
        "insufficient_evidence": None,
        "sources": sources,
        "retrieval": {"mode": retrieval_mode, "grade": grade, "correction": retrieval["correction"]},
    }

    # --- Gates that skip generation entirely (nothing to ground an answer in) ---
    gate = None
    retrieval_score = retrieval_confidence(documents)
    if not documents:
        gate = insufficient_response("retrieval", "No chunks were retrieved.", None,
                                     ["Everything; retrieval returned no chunks."], sources)
    elif (min_retrieval_confidence is not None and retrieval_score is not None
          and retrieval_score < min_retrieval_confidence):
        gate = insufficient_response(
            "retrieval", f"Retrieval confidence {retrieval_score:.3f} is below "
                         f"MIN_RETRIEVAL_CONFIDENCE={min_retrieval_confidence}.",
            None, ["The question could not be matched to relevant chunks."], sources)
    elif grade and grade["verdict"] == "insufficient":
        relevant = [r for r in grade["relevant_blocks"] if 1 <= r <= len(documents)]
        found = (f"blocks {', '.join(f'[{r}]' for r in relevant)} are related but do not answer it"
                 if relevant else None)
        gate = insufficient_response("grader", grade["reason"], found,
                                     [grade["missing_information"]] if grade["missing_information"] else [],
                                     sources, set(relevant) or None)
    if gate:
        result.update(gate)
        result["evidence_status"] = "insufficient"
        result["confidence"] = score_confidence(documents)
        return result

    # --- Grounded generation ---
    answer = generate_answer(question, documents)
    result["evidence_status"] = answer.evidence_status
    result["parts"] = [part.model_dump() for part in answer.parts]
    result["verification"] = verify_citations(answer, documents) if verify else None
    result["confidence"] = score_confidence(documents, answer, result["verification"])

    if answer.evidence_status == "insufficient":
        missing = [p.missing for p in answer.parts if p.missing]
        cited = {ref for p in answer.parts for c in p.claims for ref in c.citations}
        result.update(insufficient_response(
            "generator", "The generator found the context insufficient to answer.",
            answer.evidence_found, missing, sources, cited or None))
        return result

    result["status"] = "answered" if answer.evidence_status == "sufficient" else "partial"
    result["answer"] = render_answer(answer)
    return result
