"""Day 5 — corrective retrieval: retrieve -> grade -> (rewrite -> retrieve -> grade) x N.

A fixed, explicit workflow with a hard retry cap, not an agent. It wraps any base
retriever `query -> list[Document]` (e.g. hybrid_retriever.retrieve), so the base
retrieval variants stay benchmarkable with and without correction.
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import GRADER_MODEL, MAX_CORRECTION_RETRIES
from hybrid_retriever import retrieve
from llm import format_context_blocks, get_structured_llm


class EvidenceGrade(BaseModel):
    verdict: Literal["adequate", "partial", "insufficient"] = Field(
        description="adequate: the blocks fully answer every part of the question; "
                    "partial: they answer some parts or only part of the needed detail; "
                    "insufficient: they do not answer the question at all."
    )
    reason: str = Field(description="One or two sentences justifying the verdict.")
    missing_information: str = Field(
        description="What the question needs that the blocks do not contain. Empty if adequate."
    )
    relevant_blocks: list[int] = Field(
        description="Numbers of the context blocks that contain useful evidence."
    )


class RewrittenQuery(BaseModel):
    query: str = Field(description="The rewritten search query.")
    rationale: str = Field(description="One sentence on what was changed and why.")


GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You grade whether retrieved documentation excerpts contain enough evidence to "
     "answer a user's question. Judge only from the excerpts; do not use outside "
     "knowledge, and do not answer the question. Topical similarity is not enough: "
     "the excerpts must actually state the facts the question asks for. If the "
     "question has several parts, every part must be covered for 'adequate'."),
    ("human", "Question:\n{question}\n\nRetrieved excerpts:\n{context}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite search queries for a hybrid (embedding + BM25 keyword) search over "
     "internal technical documentation. The previous search did not find enough "
     "evidence. Write ONE improved query that keeps the user's original intent and "
     "targets the missing information: use likely documentation terminology, "
     "synonyms, expanded acronyms and specific identifiers. Do not guess the answer, "
     "do not add facts, and do not broaden the question to a different topic."),
    ("human",
     "Original question:\n{question}\n\n"
     "Queries already tried:\n{tried}\n\n"
     "Grader's assessment of the last results: {verdict} — {reason}\n"
     "Missing information: {missing}"),
])

VERDICT_RANK = {"insufficient": 0, "partial": 1, "adequate": 2}


def grade_evidence(question, documents):
    """Grade retrieved chunks against the ORIGINAL question (never a rewrite)."""
    if not documents:
        return EvidenceGrade(verdict="insufficient", reason="No chunks were retrieved.",
                             missing_information="Everything; retrieval returned nothing.",
                             relevant_blocks=[])
    chain = GRADE_PROMPT | get_structured_llm(GRADER_MODEL, EvidenceGrade)
    return chain.invoke({"question": question, "context": format_context_blocks(documents)})


def rewrite_query(question, tried_queries, grade):
    chain = REWRITE_PROMPT | get_structured_llm(GRADER_MODEL, RewrittenQuery)
    return chain.invoke({
        "question": question,
        "tried": "\n".join(f"- {q}" for q in tried_queries),
        "verdict": grade.verdict,
        "reason": grade.reason,
        "missing": grade.missing_information or "not specified",
    })


def is_weak(grade):
    return grade.verdict != "adequate"


def corrective_retrieve(question, retriever=retrieve, max_retries=MAX_CORRECTION_RETRIES):
    """Retrieve, grade, and re-retrieve with a rewritten query at most `max_retries` times.

    The returned documents are those of the best-graded attempt; on a tie the earlier
    attempt wins, so a rewrite only replaces evidence when it is graded strictly better.
    """
    documents = retriever(question)
    grade = grade_evidence(question, documents)
    attempts = [{"query": question, "rewrite_rationale": None,
                 "documents": documents, "grade": grade}]

    while is_weak(attempts[-1]["grade"]) and len(attempts) - 1 < max_retries:
        rewrite = rewrite_query(question, [a["query"] for a in attempts], attempts[-1]["grade"])
        documents = retriever(rewrite.query)  # full base pipeline again, incl. reranking
        attempts.append({"query": rewrite.query, "rewrite_rationale": rewrite.rationale,
                         "documents": documents,
                         "grade": grade_evidence(question, documents)})

    best = max(range(len(attempts)),
               key=lambda i: (VERDICT_RANK[attempts[i]["grade"].verdict], -i))
    first_grade = attempts[0]["grade"]
    rewritten = [a["query"] for a in attempts[1:]]

    return {
        "documents": attempts[best]["documents"],
        "grade": attempts[best]["grade"].model_dump(),
        "correction": {
            "enabled": True,
            "triggered": is_weak(first_grade),
            "trigger_reason": (f"{first_grade.verdict}: {first_grade.reason}"
                               if is_weak(first_grade) else None),
            "original_query": question,
            "rewritten_query": rewritten[-1] if rewritten else None,
            "retry_count": len(rewritten),
            "max_retries": max_retries,
            "selected_attempt": best,
            "final_query": attempts[best]["query"],
            "attempts": [{
                "query": a["query"],
                "rewrite_rationale": a["rewrite_rationale"],
                "grade": a["grade"].model_dump(),
                "chunk_ids": [d.metadata.get("chunk_id") for d in a["documents"]],
            } for a in attempts],
        },
    }
