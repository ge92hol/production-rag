"""Day 6 — grounded answer generation over numbered context blocks.

The model returns structured claims (each with its own [n] citations) grouped by
question part, so citations can be verified per claim without parsing prose. The
user-facing answer text is rendered deterministically from those claims.
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import GENERATION_MODEL
from llm import format_context_blocks, get_structured_llm


class Claim(BaseModel):
    statement: str = Field(description="One self-contained factual statement, taken only from the context.")
    citations: list[int] = Field(description="Numbers of the context blocks that directly support this statement.")


class AnswerPart(BaseModel):
    question_part: str = Field(description="The sub-question this part answers (the whole question if it has one part).")
    status: Literal["answered", "partially_answered", "not_answered"]
    claims: list[Claim] = Field(description="Supported statements answering this part. Empty if not_answered.")
    missing: str = Field(description="What this part needs that the context does not establish. Empty if answered.")


class GroundedAnswer(BaseModel):
    evidence_status: Literal["sufficient", "partial", "insufficient"] = Field(
        description="sufficient: every part answered; partial: some parts or details missing; "
                    "insufficient: the context does not answer the question."
    )
    parts: list[AnswerPart]
    evidence_found: str = Field(
        description="Brief summary of the relevant evidence the context does contain, with [n] references. "
                    "Empty if nothing relevant was found."
    )


SYSTEM_PROMPT = """You answer questions about internal documentation using ONLY the numbered context blocks provided.

Rules:
- Every factual statement must come from the context. Never use outside knowledge, never guess, never fill gaps.
- Cite each statement with the number(s) of the block(s) that directly support it. Cite only blocks that actually state the fact.
- Split the question into its distinct parts (one part if it is a single question) and answer each part separately.
- If the context does not establish something the question asks, mark that part not_answered or partially_answered and say precisely what is missing. Do not answer it anyway.
- Keep statements atomic: one fact per statement, so each can be checked against its citations.
- Do not refer to "the context" or "the blocks" in statements; state the facts directly."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context blocks:\n{context}\n\nQuestion:\n{question}"),
])


def generate_answer(question, documents):
    chain = PROMPT | get_structured_llm(GENERATION_MODEL, GroundedAnswer)
    return chain.invoke({"question": question, "context": format_context_blocks(documents)})


def render_claims(claims):
    return " ".join(
        claim.statement + "".join(f" [{ref}]" for ref in claim.citations)
        for claim in claims
    )


def render_answer(answer):
    """Turn structured parts/claims into answer text with inline [n] citations."""
    lines = []
    for part in answer.parts:
        text = render_claims(part.claims)
        if part.status != "answered" and part.missing:
            gap = f"The available documents do not establish: {part.missing}"
            text = f"{text} {gap}".strip()
        if len(answer.parts) > 1:
            text = f"- {part.question_part}: {text}"
        lines.append(text)
    return "\n".join(lines)
