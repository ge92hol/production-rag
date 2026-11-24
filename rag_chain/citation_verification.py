"""Day 6 — verify that each cited chunk actually supports the claim it is attached to.

One LLM-as-judge call per (claim, cited chunk) pair. A citation counts only if the
judge says the chunk supports the claim; syntactic presence of [n] is never enough.
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import JUDGE_MODEL, VERIFY_MAX_CONCURRENCY
from llm import chunk_reference, get_structured_llm


class SupportJudgement(BaseModel):
    verdict: Literal["supported", "partially_supported", "not_supported"] = Field(
        description="supported: the excerpt states everything in the claim (paraphrase is fine); "
                    "partially_supported: it states some but not all of the claim; "
                    "not_supported: it does not state the claim, or contradicts it."
    )
    reason: str = Field(description="One sentence pointing to what in the excerpt does or does not support the claim.")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check citations. Decide whether the excerpt, on its own, supports the claim. "
     "Use only the excerpt — no outside knowledge, no assumptions about what the "
     "author probably meant. Being on the same topic is not support; the excerpt must "
     "state the claimed facts. Specific details (numbers, names, conditions) must match."),
    ("human", "Claim:\n{claim}\n\nExcerpt:\n{excerpt}"),
])

# The claim takes the strongest verdict among its cited chunks. Partial support from
# several chunks is NOT combined into full support (conservative by design).
VERDICT_ORDER = ("supported", "partially_supported", "not_supported")


def claim_status(citation_results):
    if not citation_results:
        return "uncited"
    verdicts = {c["verdict"] for c in citation_results}
    for verdict in VERDICT_ORDER:
        if verdict in verdicts:
            return verdict
    return "unverified"  # cited, but every judge call for it failed


def verify_citations(answer, documents):
    """Judge every claim<->citation pair in a GroundedAnswer against the cited chunks."""
    claims, pairs = [], []

    for part_index, part in enumerate(answer.parts):
        for claim in part.claims:
            refs = list(dict.fromkeys(claim.citations))  # dedupe, keep order
            valid = [r for r in refs if 1 <= r <= len(documents)]
            claims.append({
                "part_index": part_index,
                "statement": claim.statement,
                "citations": [],
                "invalid_refs": [r for r in refs if r not in valid],
            })
            pairs.extend((len(claims) - 1, ref) for ref in valid)

    judge = JUDGE_PROMPT | get_structured_llm(JUDGE_MODEL, SupportJudgement)
    results = judge.batch(
        [{"claim": claims[i]["statement"], "excerpt": documents[ref - 1].page_content}
         for i, ref in pairs],
        config={"max_concurrency": VERIFY_MAX_CONCURRENCY},
        return_exceptions=True,
    ) if pairs else []

    for (claim_index, ref), result in zip(pairs, results):
        reference = chunk_reference(ref, documents[ref - 1])
        if isinstance(result, Exception):
            verdict, reason = "error", f"judge call failed: {type(result).__name__}: {result}"
        else:
            verdict, reason = result.verdict, result.reason
        claims[claim_index]["citations"].append({
            "ref": ref,
            "chunk_id": reference["chunk_id"],
            "source_document": reference["source_document"],
            "verdict": verdict,
            "reason": reason,
        })

    flagged = []
    for claim in claims:
        claim["status"] = claim_status(claim["citations"])
        for citation in claim["citations"]:
            if citation["verdict"] in ("not_supported", "partially_supported", "error"):
                flagged.append({"statement": claim["statement"], "ref": citation["ref"],
                                "problem": citation["verdict"], "reason": citation["reason"]})
        for ref in claim["invalid_refs"]:
            flagged.append({"statement": claim["statement"], "ref": ref,
                            "problem": "invalid_ref", "reason": "No context block with this number."})
        if claim["status"] == "uncited":
            flagged.append({"statement": claim["statement"], "ref": None,
                            "problem": "uncited", "reason": "Claim has no valid citation."})

    statuses = [c["status"] for c in claims]
    all_citations = [c for claim in claims for c in claim["citations"]]
    return {
        "claims": claims,
        "flagged": flagged,
        "summary": {
            "total_claims": len(claims),
            "supported": statuses.count("supported"),
            "partially_supported": statuses.count("partially_supported"),
            "not_supported": statuses.count("not_supported"),
            "uncited": statuses.count("uncited"),
            "unverified": statuses.count("unverified"),
            "total_citations": len(all_citations) + sum(len(c["invalid_refs"]) for c in claims),
            "supported_citations": sum(c["verdict"] == "supported" for c in all_citations),
            "flagged_citations": sum(f["ref"] is not None for f in flagged),
        },
    }
