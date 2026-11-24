"""Shared Gemini chat-model access and numbered-context helpers.

Every LLM call in the pipeline (grading, rewriting, generation, citation judging)
returns a Pydantic object via structured output; nothing parses free-form text.
"""
from functools import lru_cache

from config import (LLM_TEMPERATURE, LLM_THINKING_LEVEL, LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES,
                    LLM_REQUESTS_PER_MINUTE)


@lru_cache(maxsize=1)
def get_rate_limiter():
    if not LLM_REQUESTS_PER_MINUTE:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter
    # One bucket shared by all roles (grader/generator/judge), so a burst of judge
    # calls cannot exceed the per-minute quota. max_bucket_size=1 means no bursting.
    return InMemoryRateLimiter(requests_per_second=LLM_REQUESTS_PER_MINUTE / 60,
                               check_every_n_seconds=0.25, max_bucket_size=1)


@lru_cache(maxsize=None)
def get_chat_model(model):
    # Lazy import + cache: ingest/retrieval-only runs never build an LLM client.
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=LLM_TEMPERATURE,
        thinking_level=LLM_THINKING_LEVEL,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        rate_limiter=get_rate_limiter(),
    )


@lru_cache(maxsize=None)
def get_structured_llm(model, schema):
    return get_chat_model(model).with_structured_output(schema)


def chunk_reference(ref, doc):
    """Metadata that maps a context-block number [ref] back to the real chunk."""
    metadata = doc.metadata
    return {
        "ref": ref,
        "chunk_id": metadata.get("chunk_id"),
        "source_document": metadata.get("source_document"),
        "section_heading": metadata.get("section_heading") or None,
        "page_number": metadata.get("page_number") or None,
        "chunk_index": metadata.get("chunk_index"),
        "chunking_strategy": metadata.get("chunking_strategy"),
        "rerank_score": metadata.get("rerank_score"),
    }


def format_context_blocks(documents):
    """Render chunks as numbered blocks [1], [2], ... (numbering = list position)."""
    parts = []
    for ref, doc in enumerate(documents, start=1):
        metadata = doc.metadata
        location = f"Source: {metadata.get('source_document', 'unknown')}"
        if metadata.get("section_heading"):
            location += f" | Section: {metadata['section_heading']}"
        if metadata.get("page_number"):
            location += f" | Page: {metadata['page_number']}"
        parts.append(f"[{ref}] {location}\n{doc.page_content}")
    return "\n\n".join(parts)
