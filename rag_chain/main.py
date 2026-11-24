import argparse
import json
from hybrid_retriever import retrieve_hybrid, RETRIEVAL_MODES
from ingestion import load_document, load_all_processed_documents
from chunking import chunk_documents
from bm25_store import load_bm25_chunks
from config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_MODE, CORRECTION_ENABLED
from vector_store import get_embedding_model, add_chunks, get_chroma_collection, reset_indexes


def index_documents(documents, chunk_size, chunk_overlap, strategy, embedding_model, preview=0):
    chunks = chunk_documents(
        documents,
        strategy,
        embedding_model,
        chunk_size,
        chunk_overlap
    )

    print(f"Documents: {len(documents)}")
    print(f"Chunks before deduplication: {len(chunks)}")

    for chunk in chunks[:preview]:
        print("\n--- CHUNK ---")
        print(chunk.metadata)
        print(chunk.page_content[:300])

    return add_chunks(chunks)


def ingest_file(file_path, chunk_size, chunk_overlap, strategy):
    print(f"Loading: {file_path}")
    documents = load_document(file_path)
    index_documents(documents, chunk_size, chunk_overlap, strategy, get_embedding_model(), preview=3)


def reindex(chunk_size, chunk_overlap, strategy):
    """Rebuild Chroma + BM25 from data/processed without re-parsing the raw files."""
    processed = load_all_processed_documents()
    if not processed:
        print("No processed documents found.")
        return

    reset_indexes()
    embedding_model = get_embedding_model()

    for documents in processed:
        if documents:
            print(f"\nRe-indexing: {documents[0].metadata['source_document']}")
            index_documents(documents, chunk_size, chunk_overlap, strategy, embedding_model)


def check_index_sync():
    chroma_count = get_chroma_collection().count()
    bm25_chunks = load_bm25_chunks()

    if chroma_count != len(bm25_chunks):
        print(f"WARNING: Chroma has {chroma_count} chunks but BM25 has {len(bm25_chunks)}. "
              "Run `python main.py reindex` to rebuild both.")
    elif any("chunk_id" not in chunk.metadata for chunk in bm25_chunks):
        print("WARNING: index was built without chunk ids. Run `python main.py reindex`.")


def query_loop():
    check_index_sync()

    while True:
        try:
            query = input("\nQuestion (or 'exit'): ").strip()
        except EOFError:
            break

        if query.lower() == "exit":
            break
        if not query:
            continue

        docs = retrieve_hybrid(query)

        print(f"\n--- RERANKED TOP {len(docs)} ---")

        for i, doc in enumerate(docs, start=1):
            print(f"\n[{i}] {doc.metadata}")
            print(doc.page_content[:500])


def print_answer(result):
    print(f"\nStatus: {result['status']}  (evidence: {result['evidence_status']})")
    print(f"\n{result['answer']}")

    correction = result["retrieval"]["correction"]
    if correction["enabled"]:
        print(f"\nCorrection triggered: {correction['triggered']}"
              f"  retries: {correction['retry_count']}/{correction['max_retries']}")
        if correction["triggered"]:
            print(f"  why: {correction['trigger_reason']}")
            print(f"  rewritten query: {correction['rewritten_query']}")

    if result["insufficient_evidence"]:
        info = result["insufficient_evidence"]
        print(f"\nInsufficient evidence (decided by {info['decided_by']}): {info['reason']}")
        for item in info["not_established"]:
            print(f"  not established: {item}")
        for doc in info["documents_to_check"]:
            print(f"  check manually: {doc['source_document']} | {doc['section_heading']} | page {doc['page_number']}")

    if result["verification"]:
        print(f"\nCitation verification: {result['verification']['summary']}")
        for flag in result["verification"]["flagged"]:
            print(f"  FLAG [{flag['ref']}] {flag['problem']}: {flag['statement']} — {flag['reason']}")

    print(f"\nConfidence: {result['confidence']}")
    print("\nSources:")
    for source in result["sources"]:
        print(f"  [{source['ref']}] {source['source_document']} | {source['section_heading']}"
              f" | page {source['page_number']} | rerank {source['rerank_score']}")


def ask(question, mode, correction, as_json):
    from pipeline import answer_question  # lazy: keeps ingest/reindex free of LLM imports
    check_index_sync()
    questions = [question] if question else iter(lambda: input("\nQuestion (or 'exit'): ").strip(), "exit")
    for q in questions:
        if not q:
            continue
        result = answer_question(q, retrieval_mode=mode, correction=correction)
        print(json.dumps(result, indent=2, default=str) if as_json else "", end="")
        if not as_json:
            print_answer(result)


def add_chunking_args(parser):
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "semantic"],
        default="fixed"
    )


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline")
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("file_path")
    add_chunking_args(ingest)

    reindex_parser = subparsers.add_parser(
        "reindex", help="Reset both indexes and rebuild them from processed documents"
    )
    add_chunking_args(reindex_parser)

    subparsers.add_parser("query")

    ask_parser = subparsers.add_parser(
        "ask", help="Grounded answer with citations (interactive if no question is given)"
    )
    ask_parser.add_argument("question", nargs="?")
    ask_parser.add_argument("--mode", choices=RETRIEVAL_MODES, default=RETRIEVAL_MODE)
    ask_parser.add_argument("--correction", action=argparse.BooleanOptionalAction,
                            default=CORRECTION_ENABLED)
    ask_parser.add_argument("--json", action="store_true", help="Print the full structured result")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_file(
            args.file_path,
            args.chunk_size,
            args.chunk_overlap,
            args.strategy
        )

    elif args.command == "reindex":
        reindex(args.chunk_size, args.chunk_overlap, args.strategy)

    elif args.command == "query":
        query_loop()

    elif args.command == "ask":
        ask(args.question, args.mode, args.correction, args.json)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
