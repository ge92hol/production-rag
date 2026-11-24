import uuid
import chromadb
import numpy as np

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from bm25_store import add_bm25_chunks, save_bm25_chunks
from config import COLLECTION_NAME, CHROMA_DIR, EMBEDDING_MODEL, DEDUP_THRESHOLD

# Explicit cosine space (applies when the collection is first created).
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def get_embedding_model():
    # task_type is left unset on purpose: the class then uses RETRIEVAL_DOCUMENT for
    # embed_documents and RETRIEVAL_QUERY for embed_query. Reads GEMINI_API_KEY from env.
    # At the default 3072 dims, gemini-embedding-001 vectors are already unit-normalized.
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_chroma_collection(client=None):
    client = client or get_chroma_client()
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata=COLLECTION_METADATA)


def get_vector_store():
    return Chroma(
        client=get_chroma_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        collection_metadata=COLLECTION_METADATA
    )


def reset_indexes():
    """Drop the Chroma collection and the BM25 corpus together (used by re-indexing)."""
    client = get_chroma_client()
    if COLLECTION_NAME in [c.name if hasattr(c, "name") else c for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    save_bm25_chunks([])


def get_existing_embeddings(collection):
    result = collection.get(include=["embeddings"])
    if result["embeddings"] is None or len(result["embeddings"]) == 0:
        return np.empty((0, 0))
    return np.asarray(result["embeddings"], dtype=np.float32)


def clean_metadata(metadata):
    # Chroma rejects None metadata values (e.g. page_number for non-PDF files).
    return {
        key: value if value is not None else ""
        for key, value in metadata.items()
    }


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def deduplicate_chunks(chunks, embedding_model, existing_embeddings):
    """Embed chunks once; skip any with cosine > DEDUP_THRESHOLD vs stored or already-accepted chunks."""
    if not chunks:
        return [], [], 0, 0

    embeddings = embedding_model.embed_documents(
        [chunk.page_content for chunk in chunks]
    )
    candidates = normalize(embeddings)

    if existing_embeddings.size:
        max_existing = (candidates @ normalize(existing_embeddings).T).max(axis=1)
    else:
        max_existing = np.zeros(len(chunks))

    accepted = []
    skipped_existing = skipped_batch = 0

    for i in range(len(chunks)):
        if max_existing[i] > DEDUP_THRESHOLD:
            skipped_existing += 1
        elif accepted and (candidates[accepted] @ candidates[i]).max() > DEDUP_THRESHOLD:
            skipped_batch += 1
        else:
            accepted.append(i)

    unique_chunks = []
    for i in accepted:
        chunk = chunks[i]
        chunk.metadata = clean_metadata(chunk.metadata)
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        unique_chunks.append(chunk)

    # Store the original API vectors (already computed) — no second embedding call.
    unique_embeddings = [embeddings[i] for i in accepted]
    return unique_chunks, unique_embeddings, skipped_existing, skipped_batch


def add_chunks(chunks):
    client = get_chroma_client()
    collection = get_chroma_collection(client)
    embedding_model = get_embedding_model()

    existing_embeddings = get_existing_embeddings(collection)
    unique_chunks, unique_embeddings, skipped_existing, skipped_batch = deduplicate_chunks(
        chunks,
        embedding_model,
        existing_embeddings
    )

    print(f"Duplicates skipped (cosine > {DEDUP_THRESHOLD}): "
          f"{skipped_existing} vs stored chunks, {skipped_batch} within this batch.")

    if not unique_chunks:
        print("No new unique chunks to store.")
        return 0

    ids = [chunk.metadata["chunk_id"] for chunk in unique_chunks]
    batch_size = client.get_max_batch_size()

    try:
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=[chunk.page_content for chunk in unique_chunks[start:end]],
                metadatas=[chunk.metadata for chunk in unique_chunks[start:end]],
                embeddings=unique_embeddings[start:end]
            )
        add_bm25_chunks(unique_chunks)
    except Exception:
        # Keep Chroma and BM25 in sync: roll back any partially added vectors.
        collection.delete(ids=ids)
        raise

    print(f"Stored {len(unique_chunks)} unique chunks in Chroma and BM25.")
    return len(unique_chunks)


def retrieve_dense(query, k):
    db = get_vector_store()
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever.invoke(query)
