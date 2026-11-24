from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Anchor all storage to the project folder so results don't depend on the cwd.
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
BM25_PATH = BASE_DIR / "bm25_index.pkl"

COLLECTION_NAME = "internal_documents"
EMBEDDING_MODEL = "gemini-embedding-001"
# All chat-model calls use Gemini (reads GEMINI_API_KEY, same as the embeddings).
# Per-role models default to LLM_MODEL so experiments can swap one role at a time.
LLM_MODEL = "gemini-3.5-flash"
GRADER_MODEL = LLM_MODEL       # evidence grading + query rewriting (Day 5)
GENERATION_MODEL = LLM_MODEL   # grounded answer generation (Day 6)
JUDGE_MODEL = LLM_MODEL        # citation verification LLM-as-judge (Day 6)
LLM_TEMPERATURE = 0
LLM_THINKING_LEVEL = "low"     # Gemini 3 reasoning depth; "low" keeps per-call latency ~2-5s
LLM_TIMEOUT_SECONDS = 90
LLM_MAX_RETRIES = 2
# Client-side throttle shared by every chat-model call (None = unthrottled). The Gemini
# free tier allows 5 generate requests/min per model; raise or disable on a paid tier.
LLM_REQUESTS_PER_MINUTE = 5

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
# Semantic chunking: split where the distance between adjacent sentence windows
# is in the top (100 - SEMANTIC_BREAKPOINT_PERCENTILE)% for that document.
SEMANTIC_BREAKPOINT_PERCENTILE = 90
# Semantic chunks smaller than this are avoided (merged) where possible; CHUNK_SIZE is the max.
MIN_SEMANTIC_CHUNK_SIZE = 200

DEDUP_THRESHOLD = 0.95
DENSE_TOP_K = 10
SPARSE_TOP_K = 10
FUSION_TOP_K = 20

DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
RRF_K = 60  # standard RRF rank constant: score = weight / (RRF_K + rank)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 5

# --- Retrieval variants (for Day 8 comparisons; see hybrid_retriever.retrieve) ---
# "dense"         A. vector only
# "hybrid"        B. dense + BM25 fused with weighted RRF
# "hybrid_rerank" C. hybrid + cross-encoder reranker (the Phase 2 default)
# Variant D is "hybrid_rerank" with CORRECTION_ENABLED=True (see pipeline.answer_question).
RETRIEVAL_MODE = "hybrid_rerank"
# Final number of chunks handed to grading/generation, the same for every mode so
# variants are compared on equal context budgets. Equals RERANK_TOP_K by default.
FINAL_TOP_K = RERANK_TOP_K

# --- Day 5: corrective retrieval ---
CORRECTION_ENABLED = True
# Hard cap on rewrite-and-retrieve rounds after the first retrieval. Never a loop.
MAX_CORRECTION_RETRIES = 1

# --- Day 6: grounded generation, verification, confidence ---
# Max concurrent LLM-judge calls (one per claim<->cited-chunk pair).
VERIFY_MAX_CONCURRENCY = 4
# Retrieval confidence = mean sigmoid(cross-encoder logit) over the top N reranked
# chunks. The ms-marco cross-encoder emits uncalibrated logits; sigmoid only maps
# them to 0..1 for display/combination — it is not a calibrated probability.
# N=1 ("how relevant is the best evidence"): averaging over more chunks penalised
# questions fully answered by a single chunk; missing parts show up in completeness.
RETRIEVAL_CONFIDENCE_TOP_N = 1
# Composite confidence = weighted mean of the available components (weights are
# renormalised over components that are not None). Edit here to rebalance.
CONFIDENCE_WEIGHTS = {
    "retrieval": 0.3,
    "citation_support": 0.4,
    "completeness": 0.3,
}
# Optional hard refusal gate on retrieval confidence (skips generation entirely).
# None = disabled: the threshold is uncalibrated until Day 7 eval data exists, so
# by default refusal is decided by the evidence grader / the generator instead.
MIN_RETRIEVAL_CONFIDENCE = None
