"""
Global Configuration Settings for UltraDoc Intelligence RAG System.
"""
# =========================
# Retrieval Configuration
# =========================

TOP_K_RETRIEVAL: int = 4
SIMILARITY_THRESHOLD: float = 0.30  # for acomodating huge variance in Questions

# =========================
# Confidence Scoring
# =========================

MIN_CONFIDENCE_SCORE: float = 0.45

# =========================
# Model Configuration
# =========================

EMBEDDING_MODEL_NAME: str = "text-embedding-3-large"
LLM_MODEL_NAME: str = "gpt-4.1"

# =========================
# Memory Configuration
# =========================

MAX_SHORT_TERM_MEMORY: int = 10
