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

# Chunking LLM - GPT-3.5 for deterministic, cost-effective structure extraction
CHUNKING_LLM_MODEL: str = "gpt-3.5-turbo"

# Main LLM - GPT-4o-mini for high-quality answer generation
MAIN_LLM_MODEL: str = "gpt-4o-mini"

# =========================
# Memory Configuration
# =========================

MAX_SHORT_TERM_MEMORY: int = 10
