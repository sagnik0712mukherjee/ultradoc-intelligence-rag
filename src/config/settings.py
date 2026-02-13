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

# Structured Extraction LLM - GPT-4o-mini for deterministic, cost-effective structure extraction
CHUNKING_LLM_MODEL: str = "gpt-4o-mini"

# Main LLM - GPT-4.1 for high-quality answer generation
MAIN_LLM_MODEL: str = "gpt-4.1"

# =========================
# Memory Configuration
# =========================

MAX_SHORT_TERM_MEMORY: int = 10
