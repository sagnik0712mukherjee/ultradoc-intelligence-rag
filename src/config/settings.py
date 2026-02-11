"""
Global Configuration Settings for UltraDoc Intelligence RAG System.
"""

# =========================
# Chunking Configuration
# =========================

MAX_CHUNK_SIZE: int = 800  # Maximum characters per chunk
CHUNK_OVERLAP: int = 150  # Overlap between chunks

# Section header keywords used for structural splitting
SECTION_KEYWORDS: list[str] = [
    "Carrier Details",
    "Customer Details",
    "Stops",
    "Pickup",
    "Drop",
    "Rate Breakdown",
    "Notes",
    "Instructions",
    "Driver Details",
    "Shipper",
    "Consignee",
    "Load ID",
    "Reference ID",
]

# =========================
# Retrieval Configuration
# =========================

TOP_K_RETRIEVAL: int = 4
SIMILARITY_THRESHOLD: float = 0.25

# =========================
# Confidence Scoring
# =========================

MIN_CONFIDENCE_SCORE: float = 0.40

# =========================
# Model Configuration
# =========================

EMBEDDING_MODEL_NAME: str = "text-embedding-3-large"
LLM_MODEL_NAME: str = "gpt-4.1"

# =========================
# Memory Configuration
# =========================

MAX_SHORT_TERM_MEMORY: int = 10
