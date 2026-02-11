"""
Application State Module

Stores shared in-memory state across API endpoints.
"""

from typing import Optional
from src.core.data.vector_store import VectorStore
from src.core.state.memory_manager import MemoryManager

VECTOR_STORE_INSTANCE: Optional[VectorStore] = None
DOCUMENT_TEXT_STORAGE: str = ""
MEMORY_MANAGER_INSTANCE: MemoryManager = MemoryManager()
