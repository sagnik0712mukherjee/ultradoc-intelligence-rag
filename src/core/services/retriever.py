"""
Retriever Module

Handles retrieval of relevant document chunks using embeddings
and FAISS similarity search.
"""

from typing import List, Dict, Tuple
from src.core.services.embedding_service import EmbeddingService
from src.core.data.vector_store import VectorStore
from src.config.settings import SIMILARITY_THRESHOLD, TOP_K_RETRIEVAL


class Retriever:
    """
    A retrieval layer that connects embedding generation
    and vector similarity search.
    """

    def __init__(
        self, embedding_service: EmbeddingService, vector_store: VectorStore
    ) -> None:
        """
        Initialize Retriever.

        Args:
            embedding_service (EmbeddingService): Embedding service instance.
            vector_store (VectorStore): FAISS vector store instance.
        """

        self.embedding_service: EmbeddingService = embedding_service
        self.vector_store: VectorStore = vector_store

    def retrieve(self, query: str) -> Tuple[List[Dict[str, str]], float]:
        """
        Retrieve relevant chunks for a given query.

        Args:
            query (str): User question.

        Returns:
            Tuple[List[Dict[str, str]], float]:
                - List of relevant chunk metadata dictionaries
                - Maximum similarity score among retrieved chunks
        """

        # Generate embedding for user query
        query_embedding: List[float] = self.embedding_service.generate_embedding(query)

        # Search vector store
        search_results: List[Tuple[Dict[str, str], float]] = self.vector_store.search(
            query_embedding, TOP_K_RETRIEVAL
        )

        filtered_chunks: List[Dict[str, str]] = []
        max_similarity_score: float = 0.0

        for metadata, similarity_score in search_results:
            # Track highest similarity score
            if similarity_score > max_similarity_score:
                max_similarity_score = similarity_score

            # Apply similarity threshold filtering
            if similarity_score >= SIMILARITY_THRESHOLD:
                filtered_chunks.append(
                    {
                        "chunk_id": metadata.get("chunk_id"),
                        "content": metadata.get("content"),
                        "similarity_score": similarity_score,
                    }
                )

        return filtered_chunks, max_similarity_score
