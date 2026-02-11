"""
Vector Store Module

Implements FAISS-based vector indexing and similarity retrieval.
"""

from typing import List, Dict, Tuple
import faiss
import numpy as np
from src.config.settings import TOP_K_RETRIEVAL


class VectorStore:
    """
    A FAISS-based vector store for similarity search.
    """

    def __init__(self, embedding_dimension: int) -> None:
        """
        Initialize FAISS index.

        Args:
            embedding_dimension (int): Dimension of embedding vectors.
        """

        self.embedding_dimension: int = embedding_dimension

        # Using Inner Product index (for cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(embedding_dimension)

        # Store metadata for each vector
        self.metadata_store: List[Dict[str, str]] = []

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length for cosine similarity.

        Args:
            vectors (np.ndarray): Input vectors.

        Returns:
            np.ndarray: Normalized vectors.
        """

        norms: np.ndarray = np.linalg.norm(vectors, axis=1, keepdims=True)

        normalized_vectors: np.ndarray = vectors / norms

        return normalized_vectors

    def add_vectors(
        self, embeddings: List[List[float]], metadata: List[Dict[str, str]]
    ) -> None:
        """
        Add embeddings and metadata to FAISS index.

        Args:
            embeddings (List[List[float]]): Embedding vectors.
            metadata (List[Dict[str, str]]): Corresponding metadata.
        """

        vectors_np: np.ndarray = np.array(embeddings).astype("float32")

        normalized_vectors: np.ndarray = self._normalize_vectors(vectors_np)

        self.index.add(normalized_vectors)

        self.metadata_store.extend(metadata)

    def search(
        self, query_embedding: List[float], top_k: int = TOP_K_RETRIEVAL
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        Search for most similar chunks.

        Args:
            query_embedding (List[float]): Query embedding vector.
            top_k (int): Number of top results to return.

        Returns:
            List[Tuple[Dict[str, str], float]]:
                List of (metadata, similarity_score).
        """

        query_np: np.ndarray = np.array([query_embedding]).astype("float32")

        normalized_query: np.ndarray = self._normalize_vectors(query_np)

        similarity_scores, indices = self.index.search(normalized_query, top_k)

        results: List[Tuple[Dict[str, str], float]] = []

        for idx, score in zip(indices[0], similarity_scores[0]):
            if idx < len(self.metadata_store):
                results.append((self.metadata_store[idx], float(score)))

        return results
