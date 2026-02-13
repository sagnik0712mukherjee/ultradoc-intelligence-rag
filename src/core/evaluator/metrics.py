"""
Retrieval Metrics Module

Implements evaluation metrics for retrieval quality.
"""

from typing import List, Tuple, Dict, Any
from src.config.settings import SIMILARITY_THRESHOLD


class RetrievalMetrics:
    """
    Computes metrics for RAG retrieval quality.
    """

    @staticmethod
    def compute_precision_at_k(
        all_results: List[Tuple[Dict[str, Any], float]], k: int
    ) -> float:
        """
        Compute Precision@K.
        Ratio of relevant items in the top K retrieved results.

        A result is considered 'relevant' if its similarity score
        is >= SIMILARITY_THRESHOLD.

        Args:
            all_results (List[Tuple[Dict[str, Any], float]]): Raw search results (metadata, score).
            k (int): Number of top results to consider.

        Returns:
            float: Precision@K score between 0 and 1.
        """

        if not all_results:
            return 0.0

        # Take top K results
        top_k = all_results[:k]

        # Count relevant items
        relevant_count = sum(1 for _, score in top_k if score >= SIMILARITY_THRESHOLD)

        return relevant_count / len(top_k)
