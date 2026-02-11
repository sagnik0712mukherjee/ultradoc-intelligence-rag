"""
Confidence Scoring Module

Implements heuristic confidence scoring based on:
- Retrieval similarity
- Chunk agreement
- Answer grounding coverage
"""

from typing import List, Dict
import re
from src.config.settings import SIMILARITY_THRESHOLD


class ConfidenceScorer:
    """
    Computes confidence score for generated answers.
    """

    def __init__(self) -> None:
        """
        Initialize ConfidenceScorer.
        """
        pass

    def compute_confidence(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, str]],
        max_similarity_score: float,
    ) -> float:
        """
        Compute overall confidence score.

        Args:
            answer (str): Generated answer text.
            retrieved_chunks (List[Dict[str, str]]): Retrieved chunks used for context.
            max_similarity_score (float): Highest similarity score.

        Returns:
            float: Confidence score between 0 and 1.
        """

        retrieval_score: float = self._compute_retrieval_score(max_similarity_score)

        agreement_score: float = self._compute_chunk_agreement_score(retrieved_chunks)

        coverage_score: float = self._compute_answer_coverage_score(
            answer, retrieved_chunks
        )

        # Weighted combination
        final_score: float = (
            0.5 * retrieval_score + 0.3 * agreement_score + 0.2 * coverage_score
        )

        # Ensure score bounded between 0 and 1
        final_score = max(0.0, min(1.0, final_score))

        return final_score

    def _compute_retrieval_score(self, max_similarity_score: float) -> float:
        """
        Normalize retrieval similarity score.

        Args:
            max_similarity_score (float): Maximum cosine similarity.

        Returns:
            float: Normalized retrieval confidence.
        """

        if max_similarity_score < SIMILARITY_THRESHOLD:
            return 0.0

        return max_similarity_score

    def _compute_chunk_agreement_score(
        self, retrieved_chunks: List[Dict[str, str]]
    ) -> float:
        """
        Compute agreement score based on number of supporting chunks.

        Args:
            retrieved_chunks (List[Dict[str, str]]): Retrieved chunks.

        Returns:
            float: Agreement score between 0 and 1.
        """

        supporting_chunks: int = len(retrieved_chunks)

        if supporting_chunks == 0:
            return 0.0

        # Normalize: more than 3 chunks considered strong agreement
        normalized_score: float = min(supporting_chunks / 3, 1.0)

        return normalized_score

    def _compute_answer_coverage_score(
        self, answer: str, retrieved_chunks: List[Dict[str, str]]
    ) -> float:
        """
        Compute coverage score based on how much of answer
        overlaps with retrieved chunk text.

        Args:
            answer (str): Generated answer.
            retrieved_chunks (List[Dict[str, str]]): Retrieved chunks.

        Returns:
            float: Coverage score between 0 and 1.
        """

        if not answer:
            return 0.0

        combined_context: str = " ".join(
            chunk.get("content", "") for chunk in retrieved_chunks
        ).lower()

        answer_words: List[str] = re.findall(r"\w+", answer.lower())

        if not answer_words:
            return 0.0

        matched_words: int = sum(1 for word in answer_words if word in combined_context)

        coverage_ratio: float = matched_words / len(answer_words)

        return coverage_ratio
