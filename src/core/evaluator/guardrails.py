"""
Guardrails Module

Implements hallucination prevention and answer validation logic.
"""

from typing import List, Dict
from src.config.settings import SIMILARITY_THRESHOLD, MIN_CONFIDENCE_SCORE


class Guardrails:
    """
    A class responsible for applying hallucination guardrails
    before allowing LLM answer generation.
    """

    def __init__(self) -> None:
        """
        Initialize Guardrails.
        """
        pass

    def validate_retrieval(
        self, retrieved_chunks: List[Dict[str, str]], max_similarity_score: float
    ) -> Dict[str, str]:
        """
        Validate retrieval results before answer generation.

        Args:
            retrieved_chunks (List[Dict[str, str]]): Retrieved document chunks.
            max_similarity_score (float): Highest similarity score.

        Returns:
            Dict[str, str]:
                - status: "allow" or "reject"
                - message: Explanation if rejected
        """

        # If no chunks retrieved
        if not retrieved_chunks:
            return {"status": "reject", "message": "Not found in document."}

        # If similarity too low
        if max_similarity_score < SIMILARITY_THRESHOLD:
            return {
                "status": "reject",
                "message": "Similarity too low. Unable to confidently answer from document.",
            }

        return {"status": "allow", "message": "Retrieval validated."}

    def validate_confidence(self, confidence_score: float) -> Dict[str, str]:
        """
        Validate final confidence score before returning answer.

        Args:
            confidence_score (float): Computed confidence score.

        Returns:
            Dict[str, str]:
                - status: "allow" or "reject"
                - message: Explanation if rejected
        """

        if confidence_score < MIN_CONFIDENCE_SCORE:
            return {
                "status": "reject",
                "message": "I have low confidence in the generated answer.",
            }

        return {"status": "allow", "message": "Confidence acceptable."}
