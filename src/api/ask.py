"""
Ask API Module
"""

from fastapi import APIRouter
from typing import Dict
from src.core.services.embedding_service import EmbeddingService
from src.core.services.retriever import Retriever
from src.core.evaluator.guardrails import Guardrails
from src.core.services.answer_generator import AnswerGenerator
from src.core.evaluator.confidence import ConfidenceScorer
import src.core.state.app_state as app_state

router = APIRouter()


@router.post("/ask")
async def ask_question(query: str, api_key: str) -> Dict:
    """
    Ask question about uploaded document.

    Args:
        query (str): User question.
        api_key (str): OpenAI API key.

    Returns:
        Dict: Answer, sources, confidence.
    """

    # Check if document has been uploaded
    if app_state.VECTOR_STORE_INSTANCE is None:
        return {"error": "No document uploaded."}

    embedding_service: EmbeddingService = EmbeddingService(api_key)

    retriever: Retriever = Retriever(embedding_service, app_state.VECTOR_STORE_INSTANCE)

    retrieved_chunks, max_similarity_score, all_results = retriever.retrieve(query)

    guardrails: Guardrails = Guardrails()

    # Calculate Precision@K metric
    from src.core.evaluator.metrics import RetrievalMetrics
    from src.config.settings import TOP_K_RETRIEVAL

    precision_at_k = RetrievalMetrics.compute_precision_at_k(
        all_results, TOP_K_RETRIEVAL
    )

    validation = guardrails.validate_retrieval(retrieved_chunks, max_similarity_score)

    if validation.get("status") == "reject":
        return {
            "answer": validation.get("message"),
            "confidence": 0.0,
            "precision_at_k": precision_at_k,
            "sources": [],
        }

    answer_generator: AnswerGenerator = AnswerGenerator(
        api_key, app_state.MEMORY_MANAGER_INSTANCE
    )

    result = answer_generator.generate_answer(query, retrieved_chunks)

    answer: str = result.get("answer")

    confidence_scorer: ConfidenceScorer = ConfidenceScorer()

    confidence_score: float = confidence_scorer.compute_confidence(
        answer, retrieved_chunks, max_similarity_score
    )

    confidence_validation = guardrails.validate_confidence(confidence_score)

    if confidence_validation.get("status") == "reject":
        return {
            "answer": answer
            if answer == "Not found in document."
            else confidence_validation.get("message"),
            "confidence": confidence_score,
            "precision_at_k": precision_at_k,
            "sources": [],
        }

    return {
        "answer": answer,
        "confidence": confidence_score,
        "precision_at_k": precision_at_k,
        "sources": result.get("sources"),
    }


@router.post("/clear_memory")
async def clear_memory() -> Dict[str, str]:
    """
    Clear conversational memory.

    Returns:
        Dict[str, str]: Status message.
    """

    app_state.MEMORY_MANAGER_INSTANCE.clear_memory()

    return {"status": "Memory cleared."}
