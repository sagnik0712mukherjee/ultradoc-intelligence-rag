"""
Extract API Module
"""

from fastapi import APIRouter
from typing import Dict, Any
from src.core.data.llm_structured_extractor import LLMStructuredExtractor
import src.core.state.app_state as app_state

router = APIRouter()


@router.post("/extract")
async def extract_structured_data(api_key: str) -> Dict:
    """
    Extract structured shipment data.

    Args:
        api_key (str): OpenAI API key.

    Returns:
        Dict: Structured JSON output.
    """

    if not app_state.DOCUMENT_TEXT_STORAGE:
        return {"error": "No document uploaded."}

    try:
        extractor: LLMStructuredExtractor = LLMStructuredExtractor(api_key)

        structured_output: Dict[str, Any] = extractor.extract(
            app_state.DOCUMENT_TEXT_STORAGE
        )

        return structured_output
    except Exception as e:
        return {"error": f"Extraction failed: {str(e)}"}
