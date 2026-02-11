"""
Extract API Module
"""

from fastapi import APIRouter
from typing import Dict
from src.core.structured_extractor import StructuredExtractor
import src.core.app_state as app_state

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

    extractor: StructuredExtractor = StructuredExtractor(api_key)

    structured_output: Dict[str, str] = extractor.extract(
        app_state.DOCUMENT_TEXT_STORAGE
    )

    return structured_output
