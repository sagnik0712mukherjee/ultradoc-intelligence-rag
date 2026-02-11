"""
LLM Structured Extractor

Uses GPT-4.1 to normalize layout-aware content
into canonical structured JSON.
"""

from src.config.settings import LLM_MODEL_NAME
from typing import Dict, Any
from openai import OpenAI
from src.core.data.schemas import StructuredDocumentModel
from pydantic import ValidationError
import json


class LLMStructuredExtractor:
    """
    Uses LLM to extract structured JSON from document text.
    """

    def __init__(self, api_key: str) -> None:

        self.client = OpenAI(api_key=api_key)

    def extract(self, document_text: str) -> Dict[str, Any]:
        """
        Extract structured JSON from document using LLM.

        Args:
            document_text (str)

        Returns:
            Dict[str, Any]
        """

        system_prompt: str = """
        You are an enterprise-grade document intelligence engine.

        Your task:
        Convert the provided document into structured JSON.

        STRICT RULES:
        1. Extract ALL key-value pairs.
        2. Preserve tables as structured key-value rows.
        3. Do NOT hallucinate fields.
        4. If uncertain, include under section "unclassified".
        5. Ensure no information is omitted.
        6. Output STRICTLY valid JSON.

        Expected schema:
        {
        "sections": [
            {
            "section_name": "...",
            "content": {
                "key": "value"
            }
            }
        ]
        }
        """

        user_prompt: str = f"""
        Document Content:

        {document_text}

        Return structured JSON.
        """

        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_output: str = response.choices[0].message.content

        try:
            parsed_json = json.loads(raw_output)
            validated = StructuredDocumentModel(**parsed_json)
            return validated.dict()

        except (json.JSONDecodeError, ValidationError):
            raise ValueError("LLM output validation failed.")
