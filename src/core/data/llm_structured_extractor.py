"""
LLM Structured Extractor

Uses GPT-4.1 to normalize layout-aware content
into canonical structured JSON.
"""

from src.config.settings import CHUNKING_LLM_MODEL
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
        1. MANDATORY: Populate "shipment_details" by mapping document fields to the canonical keys.
           - Shipment_id: look for "Load ID", "Order #", "Shipment #", etc.
           - shipper/consignee: extract the main parties listed.
           - weight: look for values with "lbs", "kg", "kgs".
           - dates: look for "Ship Date", "Delivery Date", "Pickup Date".
        2. If the document is LOGISTICS/CARRIER related (Bills of Lading, Freight Invoices), you MUST populate the details.
        3. If it is NOT a logistics document (e.g., a personal phone bill), set "shipment_details" fields to null.
        4. Output STRICTLY valid JSON. Do NOT include any other keys like "sections".

        Expected schema:
        {
          "shipment_details": {
            "Shipment_id": "...",
            "shipper": "...",
            "consignee": "...",
            "pickup_datetime": "...",
            "delivery_datetime": "...",
            "equipment_type": "...",
            "mode": "...",
            "rate": "...",
            "currency": "...",
            "weight": "...",
            "carrier_name": "..."
          }
        }
        """

        user_prompt: str = f"""
        Document Content:

        {document_text}

        Return structured JSON.
        """

        response = self.client.chat.completions.create(
            model=CHUNKING_LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_output: str = response.choices[0].message.content

        # Robust JSON extraction (handle markdown blocks)
        if "```json" in raw_output:
            raw_output = raw_output.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_output:
            raw_output = raw_output.split("```")[1].split("```")[0].strip()

        try:
            parsed_json = json.loads(raw_output)
            validated = StructuredDocumentModel(**parsed_json)
            return validated.dict()

        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(
                f"LLM output validation failed: {str(e)}. Raw output: {raw_output[:100]}..."
            )
