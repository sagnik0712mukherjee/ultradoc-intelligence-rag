"""
Structured Extraction Module

Extracts structured shipment data from document context
using LLM with strict JSON schema enforcement.
"""

from typing import Dict
from openai import OpenAI
import json
from src.config.settings import LLM_MODEL_NAME


class StructuredExtractor:
    """
    Extracts structured shipment information from document text.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize StructuredExtractor with OpenAI API key.

        Args:
            api_key (str): OpenAI API key.
        """

        self.client: OpenAI = OpenAI(api_key=api_key)

    def extract(self, document_text: str) -> Dict[str, str]:
        """
        Extract structured shipment fields from document text.

        Args:
            document_text (str): Full document text.

        Returns:
            Dict[str, str]: Extracted shipment data with null values if missing.
        """

        system_prompt: str = """
            You are an information extraction assistant.

            Extract ONLY the fields listed below from the provided logistics document.
            If a field is missing, return null.

            Return STRICTLY valid JSON only.

            Required fields:
            {
            "shipment_id": string | null,
            "shipper": string | null,
            "consignee": string | null,
            "pickup_datetime": string | null,
            "delivery_datetime": string | null,
            "equipment_type": string | null,
            "mode": string | null,
            "rate": string | null,
            "currency": string | null,
            "weight": string | null,
            "carrier_name": string | null
            }
        """

        user_prompt: str = f"""
            Document:
            {document_text}
        """

        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        raw_output: str = response.choices[0].message.content

        cleaned_output: Dict[str, str] = self._safe_json_parse(raw_output)

        return cleaned_output

    def _safe_json_parse(self, raw_output: str) -> Dict[str, str]:
        """
        Safely parse JSON output from LLM.
        If parsing fails, return all-null dictionary.

        Args:
            raw_output (str): Raw LLM output.

        Returns:
            Dict[str, str]: Parsed JSON dictionary.
        """

        try:
            parsed: Dict[str, str] = json.loads(raw_output)
            return parsed

        except Exception:
            # Return null-filled schema if parsing fails
            return {
                "shipment_id": None,
                "shipper": None,
                "consignee": None,
                "pickup_datetime": None,
                "delivery_datetime": None,
                "equipment_type": None,
                "mode": None,
                "rate": None,
                "currency": None,
                "weight": None,
                "carrier_name": None,
            }
