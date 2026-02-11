"""
Field-Level Semantic Chunker
"""

from typing import List, Dict, Any
import json


class StructureAwareChunker:
    """
    Breaks structured JSON into field-level semantic chunks.
    """

    def __init__(self) -> None:
        pass

    def chunk_document(self, structured_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert structured document into field-level chunks.

        Args:
            structured_data (Dict[str, Any])

        Returns:
            List[Dict[str, str]]
        """

        chunks: List[Dict[str, str]] = []
        chunk_id: int = 0

        sections = structured_data.get("sections", [])

        for section in sections:
            section_name: str = section.get("section_name", "Unknown")
            content: Dict[str, Any] = section.get("content", {})

            for key, value in content.items():
                # Handle nested dictionaries
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        chunk_text: str = (
                            f"Section: {section_name} | {key} - {sub_key}: {sub_value}"
                        )

                        chunks.append(
                            {"chunk_id": str(chunk_id), "content": chunk_text}
                        )

                        chunk_id += 1

                # Handle list values (e.g., Stops)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            chunk_text: str = (
                                f"Section: {section_name} | {json.dumps(item)}"
                            )

                            chunks.append(
                                {"chunk_id": str(chunk_id), "content": chunk_text}
                            )

                            chunk_id += 1

                # Simple key-value
                else:
                    chunk_text: str = f"Section: {section_name} | {key}: {value}"

                    chunks.append({"chunk_id": str(chunk_id), "content": chunk_text})

                    chunk_id += 1

        return chunks
