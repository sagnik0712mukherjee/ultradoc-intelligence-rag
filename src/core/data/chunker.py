"""
Field-Level Semantic Chunker
"""

from typing import List, Dict, Any


class StructureAwareChunker:
    """
    Breaks structured JSON into field-level semantic chunks.
    """

    # Aliases to help vector search match common industry terms
    FIELD_ALIASES = {
        "Shipment_id": ["Load ID", "Order Number", "Shipment Number", "Reference ID"],
        "shipper": ["Sender", "From"],
        "consignee": ["Receiver", "Recipient", "To"],
        "pickup_datetime": ["Pickup Date", "Ship Date"],
        "delivery_datetime": ["Delivery Date", "Arrival Date"],
        "weight": ["Total Weight", "Gross Weight"],
        "carrier_name": ["Trucking Company", "Transporter"],
    }

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

        details = structured_data.get("shipment_details", {})

        # Process details as chunks
        if details:
            for key, value in details.items():
                if value:
                    aliases = self.FIELD_ALIASES.get(key, [])
                    alias_str = (
                        f" (also known as: {', '.join(aliases)})" if aliases else ""
                    )

                    # Natural language format is often better for semantic search
                    chunk_text: str = f"The {key}{alias_str} is {value}."
                    chunks.append({"chunk_id": str(chunk_id), "content": chunk_text})
                    chunk_id += 1

        return chunks
