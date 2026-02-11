"""
Pydantic Schemas for Structured Document
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List


class SectionModel(BaseModel):
    """
    Represents a structured document section.
    """

    section_name: str = Field(..., description="Logical section name")
    content: Dict[str, Any] = Field(..., description="Key-value structured content")


class StructuredDocumentModel(BaseModel):
    """
    Represents full structured document.
    """

    sections: List[SectionModel]
