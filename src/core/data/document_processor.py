"""
Document Processor Module

Uses pdfplumber for layout-aware PDF extraction
and preserves structural line breaks.
"""

import os
from typing import List
import pdfplumber
from docx import Document


class DocumentProcessor:
    """
    Extracts structured text from documents.
    """

    def __init__(self) -> None:
        pass

    def extract_text(self, file_path: str) -> str:
        """
        Extract text based on file type.

        Args:
            file_path (str)

        Returns:
            str
        """

        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension == ".pdf":
            extracted_text: str = self._extract_from_pdf(file_path)

        elif file_extension == ".docx":
            extracted_text: str = self._extract_from_docx(file_path)

        elif file_extension == ".txt":
            extracted_text: str = self._extract_from_txt(file_path)

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return extracted_text

    def _extract_from_pdf(self, file_path: str) -> str:
        """
        Layout-aware PDF extraction using pdfplumber.

        Args:
            file_path (str)

        Returns:
            str
        """

        pages_text: List[str] = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Extract text with layout preserved
                text: str = page.extract_text(x_tolerance=2, y_tolerance=2)

                if text:
                    pages_text.append(text)

        return "\n".join(pages_text)

    def _extract_from_docx(self, file_path: str) -> str:
        document: Document = Document(file_path)

        paragraphs: List[str] = [para.text for para in document.paragraphs]

        return "\n".join(paragraphs)

    def _extract_from_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
