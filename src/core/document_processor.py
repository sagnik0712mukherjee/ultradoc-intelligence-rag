"""
Document Processor Module

Handles parsing of supported document formats (PDF, DOCX, TXT)
and returns normalized text content.
"""

import os
from typing import List
from pypdf import PdfReader
from docx import Document


class DocumentProcessor:
    """
    A class responsible for extracting text from various document formats.
    """

    def __init__(self) -> None:
        """
        Initializes DocumentProcessor.
        """
        pass

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a document based on file extension.

        Args:
            file_path (str): Absolute or relative path of the document.

        Returns:
            str: Extracted and cleaned text content.
        """

        # Determine file extension
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

        # Normalize whitespace
        cleaned_text: str = self._normalize_text(extracted_text)

        return cleaned_text

    def _extract_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path (str): Path to PDF file.

        Returns:
            str: Extracted raw text from PDF.
        """

        reader: PdfReader = PdfReader(file_path)

        text_chunks: List[str] = []

        for page in reader.pages:
            page_text: str = page.extract_text()

            if page_text:
                text_chunks.append(page_text)

        combined_text: str = "\n".join(text_chunks)

        return combined_text

    def _extract_from_docx(self, file_path: str) -> str:
        """
        Extract text from a DOCX file.

        Args:
            file_path (str): Path to DOCX file.

        Returns:
            str: Extracted raw text.
        """

        document: Document = Document(file_path)

        paragraphs: List[str] = []

        for para in document.paragraphs:
            paragraphs.append(para.text)

        combined_text: str = "\n".join(paragraphs)

        return combined_text

    def _extract_from_txt(self, file_path: str) -> str:
        """
        Extract text from a TXT file.

        Args:
            file_path (str): Path to TXT file.

        Returns:
            str: Extracted raw text.
        """

        with open(file_path, "r", encoding="utf-8") as file:
            content: str = file.read()

        return content

    def _normalize_text(self, text: str) -> str:
        """
        Normalize whitespace and clean text.

        Args:
            text (str): Raw extracted text.

        Returns:
            str: Cleaned text.
        """

        # Replace multiple spaces with single space
        normalized: str = " ".join(text.split())

        return normalized
