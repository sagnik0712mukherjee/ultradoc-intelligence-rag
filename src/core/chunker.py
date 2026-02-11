"""
Chunker Module

Implements structure-aware chunking for logistics documents.
"""

from typing import List, Dict
from src.config.settings import MAX_CHUNK_SIZE, CHUNK_OVERLAP, SECTION_KEYWORDS


class StructureAwareChunker:
    """
    A chunker that splits document text based on structural section markers
    and applies size-based chunking with overlap.
    """

    def __init__(self) -> None:
        """
        Initializes the chunker.
        """
        pass

    def chunk_document(self, document_text: str) -> List[Dict[str, str]]:
        """
        Split document into structured chunks.

        Args:
            document_text (str): Cleaned document text.

        Returns:
            List[Dict[str, str]]: List of chunk dictionaries with metadata.
        """

        sections: List[str] = self._split_by_sections(document_text)

        final_chunks: List[Dict[str, str]] = []

        chunk_counter: int = 0

        for section in sections:
            section_chunks: List[str] = self._apply_size_chunking(section)

            for chunk in section_chunks:
                final_chunks.append({"chunk_id": str(chunk_counter), "content": chunk})

                chunk_counter += 1

        return final_chunks

    def _split_by_sections(self, text: str) -> List[str]:
        """
        Split text using predefined section keywords.

        Args:
            text (str): Full document text.

        Returns:
            List[str]: Section-wise split text blocks.
        """

        sections: List[str] = []
        current_section: str = ""

        words: List[str] = text.split(" ")

        for word in words:
            if any(keyword in word for keyword in SECTION_KEYWORDS):
                if current_section:
                    sections.append(current_section.strip())
                current_section = word + " "
            else:
                current_section += word + " "

        if current_section:
            sections.append(current_section.strip())

        return sections

    def _apply_size_chunking(self, text: str) -> List[str]:
        """
        Apply size-based chunking with overlap.

        Args:
            text (str): Section text.

        Returns:
            List[str]: Size-constrained chunks.
        """

        chunks: List[str] = []
        start: int = 0
        text_length: int = len(text)

        while start < text_length:
            end: int = start + MAX_CHUNK_SIZE

            chunk: str = text[start:end]

            chunks.append(chunk)

            start = end - CHUNK_OVERLAP

        return chunks
