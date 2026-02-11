"""
Table-First Generic Structured Extractor

Converts layout-aware PDF tables into clean semantic JSON.
No hardcoding. Fully generic.
"""

from typing import Dict, Any, List
import pdfplumber


class StructuredExtractor:
    """
    Extract structured JSON using table-first strategy.
    """

    def __init__(self) -> None:
        pass

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract structured representation from PDF.

        Args:
            file_path (str)

        Returns:
            Dict[str, Any]
        """

        structured: Dict[str, Any] = {"sections": []}

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()

                for table in tables:
                    if not table or len(table) < 1:
                        continue

                    cleaned_table = self._clean_table(table)

                    if not cleaned_table:
                        continue

                    parsed_section = self._parse_table(cleaned_table)

                    if parsed_section:
                        structured["sections"].append(parsed_section)

        return structured

    def _clean_table(self, table: List[List[str]]) -> List[List[str]]:
        """
        Remove empty rows and normalize cells.

        Args:
            table (List[List[str]])

        Returns:
            List[List[str]]
        """

        cleaned = []

        for row in table:
            if not row:
                continue

            cleaned_row = [cell.strip() if cell else "" for cell in row]

            if any(cleaned_row):
                cleaned.append(cleaned_row)

        return cleaned

    def _parse_table(self, table: List[List[str]]) -> Any:
        """
        Convert table into structured JSON.

        Args:
            table (List[List[str]])

        Returns:
            Dict or List
        """

        # Case 1: Two-column key-value style table
        if all(len(row) == 2 for row in table):
            section_dict = {}

            for row in table:
                key, value = row

                if key and value:
                    section_dict[key] = value

            return section_dict if section_dict else None

        # Case 2: Header + rows table
        headers = table[0]
        rows = table[1:]

        if not headers or not rows:
            return None

        structured_rows = []

        for row in rows:
            if len(row) != len(headers):
                continue

            row_dict = {}

            for idx, header in enumerate(headers):
                if header:
                    row_dict[header] = row[idx]

            if row_dict:
                structured_rows.append(row_dict)

        return structured_rows if structured_rows else None
