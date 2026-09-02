"""
Legal Contract Clause Segmentation & Preprocessing Engine
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ContractClause:
    index: int
    title: Optional[str]
    text: str
    section_number: Optional[str] = None


class ContractSegmenter:
    """
    Robust Legal Document Segmenter.
    Splits contracts into discrete clauses based on numbering patterns, section headers, and paragraph delimiters.
    """

    SECTION_PATTERN = re.compile(
        r"^(?:Section\s+\d+|Article\s+[IVXLCDM\d]+|\d+\.\d+|\d+\.|\([a-z\d]\))\s*[:\-\.]?\s*(.*)$",
        re.IGNORECASE | re.MULTILINE
    )

    BOILERPLATE_PATTERNS = [
        r"^in witness whereof.*$",
        r"^executed as of the date.*$",
        r"^\[signature page follows\].*$",
        r"^table of contents.*$",
    ]

    def __init__(self, min_clause_length: int = 25):
        self.min_clause_length = min_clause_length

    def clean_text(self, text: str) -> str:
        """Removes duplicate whitespace and normalizes line breaks."""
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def segment(self, contract_text: str) -> List[ContractClause]:
        """
        Segments raw contract text into structured ContractClause objects.
        """
        cleaned = self.clean_text(contract_text)
        if not cleaned:
            return []

        # Split by double newline as primary clause candidates
        raw_blocks = [b.strip() for b in cleaned.split("\n\n") if len(b.strip()) >= self.min_clause_length]

        clauses: List[ContractClause] = []
        clause_idx = 1

        for block in raw_blocks:
            # Check for boilerplate headers to ignore
            is_boilerplate = any(
                re.match(p, block.strip(), re.IGNORECASE) for p in self.BOILERPLATE_PATTERNS
            )
            if is_boilerplate:
                continue

            # Check if block starts with section numbering
            lines = block.split("\n")
            first_line = lines[0].strip()
            section_match = self.SECTION_PATTERN.match(first_line)

            if section_match:
                section_header = first_line
                clause_text = "\n".join(lines[1:]).strip() if len(lines) > 1 else first_line
                if not clause_text:
                    clause_text = section_header
                clauses.append(
                    ContractClause(
                        index=clause_idx,
                        title=section_header[:60],
                        text=clause_text,
                        section_number=section_header.split()[0] if section_header.split() else None,
                    )
                )
            else:
                clauses.append(
                    ContractClause(
                        index=clause_idx,
                        title=f"Clause {clause_idx}",
                        text=block,
                        section_number=None,
                    )
                )
            clause_idx += 1

        return clauses
