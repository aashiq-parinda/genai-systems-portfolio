"""
Unit tests for Legal Contract Segmenter
"""
import pytest
from src.core.segmenter import ContractSegmenter, ContractClause


@pytest.fixture
def segmenter():
    return ContractSegmenter()


def test_segmenter_numbered_sections(segmenter):
    contract = """
1.1 Indemnification. Vendor agrees to indemnify Customer against all third-party claims and liabilities.

1.2 Limitation of Liability. In no event shall either party be liable for indirect damages.

1.3 Governing Law. This Agreement shall be governed by the laws of New York.
"""
    clauses = segmenter.segment(contract)
    assert len(clauses) == 3
    assert clauses[0].index == 1
    assert "Indemnification" in clauses[0].title
    assert "indemnify" in clauses[0].text


def test_segmenter_boilerplate_filter(segmenter):
    contract = """
Section 1. Confidentiality. Parties agree to keep information secret.

[Signature Page Follows]

In Witness Whereof, the parties have executed this Agreement.
"""
    clauses = segmenter.segment(contract)
    assert len(clauses) == 1
    assert "Confidentiality" in clauses[0].title


def test_segmenter_empty_input(segmenter):
    clauses = segmenter.segment("")
    assert clauses == []
