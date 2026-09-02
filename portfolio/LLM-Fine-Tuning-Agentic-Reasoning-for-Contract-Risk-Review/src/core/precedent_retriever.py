"""
Precedent Legal Knowledge Retrieval Engine (Vector & SEC Filings Grounding)
"""
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class PrecedentClause:
    id: str
    category: str
    source: str
    precedent_text: str
    standard_risk_guidance: str


PRECEDENT_DATABASE: List[PrecedentClause] = [
    PrecedentClause(
        id="prec_indem_01",
        category="Indemnification",
        source="SEC EDGAR 10-K Precedent (Standard Mutual)",
        precedent_text="Each party agrees to defend, indemnify, and hold harmless the other party from third-party claims arising from gross negligence or willful misconduct, subject to prompt written notice.",
        standard_risk_guidance="Standard enterprise clause requires mutual indemnity and an explicit exclusion for ordinary negligence."
    ),
    PrecedentClause(
        id="prec_liab_01",
        category="Limitation of Liability",
        source="American Bar Association Commercial Model",
        precedent_text="In no event shall either party's aggregate liability exceed the total fees paid under this Agreement in the twelve (12) months preceding the claim, excluding gross negligence and breach of confidentiality.",
        standard_risk_guidance="Standard liability cap is 12 months fees paid. Watch out for uncapped or asymmetrical liabilities."
    ),
    PrecedentClause(
        id="prec_term_01",
        category="Termination",
        source="SEC EDGAR 10-K Precedent (SaaS Master Agreement)",
        precedent_text="Either party may terminate this Agreement upon thirty (30) days prior written notice in the event of a material breach remaining uncured after such notice period.",
        standard_risk_guidance="Require a minimum 30-day cure period for material breach. Reject immediate termination without cure."
    ),
    PrecedentClause(
        id="prec_gov_01",
        category="Governing Law",
        source="Delaware Corporate Precedent",
        precedent_text="This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflict of law principles.",
        standard_risk_guidance="Delaware or New York law is standard for commercial agreements. Scrutinize offshore or unfamiliar jurisdictions."
    ),
    PrecedentClause(
        id="prec_conf_01",
        category="Confidentiality",
        source="Standard NDA Precedent",
        precedent_text="Recipient agrees to protect Discloser's Confidential Information with the same degree of care it uses for its own confidential information, but in no event less than reasonable care.",
        standard_risk_guidance="Verify 3 to 5 year confidentiality sunset period and standard exceptions (public domain, independent creation)."
    ),
    PrecedentClause(
        id="prec_ip_01",
        category="Intellectual Property",
        source="Enterprise Tech Licensing Playbook",
        precedent_text="Customer retains all right, title, and interest in and to Customer Data. Vendor retains all rights to the underlying platform and generic algorithms.",
        standard_risk_guidance="Ensure clear boundary protecting customer data ownership and preventing vendor model training on customer IP."
    ),
]


def tokenize_legal_text(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class PrecedentRetriever:
    """
    Retrieves nearest precedent clauses and legal playbook guidance to ground risk explanations.
    """

    def __init__(self, precedents: Optional[List[PrecedentClause]] = None):
        self.precedents = precedents or PRECEDENT_DATABASE
        self._build_vocab()

    def _build_vocab(self):
        self.doc_tokens = [tokenize_legal_text(f"{p.category} {p.precedent_text} {p.standard_risk_guidance}") for p in self.precedents]
        self.vocab = {}
        for tokens in self.doc_tokens:
            for t in tokens:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)

    def _vectorize(self, tokens: List[str]) -> Dict[int, float]:
        vec = {}
        for t in tokens:
            if t in self.vocab:
                idx = self.vocab[t]
                vec[idx] = vec.get(idx, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def retrieve_precedent(self, category: str, clause_text: str) -> Optional[Dict[str, Any]]:
        """
        Finds the closest precedent clause matching the category and semantic context.
        """
        # Exact category match filter
        category_matches = [p for p in self.precedents if p.category == category]
        if not category_matches:
            category_matches = self.precedents

        query_tokens = tokenize_legal_text(clause_text)
        query_vec = self._vectorize(query_tokens)

        best_score = -1.0
        best_doc = category_matches[0]

        for prec in category_matches:
            doc_vec = self._vectorize(tokenize_legal_text(f"{prec.precedent_text} {prec.standard_risk_guidance}"))
            score = sum(v * doc_vec.get(k, 0.0) for k, v in query_vec.items())
            if score > best_score:
                best_score = score
                best_doc = prec

        return {
            "precedent_id": best_doc.id,
            "category": best_doc.category,
            "source": best_doc.source,
            "precedent_text": best_doc.precedent_text,
            "standard_risk_guidance": best_doc.standard_risk_guidance,
            "similarity_score": round(max(0.1, best_score), 4),
        }
