"""
Enterprise Hybrid Policy Retriever (BM25 Sparse + Dense Vector Cosine Similarity + Reciprocal Rank Fusion)
"""
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


@dataclass
class Document:
    id: str
    category: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


ENTERPRISE_KNOWLEDGE_BASE: List[Document] = [
    Document(
        id="kb_ord_001",
        category="Order Status",
        title="Shipping & Fulfillment Policy",
        content="Standard orders are processed within 1-2 business days. Express orders ship the same day if placed before 1 PM EST. Tracking numbers are transmitted via email and SMS upon carrier pickup.",
        metadata={"priority": 1, "department": "Logistics"}
    ),
    Document(
        id="kb_ord_002",
        category="Order Status",
        title="International Delivery & Customs",
        content="International shipments take 7-14 business days. Customs duties and import taxes are calculated at checkout and prepaid for DDP destinations.",
        metadata={"priority": 2, "department": "Logistics"}
    ),
    Document(
        id="kb_ref_001",
        category="Refund Request",
        title="Standard 30-Day Return Policy",
        content="Items in original, unworn condition with tags attached can be returned within 30 days of delivery for a full refund to the original payment method. Return shipping is free.",
        metadata={"priority": 1, "department": "Billing"}
    ),
    Document(
        id="kb_ref_002",
        category="Refund Request",
        title="Non-Refundable & Final Sale Items",
        content="Gift cards, customized/engraved merchandise, and clearance items marked Final Sale are ineligible for refunds unless arriving defective or damaged in transit.",
        metadata={"priority": 2, "department": "Billing"}
    ),
    Document(
        id="kb_tech_001",
        category="Technical Support",
        title="Mobile App Crash & Performance Diagnostics",
        content="For persistent mobile crashes, clear cache via Settings > Storage, verify iOS/Android OS version compatibility, or reinstall app v4.8+. Diagnostic logs can be exported from Settings > Help.",
        metadata={"priority": 1, "department": "Engineering"}
    ),
    Document(
        id="kb_tech_002",
        category="Technical Support",
        title="Web Portal Browser Compatibility",
        content="The web portal supports Chromium-based browsers (Chrome, Edge, Brave), Firefox 115+, and Safari 16+. Disable conflicting ad-blocker extensions if checkout page fails to render.",
        metadata={"priority": 2, "department": "Engineering"}
    ),
    Document(
        id="kb_sec_001",
        category="Account Security",
        title="Two-Factor Authentication & Account Recovery",
        content="Users can reset passwords via verified email magic links or SMS OTP. If 2FA authenticator device is lost, recovery requires verifying government ID via secure verification portal.",
        metadata={"priority": 1, "department": "Security"}
    ),
    Document(
        id="kb_bil_001",
        category="Billing Dispute",
        title="Duplicate Charges & Invoicing Reconciliation",
        content="Pending authorization holds normally clear within 48-72 business hours. If duplicate charges post to your bank statement, submit statement PDF for an immediate manual credit reversal.",
        metadata={"priority": 1, "department": "Billing"}
    ),
    Document(
        id="kb_gen_001",
        category="General Inquiry",
        title="Support Operational Hours & SLA",
        content="Live chat support operates 24/7. Phone support is available Monday through Friday from 8 AM to 8 PM EST with an average response time of under 3 minutes.",
        metadata={"priority": 1, "department": "Support Ops"}
    ),
]


def tokenize(text: str) -> List[str]:
    """Tokenizes text into normalized lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class HybridPolicyRetriever:
    """
    Production Multi-stage Hybrid Retriever:
    1. Sparse retrieval: BM25 keyword score
    2. Dense retrieval: Cosine semantic overlap score
    3. Fusion: Reciprocal Rank Fusion (RRF) with category weighting
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60
    ):
        self.documents = documents or ENTERPRISE_KNOWLEDGE_BASE
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self._build_index()

    def _build_index(self):
        """Initializes BM25 and term vector indices."""
        self.corpus_tokens = [tokenize(f"{d.title} {d.content}") for d in self.documents]
        if HAS_BM25:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

        # Build vocabulary for dense vector TF-IDF cosine approximation
        self.vocab: Dict[str, int] = {}
        for doc_tokens in self.corpus_tokens:
            for token in doc_tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        self.doc_vectors = [self._compute_vector(tokens) for tokens in self.corpus_tokens]

    def _compute_vector(self, tokens: List[str]) -> Dict[int, float]:
        """Calculates normalized term frequency vector."""
        counts: Dict[int, float] = {}
        for token in tokens:
            if token in self.vocab:
                idx = self.vocab[token]
                counts[idx] = counts.get(idx, 0.0) + 1.0
        # Normalize
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return {k: v / norm for k, v in counts.items()}

    def _cosine_similarity(self, vec1: Dict[int, float], vec2: Dict[int, float]) -> float:
        """Calculates dot product cosine similarity between sparse representation vectors."""
        score = 0.0
        for k, v in vec1.items():
            if k in vec2:
                score += v * vec2[k]
        return score

    def _bm25_search(self, query_tokens: List[str]) -> List[float]:
        """Returns BM25 scores for all documents in index."""
        if HAS_BM25 and self.bm25:
            return list(self.bm25.get_scores(query_tokens))
        
        # Pure-Python fallback BM25
        scores = []
        n_docs = len(self.documents)
        avgdl = sum(len(d) for d in self.corpus_tokens) / max(1, n_docs)
        k1 = 1.5
        b = 0.75

        for doc_tokens in self.corpus_tokens:
            doc_len = len(doc_tokens)
            doc_score = 0.0
            for q in query_tokens:
                if q in self.vocab:
                    n_containing = sum(1 for d in self.corpus_tokens if q in d)
                    idf = math.log((n_docs - n_containing + 0.5) / (n_containing + 0.5) + 1.0)
                    freq = doc_tokens.count(q)
                    tf = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (doc_len / avgdl)))
                    doc_score += idf * tf
            scores.append(doc_score)
        return scores

    def retrieve(
        self,
        query: str,
        category_filter: Optional[str] = None,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid BM25 + dense search with Reciprocal Rank Fusion (RRF) and metadata filtering.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # 1. Sparse Scores & Ranking
        sparse_scores = self._bm25_search(query_tokens)
        sparse_ranked_indices = sorted(range(len(self.documents)), key=lambda i: sparse_scores[i], reverse=True)

        # 2. Dense Scores & Ranking
        query_vec = self._compute_vector(query_tokens)
        dense_scores = [self._cosine_similarity(query_vec, doc_vec) for doc_vec in self.doc_vectors]
        dense_ranked_indices = sorted(range(len(self.documents)), key=lambda i: dense_scores[i], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(sparse_ranked_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))

        for rank, idx in enumerate(dense_ranked_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self.dense_weight * (1.0 / (self.rrf_k + rank + 1))

        # Category boost
        if category_filter:
            for idx, doc in enumerate(self.documents):
                if doc.category == category_filter:
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) * 1.5

        # Sort combined results
        final_ranked = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results = []
        for idx in final_ranked:
            doc = self.documents[idx]
            # If category filter provided, prioritize matching categories
            if category_filter and doc.category != category_filter and len(results) >= top_k:
                continue
            results.append({
                "id": doc.id,
                "category": doc.category,
                "title": doc.title,
                "content": doc.content,
                "metadata": doc.metadata,
                "hybrid_score": round(rrf_scores[idx], 5),
                "bm25_score": round(sparse_scores[idx], 4),
                "dense_score": round(dense_scores[idx], 4),
            })
            if len(results) >= top_k:
                break

        return results
