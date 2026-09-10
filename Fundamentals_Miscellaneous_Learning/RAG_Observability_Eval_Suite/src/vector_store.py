import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

class VectorStore:
    """
    Lightweight vector database supporting text chunking, deterministic cosine similarity,
    and simulated retrieval for CloudFlow internal documentation.
    """
    def __init__(self, docs_dir: Optional[Path] = None):
        self.docs_dir = docs_dir
        self.chunks: List[DocumentChunk] = []
        self._vocabulary: Dict[str, int] = {}
        if self.docs_dir and self.docs_dir.exists():
            self.load_and_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

    def _embed_text(self, text: str, dim: int = 64) -> List[float]:
        """
        Deterministic, dependency-free embedding function using token frequency
        and locality-sensitive hashing for fast, accurate cosine similarity.
        """
        tokens = self._tokenize(text)
        vec = [0.0] * dim
        if not tokens:
            return vec
        for token in tokens:
            # Deterministic hash mapping
            h = hash(token) % dim
            vec[h] += 1.0
        # L2 normalization
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2))

    def load_and_index(self):
        """Parse markdown files in docs_dir, chunk by headings, and index."""
        self.chunks.clear()
        doc_files = list(self.docs_dir.glob("*.md"))
        for doc_file in doc_files:
            text = doc_file.read_text(encoding="utf-8")
            # Chunk by section headers (## or ###)
            sections = re.split(r"(?=\n##\s+)", text)
            for idx, section in enumerate(sections):
                stripped = section.strip()
                if not stripped:
                    continue
                # Extract header line
                first_line = stripped.split("\n", 1)[0].replace("#", "").strip()
                chunk_id = f"{doc_file.stem}_chunk_{idx}"
                self.chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        source_file=doc_file.name,
                        title=first_line or doc_file.stem,
                        content=stripped,
                        metadata={"source": doc_file.name, "section": first_line}
                    )
                )

    def retrieve(self, query: str, top_k: int = 3, force_irrelevant: bool = False) -> List[DocumentChunk]:
        """
        Retrieve relevant chunks for a user query.
        If `force_irrelevant` is True (Failure Injection 1), returns mismatched out-of-domain chunks.
        """
        if not self.chunks:
            return []

        if force_irrelevant:
            # Deliberately return completely unrelated chunks with low relevance
            # e.g., only returning the document header or non-matching section
            irrelevant_chunks = [
                DocumentChunk(
                    chunk_id="corrupted_chunk_0",
                    source_file="cookie_banner_terms.md",
                    title="Third-Party Tracking Cookies",
                    content="We use cookies to analyze web traffic. CloudFlow uses analytics providers to monitor bounce rates on public marketing landing pages.",
                    metadata={"source": "cookie_banner_terms.md", "injected_fault": "bad_retrieval"},
                    score=0.08
                )
            ]
            return irrelevant_chunks

        query_vec = self._embed_text(query)
        scored_chunks: List[DocumentChunk] = []

        for chunk in self.chunks:
            chunk_vec = self._embed_text(chunk.content + " " + chunk.title)
            score = self._cosine_similarity(query_vec, chunk_vec)
            scored_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                title=chunk.title,
                content=chunk.content,
                metadata=chunk.metadata,
                score=round(score, 4)
            )
            scored_chunks.append(scored_chunk)

        # Sort descending by cosine similarity
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        return scored_chunks[:top_k]
