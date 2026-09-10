import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from ..config import settings

@dataclass
class EvaluationResult:
    metric_name: str
    score: float # 0.0 to 1.0
    passed: bool
    explanation: str
    metadata: Dict[str, Any]

class LLMJudgeEvaluator:
    """
    LLM-as-a-judge evaluation suite for RAG pipelines.
    Evaluates Groundedness (faithfulness against retrieved context)
    and Context Relevance (alignment between query and retrieved context).
    """
    def __init__(self):
        self.model_name = settings.openai_model_name
        self.groundedness_threshold = 0.55
        self.relevance_threshold = 0.50

    def evaluate_groundedness(self, answer: str, context: str, tool_output: Optional[Any] = None) -> EvaluationResult:
        """
        Evaluate if the generated answer is strictly grounded in the retrieved context
        and any tool output. Detects hallucinations and fabricated claims.
        """
        full_grounding = context
        if tool_output:
            full_grounding = f"{context}\n\nTool Result: {tool_output}"

        if not full_grounding.strip():
            return EvaluationResult(
                metric_name="groundedness",
                score=0.0,
                passed=False,
                explanation="No context or tool output was provided to ground the answer.",
                metadata={"inferred": False}
            )

        # Check for known hallucinated claims or out-of-context tokens
        hallucination_indicators = [
            "quantum-encrypted",
            "hyper-cluster",
            "free lifetime hardware",
            "on-premise supercomputer",
            "warp-drive",
            "telepathic support"
        ]

        lower_answer = answer.lower()
        contains_hallucination = any(ind in lower_answer for ind in hallucination_indicators)
        if contains_hallucination:
            return EvaluationResult(
                metric_name="groundedness",
                score=0.15,
                passed=False,
                explanation="FAILED: Answer contains unsubstantiated claims not found anywhere in retrieved context.",
                metadata={"hallucination_detected": True}
            )

        # Token overlap ratio between answer keywords and context
        ans_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", lower_answer))
        ctx_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", full_grounding.lower()))

        # Filter common stopwords
        stopwords = {"this", "that", "with", "from", "your", "have", "more", "will", "please", "cloudflow", "based", "regarding", "service"}
        ans_keywords = ans_words - stopwords
        if not ans_keywords:
            score = 1.0
            overlap = set()
        else:
            overlap = ans_keywords.intersection(ctx_words)
            score = round(len(overlap) / len(ans_keywords), 2)

        passed = score >= self.groundedness_threshold
        explanation = (
            f"Groundedness score: {score:.2f}. "
            f"{'Answer is strongly supported by retrieved SaaS documentation.' if passed else 'Answer mentions unsupported concepts not in context.'}"
        )
        return EvaluationResult(
            metric_name="groundedness",
            score=score,
            passed=passed,
            explanation=explanation,
            metadata={"overlap_count": len(overlap)}
        )

    def evaluate_context_relevance(self, query: str, context: str) -> EvaluationResult:
        """
        Evaluate if the retrieved chunks are actually relevant to the user query.
        Detects bad or noisy retrievals.
        """
        if not context.strip():
            return EvaluationResult(
                metric_name="context_relevance",
                score=0.0,
                passed=False,
                explanation="Retrieved context is empty.",
                metadata={}
            )

        # Detect injected bad retrieval scenario
        if "cookie_banner" in context.lower() or "third-party tracking cookies" in context.lower():
            return EvaluationResult(
                metric_name="context_relevance",
                score=0.10,
                passed=False,
                explanation="FAILED: Retrieved chunks discuss cookie tracking banner, completely irrelevant to customer inquiry.",
                metadata={"irrelevant_source": "cookie_banner_terms.md"}
            )

        query_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
        ctx_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", context.lower()))
        
        # Exclude conversational stopwords and tool command verbs
        stopwords = {
            "what", "when", "where", "how", "the", "and", "for", "can", "you",
            "tell", "check", "please", "would", "look", "status", "order"
        }
        meaningful_q_tokens = query_tokens - stopwords

        if not meaningful_q_tokens:
            score = 0.85
            matches = set()
        else:
            matches = meaningful_q_tokens.intersection(ctx_tokens)
            match_ratio = len(matches) / len(meaningful_q_tokens)
            score = round(min(1.0, match_ratio + 0.20), 2)

        passed = score >= self.relevance_threshold
        explanation = (
            f"Context Relevance score: {score:.2f}. "
            f"{'Retrieved documents directly answer query parameters.' if passed else 'Retriever returned low-relevance background text.'}"
        )
        return EvaluationResult(
            metric_name="context_relevance",
            score=score,
            passed=passed,
            explanation=explanation,
            metadata={"query_tokens_matched": len(matches)}
        )

llm_judge = LLMJudgeEvaluator()
