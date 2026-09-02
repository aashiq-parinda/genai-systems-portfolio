"""
Stateful Agentic Contract Risk Review Pipeline (LangGraph-style State Machine)
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

from src.core.segmenter import ContractSegmenter, ContractClause
from src.core.lora_classifier import LoRAClauseClassifier, LoRAConfig
from src.core.risk_taxonomy import assess_clause_risk, RiskTier
from src.core.precedent_retriever import PrecedentRetriever


class ReviewStatus(str, Enum):
    AUTO_FINALIZED = "AUTO_FINALIZED"
    FLAGGED_FOR_HUMAN_REVIEW = "FLAGGED_FOR_HUMAN_REVIEW"
    CRITICAL_RISK_ESCALATED = "CRITICAL_RISK_ESCALATED"


@dataclass
class ClauseAnalysisResult:
    clause_index: int
    title: str
    clause_text: str
    predicted_category: str
    confidence: float
    top_3_predictions: List[Dict[str, Any]]
    risk_tier: str
    is_high_risk: bool
    risk_triggers: List[str]
    precedent: Optional[Dict[str, Any]]
    risk_explanation: str
    status: ReviewStatus
    flag_reason: Optional[str]


@dataclass
class ContractReviewReport:
    total_clauses: int
    overall_risk_tier: str
    high_risk_clause_count: int
    human_review_queue_count: int
    auto_finalized_count: int
    total_processing_time_ms: float
    clause_analyses: List[ClauseAnalysisResult]
    executive_summary: str


class ContractRiskReviewAgent:
    """
    Production Multi-Stage Agentic Contract Review Engine:
    1. Legal Document Clause Segmentation
    2. LoRA PEFT Clause Category Classification
    3. Deterministic Risk Tier & Trigger Rule Assessment
    4. Precedent Vector Knowledge Retrieval
    5. Plain-English Risk Explanation Synthesis
    6. Confidence Guardrail & Human Review Queue Routing
    7. Overall Contract Summary Compilation
    """

    def __init__(
        self,
        segmenter: Optional[ContractSegmenter] = None,
        classifier: Optional[LoRAClauseClassifier] = None,
        retriever: Optional[PrecedentRetriever] = None,
        confidence_threshold: float = 0.75,
    ):
        self.segmenter = segmenter or ContractSegmenter()
        self.classifier = classifier or LoRAClauseClassifier(confidence_threshold=confidence_threshold)
        self.retriever = retriever or PrecedentRetriever()
        self.confidence_threshold = confidence_threshold

    def _generate_explanation(
        self,
        clause_text: str,
        category: str,
        risk_tier: str,
        triggers: List[str],
        precedent: Optional[Dict[str, Any]],
    ) -> str:
        """Synthesizes plain-English risk explanation and negotiation guidance."""
        if triggers:
            trigger_summary = " ".join(triggers)
            guidance = precedent["standard_risk_guidance"] if precedent else "Recommend legal redline review."
            return (
                f"**Risk Flag ({risk_tier})**: {trigger_summary} "
                f"**Standard Playbook Comparison**: {guidance}"
            )

        if risk_tier in [RiskTier.CRITICAL.value, RiskTier.HIGH.value]:
            return (
                f"This clause addresses **{category}** under a **{risk_tier}** risk profile. "
                f"Ensure liability caps and reciprocal obligations conform to company contracting standards."
            )

        return (
            f"Standard **{category}** clause conforming to normal commercial contracting terms."
        )

    def analyze_single_clause(self, clause: ContractClause) -> ClauseAnalysisResult:
        """Executes full review DAG for an individual clause."""
        # 1. Classification
        clf_result = self.classifier.classify_clause(clause.text)
        category = clf_result["predicted_category"]
        confidence = clf_result["confidence"]
        top_3 = clf_result["top_3_predictions"]

        # 2. Risk Assessment
        risk_result = assess_clause_risk(category, clause.text)
        risk_tier = risk_result["risk_tier"]
        is_high_risk = risk_result["is_high_risk"]
        triggers = risk_result["triggers"]

        # 3. Precedent Retrieval
        precedent = self.retriever.retrieve_precedent(category, clause.text)

        # 4. Plain-English Explanation
        explanation = self._generate_explanation(
            clause_text=clause.text,
            category=category,
            risk_tier=risk_tier,
            triggers=triggers,
            precedent=precedent,
        )

        # 5. Guardrail check & routing
        if risk_tier == RiskTier.CRITICAL.value:
            status = ReviewStatus.CRITICAL_RISK_ESCALATED
            flag_reason = "Critical risk tier or severe trigger terms detected."
        elif confidence < self.confidence_threshold:
            status = ReviewStatus.FLAGGED_FOR_HUMAN_REVIEW
            flag_reason = f"Classification confidence ({confidence:.2f}) below threshold ({self.confidence_threshold:.2f})."
        elif is_high_risk:
            status = ReviewStatus.FLAGGED_FOR_HUMAN_REVIEW
            flag_reason = "High-risk category requires secondary review."
        else:
            status = ReviewStatus.AUTO_FINALIZED
            flag_reason = None

        return ClauseAnalysisResult(
            clause_index=clause.index,
            title=clause.title or f"Clause {clause.index}",
            clause_text=clause.text,
            predicted_category=category,
            confidence=confidence,
            top_3_predictions=top_3,
            risk_tier=risk_tier,
            is_high_risk=is_high_risk,
            risk_triggers=triggers,
            precedent=precedent,
            risk_explanation=explanation,
            status=status,
            flag_reason=flag_reason,
        )

    def review_contract(self, contract_text: str) -> ContractReviewReport:
        """
        Executes end-to-end multi-agent contract review over complete document.
        """
        start_time = time.perf_counter()

        # Step 1: Segmentation
        clauses = self.segmenter.segment(contract_text)
        if not clauses:
            # Fallback if unsegmented
            clauses = [ContractClause(index=1, title="Full Text", text=contract_text)]

        # Step 2: Loop over clauses
        results: List[ClauseAnalysisResult] = []
        high_risk_count = 0
        queue_count = 0
        finalized_count = 0

        for clause in clauses:
            res = self.analyze_single_clause(clause)
            results.append(res)
            if res.is_high_risk:
                high_risk_count += 1
            if res.status != ReviewStatus.AUTO_FINALIZED:
                queue_count += 1
            else:
                finalized_count += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Overall contract risk determination
        if any(r.risk_tier == RiskTier.CRITICAL.value for r in results):
            overall_tier = RiskTier.CRITICAL.value
        elif high_risk_count > 0:
            overall_tier = RiskTier.HIGH.value
        elif any(r.risk_tier == RiskTier.MEDIUM.value for r in results):
            overall_tier = RiskTier.MEDIUM.value
        else:
            overall_tier = RiskTier.LOW.value

        summary = (
            f"Contract contains {len(clauses)} analyzed clauses with an overall **{overall_tier}** risk assessment. "
            f"{high_risk_count} high/critical risk clauses detected. "
            f"{queue_count} clauses routed to the Human Review Queue for attorney audit; "
            f"{finalized_count} clauses auto-finalized."
        )

        return ContractReviewReport(
            total_clauses=len(clauses),
            overall_risk_tier=overall_tier,
            high_risk_clause_count=high_risk_count,
            human_review_queue_count=queue_count,
            auto_finalized_count=finalized_count,
            total_processing_time_ms=round(elapsed_ms, 2),
            clause_analyses=results,
            executive_summary=summary,
        )
