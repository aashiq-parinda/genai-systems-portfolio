"""
Unit tests for LoRA Clause Classifier & Calibration
"""
import pytest
from src.core.lora_classifier import LoRAClauseClassifier, LoRAConfig


@pytest.fixture
def classifier():
    return LoRAClauseClassifier(confidence_threshold=0.75)


def test_lora_config_dict():
    config = LoRAConfig(r=8, lora_alpha=16)
    d = config.to_dict()
    assert d["r"] == 8
    assert d["lora_alpha"] == 16
    assert "query_proj" in d["target_modules"]


def test_classify_indemnification(classifier):
    text = "Vendor shall defend, indemnify, and hold harmless Customer from any liabilities and third-party claims."
    res = classifier.classify_clause(text)
    assert res["predicted_category"] == "Indemnification"
    assert res["confidence"] >= 0.75
    assert len(res["top_3_predictions"]) == 3
    assert res["top_3_predictions"][0]["category"] == "Indemnification"


def test_classify_limitation_of_liability(classifier):
    text = "Neither party shall be liable for indirect, incidental, or consequential damages. Aggregate liability is capped."
    res = classifier.classify_clause(text)
    assert res["predicted_category"] == "Limitation of Liability"


def test_classify_low_confidence(classifier):
    text = "Random generic text without legal terminology xyz."
    res = classifier.classify_clause(text)
    assert res["requires_human_review"] is True
    assert "below threshold" in res["review_reason"]
