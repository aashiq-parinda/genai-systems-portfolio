"""Unit tests for FinOps TCO & ROI simulation engine."""

import pytest
from src.finops.tco_engine import FinOpsEngine


def test_tco_comparison_generation():
    engine = FinOpsEngine(consulting_fee_cr=4.80)
    options = engine.generate_comparative_tco()

    assert "Option_A_Train_From_Scratch" in options
    assert "Option_B_Public_SaaS_APIs" in options
    assert "Option_C_Private_Routed_Platform" in options

    # Option A (Scratch) is the most expensive
    assert options["Option_A_Train_From_Scratch"].three_year_total_cr >= 700.0

    # Option C (Private platform) is the most cost-effective long term
    assert options["Option_C_Private_Routed_Platform"].three_year_total_cr <= 150.0


def test_business_roi_metrics():
    engine = FinOpsEngine(consulting_fee_cr=4.80)
    roi = engine.calculate_business_roi()

    assert 70.0 <= roi.annual_infrastructure_savings_cr <= 90.0
    assert 120.0 <= roi.annual_platform_revenue_cr <= 150.0
    assert roi.roi_multiple_on_consulting_fee > 100.0  # Over 100x return
