"""Enterprise GenAI FinOps & Total Cost of Ownership (TCO) Simulator.

Compares multi-year Capex/Opex economics across:
1. Option A: Training Foundation Model from Scratch (~500 - 2,000+ Cr)
2. Option B: Unmanaged Public SaaS APIs (~120 - 160 Cr/yr at 10M scale)
3. Option C: Private Isolated Inference + Dynamic Routing (~25 - 40 Cr/yr)

Quantifies business ROI, infrastructure cost avoidance, and annualized platform revenue.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class TCOOption:
    name: str
    year_1_capex_cr: float
    year_1_opex_cr: float
    year_2_opex_cr: float
    year_3_opex_cr: float
    three_year_total_cr: float
    data_privacy_level: str
    risk_profile: str
    time_to_market_months: int


@dataclass
class BusinessROISummary:
    annual_infrastructure_savings_cr: float
    three_year_savings_cr: float
    annual_platform_revenue_cr: float
    projected_three_year_net_value_cr: float
    roi_multiple_on_consulting_fee: float


class FinOpsEngine:
    """Enterprise FinOps & TCO evaluation engine."""

    def __init__(self, consulting_fee_cr: float = 4.80):
        self.consulting_fee_cr = consulting_fee_cr

    def generate_comparative_tco(self) -> Dict[str, TCOOption]:
        """Generate 3-year TCO projections across architectural strategies."""
        options = {
            "Option_A_Train_From_Scratch": TCOOption(
                name="Train Custom Frontier Model From Scratch",
                year_1_capex_cr=450.0,   # Massive GPU cluster acquisition / supercomputer lease
                year_1_opex_cr=120.0,    # Top AI research talent, power, data curation
                year_2_opex_cr=90.0,     # Post-training, continuous pretraining, alignment
                year_3_opex_cr=90.0,
                three_year_total_cr=750.0,
                data_privacy_level="High",
                risk_profile="Extreme (high probability of model underperforming frontier)",
                time_to_market_months=18
            ),
            "Option_B_Public_SaaS_APIs": TCOOption(
                name="Unmanaged Third-Party Public SaaS APIs",
                year_1_capex_cr=2.0,     # Basic integration & gateway setup
                year_1_opex_cr=135.0,    # Raw token billing for 10M users / 100K DAU
                year_2_opex_cr=155.0,    # Scaling token consumption
                year_3_opex_cr=180.0,
                three_year_total_cr=472.0,
                data_privacy_level="Low (Enterprise IP & telemetry shared with external vendor)",
                risk_profile="High (Data sovereignty & regulatory non-compliance)",
                time_to_market_months=2
            ),
            "Option_C_Private_Routed_Platform": TCOOption(
                name="Private Licensed Inference + Dynamic Model Routing",
                year_1_capex_cr=12.0,    # Air-gapped GPU cluster deployment & enterprise gateway
                year_1_opex_cr=28.0,     # Model weight licensing + private GPU hosting + FinOps routing
                year_2_opex_cr=32.0,     # Scaled capacity for 50+ bots
                year_3_opex_cr=36.0,
                three_year_total_cr=108.0,
                data_privacy_level="Maximum (100% On-prem / Air-gapped VPC, Zero Data Leakage)",
                risk_profile="Low (Proven frontier weights + SLA isolation)",
                time_to_market_months=3
            )
        }
        return options

    def calculate_business_roi(self) -> BusinessROISummary:
        """Calculate quantified cost savings and platform revenue impact."""
        # Baseline Public API vs Private Routed Platform
        annual_savings_cr = 135.0 - 28.0  # ~107 Cr nominal, conservatively modeled as 70-90 Cr
        conservative_annual_savings = 82.5  # Midpoint of ₹70 - ₹90 Cr
        three_year_savings = conservative_annual_savings * 3

        # Projected annualized enterprise subscription / internal cross-charge revenue
        conservative_annual_revenue = 135.0  # Midpoint of ₹120 - ₹150 Cr
        
        # 3-Year Net Economic Value Created
        total_value_3yr = three_year_savings + (conservative_annual_revenue * 3) - 108.0  # minus platform cost
        
        # Multiple on the ₹4.8 Cr architecture fee
        roi_multiple = round(total_value_3yr / self.consulting_fee_cr, 1)

        return BusinessROISummary(
            annual_infrastructure_savings_cr=conservative_annual_savings,
            three_year_savings_cr=round(three_year_savings, 2),
            annual_platform_revenue_cr=conservative_annual_revenue,
            projected_three_year_net_value_cr=round(total_value_3yr, 2),
            roi_multiple_on_consulting_fee=roi_multiple,
        )
