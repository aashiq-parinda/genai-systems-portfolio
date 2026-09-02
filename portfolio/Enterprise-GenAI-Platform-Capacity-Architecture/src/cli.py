"""Interactive CLI for Enterprise GenAI Capacity Sizing & FinOps Simulation."""

import argparse
import json
import os
import sys

from sizing.gpu_capacity_calculator import (
    GPUCapacityCalculator,
    MODEL_PROFILES,
    GPU_PROFILES,
)
from sizing.latency_simulator import LatencySimulator
from finops.tco_engine import FinOpsEngine


def cmd_size_cluster(args):
    """Run GPU cluster sizing calculation."""
    calc = GPUCapacityCalculator()
    model = MODEL_PROFILES.get(args.model, MODEL_PROFILES["Claude-X-Frontier-70B"])
    gpu = GPU_PROFILES.get(args.gpu, GPU_PROFILES["NVIDIA-H100-SXM-80GB"])

    result = calc.size_cluster(
        model=model,
        gpu=gpu,
        target_concurrency=args.concurrency,
        precision=args.precision,
        kv_precision=args.kv_precision,
        average_context_tokens=args.context_tokens,
    )

    print("\n" + "=" * 70)
    print(f"🚀 GPU CLUSTER SIZING SPECIFICATION: {result.model_name}")
    print("=" * 70)
    print(f"• Accelerator Target           : {result.gpu_name}")
    print(f"• Weights Precision            : {result.precision.upper()} ({result.model_weights_vram_gb} GB VRAM)")
    print(f"• Target Concurrency           : {result.target_concurrency:,} concurrent streams")
    print(f"• Average Context Length       : {result.average_context_tokens:,} tokens")
    print(f"• KV-Cache per Request         : {result.kv_cache_per_request_mb} MB")
    print(f"• Total Cluster KV-Cache       : {result.total_kv_cache_gb:,.2f} GB")
    print("-" * 70)
    print(f"• Min Tensor Parallelism (TP)  : {result.min_tensor_parallelism}x")
    print(f"• GPUs per Serving Replica     : {result.gpus_per_replica}")
    print(f"• Active Replicas Needed       : {result.active_replicas_needed}")
    print(f"• Total Replicas (N+1 Failover): {result.n_plus_one_replicas}")
    print(f"• Total GPUs Required          : {result.total_gpus_required} GPUs")
    print(f"• 8-way HGX Nodes Required     : {result.total_nodes_8x} nodes")
    print(f"• Peak Aggregate Throughput    : {result.peak_aggregate_tokens_per_sec:,.0f} tokens/sec")
    print(f"• Est. Monthly Compute Cost    : ₹{result.estimated_monthly_compute_cost_inr_cr:.2f} Crores/month")
    print("=" * 70 + "\n")


def cmd_simulate_latency(args):
    """Simulate latency and SLA distribution."""
    sim = LatencySimulator()
    profile = sim.simulate_profile(
        concurrency=args.concurrency,
        active_gpu_replicas=args.replicas,
        prompt_tokens=args.prompt_tokens,
        generation_tokens=args.gen_tokens
    )

    print("\n" + "=" * 70)
    print(f"⚡ LATENCY & SLA PROFILE (Concurrency: {profile.concurrency_load:,})")
    print("=" * 70)
    print(f"• Time-To-First-Token (P50/P95/P99) : {profile.ttft_p50_ms} ms / {profile.ttft_p95_ms} ms / {profile.ttft_p99_ms} ms")
    print(f"• Inter-Token Latency (P50/P95/P99)  : {profile.itl_p50_ms} ms / {profile.itl_p95_ms} ms / {profile.itl_p99_ms} ms")
    print(f"• End-to-End Latency  (P50/P95/P99)  : {profile.e2e_p50_s} s / {profile.e2e_p95_s} s / {profile.e2e_p99_s} s")
    print(f"• SLA Violation Rate (TTFT > 1.0s)  : {profile.sla_violation_rate * 100:.2f}%")
    print("=" * 70 + "\n")


def cmd_run_tco(args):
    """Run FinOps 3-Year TCO & ROI model."""
    engine = FinOpsEngine(consulting_fee_cr=args.fee_cr)
    tco_options = engine.generate_comparative_tco()
    roi = engine.calculate_business_roi()

    print("\n" + "=" * 80)
    print("💰 3-YEAR ENTERPRISE FINOPS TCO COMPARISON (₹ CRORES)")
    print("=" * 80)
    for key, opt in tco_options.items():
        print(f"\n▶ {opt.name}")
        print(f"  - Year 1 Capex/Opex : ₹{opt.year_1_capex_cr:.1f} Cr Capex + ₹{opt.year_1_opex_cr:.1f} Cr Opex")
        print(f"  - 3-Year Total Cost : ₹{opt.three_year_total_cr:.1f} Crores")
        print(f"  - Data Isolation    : {opt.data_privacy_level}")
        print(f"  - Time to Market    : {opt.time_to_market_months} months")
    
    print("-" * 80)
    print("📊 QUANTIFIED VALUE & CONSULTING ROI:")
    print(f"• Annualized Infra Cost Savings  : ₹{roi.annual_infrastructure_savings_cr:.1f} Crores / year")
    print(f"• 3-Year Cumulative Savings      : ₹{roi.three_year_savings_cr:.1f} Crores")
    print(f"• Annualized Platform Run-Rate   : ₹{roi.annual_platform_revenue_cr:.1f} Crores / year")
    print(f"• 3-Year Net Economic Value      : ₹{roi.projected_three_year_net_value_cr:.1f} Crores")
    print(f"• Multiple on Architecture Fee   : {roi.roi_multiple_on_consulting_fee}x Return")
    print("=" * 80 + "\n")


def cmd_export_receipts(args):
    """Export benchmark receipts to JSON."""
    calc = GPUCapacityCalculator()
    sim = LatencySimulator()
    engine = FinOpsEngine()

    os.makedirs("results", exist_ok=True)

    # 1. Capacity Receipts across scales (10K, 50K, 100K, 1M)
    scales = [10_000, 50_000, 100_000, 1_000_000]
    sizing_receipts = []
    for scale in scales:
        res = calc.size_cluster(
            model=MODEL_PROFILES["Claude-X-Frontier-70B"],
            gpu=GPU_PROFILES["NVIDIA-H100-SXM-80GB"],
            target_concurrency=scale,
            precision="fp8",
            kv_precision="fp8"
        )
        lat = sim.simulate_profile(concurrency=scale, active_gpu_replicas=res.active_replicas_needed)
        sizing_receipts.append({
            "target_concurrency": scale,
            "total_gpus_required": res.total_gpus_required,
            "hgx_nodes_8x": res.total_nodes_8x,
            "ttft_p95_ms": lat.ttft_p95_ms,
            "e2e_p95_s": lat.e2e_p95_s,
            "monthly_cost_inr_cr": res.estimated_monthly_compute_cost_inr_cr
        })

    with open("results/capacity_sizing_receipts.json", "w") as f:
        json.dump(sizing_receipts, f, indent=2)

    # 2. FinOps Receipts
    tco_data = {k: v.__dict__ for k, v in engine.generate_comparative_tco().items()}
    roi_data = engine.calculate_business_roi().__dict__
    
    with open("results/tco_comparison_receipts.json", "w") as f:
        json.dump({"tco_options": tco_data, "business_roi": roi_data}, f, indent=2)

    print("✅ Verified receipts exported to results/capacity_sizing_receipts.json and results/tco_comparison_receipts.json")


def main():
    parser = argparse.ArgumentParser(description="Enterprise GenAI Capacity Sizing & FinOps CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # size-cluster
    p_size = subparsers.add_parser("size-cluster", help="Calculate GPU cluster topology")
    p_size.add_argument("--model", default="Claude-X-Frontier-70B", choices=list(MODEL_PROFILES.keys()))
    p_size.add_argument("--gpu", default="NVIDIA-H100-SXM-80GB", choices=list(GPU_PROFILES.keys()))
    p_size.add_argument("--concurrency", type=int, default=10000)
    p_size.add_argument("--precision", default="fp8", choices=["fp16", "bf16", "fp8", "int4"])
    p_size.add_argument("--kv-precision", default="fp8", choices=["fp16", "bf16", "fp8", "int4"])
    p_size.add_argument("--context-tokens", type=int, default=4096)
    p_size.set_defaults(func=cmd_size_cluster)

    # simulate-latency
    p_lat = subparsers.add_parser("simulate-latency", help="Simulate TTFT and ITL latencies")
    p_lat.add_argument("--concurrency", type=int, default=10000)
    p_lat.add_argument("--replicas", type=int, default=32)
    p_lat.add_argument("--prompt-tokens", type=int, default=2048)
    p_lat.add_argument("--gen-tokens", type=int, default=512)
    p_lat.set_defaults(func=cmd_simulate_latency)

    # run-tco
    p_tco = subparsers.add_parser("run-tco", help="Run comparative FinOps TCO & ROI model")
    p_tco.add_argument("--fee-cr", type=float, default=4.80)
    p_tco.set_defaults(func=cmd_run_tco)

    # export-receipts
    p_exp = subparsers.add_parser("export-receipts", help="Export ground truth benchmark receipts")
    p_exp.set_defaults(func=cmd_export_receipts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
