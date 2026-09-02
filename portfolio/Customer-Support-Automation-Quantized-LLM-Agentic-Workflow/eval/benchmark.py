"""
Benchmarking Entrypoint Forwarder
"""
import sys
from src.eval.benchmark import run_comprehensive_benchmark, save_benchmark_artifacts

if __name__ == "__main__":
    results = run_comprehensive_benchmark(iterations=5)
    save_benchmark_artifacts(results, output_dir="results")
