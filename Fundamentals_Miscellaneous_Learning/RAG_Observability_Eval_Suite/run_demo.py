#!/usr/bin/env python3
"""
Entrypoint demo runner for CloudFlow Customer Support RAG Observability Suite.
Executes baseline query, runs the 3 injected failure modes, logs OpenTelemetry traces,
and prints the observability comparison matrix.
"""
import sys
from pathlib import Path

# Ensure root is in sys.path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from dashboard.comparison_report import run_comparative_benchmark

def main():
    report_md = run_comparative_benchmark()
    print("\n✅ Observability benchmark execution complete!")
    print(f"📄 Full comparison dashboard saved to: {current_dir / 'dashboard' / 'COMPARISON_DASHBOARD.md'}")

if __name__ == "__main__":
    main()
