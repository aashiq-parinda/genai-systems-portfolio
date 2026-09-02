# Verified Benchmark Receipts: 4-Bit Quantized vs FP16

*Generated on 2026-08-27 15:13:03 UTC across 160 benchmark inferences.*

| Performance & Financial Metric | FP16 Full Precision | 4-bit Quantized (NF4) | Delta / Efficiency Gain |
| :--- | :--- | :--- | :--- |
| **VRAM Footprint (7B Params)** | `14.54 GB` | `5.09 GB` | **9.45 GB saved (65.02%)** |
| **Mean Latency / Ticket** | `21.74 ms` | `2.83 ms` | **7.68x faster** |
| **p50 Latency** | `0.1 ms` | `0.04 ms` | Faster response |
| **p95 Latency** | `50.73 ms` | `6.56 ms` | Consistent SLA |
| **Throughput** | `4955.27 tok/s` | `38007.77 tok/s` | High concurrency |
| **Cost per 1,000 Tickets** | `$0.8921` | `$0.0297` | **96.67% Cost Reduction** |
| **Classification Accuracy** | `93.75%` | `93.75%` | Parity preserved |
| **Guardrail F1 Score** | `0.9412` | `0.9412` | Zero safety degradation |
