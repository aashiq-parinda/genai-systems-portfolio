#!/usr/bin/env python3
"""
LangFlow Visual Dashboard Launcher & Manager.
Manages launching the official LangFlow visual canvas (http://127.0.0.1:7860)
and validates pre-packaged flow JSON blueprints.
"""

import sys
import os
import shutil
import socket
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import Config
from src.langflow_flows.flow_exporter import validate_flow_json

FLOWS_DIR = BASE_DIR / "flows"

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def check_flows():
    print("\n🔍 Validating Pre-Packaged LangFlow Visual Flows:")
    flow_files = list(FLOWS_DIR.glob("*.json"))
    if not flow_files:
        print("  ⚠️ No flow JSON files found in flows/ directory.")
        return

    for flow_path in sorted(flow_files):
        validation = validate_flow_json(flow_path)
        status_icon = "✅" if validation["is_valid"] else "❌"
        print(f"  {status_icon} [{flow_path.name}] - '{validation['flow_name']}'")
        print(f"     Nodes: {validation['node_count']} | Edges: {validation['edge_count']}")
        print(f"     Components: {', '.join(validation['nodes'][:4])}...")
        if not validation["is_valid"]:
            print(f"     ⚠️ Dangling edges detected: {validation['dangling_edges']}")

def print_banner():
    print("=" * 70)
    print(" 🌊 OFFICIAL LANGFLOW VISUAL DASHBOARD LAUNCHER")
    print("=" * 70)
    print(f" Target Web Canvas: http://{Config.LANGFLOW_HOST}:{Config.LANGFLOW_PORT}")
    print(f" Flows Directory:   {FLOWS_DIR}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Launch or inspect the official LangFlow dashboard.")
    parser.add_argument("--check-only", action="store_true", help="Validate flows and environment without launching.")
    parser.add_argument("--port", type=int, default=Config.LANGFLOW_PORT, help="Port to bind LangFlow server.")
    parser.add_argument("--host", type=str, default=Config.LANGFLOW_HOST, help="Host address to bind.")
    parser.add_argument("--docker", action="store_true", help="Print Docker launch command for containerized LangFlow.")
    args = parser.parse_args()

    print_banner()
    check_flows()

    # Check if port is already active
    port_busy = is_port_in_use(args.port, args.host)
    if port_busy:
        print(f"\n🟢 Notice: Port {args.port} is already active!")
        print(f"👉 You can access the running dashboard directly at: http://{args.host}:{args.port}")
        return

    # Check Docker command if requested
    if args.docker:
        print("\n🐳 Containerized LangFlow Launch Command:")
        print(f"  docker run -p {args.port}:7860 --name langflow logspace/langflow:latest")
        print(f"\nOnce started, navigate to: http://127.0.0.1:{args.port}")
        return

    # Check if uvx is available
    uvx_cmd = shutil.which("uvx") or (Path.home() / ".local" / "bin" / "uvx" if (Path.home() / ".local" / "bin" / "uvx").exists() else None)

    # Check if langflow executable is installed
    langflow_cmd = shutil.which("langflow")
    if not langflow_cmd:
        # Check in current python environment
        py_langflow = Path(sys.executable).parent / "langflow"
        if py_langflow.exists():
            langflow_cmd = str(py_langflow)

    print("\n📦 Environment Inspection:")
    if langflow_cmd:
        print(f"  ✅ LangFlow binary found at: {langflow_cmd}")
        if args.check_only:
            print("  [--check-only mode active] Exiting without spawning server.")
            return

        print(f"\n🚀 Launching official LangFlow visual server on http://{args.host}:{args.port}...")
        print("   (Press Ctrl+C to stop the dashboard server)\n")
        try:
            subprocess.run([langflow_cmd, "run", "--host", args.host, "--port", str(args.port)])
        except KeyboardInterrupt:
            print("\n🛑 LangFlow server stopped by user.")
    elif uvx_cmd:
        print(f"  ⚡ Fast runner 'uvx' detected at: {uvx_cmd}")
        print("\n🚀 You can run LangFlow directly with uvx (no manual virtualenv install needed):")
        print(f"   uvx langflow run --host {args.host} --port {args.port}")
        print("\nOr install directly into your active virtualenv:")
        print("   pip install langflow")
        print(f"   python3 launch_langflow.py --port {args.port}")
    else:
        print("  ⚠️ 'langflow' is not yet installed in this Python environment.")
        print("\n📥 Installation Options:")
        print("  Option 1 (pip install into virtual environment):")
        print("     pip install langflow")
        print(f"     python3 launch_langflow.py --port {args.port}")
        print("\n  Option 2 (run immediately via Docker container):")
        print(f"     docker run -p {args.port}:7860 -it logspace/langflow:latest")

    print(f"\nOnce the LangFlow web dashboard opens at http://127.0.0.1:{args.port}:")
    print("  1. Click 'New Flow' or 'Upload / Import'")
    print("  2. Select any flow from the `flows/` folder in this project:")
    print("     - flows/01_lcel_conversational_rag.json")
    print("     - flows/02_langgraph_cyclic_agent.json")
    print("  3. Drag, connect, and interact with the nodes visually on the canvas!")

if __name__ == "__main__":
    main()
