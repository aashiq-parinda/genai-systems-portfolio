"""
LangFlow Flow Exporter and Schema Validator.
Demonstrates:
1. Programmatic creation and validation of LangFlow JSON flow schemas
2. Inspecting nodes, edges, inputs, outputs, and component connections
3. Exporting flows for one-click drag-and-drop import into LangFlow's web dashboard
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

class FlowBuilder:
    """Helper to programmatically build and validate LangFlow graph topologies."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []

    def add_node(
        self,
        node_id: str,
        display_name: str,
        node_type: str,
        position: Dict[str, float],
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[List[str]] = None
    ):
        node = {
            "id": node_id,
            "type": "genericNode",
            "position": position,
            "data": {
                "type": node_type,
                "node": {
                    "display_name": display_name,
                    "description": f"{display_name} Component",
                    "template": inputs or {},
                    "outputs": [{"name": o, "display_name": o.capitalize()} for o in (outputs or ["output"])]
                }
            }
        }
        self.nodes.append(node)
        return self

    def add_edge(self, source_id: str, source_handle: str, target_id: str, target_handle: str):
        edge = {
            "id": f"reactflow__edge-{source_id}{source_handle}-{target_id}{target_handle}",
            "source": source_id,
            "target": target_id,
            "sourceHandle": source_handle,
            "targetHandle": target_handle,
            "style": {"stroke": "#818cf8", "strokeWidth": 2}
        }
        self.edges.append(edge)
        return self

    def build_flow_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "data": {
                "nodes": self.nodes,
                "edges": self.edges,
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        }

    def export_to_file(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.build_flow_dict(), f, indent=2)

def validate_flow_json(file_path: Path) -> Dict[str, Any]:
    """Validates that a JSON flow file meets LangFlow topological requirements."""
    if not file_path.exists():
        raise FileNotFoundError(f"Flow file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        flow_data = json.load(f)

    data = flow_data.get("data", {})
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    node_ids = {n.get("id") for n in nodes}
    dangling_edges = []

    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in node_ids or tgt not in node_ids:
            dangling_edges.append(edge.get("id"))

    return {
        "flow_name": flow_data.get("name"),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [n.get("data", {}).get("node", {}).get("display_name", n.get("id")) for n in nodes],
        "is_valid": len(dangling_edges) == 0,
        "dangling_edges": dangling_edges
    }
