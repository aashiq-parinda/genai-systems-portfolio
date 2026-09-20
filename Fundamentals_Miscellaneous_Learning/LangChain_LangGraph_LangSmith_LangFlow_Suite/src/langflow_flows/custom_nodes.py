"""
LangFlow Custom Component Blueprint.
Demonstrates:
1. How to create native custom components for LangFlow's visual canvas
2. Input field types (Prompt, Model, Threshold, Dropdown)
3. The build() method execution protocol
"""

class BaseLangFlowComponent:
    """Mock interface reflecting LangFlow's native CustomComponent SDK."""
    display_name: str = "Custom Component"
    description: str = "Component Description"
    icon: str = "Code"

    def build_config(self):
        return {}

    def build(self, *args, **kwargs):
        raise NotImplementedError

class VectorDBSwitcherComponent(BaseLangFlowComponent):
    """
    Production-grade custom LangFlow component:
    Dynamically switches between Pinecone, Qdrant, and Milvus vector retrievers
    based on tenancy, metadata, or query requirements.
    """
    display_name = "Dynamic Vector DB Switcher"
    description = "Routes retrieval queries to either Qdrant, Pinecone, or Milvus."
    icon = "Database"

    def build_config(self):
        return {
            "query": {"display_name": "Search Query", "field_type": "str"},
            "provider": {
                "display_name": "Target Vector DB",
                "field_type": "str",
                "options": ["Qdrant", "Pinecone", "Milvus"],
                "value": "Qdrant"
            },
            "top_k": {"display_name": "Top K Documents", "field_type": "int", "value": 4},
            "score_threshold": {"display_name": "Min Similarity Score", "field_type": "float", "value": 0.75}
        }

    def build(self, query: str, provider: str = "Qdrant", top_k: int = 4, score_threshold: float = 0.75) -> dict:
        """Executes retrieval against the selected vector store."""
        simulated_docs = [
            {"doc_id": f"{provider.lower()}_chunk_1", "text": f"Index optimized in {provider} with HNSW parameters.", "score": 0.93},
            {"doc_id": f"{provider.lower()}_chunk_2", "text": f"Filter applied: tenant_id == 'production_acme'.", "score": 0.88},
        ]
        return {
            "selected_provider": provider,
            "query": query,
            "top_k": top_k,
            "retrieved_documents": simulated_docs
        }

class HumanApprovalGateComponent(BaseLangFlowComponent):
    """
    LangFlow Custom Component for pausing workflows until a human approves the draft.
    """
    display_name = "Human Approval Gate"
    description = "Pauses pipeline execution and requests human confirmation before publishing."
    icon = "UserCheck"

    def build_config(self):
        return {
            "draft_text": {"display_name": "Generated Draft", "field_type": "str"},
            "reviewer_email": {"display_name": "Reviewer Email", "field_type": "str"},
            "auto_approve_threshold": {"display_name": "Auto Approve Score", "field_type": "float", "value": 0.95}
        }

    def build(self, draft_text: str, reviewer_email: str, auto_approve_threshold: float = 0.95) -> dict:
        return {
            "draft": draft_text,
            "status": "awaiting_approval",
            "reviewer": reviewer_email,
            "requires_action": True
        }
