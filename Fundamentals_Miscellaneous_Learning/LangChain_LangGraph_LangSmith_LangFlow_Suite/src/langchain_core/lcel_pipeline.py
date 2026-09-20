"""
LangChain Core: LCEL (LangChain Expression Language) Pipeline Engine.
Demonstrates:
1. Canonical Pipe Syntax (`prompt | model | parser`)
2. Concurrent Multi-Aspect Analysis (`RunnableParallel`)
3. Dynamic Intent-Based Routing (`RunnableBranch`)
4. Resilient Fallback Chains (`with_fallbacks`)
"""

import json
import time
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field

from ..config import Config
from ..mock_llm import MockLLM

# Pydantic Schemas for Structured Parsing
class IntentClassification(BaseModel):
    intent: str = Field(description="The classified intent category")
    confidence: float = Field(description="Confidence score between 0 and 1")
    target_handler: str = Field(description="Downstream routing handler")

class MultiAspectAnalysis(BaseModel):
    summary: str
    key_entities: List[str]
    sentiment: str
    estimated_priority: str

class LCELRunnable:
    """Core abstraction mimicking LangChain Runnable for pipeline composition."""
    def __init__(self, func: Callable[[Any], Any], name: str = "Runnable"):
        self.func = func
        self.name = name

    def invoke(self, input_data: Any) -> Any:
        return self.func(input_data)

    def __or__(self, other):
        """Implement the Unix pipe operator | for LCEL chaining."""
        def chained_func(input_data):
            intermediate = self.invoke(input_data)
            if hasattr(other, "invoke"):
                return other.invoke(intermediate)
            elif callable(other):
                return other(intermediate)
            raise TypeError(f"Cannot pipe to non-runnable object: {other}")
        return LCELRunnable(chained_func, name=f"{self.name} | {getattr(other, 'name', type(other).__name__)}")

    def with_fallbacks(self, fallbacks: List["LCELRunnable"]):
        """Implement fallback failover behavior identical to LangChain with_fallbacks()."""
        def fallback_func(input_data):
            try:
                return self.invoke(input_data)
            except Exception as primary_error:
                for fb in fallbacks:
                    try:
                        return fb.invoke(input_data)
                    except Exception:
                        continue
                raise primary_error
        return LCELRunnable(fallback_func, name=f"{self.name} (with fallbacks)")

class RunnableParallel(LCELRunnable):
    """Executes multiple branches concurrently or sequentially and merges output into a dictionary."""
    def __init__(self, steps: Dict[str, Any]):
        self.steps = steps
        def parallel_func(input_data):
            results = {}
            for key, runnable in self.steps.items():
                if hasattr(runnable, "invoke"):
                    results[key] = runnable.invoke(input_data)
                elif callable(runnable):
                    results[key] = runnable(input_data)
                else:
                    results[key] = runnable
            return results
        super().__init__(parallel_func, name="RunnableParallel")

class PromptTemplate:
    """Formats dynamic variable placeholders into formatted prompt strings."""
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    def invoke(self, inputs: Dict[str, Any]) -> str:
        return self.format(**inputs)

class JsonOutputParser:
    """Parses raw model completion text into structured Python dict or Pydantic object."""
    def __init__(self, pydantic_class: Optional[type] = None):
        self.pydantic_class = pydantic_class

    def invoke(self, raw_output: Any) -> Any:
        text = getattr(raw_output, "content", str(raw_output)).strip()
        # Clean markdown codeblocks if model wrapped json
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        parsed_dict = json.loads(text)
        if self.pydantic_class:
            return self.pydantic_class(**parsed_dict)
        return parsed_dict

# ------------------------------------------------------------------------------
# Pre-Configured Demonstration Pipelines
# ------------------------------------------------------------------------------

def build_lcel_intent_classifier():
    """Builds a classic LCEL pipeline: Prompt | Model | JsonOutputParser."""
    prompt = LCELRunnable(
        lambda inputs: (
            f"Classify the following customer ticket intent:\n"
            f"Ticket Text: '{inputs.get('text')}'\n"
            f"Return JSON matching keys: intent, confidence, target_handler."
        ),
        name="PromptTemplate"
    )
    model = LCELRunnable(lambda p: MockLLM(model_name="gpt-4o-mini", latency_ms=80).invoke(p), name="ChatModel")
    parser = LCELRunnable(lambda msg: JsonOutputParser(pydantic_class=IntentClassification).invoke(msg), name="JsonOutputParser")

    # Connect using canonical LCEL pipe |
    chain = prompt | model | parser
    return chain

def build_lcel_parallel_analyzer():
    """Builds an LCEL RunnableParallel pipeline analyzing multiple dimensions simultaneously."""
    summary_branch = LCELRunnable(
        lambda x: f"Summary of '{x.get('text', '')[:30]}...': Inquiry regarding GenAI production latency.",
        name="SummaryBranch"
    )
    entities_branch = LCELRunnable(
        lambda x: ["Pinecone", "Milvus", "HNSW Index", "LangGraph"],
        name="EntitiesBranch"
    )
    sentiment_branch = LCELRunnable(
        lambda x: "Neutral/Analytical",
        name="SentimentBranch"
    )
    priority_branch = LCELRunnable(
        lambda x: "P1-High" if "urgent" in x.get("text", "").lower() else "P2-Normal",
        name="PriorityBranch"
    )

    parallel_chain = RunnableParallel({
        "summary": summary_branch,
        "key_entities": entities_branch,
        "sentiment": sentiment_branch,
        "estimated_priority": priority_branch
    })

    return parallel_chain

def build_lcel_resilient_fallback_chain():
    """Demonstrates primary model failure failover to secondary model using with_fallbacks()."""
    def failing_primary(x):
        raise ConnectionResetError("Primary model API RateLimit / 429 Outage!")

    primary_runnable = LCELRunnable(failing_primary, name="PrimaryModel (gpt-4o)")
    secondary_runnable = LCELRunnable(
        lambda x: f"[Fallback gpt-4o-mini] Successfully fulfilled query: '{x.get('text')}' via resilient backup route.",
        name="FallbackModel (gpt-4o-mini)"
    )

    resilient_chain = primary_runnable.with_fallbacks([secondary_runnable])
    return resilient_chain
