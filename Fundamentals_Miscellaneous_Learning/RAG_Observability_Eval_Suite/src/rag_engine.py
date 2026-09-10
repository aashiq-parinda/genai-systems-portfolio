import time
import re
from typing import Dict, Any, List, Optional
from .config import settings
from .vector_store import VectorStore, DocumentChunk
from .tools import check_order_status, OrderLookupError
from .telemetry.otel_tracer import (
    global_tracer,
    GEN_AI_SYSTEM,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_PROMPT_TOKENS,
    GEN_AI_USAGE_COMPLETION_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    GEN_AI_OPERATION_NAME,
    GEN_AI_RETRIEVAL_QUERY,
    GEN_AI_RETRIEVAL_TOP_K,
    GEN_AI_RETRIEVAL_CHUNKS_RETURNED,
    GEN_AI_RETRIEVAL_TOP_SCORE,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_ARGS,
    GEN_AI_TOOL_STATUS,
    GEN_AI_TOOL_ERROR
)
from .telemetry.langfuse_client import langfuse_client
from .telemetry.phoenix_client import phoenix_client
from .telemetry.helicone_client import helicone_client
from .evaluation.llm_judge import llm_judge, EvaluationResult
from .failure_injection import FailureScenario

class RAGPipeline:
    """
    Production-style Customer Support RAG Pipeline instrumented with:
    - OpenTelemetry GenAI Semantic Conventions
    - Langfuse (Prompt versions, traces, eval scores)
    - Arize Phoenix (Retrieval quality & drift)
    - Helicone (Edge proxy token and cost tracking)
    - LLM-as-a-Judge evaluation suite
    """
    def __init__(self):
        self.vector_store = VectorStore(docs_dir=settings.docs_dir)
        self.tracer = global_tracer

    def _extract_order_id(self, query: str) -> Optional[str]:
        match = re.search(r"\b(ORD-\d+|SUB-\d+)\b", query, re.IGNORECASE)
        return match.group(1).upper() if match else None

    def execute_query(
        self,
        query: str,
        scenario: FailureScenario = FailureScenario.BASELINE,
        user_id: str = "customer_vip_007",
        session_id: str = "support_sess_492"
    ) -> Dict[str, Any]:
        """
        Execute the full RAG customer support pipeline with telemetry instrumentation.
        """
        trace_id = self.tracer.start_trace()
        start_wall_clock = time.time()
        
        # 1. Register Langfuse trace
        langfuse_client.create_trace(
            trace_id=trace_id,
            name=f"customer_support_rag_{scenario.value}",
            user_id=user_id,
            session_id=session_id,
            tags=["customer-support", scenario.value]
        )

        retrieved_chunks: List[DocumentChunk] = []
        tool_result: Optional[Dict[str, Any]] = None
        tool_error: Optional[str] = None
        generated_answer: str = ""
        prompt_tokens = 0
        completion_tokens = 0

        # Fetch version-controlled prompt from Langfuse
        prompt_version = langfuse_client.get_prompt("support_rag_system")

        # ======================================================================
        # STAGE 1: Vector Retrieval (OpenTelemetry Span)
        # ======================================================================
        force_bad_retrieval = (scenario == FailureScenario.BAD_RETRIEVAL)
        with self.tracer.start_span("retrieve_context") as span:
            span.set_attribute(GEN_AI_OPERATION_NAME, "retrieval")
            span.set_attribute(GEN_AI_RETRIEVAL_QUERY, query)
            span.set_attribute(GEN_AI_RETRIEVAL_TOP_K, 3)

            retrieved_chunks = self.vector_store.retrieve(
                query=query,
                top_k=3,
                force_irrelevant=force_bad_retrieval
            )

            top_score = retrieved_chunks[0].score if retrieved_chunks else 0.0
            span.set_attribute(GEN_AI_RETRIEVAL_CHUNKS_RETURNED, len(retrieved_chunks))
            span.set_attribute(GEN_AI_RETRIEVAL_TOP_SCORE, top_score)
            span.set_attribute("gen_ai.retrieval.sources", [c.source_file for c in retrieved_chunks])

        # Combine retrieved context
        context_text = "\n\n---\n\n".join([f"[{c.source_file}]: {c.content}" for c in retrieved_chunks])

        # ======================================================================
        # STAGE 2: Tool Execution (check_order_status if order mentioned)
        # ======================================================================
        order_id = self._extract_order_id(query)
        if order_id:
            force_tool_failure = (scenario == FailureScenario.FAILED_TOOL)
            try:
                with self.tracer.start_span("tool_call:check_order_status") as tool_span:
                    tool_span.set_attribute(GEN_AI_OPERATION_NAME, "tool_execution")
                    tool_span.set_attribute(GEN_AI_TOOL_NAME, "check_order_status")
                    tool_span.set_attribute(GEN_AI_TOOL_ARGS, {"order_id": order_id})

                    tool_result = check_order_status(
                        order_id=order_id,
                        force_failure=force_tool_failure,
                        failure_type="timeout"
                    )
                    tool_span.set_attribute(GEN_AI_TOOL_STATUS, "SUCCESS")
                    tool_span.set_attribute("gen_ai.tool.output", tool_result)
            except Exception as exc:
                tool_error = str(exc)
                # OTel automatically catches and sets status ERROR on the span

        # ======================================================================
        # STAGE 3: LLM Generation / Synthesis (OpenTelemetry GenAI Conventions)
        # ======================================================================
        with self.tracer.start_span(f"chat {settings.openai_model_name}") as gen_span:
            gen_span.set_attribute(GEN_AI_SYSTEM, "openai")
            gen_span.set_attribute(GEN_AI_REQUEST_MODEL, settings.openai_model_name)
            gen_span.set_attribute(GEN_AI_RESPONSE_MODEL, settings.openai_model_name)
            gen_span.set_attribute(GEN_AI_OPERATION_NAME, "chat")

            # Determine response synthesis
            if scenario == FailureScenario.HALLUCINATION:
                # Deliberate hallucination
                generated_answer = (
                    "CloudFlow offers exclusive Quantum-Encrypted On-Premise Hyper-Clusters with free lifetime "
                    "hardware upgrades for all Enterprise customers. You can deploy this on your private server "
                    "racks with unlimited warp-drive parallel execution nodes."
                )
            elif tool_error:
                generated_answer = (
                    f"I attempted to look up your order {order_id}, but the internal order lookup service "
                    f"encountered a temporary system error ({tool_error}). Please try again shortly or contact support."
                )
            elif tool_result:
                status_text = tool_result.get("status")
                plan = tool_result.get("plan")
                seats = tool_result.get("seats_allocated")
                generated_answer = (
                    f"Your order {order_id} is currently {status_text.upper()} on the {plan} tier with {seats} seats. "
                    f"Based on CloudFlow documentation, the Enterprise tier starts at $1,200 per month with 24/7 dedicated support and custom volume."
                )
            elif force_bad_retrieval:
                generated_answer = (
                    "According to CloudFlow documentation, we use third-party tracking cookies to monitor bounce rates "
                    "on public marketing landing pages. Please accept our cookie banner to proceed."
                )
            else:
                # Standard grounded response from retrieved docs
                first_chunk = retrieved_chunks[0] if retrieved_chunks else None
                summary = first_chunk.content[:250].replace("\n", " ") if first_chunk else "Standard cloud services."
                generated_answer = (
                    f"Based on CloudFlow documentation regarding {first_chunk.title if first_chunk else 'our service'}: "
                    f"{summary}..."
                )

            # Calculate token telemetry
            prompt_text = prompt_version.template.format(
                retrieved_context=context_text,
                user_query=query
            )
            prompt_tokens = len(prompt_text.split()) * 2
            completion_tokens = len(generated_answer.split()) * 2
            total_tokens = prompt_tokens + completion_tokens

            gen_span.set_attribute(GEN_AI_USAGE_PROMPT_TOKENS, prompt_tokens)
            gen_span.set_attribute(GEN_AI_USAGE_COMPLETION_TOKENS, completion_tokens)
            gen_span.set_attribute(GEN_AI_USAGE_TOTAL_TOKENS, total_tokens)

        total_latency_ms = (time.time() - start_wall_clock) * 1000.0

        # ======================================================================
        # STAGE 4: Helicone Edge Proxy Logging
        # ======================================================================
        proxy_status_code = 504 if scenario == FailureScenario.FAILED_TOOL else 200
        helicone_req = helicone_client.record_proxy_call(
            request_id=trace_id,
            model=settings.openai_model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=total_latency_ms,
            status_code=proxy_status_code,
            scenario_tag=scenario.value,
            user_id=user_id,
            session_id=session_id
        )

        # ======================================================================
        # STAGE 5: Multi-Sink Telemetry Sync (Langfuse & Arize Phoenix)
        # ======================================================================
        spans = self.tracer.get_spans_for_trace(trace_id)
        
        # Sync to Langfuse
        langfuse_client.sync_otel_spans(trace_id=trace_id, spans=spans)
        
        # Sync to Arize Phoenix
        phoenix_client.log_trace(
            trace_id=trace_id,
            spans=spans,
            query=query,
            retrieved_chunks=retrieved_chunks
        )

        # ======================================================================
        # STAGE 6: LLM-as-a-Judge Evaluation & Langfuse Score Logging
        # ======================================================================
        groundedness_eval = llm_judge.evaluate_groundedness(
            answer=generated_answer,
            context=context_text,
            tool_output=tool_result
        )
        relevance_eval = llm_judge.evaluate_context_relevance(
            query=query,
            context=context_text
        )

        # Log evaluation scores back into Langfuse as custom metrics
        langfuse_client.record_score(
            trace_id=trace_id,
            name="groundedness",
            value=groundedness_eval.score,
            comment=groundedness_eval.explanation
        )
        langfuse_client.record_score(
            trace_id=trace_id,
            name="context_relevance",
            value=relevance_eval.score,
            comment=relevance_eval.explanation
        )
        if tool_error:
            langfuse_client.record_score(
                trace_id=trace_id,
                name="tool_execution_success",
                value=0.0,
                comment=f"Tool failed with {tool_error}"
            )
        elif order_id:
            langfuse_client.record_score(
                trace_id=trace_id,
                name="tool_execution_success",
                value=1.0,
                comment="Tool executed successfully"
            )

        # Also register evaluations on Phoenix trace
        phoenix_client.log_evaluation(
            trace_id=trace_id,
            evaluation_name="groundedness",
            score=groundedness_eval.score,
            explanation=groundedness_eval.explanation
        )
        phoenix_client.log_evaluation(
            trace_id=trace_id,
            evaluation_name="context_relevance",
            score=relevance_eval.score,
            explanation=relevance_eval.explanation
        )

        return {
            "trace_id": trace_id,
            "scenario": scenario.value,
            "query": query,
            "answer": generated_answer,
            "retrieved_chunks_count": len(retrieved_chunks),
            "tool_result": tool_result,
            "tool_error": tool_error,
            "evaluations": {
                "groundedness": {
                    "score": groundedness_eval.score,
                    "passed": groundedness_eval.passed,
                    "explanation": groundedness_eval.explanation
                },
                "context_relevance": {
                    "score": relevance_eval.score,
                    "passed": relevance_eval.passed,
                    "explanation": relevance_eval.explanation
                }
            },
            "helicone_metrics": {
                "cost_usd": helicone_req.cost_usd,
                "total_tokens": helicone_req.total_tokens,
                "latency_ms": helicone_req.latency_ms
            },
            "spans_count": len(spans)
        }
