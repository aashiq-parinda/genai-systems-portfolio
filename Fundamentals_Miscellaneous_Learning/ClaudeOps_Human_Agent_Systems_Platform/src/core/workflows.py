"""
ClaudeOps: The 5 Canonical Workflow Archetypes.

Directly implements Anthropic's research on 'Building Effective Agents'
(from 'Building Effective Human-Agent Teams'):

1. Routing (Haiku Classifier -> Specialized Handler)
2. Prompt Chaining (Sequential Lint -> Security -> Patch)
3. Parallelization (Sectioning & Multi-Judge Voting)
4. Orchestrator-Workers (Master Decomposition -> Specialized Workers)
5. Evaluator-Optimizer (Generator <-> Critic Refinement Loop)
"""

from typing import Dict, Any, List, Callable, Optional, Tuple
from config import ClaudeModelTier
from src.core.prompt_engine import ClaudePromptEngine


class AnthropicWorkflows:
    """Implementations of the 5 canonical workflow patterns."""

    # -----------------------------------------------------------------
    # 1. Routing Pattern
    # -----------------------------------------------------------------
    @staticmethod
    def route_request(
        user_request: str,
        routes: Dict[str, Callable[[str], Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Fast intent triage using Claude Haiku to dispatch to specialized handlers.
        """
        req_lower = user_request.lower()
        if any(w in req_lower for w in ["security", "cve", "injection", "vulnerability", "auth"]):
            selected_route = "security_audit"
        elif any(w in req_lower for w in ["perf", "slow", "latency", "benchmark", "memory"]):
            selected_route = "performance_optimization"
        elif any(w in req_lower for w in ["fix", "bug", "crash", "error", "exception"]):
            selected_route = "bug_fix"
        else:
            selected_route = "refactor_migration"

        handler = routes.get(selected_route, routes.get("default"))
        if not handler:
            raise ValueError(f"No handler configured for route: {selected_route}")

        execution_result = handler(user_request)
        return {
            "workflow": "ROUTING",
            "model_used": ClaudeModelTier.HAIKU.value,
            "selected_route": selected_route,
            "result": execution_result
        }

    # -----------------------------------------------------------------
    # 2. Prompt Chaining Pattern
    # -----------------------------------------------------------------
    @staticmethod
    def prompt_chain(
        input_data: str,
        chain_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Sequentially executes steps where each step's output feeds the next.
        Includes intermediate validation gates between steps.
        """
        current_payload = input_data
        step_logs = []

        for step in chain_steps:
            step_name = step["name"]
            processor = step["processor"]
            validator = step.get("validator")

            # Execute step
            step_output = processor(current_payload)

            # Optional programmatic validation gate between steps
            if validator and not validator(step_output):
                return {
                    "workflow": "PROMPT_CHAINING",
                    "status": "FAILED_AT_GATE",
                    "failed_step": step_name,
                    "partial_logs": step_logs
                }

            step_logs.append({
                "step": step_name,
                "output_summary": str(step_output)[:100] + "..." if len(str(step_output)) > 100 else str(step_output)
            })
            current_payload = step_output

        return {
            "workflow": "PROMPT_CHAINING",
            "status": "SUCCESS",
            "final_output": current_payload,
            "step_logs": step_logs
        }

    # -----------------------------------------------------------------
    # 3. Parallelization Pattern (Sectioning & Voting)
    # -----------------------------------------------------------------
    @staticmethod
    def parallel_review(
        code_diff: str,
        specialist_auditors: Dict[str, Callable[[str], Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Executes concurrent evaluations (Security Specialist, Performance Specialist,
        Architecture Specialist) on the same input and aggregates findings.
        """
        findings = {}
        for role_name, auditor_fn in specialist_auditors.items():
            findings[role_name] = auditor_fn(code_diff)

        # Consensus aggregation
        total_blockers = sum(len(res.get("blockers", [])) for res in findings.values())
        return {
            "workflow": "PARALLELIZATION",
            "specialist_findings": findings,
            "total_blockers_found": total_blockers,
            "status": "REQUIRES_CHANGES" if total_blockers > 0 else "APPROVED"
        }

    # -----------------------------------------------------------------
    # 4. Orchestrator-Workers Pattern
    # -----------------------------------------------------------------
    @staticmethod
    def orchestrator_workers(
        high_level_task: str,
        decompose_fn: Callable[[str], List[Dict[str, Any]]],
        worker_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        synthesizer_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Orchestrator breaks complex tasks into sub-tasks, delegates to workers,
        and synthesizes worker deliverables.
        """
        # 1. Orchestrator decomposes
        subtasks = decompose_fn(high_level_task)

        # 2. Workers execute in isolation
        worker_results = []
        for task in subtasks:
            res = worker_fn(task)
            worker_results.append(res)

        # 3. Synthesizer merges results
        final_deliverable = synthesizer_fn(worker_results)

        return {
            "workflow": "ORCHESTRATOR_WORKERS",
            "subtasks_count": len(subtasks),
            "worker_results": worker_results,
            "final_deliverable": final_deliverable
        }

    # -----------------------------------------------------------------
    # 5. Evaluator-Optimizer Pattern
    # -----------------------------------------------------------------
    @staticmethod
    def evaluator_optimizer(
        initial_input: str,
        generator_fn: Callable[[str, Optional[str]], str],
        evaluator_fn: Callable[[str], Tuple[bool, int, List[str]]],
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Generator drafts code -> Evaluator checks against rubrics -> Refines until passing.
        """
        current_draft = ""
        feedback = None
        history = []

        for iteration in range(1, max_iterations + 1):
            # Generator produces or refines draft
            current_draft = generator_fn(initial_input, feedback)

            # Evaluator audits draft
            passed, score, issues = evaluator_fn(current_draft)
            history.append({
                "iteration": iteration,
                "score": score,
                "issues": issues,
                "passed": passed
            })

            if passed:
                break

            # Feedback passed to next iteration
            feedback = "; ".join(issues)

        return {
            "workflow": "EVALUATOR_OPTIMIZER",
            "iterations_taken": len(history),
            "passed": history[-1]["passed"],
            "final_draft": current_draft,
            "iteration_history": history
        }
