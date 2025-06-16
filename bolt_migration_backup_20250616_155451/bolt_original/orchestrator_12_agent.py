"""
12-Agent Orchestrator for M4 Pro

Utilizes all 12 cores (8 P-cores + 4 E-cores) with intelligent
task distribution and dynamic token optimization.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .agents.agent_pool import TaskPriority, WorkStealingAgentPool, WorkStealingTask
from .core.cpu_optimizer import get_cpu_optimizer
from .core.dynamic_token_optimizer import (
    ResponseComplexity,
    TaskContext,
    get_token_optimizer,
)
from .core.output_token_manager import TokenBudget

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Specialized roles for 12-agent system."""

    # P-core agents (heavy compute)
    ANALYZER = "analyzer"  # Deep analysis tasks
    ARCHITECT = "architect"  # System design
    OPTIMIZER = "optimizer"  # Performance optimization
    GENERATOR = "generator"  # Code generation
    VALIDATOR = "validator"  # Testing and validation
    INTEGRATOR = "integrator"  # Cross-component work
    RESEARCHER = "researcher"  # Complex searches
    SYNTHESIZER = "synthesizer"  # Result synthesis

    # E-core agents (coordination/IO)
    COORDINATOR = "coordinator"  # Task orchestration
    DOCUMENTER = "documenter"  # Documentation
    MONITOR = "monitor"  # System monitoring
    REPORTER = "reporter"  # Status reporting


@dataclass
class EnhancedTask:
    """Task with token budget and complexity awareness."""

    base_task: WorkStealingTask
    token_budget: int
    complexity: ResponseComplexity
    assigned_role: AgentRole
    expected_output_size: int


class Orchestrator12Agent:
    """
    12-agent orchestrator leveraging all M4 Pro cores with
    intelligent token utilization and drift management.
    """

    def __init__(self):
        self.agent_pool: WorkStealingAgentPool | None = None
        self.token_optimizer = get_token_optimizer()
        self.cpu_optimizer = get_cpu_optimizer()

        # Role assignments for 12 agents
        self.agent_roles = {
            0: AgentRole.ANALYZER,
            1: AgentRole.ARCHITECT,
            2: AgentRole.OPTIMIZER,
            3: AgentRole.GENERATOR,
            4: AgentRole.VALIDATOR,
            5: AgentRole.INTEGRATOR,
            6: AgentRole.RESEARCHER,
            7: AgentRole.SYNTHESIZER,
            8: AgentRole.COORDINATOR,
            9: AgentRole.DOCUMENTER,
            10: AgentRole.MONITOR,
            11: AgentRole.REPORTER,
        }

        # Performance tracking
        self.execution_metrics = {
            "total_tokens_generated": 0,
            "average_token_efficiency": 0.0,
            "task_completion_times": [],
            "core_utilization": {},
            "drift_events": 0,
        }

    async def initialize(self) -> None:
        """Initialize 12-agent pool with CPU optimization."""
        logger.info("Initializing 12-agent orchestrator for M4 Pro")

        # Initialize optimized agent pool
        self.agent_pool = WorkStealingAgentPool(
            num_agents=12, enable_work_stealing=True
        )
        await self.agent_pool.initialize()

        # Apply CPU optimizations
        self.cpu_optimizer.optimize_for_throughput()
        self.cpu_optimizer.start_monitoring()

        # Assign agents to cores
        core_assignments = self.cpu_optimizer.assign_agent_pool_cores(12)
        logger.info(f"CPU core assignments: {core_assignments}")

        logger.info("12-agent orchestrator initialized successfully")

    async def execute_complex_task(
        self, instruction: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute a complex task using all 12 agents with dynamic token optimization.
        """
        start_time = time.time()

        # Analyze task complexity and allocate tokens
        task_context = self.token_optimizer.analyze_task(instruction, context)
        token_budget = self.token_optimizer.allocate_tokens(task_context)

        logger.info(f"Task complexity: {token_budget.complexity.value}")
        logger.info(f"Token allocation: {token_budget.target_tokens:,} tokens")

        # Decompose into 12 specialized subtasks
        subtasks = await self._decompose_for_12_agents(
            instruction, task_context, token_budget
        )

        # Submit all tasks in parallel
        logger.info(f"Submitting {len(subtasks)} tasks to 12-agent pool")

        task_futures = []
        for enhanced_task in subtasks:
            await self.agent_pool.submit_task(enhanced_task.base_task)
            task_futures.append(
                (
                    enhanced_task,
                    self.agent_pool.wait_for_task_completion(
                        enhanced_task.base_task.id
                    ),
                )
            )

        # Collect results as they complete
        results = {}
        completed = 0

        for enhanced_task, future in task_futures:
            result = await future
            results[enhanced_task.base_task.id] = {
                "task": enhanced_task,
                "result": result,
                "role": enhanced_task.assigned_role.value,
                "tokens_allocated": enhanced_task.token_budget,
            }
            completed += 1

            # Update drift tracking
            if result.get("output_size"):
                self.execution_metrics["total_tokens_generated"] += result[
                    "output_size"
                ]

        # Synthesize results
        synthesis = await self._synthesize_results(results, token_budget)

        total_duration = time.time() - start_time

        # Update metrics
        self.execution_metrics["task_completion_times"].append(total_duration)
        self.execution_metrics["average_token_efficiency"] = self.execution_metrics[
            "total_tokens_generated"
        ] / (64000 * len(self.execution_metrics["task_completion_times"]))

        return {
            "success": True,
            "instruction": instruction,
            "complexity": token_budget.complexity.value,
            "token_budget": {
                "allocated": token_budget.target_tokens,
                "used": synthesis.get("total_tokens_used", 0),
                "efficiency": f"{token_budget.efficiency_ratio:.1%}",
            },
            "agents_used": 12,
            "duration": total_duration,
            "results": synthesis,
            "performance": {
                "parallel_efficiency": f"{12 / (total_duration * 12):.1%}",
                "cpu_utilization": self.cpu_optimizer.get_metrics().utilization_percent,
                "drift_compensation": task_context.drift_compensation,
            },
        }

    async def _decompose_for_12_agents(
        self, instruction: str, task_context: "TaskContext", token_budget: "TokenBudget"
    ) -> list[EnhancedTask]:
        """Decompose task into 12 specialized subtasks with token allocations."""

        # Base token allocation per agent
        base_tokens_per_agent = token_budget.target_tokens // 12

        # Create specialized tasks based on complexity
        subtasks = []

        if task_context.code_generation:
            # Code generation workflow (uses most agents)
            subtasks.extend(
                [
                    self._create_task(
                        "Analyze requirements and constraints",
                        AgentRole.ANALYZER,
                        base_tokens_per_agent * 1.5,
                    ),
                    self._create_task(
                        "Design system architecture",
                        AgentRole.ARCHITECT,
                        base_tokens_per_agent * 2,
                    ),
                    self._create_task(
                        "Optimize performance strategies",
                        AgentRole.OPTIMIZER,
                        base_tokens_per_agent * 1.5,
                    ),
                    self._create_task(
                        "Generate core implementation",
                        AgentRole.GENERATOR,
                        base_tokens_per_agent * 3,
                    ),
                    self._create_task(
                        "Create validation and tests",
                        AgentRole.VALIDATOR,
                        base_tokens_per_agent * 2,
                    ),
                    self._create_task(
                        "Integrate components",
                        AgentRole.INTEGRATOR,
                        base_tokens_per_agent * 1.5,
                    ),
                    self._create_task(
                        "Research best practices",
                        AgentRole.RESEARCHER,
                        base_tokens_per_agent,
                    ),
                    self._create_task(
                        "Synthesize final solution",
                        AgentRole.SYNTHESIZER,
                        base_tokens_per_agent * 2,
                    ),
                    self._create_task(
                        "Coordinate workflow",
                        AgentRole.COORDINATOR,
                        base_tokens_per_agent * 0.5,
                    ),
                    self._create_task(
                        "Generate documentation",
                        AgentRole.DOCUMENTER,
                        base_tokens_per_agent * 1.5,
                    ),
                    self._create_task(
                        "Monitor progress",
                        AgentRole.MONITOR,
                        base_tokens_per_agent * 0.3,
                    ),
                    self._create_task(
                        "Compile status report",
                        AgentRole.REPORTER,
                        base_tokens_per_agent * 0.7,
                    ),
                ]
            )

        elif task_context.multi_step_reasoning:
            # Multi-step analysis workflow
            subtasks.extend(
                [
                    self._create_task(
                        "Break down problem steps",
                        AgentRole.ANALYZER,
                        base_tokens_per_agent * 2,
                    ),
                    self._create_task(
                        "Design solution approach",
                        AgentRole.ARCHITECT,
                        base_tokens_per_agent * 1.5,
                    ),
                    self._create_task(
                        "Optimize reasoning chain",
                        AgentRole.OPTIMIZER,
                        base_tokens_per_agent,
                    ),
                    self._create_task(
                        "Generate step solutions",
                        AgentRole.GENERATOR,
                        base_tokens_per_agent * 2,
                    ),
                    self._create_task(
                        "Validate logic flow",
                        AgentRole.VALIDATOR,
                        base_tokens_per_agent * 1.5,
                    ),
                    self._create_task(
                        "Connect step dependencies",
                        AgentRole.INTEGRATOR,
                        base_tokens_per_agent * 1.5,
                    ),
                    self._create_task(
                        "Research edge cases",
                        AgentRole.RESEARCHER,
                        base_tokens_per_agent * 1.5,
                    ),
                    self._create_task(
                        "Synthesize conclusions",
                        AgentRole.SYNTHESIZER,
                        base_tokens_per_agent * 2,
                    ),
                    self._create_task(
                        "Manage step execution",
                        AgentRole.COORDINATOR,
                        base_tokens_per_agent * 0.5,
                    ),
                    self._create_task(
                        "Document reasoning",
                        AgentRole.DOCUMENTER,
                        base_tokens_per_agent,
                    ),
                    self._create_task(
                        "Track step progress",
                        AgentRole.MONITOR,
                        base_tokens_per_agent * 0.5,
                    ),
                    self._create_task(
                        "Summarize findings", AgentRole.REPORTER, base_tokens_per_agent
                    ),
                ]
            )

        else:
            # General complex task workflow
            subtasks.extend(
                [
                    self._create_task(
                        f"Analyze aspect 1: {instruction[:50]}...",
                        AgentRole.ANALYZER,
                        base_tokens_per_agent * 1.5,
                    ),
                    self._create_task(
                        "Examine system structure",
                        AgentRole.ARCHITECT,
                        base_tokens_per_agent,
                    ),
                    self._create_task(
                        "Find optimization opportunities",
                        AgentRole.OPTIMIZER,
                        base_tokens_per_agent,
                    ),
                    self._create_task(
                        "Generate recommendations",
                        AgentRole.GENERATOR,
                        base_tokens_per_agent * 1.5,
                    ),
                    self._create_task(
                        "Validate findings", AgentRole.VALIDATOR, base_tokens_per_agent
                    ),
                    self._create_task(
                        "Cross-reference components",
                        AgentRole.INTEGRATOR,
                        base_tokens_per_agent,
                    ),
                    self._create_task(
                        "Deep dive research",
                        AgentRole.RESEARCHER,
                        base_tokens_per_agent * 1.5,
                    ),
                    self._create_task(
                        "Compile final analysis",
                        AgentRole.SYNTHESIZER,
                        base_tokens_per_agent * 2,
                    ),
                    self._create_task(
                        "Orchestrate workflow",
                        AgentRole.COORDINATOR,
                        base_tokens_per_agent * 0.5,
                    ),
                    self._create_task(
                        "Create documentation",
                        AgentRole.DOCUMENTER,
                        base_tokens_per_agent,
                    ),
                    self._create_task(
                        "Monitor execution",
                        AgentRole.MONITOR,
                        base_tokens_per_agent * 0.3,
                    ),
                    self._create_task(
                        "Generate report",
                        AgentRole.REPORTER,
                        base_tokens_per_agent * 0.7,
                    ),
                ]
            )

        # Apply drift compensation
        if task_context.drift_compensation > 0.1:
            logger.info(
                f"Applying drift compensation: {task_context.drift_compensation:.2f}"
            )
            for task in subtasks:
                task.token_budget = int(
                    task.token_budget * (1 + task_context.drift_compensation)
                )

        return subtasks

    def _create_task(
        self, description: str, role: AgentRole, token_budget: int
    ) -> EnhancedTask:
        """Create an enhanced task with token budget."""
        base_task = WorkStealingTask(
            id=f"{role.value}_{int(time.time() * 1000)}",
            description=description,
            priority=TaskPriority.HIGH
            if role in [AgentRole.ANALYZER, AgentRole.SYNTHESIZER]
            else TaskPriority.NORMAL,
            metadata={
                "role": role.value,
                "token_budget": token_budget,
                "expected_complexity": "high" if token_budget > 5000 else "moderate",
            },
        )

        # Determine complexity based on token budget
        if token_budget < 3000:
            complexity = ResponseComplexity.SIMPLE
        elif token_budget < 8000:
            complexity = ResponseComplexity.MODERATE
        elif token_budget < 20000:
            complexity = ResponseComplexity.DETAILED
        else:
            complexity = ResponseComplexity.EXHAUSTIVE

        return EnhancedTask(
            base_task=base_task,
            token_budget=token_budget,
            complexity=complexity,
            assigned_role=role,
            expected_output_size=token_budget,
        )

    async def _synthesize_results(
        self, results: dict[str, Any], token_budget: "TokenBudget"
    ) -> dict[str, Any]:
        """Synthesize results from all 12 agents."""
        synthesis = {
            "summary": f"12-agent parallel execution completed with {token_budget.complexity.value} complexity",
            "total_tokens_used": 0,
            "agent_contributions": {},
            "key_findings": [],
            "recommendations": [],
            "performance_stats": {},
        }

        # Aggregate results by role
        for _task_id, result_data in results.items():
            role = result_data["role"]
            task = result_data["task"]
            result = result_data["result"]

            synthesis["agent_contributions"][role] = {
                "tokens_allocated": task.token_budget,
                "success": result.get("success", False),
                "duration": result.get("duration", 0),
                "contribution": result.get("result", ""),
            }

            # Extract key findings based on role
            if task.assigned_role == AgentRole.ANALYZER:
                synthesis["key_findings"].append(
                    f"Analysis: {result.get('result', '')[:100]}..."
                )
            elif task.assigned_role == AgentRole.OPTIMIZER:
                synthesis["recommendations"].append(
                    f"Optimization: {result.get('result', '')[:100]}..."
                )

        # Calculate performance statistics
        total_duration = max(r["result"].get("duration", 0) for r in results.values())
        successful_agents = sum(
            1 for r in results.values() if r["result"].get("success", False)
        )

        synthesis["performance_stats"] = {
            "agents_successful": f"{successful_agents}/12",
            "parallel_execution_time": f"{total_duration:.3f}s",
            "average_agent_time": f"{sum(r['result'].get('duration', 0) for r in results.values()) / 12:.3f}s",
            "token_efficiency": f"{token_budget.efficiency_ratio:.1%}",
        }

        return synthesis

    async def shutdown(self) -> None:
        """Graceful shutdown of 12-agent orchestrator."""
        if self.agent_pool:
            await self.agent_pool.shutdown()

        self.cpu_optimizer.stop_monitoring()

        # Log final metrics
        logger.info("12-agent orchestrator shutdown complete")
        logger.info(
            f"Total tokens generated: {self.execution_metrics['total_tokens_generated']:,}"
        )
        logger.info(
            f"Average token efficiency: {self.execution_metrics['average_token_efficiency']:.1%}"
        )


# Convenience functions
async def execute_with_12_agents(
    instruction: str, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute a task using the 12-agent orchestrator."""
    orchestrator = Orchestrator12Agent()
    await orchestrator.initialize()

    try:
        result = await orchestrator.execute_complex_task(instruction, context)
        return result
    finally:
        await orchestrator.shutdown()
