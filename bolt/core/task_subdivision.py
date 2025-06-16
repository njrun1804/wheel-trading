#!/usr/bin/env python3
"""
Production-Ready Dynamic Task Subdivision System

Intelligent task decomposition and workload distribution for Bolt agents.
Optimized for M4 Pro heterogeneous architecture.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SubdivisionStrategy(Enum):
    """Task subdivision strategies."""

    NONE = "none"  # No subdivision
    UNIFORM = "uniform"  # Equal parts
    ADAPTIVE = "adaptive"  # Based on agent performance
    WORKLOAD_AWARE = "workload_aware"  # Based on task complexity
    ML_PREDICTED = "ml_predicted"  # Machine learning based


class TaskType(Enum):
    """Task types for subdivision optimization."""

    CPU_INTENSIVE = "cpu_intensive"
    IO_BOUND = "io_bound"
    GPU_ACCELERATED = "gpu_accelerated"
    MEMORY_BOUND = "memory_bound"
    MIXED_WORKLOAD = "mixed_workload"
    SEARCH_OPERATION = "search_operation"
    ANALYSIS_OPERATION = "analysis_operation"


@dataclass
class SubdivisionMetrics:
    """Metrics for subdivision performance analysis."""

    original_duration_estimate: float
    subdivision_overhead: float
    predicted_speedup: float
    actual_speedup: float | None = None
    parallel_efficiency: float | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class TaskCharacteristics:
    """Characteristics used for subdivision decisions."""

    estimated_duration: float
    complexity_score: float
    memory_requirements: int  # bytes
    cpu_intensity: float  # 0.0 to 1.0
    io_requirements: float  # 0.0 to 1.0
    parallelizable: bool
    data_size: int  # bytes
    dependencies: list[str]


class SubdivisionDecisionEngine:
    """Intelligent decision engine for task subdivision."""

    def __init__(self):
        self.subdivision_history: list[
            tuple[TaskCharacteristics, SubdivisionMetrics]
        ] = []
        self.performance_thresholds = {
            "min_duration_for_subdivision": 2.0,  # seconds
            "min_speedup_threshold": 1.3,  # 30% improvement
            "max_subdivision_overhead": 0.5,  # seconds
            "min_parallel_efficiency": 0.7,  # 70% efficiency
        }

    def should_subdivide(
        self,
        task_chars: TaskCharacteristics,
        available_agents: int,
        current_system_load: float,
    ) -> tuple[bool, SubdivisionStrategy, int]:
        """Determine if and how to subdivide a task."""

        # Basic eligibility checks
        if not task_chars.parallelizable:
            return False, SubdivisionStrategy.NONE, 1

        if (
            task_chars.estimated_duration
            < self.performance_thresholds["min_duration_for_subdivision"]
        ):
            return False, SubdivisionStrategy.NONE, 1

        if available_agents < 2:
            return False, SubdivisionStrategy.NONE, 1

        # Calculate optimal subdivision
        strategy, num_parts = self._calculate_optimal_subdivision(
            task_chars, available_agents, current_system_load
        )

        if num_parts <= 1:
            return False, SubdivisionStrategy.NONE, 1

        # Predict performance gain
        predicted_speedup = self._predict_subdivision_speedup(
            task_chars, strategy, num_parts
        )

        should_subdivide = (
            predicted_speedup >= self.performance_thresholds["min_speedup_threshold"]
        )

        return should_subdivide, strategy, num_parts

    def _calculate_optimal_subdivision(
        self, task_chars: TaskCharacteristics, available_agents: int, system_load: float
    ) -> tuple[SubdivisionStrategy, int]:
        """Calculate optimal subdivision strategy and number of parts."""

        # Adjust for system load
        effective_agents = max(1, int(available_agents * (1.0 - system_load)))

        # Choose strategy based on task characteristics
        if task_chars.cpu_intensity > 0.8:
            # CPU-intensive tasks benefit from P-core distribution
            strategy = SubdivisionStrategy.WORKLOAD_AWARE
            num_parts = min(effective_agents, 8)  # Max P-cores

        elif task_chars.io_requirements > 0.6:
            # I/O bound tasks can use more agents
            strategy = SubdivisionStrategy.ADAPTIVE
            num_parts = min(effective_agents, 12)  # All CPU cores

        elif task_chars.data_size > 100 * 1024 * 1024:  # > 100MB
            # Large data operations need memory-aware subdivision
            strategy = SubdivisionStrategy.WORKLOAD_AWARE
            num_parts = min(effective_agents, 4)  # Memory bandwidth limited

        else:
            # Default to uniform subdivision
            strategy = SubdivisionStrategy.UNIFORM
            num_parts = min(effective_agents, 6)

        return strategy, num_parts

    def _predict_subdivision_speedup(
        self,
        task_chars: TaskCharacteristics,
        strategy: SubdivisionStrategy,
        num_parts: int,
    ) -> float:
        """Predict speedup from subdivision."""

        # Base speedup from parallelization
        base_speedup = self._calculate_amdahl_speedup(
            task_chars.cpu_intensity, num_parts
        )

        # Adjust for overhead
        coordination_overhead = 0.1 * num_parts  # Linear overhead
        memory_overhead = min(
            0.2, task_chars.data_size / (1024**3)
        )  # Up to 20% for large data

        total_overhead = coordination_overhead + memory_overhead
        adjusted_speedup = base_speedup / (1.0 + total_overhead)

        # Apply strategy-specific adjustments
        if strategy == SubdivisionStrategy.WORKLOAD_AWARE:
            adjusted_speedup *= 1.1  # 10% bonus for intelligent distribution
        elif strategy == SubdivisionStrategy.ADAPTIVE:
            adjusted_speedup *= 1.05  # 5% bonus for adaptive approach

        return adjusted_speedup

    def _calculate_amdahl_speedup(
        self, parallel_fraction: float, num_cores: int
    ) -> float:
        """Calculate theoretical speedup using Amdahl's Law."""
        serial_fraction = 1.0 - parallel_fraction
        speedup = 1.0 / (serial_fraction + parallel_fraction / num_cores)
        return speedup

    def record_subdivision_result(
        self, task_chars: TaskCharacteristics, subdivision_metrics: SubdivisionMetrics
    ):
        """Record subdivision results for learning."""
        self.subdivision_history.append((task_chars, subdivision_metrics))

        # Keep only recent history
        if len(self.subdivision_history) > 1000:
            self.subdivision_history = self.subdivision_history[-500:]


class TaskSubdivider(ABC):
    """Abstract base class for task subdivision."""

    @abstractmethod
    async def subdivide(
        self, task: Any, num_parts: int, strategy: SubdivisionStrategy
    ) -> list[Any]:
        """Subdivide task into parts."""
        pass

    @abstractmethod
    def get_task_characteristics(self, task: Any) -> TaskCharacteristics:
        """Extract characteristics from task."""
        pass


class SearchTaskSubdivider(TaskSubdivider):
    """Subdivider for search operations."""

    async def subdivide(
        self, task: Any, num_parts: int, strategy: SubdivisionStrategy
    ) -> list[Any]:
        """Subdivide search task into parallel queries."""

        if hasattr(task, "query_batch"):
            # Batch search subdivision
            queries = task.query_batch
            chunk_size = max(1, len(queries) // num_parts)

            subtasks = []
            for i in range(0, len(queries), chunk_size):
                chunk = queries[i : i + chunk_size]
                if chunk:
                    subtask = type(task)(
                        id=f"{task.id}_search_part_{i//chunk_size}",
                        description=f"Search batch {i//chunk_size + 1}/{num_parts}",
                        query_batch=chunk,
                        metadata={**task.metadata, "subdivision_part": i // chunk_size},
                    )
                    subtasks.append(subtask)

            return subtasks

        # Single query subdivision (by search space)
        return await self._subdivide_search_space(task, num_parts)

    async def _subdivide_search_space(self, task: Any, num_parts: int) -> list[Any]:
        """Subdivide by search space partitioning."""
        # Implementation would depend on specific search algorithm
        subtasks = []
        for i in range(num_parts):
            subtask = type(task)(
                id=f"{task.id}_space_part_{i}",
                description=f"Search space partition {i+1}/{num_parts}",
                search_partition=i,
                total_partitions=num_parts,
                metadata={**task.metadata, "space_partition": i},
            )
            subtasks.append(subtask)

        return subtasks

    def get_task_characteristics(self, task: Any) -> TaskCharacteristics:
        """Extract characteristics from search task."""
        query_complexity = 1.0
        data_size = 1024 * 1024  # Default 1MB

        if hasattr(task, "query_batch"):
            query_complexity = len(task.query_batch) * 0.1
            data_size = len(task.query_batch) * 1024

        return TaskCharacteristics(
            estimated_duration=query_complexity,
            complexity_score=query_complexity,
            memory_requirements=data_size,
            cpu_intensity=0.6,  # Moderate CPU usage
            io_requirements=0.3,  # Some I/O for index access
            parallelizable=True,
            data_size=data_size,
            dependencies=getattr(task, "dependencies", []),
        )


class AnalysisTaskSubdivider(TaskSubdivider):
    """Subdivider for analysis operations."""

    async def subdivide(
        self, task: Any, num_parts: int, strategy: SubdivisionStrategy
    ) -> list[Any]:
        """Subdivide analysis task by data partitioning."""

        if hasattr(task, "file_list"):
            # File-based analysis subdivision
            files = task.file_list
            chunk_size = max(1, len(files) // num_parts)

            subtasks = []
            for i in range(0, len(files), chunk_size):
                chunk = files[i : i + chunk_size]
                if chunk:
                    subtask = type(task)(
                        id=f"{task.id}_analysis_part_{i//chunk_size}",
                        description=f"Analysis batch {i//chunk_size + 1}/{num_parts}",
                        file_list=chunk,
                        metadata={**task.metadata, "file_partition": i // chunk_size},
                    )
                    subtasks.append(subtask)

            return subtasks

        # Default subdivision for other analysis types
        return await self._subdivide_by_scope(task, num_parts)

    async def _subdivide_by_scope(self, task: Any, num_parts: int) -> list[Any]:
        """Subdivide by analysis scope."""
        subtasks = []
        for i in range(num_parts):
            subtask = type(task)(
                id=f"{task.id}_scope_part_{i}",
                description=f"Analysis scope {i+1}/{num_parts}",
                scope_partition=i,
                total_partitions=num_parts,
                metadata={**task.metadata, "scope_partition": i},
            )
            subtasks.append(subtask)

        return subtasks

    def get_task_characteristics(self, task: Any) -> TaskCharacteristics:
        """Extract characteristics from analysis task."""
        file_count = len(getattr(task, "file_list", [1]))
        complexity = file_count * 0.5
        data_size = file_count * 50 * 1024  # ~50KB per file

        return TaskCharacteristics(
            estimated_duration=complexity,
            complexity_score=complexity,
            memory_requirements=data_size,
            cpu_intensity=0.8,  # High CPU usage for analysis
            io_requirements=0.4,  # File I/O required
            parallelizable=True,
            data_size=data_size,
            dependencies=getattr(task, "dependencies", []),
        )


class DynamicTaskSubdivisionSystem:
    """Production-ready dynamic task subdivision system."""

    def __init__(self):
        self.decision_engine = SubdivisionDecisionEngine()
        self.subdividers = {
            TaskType.SEARCH_OPERATION: SearchTaskSubdivider(),
            TaskType.ANALYSIS_OPERATION: AnalysisTaskSubdivider(),
            # Add more subdividers as needed
        }

        self.performance_tracker = {
            "total_subdivisions": 0,
            "successful_subdivisions": 0,
            "average_speedup": 0.0,
            "subdivision_overhead_avg": 0.0,
        }

        logger.info("Dynamic Task Subdivision System initialized")

    async def analyze_and_subdivide(
        self, task: Any, available_agents: int, current_system_load: float
    ) -> tuple[bool, list[Any], SubdivisionMetrics]:
        """Analyze task and potentially subdivide it."""

        # Determine task type
        task_type = self._classify_task(task)

        if task_type not in self.subdividers:
            # No subdivider available for this task type
            return (
                False,
                [task],
                SubdivisionMetrics(
                    original_duration_estimate=getattr(task, "estimated_duration", 1.0),
                    subdivision_overhead=0.0,
                    predicted_speedup=1.0,
                ),
            )

        subdivider = self.subdividers[task_type]

        # Get task characteristics
        task_chars = subdivider.get_task_characteristics(task)

        # Make subdivision decision
        should_subdivide, strategy, num_parts = self.decision_engine.should_subdivide(
            task_chars, available_agents, current_system_load
        )

        if not should_subdivide:
            return (
                False,
                [task],
                SubdivisionMetrics(
                    original_duration_estimate=task_chars.estimated_duration,
                    subdivision_overhead=0.0,
                    predicted_speedup=1.0,
                ),
            )

        # Perform subdivision
        start_time = time.time()
        subtasks = await subdivider.subdivide(task, num_parts, strategy)
        subdivision_overhead = time.time() - start_time

        # Calculate predicted speedup
        predicted_speedup = self.decision_engine._predict_subdivision_speedup(
            task_chars, strategy, num_parts
        )

        metrics = SubdivisionMetrics(
            original_duration_estimate=task_chars.estimated_duration,
            subdivision_overhead=subdivision_overhead,
            predicted_speedup=predicted_speedup,
        )

        # Update performance tracking
        self.performance_tracker["total_subdivisions"] += 1
        self.performance_tracker["subdivision_overhead_avg"] = (
            self.performance_tracker["subdivision_overhead_avg"]
            * (self.performance_tracker["total_subdivisions"] - 1)
            + subdivision_overhead
        ) / self.performance_tracker["total_subdivisions"]

        logger.info(
            f"Subdivided task {task.id} into {len(subtasks)} parts with {strategy.value} strategy"
        )

        return True, subtasks, metrics

    def _classify_task(self, task: Any) -> TaskType:
        """Classify task type for appropriate subdivision."""
        task_desc = getattr(task, "description", "").lower()

        if "search" in task_desc or "find" in task_desc:
            return TaskType.SEARCH_OPERATION
        elif "analyze" in task_desc or "analysis" in task_desc:
            return TaskType.ANALYSIS_OPERATION
        elif hasattr(task, "metadata"):
            task_type_str = task.metadata.get("type", "")
            if "search" in task_type_str:
                return TaskType.SEARCH_OPERATION
            elif "analysis" in task_type_str:
                return TaskType.ANALYSIS_OPERATION

        # Default classification
        return TaskType.MIXED_WORKLOAD

    def record_execution_result(
        self,
        original_task: Any,
        subtasks: list[Any],
        metrics: SubdivisionMetrics,
        actual_duration: float,
    ):
        """Record actual execution results for learning."""

        if metrics.original_duration_estimate > 0:
            actual_speedup = metrics.original_duration_estimate / actual_duration
            metrics.actual_speedup = actual_speedup

            # Update tracking
            if actual_speedup >= 1.1:  # At least 10% improvement
                self.performance_tracker["successful_subdivisions"] += 1

            # Update average speedup
            current_avg = self.performance_tracker["average_speedup"]
            total_subdivisions = self.performance_tracker["total_subdivisions"]
            self.performance_tracker["average_speedup"] = (
                current_avg * (total_subdivisions - 1) + actual_speedup
            ) / total_subdivisions

            # Record for decision engine learning
            task_type = self._classify_task(original_task)
            if task_type in self.subdividers:
                subdivider = self.subdividers[task_type]
                task_chars = subdivider.get_task_characteristics(original_task)
                self.decision_engine.record_subdivision_result(task_chars, metrics)

            logger.debug(
                f"Recorded subdivision result: predicted={metrics.predicted_speedup:.2f}, actual={actual_speedup:.2f}"
            )

    def get_performance_report(self) -> dict[str, Any]:
        """Get subdivision system performance report."""
        total = self.performance_tracker["total_subdivisions"]
        successful = self.performance_tracker["successful_subdivisions"]
        success_rate = successful / total if total > 0 else 0.0

        return {
            "total_subdivisions": total,
            "successful_subdivisions": successful,
            "success_rate": success_rate,
            "average_speedup": self.performance_tracker["average_speedup"],
            "average_subdivision_overhead_ms": self.performance_tracker[
                "subdivision_overhead_avg"
            ]
            * 1000,
            "supported_task_types": list(self.subdividers.keys()),
            "decision_engine_history_size": len(
                self.decision_engine.subdivision_history
            ),
        }


# Global instance
_subdivision_system: DynamicTaskSubdivisionSystem | None = None


def get_subdivision_system() -> DynamicTaskSubdivisionSystem:
    """Get global subdivision system instance."""
    global _subdivision_system
    if _subdivision_system is None:
        _subdivision_system = DynamicTaskSubdivisionSystem()
    return _subdivision_system
