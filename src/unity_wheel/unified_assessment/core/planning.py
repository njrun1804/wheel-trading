"""
Action Planning Layer - Task decomposition and resource optimization.

This module creates optimized execution plans by:
- Decomposing complex intents into atomic tasks
- Analyzing task dependencies
- Allocating resources efficiently
- Optimizing for M4 Pro hardware
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from ..schemas.command import TaskType
from .context import GatheredContext
from .intent import IntentAnalysis, Intent, IntentCategory

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task execution priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionStrategy(Enum):
    """Execution strategy for tasks."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    HYBRID = "hybrid"


@dataclass
class ResourceRequirement:
    """Resource requirements for a task."""
    
    cpu_cores: int = 1
    memory_mb: int = 100
    gpu_required: bool = False
    disk_io_intensive: bool = False
    network_required: bool = False
    estimated_duration_ms: float = 1000.0


@dataclass
class ActionTask:
    """Individual task in an execution plan."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    task_type: TaskType = TaskType.ANALYZE
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    dependents: List[str] = field(default_factory=list)   # Task IDs that depend on this
    
    # Execution details
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration_ms: float = 1000.0
    required_tools: List[str] = field(default_factory=list)
    
    # Resource requirements
    resources: ResourceRequirement = field(default_factory=ResourceRequirement)
    
    # Context and parameters
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # State tracking
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task can execute given completed dependencies."""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)
    
    def mark_completed(self, result: Dict[str, Any]):
        """Mark task as completed with result."""
        self.status = "completed"
        self.result = result
    
    def mark_failed(self, error: str):
        """Mark task as failed with error."""
        self.status = "failed"
        self.error = error


@dataclass
class ExecutionGroup:
    """Group of tasks that can execute in parallel."""
    
    tasks: List[ActionTask]
    strategy: ExecutionStrategy
    max_parallelism: int = 4
    estimated_duration_ms: float = 0.0
    
    def __post_init__(self):
        """Calculate estimated duration for the group."""
        if self.strategy == ExecutionStrategy.PARALLEL:
            # For parallel execution, duration is the maximum task duration
            self.estimated_duration_ms = max(
                (task.estimated_duration_ms for task in self.tasks), 
                default=0.0
            )
        else:
            # For sequential execution, duration is sum of task durations
            self.estimated_duration_ms = sum(
                task.estimated_duration_ms for task in self.tasks
            )


@dataclass
class FallbackStrategy:
    """Fallback strategy for handling task failures."""
    
    strategy_type: str  # retry, skip, alternative, abort
    max_retries: int = 3
    alternative_tasks: List[ActionTask] = field(default_factory=list)
    abort_on_failure: bool = False


@dataclass
class ExecutionPlan:
    """Complete execution plan with optimized task ordering."""
    
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Tasks and execution order
    tasks: List[ActionTask] = field(default_factory=list)
    execution_groups: List[ExecutionGroup] = field(default_factory=list)
    
    # Resource allocation
    total_estimated_duration_ms: float = 0.0
    peak_cpu_cores: int = 1
    peak_memory_mb: int = 100
    requires_gpu: bool = False
    
    # Fallback strategies
    fallback_strategies: Dict[str, FallbackStrategy] = field(default_factory=dict)
    
    # Metadata
    created_for_intent: Optional[str] = None
    optimization_target: str = "balanced"  # speed, accuracy, balanced
    hardware_profile: str = "m4_pro"
    
    def get_task_by_id(self, task_id: str) -> Optional[ActionTask]:
        """Get task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[ActionTask]:
        """Get tasks that are ready to execute."""
        return [
            task for task in self.tasks
            if task.status == "pending" and task.can_execute(completed_tasks)
        ]
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.status in ["completed", "failed"] for task in self.tasks)
    
    def get_success_rate(self) -> float:
        """Get success rate of completed tasks."""
        completed = sum(1 for task in self.tasks if task.status == "completed")
        failed = sum(1 for task in self.tasks if task.status == "failed")
        total_finished = completed + failed
        
        return completed / total_finished if total_finished > 0 else 0.0


class TaskDecomposer:
    """Decomposes high-level intents into atomic tasks."""
    
    def __init__(self):
        self.task_templates = {
            IntentCategory.FIX: self._create_fix_tasks,
            IntentCategory.CREATE: self._create_create_tasks,
            IntentCategory.OPTIMIZE: self._create_optimize_tasks,
            IntentCategory.ANALYZE: self._create_analyze_tasks,
            IntentCategory.REFACTOR: self._create_refactor_tasks,
            IntentCategory.TEST: self._create_test_tasks,
            IntentCategory.DEPLOY: self._create_deploy_tasks,
            IntentCategory.MONITOR: self._create_monitor_tasks,
            IntentCategory.QUERY: self._create_query_tasks,
        }
    
    def decompose_intent(
        self, 
        intent: Intent, 
        context: GatheredContext
    ) -> List[ActionTask]:
        """Decompose intent into atomic tasks."""
        
        if intent.category in self.task_templates:
            return self.task_templates[intent.category](intent, context)
        else:
            # Fallback for unknown intents
            return self._create_generic_tasks(intent, context)
    
    def _create_fix_tasks(self, intent: Intent, context: GatheredContext) -> List[ActionTask]:
        """Create tasks for fixing issues."""
        tasks = []
        
        # Task 1: Analyze the problem
        analyze_task = ActionTask(
            description="Analyze the issue to understand the problem",
            task_type=TaskType.ANALYZE,
            priority=TaskPriority.HIGH,
            estimated_duration_ms=2000.0,
            required_tools=["einstein_search", "python_analysis"],
            resources=ResourceRequirement(cpu_cores=2, memory_mb=200),
            parameters={"analysis_depth": "detailed", "include_stack_traces": True}
        )
        tasks.append(analyze_task)
        
        # Task 2: Identify affected files
        identify_task = ActionTask(
            description="Identify files affected by the issue",
            task_type=TaskType.SEARCH,
            dependencies=[analyze_task.id],
            estimated_duration_ms=1500.0,
            required_tools=["dependency_graph", "ripgrep"],
            resources=ResourceRequirement(cpu_cores=1, memory_mb=150)
        )
        tasks.append(identify_task)
        
        # Task 3: Generate fix
        fix_task = ActionTask(
            description="Generate and apply the fix",
            task_type=TaskType.MODIFY,
            dependencies=[identify_task.id],
            priority=TaskPriority.HIGH,
            estimated_duration_ms=3000.0,
            required_tools=["code_editor", "syntax_validator"],
            resources=ResourceRequirement(cpu_cores=1, memory_mb=100)
        )
        tasks.append(fix_task)
        
        # Task 4: Validate fix
        validate_task = ActionTask(
            description="Validate that the fix works correctly",
            task_type=TaskType.VALIDATE,
            dependencies=[fix_task.id],
            estimated_duration_ms=2000.0,
            required_tools=["test_runner", "static_analyzer"],
            resources=ResourceRequirement(cpu_cores=2, memory_mb=300)
        )
        tasks.append(validate_task)
        
        return tasks
    
    def _create_create_tasks(self, intent: Intent, context: GatheredContext) -> List[ActionTask]:
        """Create tasks for creating new components."""
        tasks = []
        
        # Task 1: Plan the new component
        plan_task = ActionTask(
            description="Plan the structure of the new component",
            task_type=TaskType.ANALYZE,
            priority=TaskPriority.NORMAL,
            estimated_duration_ms=1500.0,
            required_tools=["pattern_analyzer", "template_engine"],
            resources=ResourceRequirement(cpu_cores=1, memory_mb=100)
        )
        tasks.append(plan_task)
        
        # Task 2: Generate boilerplate
        generate_task = ActionTask(
            description="Generate boilerplate code and structure",
            task_type=TaskType.CREATE,
            dependencies=[plan_task.id],
            estimated_duration_ms=2000.0,
            required_tools=["code_generator", "template_engine"],
            resources=ResourceRequirement(cpu_cores=1, memory_mb=150)
        )
        tasks.append(generate_task)
        
        # Task 3: Implement core logic
        implement_task = ActionTask(
            description="Implement the core functionality",
            task_type=TaskType.CREATE,
            dependencies=[generate_task.id],
            priority=TaskPriority.HIGH,
            estimated_duration_ms=5000.0,
            required_tools=["code_editor", "syntax_validator"],
            resources=ResourceRequirement(cpu_cores=2, memory_mb=200)
        )
        tasks.append(implement_task)
        
        # Task 4: Add tests
        test_task = ActionTask(
            description="Create tests for the new component",
            task_type=TaskType.TEST,
            dependencies=[implement_task.id],
            estimated_duration_ms=3000.0,
            required_tools=["test_generator", "test_runner"],
            resources=ResourceRequirement(cpu_cores=1, memory_mb=200)
        )
        tasks.append(test_task)
        
        return tasks
    
    def _create_optimize_tasks(self, intent: Intent, context: GatheredContext) -> List[ActionTask]:
        """Create tasks for optimization."""
        tasks = []
        
        # Task 1: Profile current performance
        profile_task = ActionTask(
            description="Profile current performance to identify bottlenecks",
            task_type=TaskType.ANALYZE,
            priority=TaskPriority.HIGH,
            estimated_duration_ms=3000.0,
            required_tools=["profiler", "performance_monitor"],
            resources=ResourceRequirement(cpu_cores=2, memory_mb=300, gpu_required=True)
        )
        tasks.append(profile_task)
        
        # Task 2: Analyze bottlenecks
        analyze_task = ActionTask(
            description="Analyze performance bottlenecks",
            task_type=TaskType.ANALYZE,
            dependencies=[profile_task.id],
            estimated_duration_ms=2000.0,
            required_tools=["performance_analyzer"],
            resources=ResourceRequirement(cpu_cores=1, memory_mb=200)
        )
        tasks.append(analyze_task)
        
        # Task 3: Apply optimizations
        optimize_task = ActionTask(
            description="Apply performance optimizations",
            task_type=TaskType.MODIFY,
            dependencies=[analyze_task.id],
            priority=TaskPriority.HIGH,
            estimated_duration_ms=4000.0,
            required_tools=["code_optimizer", "syntax_validator"],
            resources=ResourceRequirement(cpu_cores=2, memory_mb=250)
        )
        tasks.append(optimize_task)
        
        # Task 4: Benchmark results
        benchmark_task = ActionTask(
            description="Benchmark the optimized code",
            task_type=TaskType.VALIDATE,
            dependencies=[optimize_task.id],
            estimated_duration_ms=2500.0,
            required_tools=["benchmark_runner", "performance_monitor"],
            resources=ResourceRequirement(cpu_cores=4, memory_mb=400, gpu_required=True)
        )
        tasks.append(benchmark_task)
        
        return tasks
    
    def _create_analyze_tasks(self, intent: Intent, context: GatheredContext) -> List[ActionTask]:
        """Create tasks for analysis."""
        tasks = []
        
        # Task 1: Gather analysis data
        gather_task = ActionTask(
            description="Gather data for analysis",
            task_type=TaskType.SEARCH,
            priority=TaskPriority.NORMAL,
            estimated_duration_ms=2000.0,
            required_tools=["einstein_search", "dependency_graph"],
            resources=ResourceRequirement(cpu_cores=2, memory_mb=200)
        )
        tasks.append(gather_task)
        
        # Task 2: Perform analysis
        analyze_task = ActionTask(
            description="Perform detailed analysis",
            task_type=TaskType.ANALYZE,
            dependencies=[gather_task.id],
            priority=TaskPriority.HIGH,
            estimated_duration_ms=3000.0,
            required_tools=["python_analysis", "pattern_analyzer"],
            resources=ResourceRequirement(cpu_cores=2, memory_mb=300)
        )
        tasks.append(analyze_task)
        
        # Task 3: Generate report
        report_task = ActionTask(
            description="Generate analysis report",
            task_type=TaskType.CREATE,
            dependencies=[analyze_task.id],
            estimated_duration_ms=1500.0,
            required_tools=["report_generator"],
            resources=ResourceRequirement(cpu_cores=1, memory_mb=100)
        )
        tasks.append(report_task)
        
        return tasks
    
    def _create_refactor_tasks(self, intent: Intent, context: GatheredContext) -> List[ActionTask]:
        """Create tasks for refactoring."""
        return self._create_fix_tasks(intent, context)  # Similar to fix tasks
    
    def _create_test_tasks(self, intent: Intent, context: GatheredContext) -> List[ActionTask]:
        """Create tasks for testing."""
        tasks = []
        
        # Single comprehensive test task
        test_task = ActionTask(
            description="Run comprehensive tests",
            task_type=TaskType.TEST,
            priority=TaskPriority.HIGH,
            estimated_duration_ms=5000.0,
            required_tools=["test_runner", "coverage_analyzer"],
            resources=ResourceRequirement(cpu_cores=4, memory_mb=500)
        )
        tasks.append(test_task)
        
        return tasks
    
    def _create_deploy_tasks(self, intent: Intent, context: GatheredContext) -> List[ActionTask]:
        """Create tasks for deployment."""
        tasks = []
        
        # Single deployment task
        deploy_task = ActionTask(
            description="Deploy to target environment",
            task_type=TaskType.DEPLOY,
            priority=TaskPriority.CRITICAL,
            estimated_duration_ms=10000.0,
            required_tools=["deployment_manager"],
            resources=ResourceRequirement(cpu_cores=2, memory_mb=300, network_required=True)
        )
        tasks.append(deploy_task)
        
        return tasks
    
    def _create_monitor_tasks(self, intent: Intent, context: GatheredContext) -> List[ActionTask]:
        """Create tasks for monitoring."""
        tasks = []
        
        # Single monitoring task
        monitor_task = ActionTask(
            description="Set up monitoring and alerts",
            task_type=TaskType.MONITOR,
            priority=TaskPriority.NORMAL,
            estimated_duration_ms=3000.0,
            required_tools=["monitoring_system"],
            resources=ResourceRequirement(cpu_cores=1, memory_mb=150, network_required=True)
        )
        tasks.append(monitor_task)
        
        return tasks
    
    def _create_query_tasks(self, intent: Intent, context: GatheredContext) -> List[ActionTask]:
        """Create tasks for queries."""
        tasks = []
        
        # Single search/query task
        query_task = ActionTask(
            description="Search and retrieve requested information",
            task_type=TaskType.SEARCH,
            priority=TaskPriority.NORMAL,
            estimated_duration_ms=1500.0,
            required_tools=["einstein_search", "ripgrep"],
            resources=ResourceRequirement(cpu_cores=2, memory_mb=200)
        )
        tasks.append(query_task)
        
        return tasks
    
    def _create_generic_tasks(self, intent: Intent, context: GatheredContext) -> List[ActionTask]:
        """Create generic tasks for unknown intents."""
        tasks = []
        
        # Generic task
        generic_task = ActionTask(
            description=f"Execute {intent.action} action",
            task_type=TaskType.ANALYZE,
            priority=TaskPriority.NORMAL,
            estimated_duration_ms=2000.0,
            required_tools=["einstein_search"],
            resources=ResourceRequirement(cpu_cores=1, memory_mb=150)
        )
        tasks.append(generic_task)
        
        return tasks


class ResourceOptimizer:
    """Optimizes task execution for M4 Pro hardware."""
    
    def __init__(self):
        self.hardware_profile = {
            "cpu_cores_performance": 8,  # P cores
            "cpu_cores_efficiency": 4,   # E cores
            "gpu_cores": 20,             # Metal GPU cores
            "unified_memory_gb": 24,
            "max_concurrent_tasks": 12,
            "thermal_limit": 0.85        # 85% thermal threshold
        }
    
    def optimize_execution_plan(self, tasks: List[ActionTask]) -> ExecutionPlan:
        """Optimize task execution plan for M4 Pro."""
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(tasks)
        
        # Create execution groups
        execution_groups = self._create_execution_groups(tasks, dependency_graph)
        
        # Calculate resource requirements
        peak_cpu, peak_memory, requires_gpu = self._calculate_peak_resources(execution_groups)
        
        # Calculate total duration
        total_duration = sum(group.estimated_duration_ms for group in execution_groups)
        
        # Create optimized plan
        plan = ExecutionPlan(
            tasks=tasks,
            execution_groups=execution_groups,
            total_estimated_duration_ms=total_duration,
            peak_cpu_cores=peak_cpu,
            peak_memory_mb=peak_memory,
            requires_gpu=requires_gpu,
            hardware_profile="m4_pro"
        )
        
        return plan
    
    def _build_dependency_graph(self, tasks: List[ActionTask]) -> Dict[str, List[str]]:
        """Build task dependency graph."""
        graph = {}
        
        for task in tasks:
            graph[task.id] = task.dependencies
            
            # Update dependents
            for dep_id in task.dependencies:
                for dep_task in tasks:
                    if dep_task.id == dep_id:
                        if task.id not in dep_task.dependents:
                            dep_task.dependents.append(task.id)
        
        return graph
    
    def _create_execution_groups(
        self, 
        tasks: List[ActionTask], 
        dependency_graph: Dict[str, List[str]]
    ) -> List[ExecutionGroup]:
        """Create optimized execution groups."""
        
        groups = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # Find tasks with no dependencies among remaining tasks
            ready_tasks = [
                task for task in remaining_tasks 
                if not any(dep_id in [t.id for t in remaining_tasks] for dep_id in task.dependencies)
            ]
            
            if not ready_tasks:
                # Handle circular dependencies by picking highest priority task
                ready_tasks = [max(remaining_tasks, key=lambda t: t.priority.value)]
            
            # Group tasks that can run in parallel
            parallel_group = []
            sequential_group = []
            
            for task in ready_tasks:
                if (task.resources.cpu_cores <= 2 and 
                    not task.resources.gpu_required and
                    len(parallel_group) < self.hardware_profile["max_concurrent_tasks"]):
                    parallel_group.append(task)
                else:
                    sequential_group.append(task)
            
            # Create execution groups
            if parallel_group:
                groups.append(ExecutionGroup(
                    tasks=parallel_group,
                    strategy=ExecutionStrategy.PARALLEL,
                    max_parallelism=min(len(parallel_group), 8)
                ))
            
            if sequential_group:
                groups.append(ExecutionGroup(
                    tasks=sequential_group,
                    strategy=ExecutionStrategy.SEQUENTIAL
                ))
            
            # Remove processed tasks
            for task in ready_tasks:
                remaining_tasks.remove(task)
        
        return groups
    
    def _calculate_peak_resources(self, groups: List[ExecutionGroup]) -> tuple[int, int, bool]:
        """Calculate peak resource requirements."""
        
        peak_cpu = 0
        peak_memory = 0
        requires_gpu = False
        
        for group in groups:
            if group.strategy == ExecutionStrategy.PARALLEL:
                group_cpu = sum(task.resources.cpu_cores for task in group.tasks)
                group_memory = sum(task.resources.memory_mb for task in group.tasks)
                group_gpu = any(task.resources.gpu_required for task in group.tasks)
            else:
                group_cpu = max((task.resources.cpu_cores for task in group.tasks), default=0)
                group_memory = max((task.resources.memory_mb for task in group.tasks), default=0)
                group_gpu = any(task.resources.gpu_required for task in group.tasks)
            
            peak_cpu = max(peak_cpu, group_cpu)
            peak_memory = max(peak_memory, group_memory)
            requires_gpu = requires_gpu or group_gpu
        
        return peak_cpu, peak_memory, requires_gpu


class ActionPlanner:
    """
    Action planner that creates optimized execution plans.
    
    Combines intent analysis with context to create detailed,
    hardware-optimized execution plans.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_decomposer = TaskDecomposer()
        self.resource_optimizer = ResourceOptimizer()
        
        # Performance tracking
        self.total_plans_created = 0
        self.average_planning_time_ms = 0.0
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize action planner."""
        if self.initialized:
            return
        
        logger.info("📋 Initializing Action Planner...")
        
        # Planning is mostly algorithmic, so initialization is quick
        self.initialized = True
        
        logger.info("✅ Action Planner initialized")
    
    async def plan_actions(
        self,
        intent_analysis: IntentAnalysis,
        context: GatheredContext
    ) -> ExecutionPlan:
        """
        Create an optimized execution plan from intent analysis.
        
        Args:
            intent_analysis: Analyzed user intent
            context: Gathered context
            
        Returns:
            ExecutionPlan with optimized task ordering and resource allocation
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            primary_intent = intent_analysis.primary_intent
            
            # Stage 1: Decompose intent into tasks
            tasks = self.task_decomposer.decompose_intent(primary_intent, context)
            
            # Stage 2: Optimize for hardware
            execution_plan = self.resource_optimizer.optimize_execution_plan(tasks)
            
            # Stage 3: Add metadata
            execution_plan.created_for_intent = primary_intent.category.value
            execution_plan.optimization_target = context.query.optimization_target
            
            # Stage 4: Create fallback strategies
            fallback_strategies = self._create_fallback_strategies(tasks, primary_intent)
            execution_plan.fallback_strategies = fallback_strategies
            
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update statistics
            self.total_plans_created += 1
            self.average_planning_time_ms = (
                (self.average_planning_time_ms * (self.total_plans_created - 1) + processing_time_ms)
                / self.total_plans_created
            )
            
            logger.debug(
                f"📋 Execution plan created in {processing_time_ms:.1f}ms "
                f"({len(tasks)} tasks, {len(execution_plan.execution_groups)} groups, "
                f"~{execution_plan.total_estimated_duration_ms/1000:.1f}s duration)"
            )
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"❌ Action planning failed: {e}")
            
            # Return minimal fallback plan
            fallback_task = ActionTask(
                description="Execute fallback action",
                task_type=TaskType.ANALYZE,
                estimated_duration_ms=1000.0,
                required_tools=["einstein_search"]
            )
            
            return ExecutionPlan(
                tasks=[fallback_task],
                execution_groups=[ExecutionGroup(
                    tasks=[fallback_task],
                    strategy=ExecutionStrategy.SEQUENTIAL
                )],
                total_estimated_duration_ms=1000.0,
                peak_cpu_cores=1,
                peak_memory_mb=100,
                requires_gpu=False
            )
    
    def _create_fallback_strategies(
        self, 
        tasks: List[ActionTask], 
        intent: Intent
    ) -> Dict[str, FallbackStrategy]:
        """Create fallback strategies for task failures."""
        
        strategies = {}
        
        # Critical tasks should retry
        for task in tasks:
            if task.priority == TaskPriority.CRITICAL:
                strategies[task.id] = FallbackStrategy(
                    strategy_type="retry",
                    max_retries=3,
                    abort_on_failure=True
                )
            elif task.priority == TaskPriority.HIGH:
                strategies[task.id] = FallbackStrategy(
                    strategy_type="retry",
                    max_retries=2,
                    abort_on_failure=False
                )
            else:
                strategies[task.id] = FallbackStrategy(
                    strategy_type="skip",
                    max_retries=1,
                    abort_on_failure=False
                )
        
        return strategies
    
    async def shutdown(self):
        """Shutdown action planner."""
        logger.debug("🔄 Shutting down Action Planner")
        self.initialized = False
        logger.debug("✅ Action Planner shutdown complete")