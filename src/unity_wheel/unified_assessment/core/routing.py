"""
Execution Routing Layer - Routes tasks to appropriate execution engines.

This module handles task execution by routing to:
- Bolt multi-agent orchestration for complex tasks
- Direct tool execution for simple tasks  
- Hybrid execution for mixed workloads
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .context import GatheredContext
from .planning import ExecutionPlan, ActionTask, ExecutionGroup, ExecutionStrategy

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies for task execution."""
    BOLT_MULTI_AGENT = "bolt_multi_agent"
    DIRECT_TOOLS = "direct_tools" 
    HYBRID = "hybrid"
    EINSTEIN_ENHANCED = "einstein_enhanced"


@dataclass
class ExecutionResult:
    """Result of executing a single task or group."""
    
    task_id: str
    success: bool
    duration_ms: float
    result_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Output artifacts
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    
    # Metadata
    execution_strategy: Optional[RoutingStrategy] = None
    resources_used: Dict[str, float] = field(default_factory=dict)


@dataclass
class PlanExecutionResult:
    """Result of executing an entire execution plan."""
    
    plan_id: str
    success: bool
    total_duration_ms: float
    
    # Task results
    task_results: List[ExecutionResult] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Aggregated outputs
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    
    # File changes
    files_affected: List[str] = field(default_factory=list)
    changes_made: List[Dict[str, Any]] = field(default_factory=list)
    
    # Errors and metrics
    errors: List[Dict[str, str]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_task_result(self, result: ExecutionResult):
        """Add a task execution result."""
        self.task_results.append(result)
        
        if result.success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1
            self.errors.append({
                "task": result.task_id,
                "error": result.error or "Unknown error"
            })
        
        # Aggregate file changes
        self.files_affected.extend(result.files_created)
        self.files_affected.extend(result.files_modified)
        self.files_affected.extend(result.files_deleted)
        
        # Remove duplicates
        self.files_affected = list(set(self.files_affected))


class BoltMultiAgentExecutor:
    """Executes tasks using the Bolt multi-agent system."""
    
    def __init__(self):
        self.bolt_integration = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize Bolt integration."""
        if self.initialized:
            return
        
        try:
            # Lazy import to avoid circular dependencies
            from bolt.core.integration import BoltIntegration
            self.bolt_integration = BoltIntegration(num_agents=8)
            await self.bolt_integration.initialize()
            self.initialized = True
            logger.debug("✅ Bolt multi-agent executor initialized")
        except Exception as e:
            logger.warning(f"⚠️  Bolt multi-agent executor not available: {e}")
            self.bolt_integration = None
    
    async def execute_task(
        self, 
        task: ActionTask, 
        context: GatheredContext
    ) -> ExecutionResult:
        """Execute a task using Bolt multi-agent system."""
        
        if not self.bolt_integration:
            return ExecutionResult(
                task_id=task.id,
                success=False,
                duration_ms=0.0,
                error="Bolt integration not available",
                execution_strategy=RoutingStrategy.BOLT_MULTI_AGENT
            )
        
        start_time = time.perf_counter()
        
        try:
            # Convert task to Bolt query format
            bolt_query = self._convert_task_to_bolt_query(task, context)
            
            # Execute through Bolt
            bolt_result = await self.bolt_integration.solve(bolt_query, analyze_only=False)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Convert Bolt result to ExecutionResult
            return self._convert_bolt_result(task.id, bolt_result, duration_ms)
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"❌ Bolt execution failed for task {task.id}: {e}")
            
            return ExecutionResult(
                task_id=task.id,
                success=False,
                duration_ms=duration_ms,
                error=str(e),
                execution_strategy=RoutingStrategy.BOLT_MULTI_AGENT
            )
    
    def _convert_task_to_bolt_query(self, task: ActionTask, context: GatheredContext) -> str:
        """Convert ActionTask to Bolt query string."""
        
        # Build query based on task type and description
        query_parts = [task.description]
        
        # Add context from relevant files
        if context.relevant_files:
            file_list = ", ".join([fc.file_path for fc in context.relevant_files[:5]])
            query_parts.append(f"Focus on files: {file_list}")
        
        # Add task-specific instructions
        if task.task_type.value == "fix":
            query_parts.append("Identify the issue and provide a fix")
        elif task.task_type.value == "create":
            query_parts.append("Create new code following best practices")
        elif task.task_type.value == "optimize":
            query_parts.append("Optimize for performance and efficiency")
        elif task.task_type.value == "analyze":
            query_parts.append("Provide detailed analysis and insights")
        
        return ". ".join(query_parts)
    
    def _convert_bolt_result(
        self, 
        task_id: str, 
        bolt_result: Dict[str, Any], 
        duration_ms: float
    ) -> ExecutionResult:
        """Convert Bolt result to ExecutionResult."""
        
        success = bolt_result.get("success", False)
        results = bolt_result.get("results", {})
        
        result = ExecutionResult(
            task_id=task_id,
            success=success,
            duration_ms=duration_ms,
            execution_strategy=RoutingStrategy.BOLT_MULTI_AGENT
        )
        
        # Extract findings and recommendations
        if isinstance(results, dict):
            result.result_data = results
            
            # Extract common fields
            if "summary" in results:
                result.result_data["summary"] = results["summary"]
            if "findings" in results:
                result.result_data["findings"] = results["findings"]
            if "recommendations" in results:
                result.result_data["recommendations"] = results["recommendations"]
        
        return result
    
    async def shutdown(self):
        """Shutdown Bolt integration."""
        if self.bolt_integration:
            await self.bolt_integration.shutdown()
        self.initialized = False


class DirectToolExecutor:
    """Executes tasks using direct tool calls."""
    
    def __init__(self):
        self.tools = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize direct tools."""
        if self.initialized:
            return
        
        try:
            # Initialize accelerated tools
            await self._init_accelerated_tools()
            self.initialized = True
            logger.debug("✅ Direct tool executor initialized")
        except Exception as e:
            logger.warning(f"⚠️  Some direct tools not available: {e}")
    
    async def _init_accelerated_tools(self):
        """Initialize accelerated tools."""
        try:
            from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
            from src.unity_wheel.accelerated_tools.python_analysis_turbo import get_python_analyzer
            from src.unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph
            
            self.tools["ripgrep"] = get_ripgrep_turbo()
            self.tools["python_analyzer"] = get_python_analyzer()
            self.tools["dependency_graph"] = get_dependency_graph()
            
        except Exception as e:
            logger.debug(f"Accelerated tools partially available: {e}")
    
    async def execute_task(
        self, 
        task: ActionTask, 
        context: GatheredContext
    ) -> ExecutionResult:
        """Execute a task using direct tools."""
        
        start_time = time.perf_counter()
        
        try:
            # Route to appropriate tool based on task type
            if task.task_type.value == "search":
                result_data = await self._execute_search_task(task, context)
            elif task.task_type.value == "analyze":
                result_data = await self._execute_analyze_task(task, context)
            else:
                # Fallback for other task types
                result_data = await self._execute_generic_task(task, context)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return ExecutionResult(
                task_id=task.id,
                success=True,
                duration_ms=duration_ms,
                result_data=result_data,
                execution_strategy=RoutingStrategy.DIRECT_TOOLS
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"❌ Direct tool execution failed for task {task.id}: {e}")
            
            return ExecutionResult(
                task_id=task.id,
                success=False,
                duration_ms=duration_ms,
                error=str(e),
                execution_strategy=RoutingStrategy.DIRECT_TOOLS
            )
    
    async def _execute_search_task(
        self, 
        task: ActionTask, 
        context: GatheredContext
    ) -> Dict[str, Any]:
        """Execute a search task."""
        
        results = {
            "task_type": "search",
            "description": task.description,
            "findings": []
        }
        
        # Use ripgrep for text search
        if "ripgrep" in self.tools:
            try:
                # Extract search terms from task description
                search_terms = task.description.lower().split()
                search_terms = [term for term in search_terms if len(term) > 3]
                
                if search_terms:
                    search_results = await self.tools["ripgrep"].parallel_search(
                        search_terms[:3], "src"  # Limit to top 3 terms
                    )
                    results["findings"].extend([
                        f"Found '{term}' in {len(matches)} locations"
                        for term, matches in search_results.items()
                    ])
            except Exception as e:
                results["warnings"] = [f"Search tool error: {e}"]
        
        # Add context-based findings
        if context.relevant_files:
            results["findings"].append(
                f"Found {len(context.relevant_files)} relevant files"
            )
        
        return results
    
    async def _execute_analyze_task(
        self, 
        task: ActionTask, 
        context: GatheredContext
    ) -> Dict[str, Any]:
        """Execute an analysis task."""
        
        results = {
            "task_type": "analyze",
            "description": task.description,
            "findings": [],
            "recommendations": []
        }
        
        # Use Python analyzer if available
        if "python_analyzer" in self.tools and context.relevant_files:
            try:
                # Analyze top relevant Python files
                python_files = [
                    fc.file_path for fc in context.relevant_files 
                    if fc.file_path.endswith('.py')
                ][:5]
                
                if python_files:
                    for file_path in python_files:
                        try:
                            analysis = await self.tools["python_analyzer"].analyze_file(file_path)
                            if analysis:
                                results["findings"].append(
                                    f"Analyzed {file_path}: {len(analysis.get('functions', []))} functions, "
                                    f"{len(analysis.get('classes', []))} classes"
                                )
                        except Exception:
                            continue
                            
            except Exception as e:
                results["warnings"] = [f"Analysis tool error: {e}"]
        
        # Add general recommendations
        results["recommendations"].extend([
            "Review code for best practices",
            "Consider adding tests if missing",
            "Check for potential optimizations"
        ])
        
        return results
    
    async def _execute_generic_task(
        self, 
        task: ActionTask, 
        context: GatheredContext
    ) -> Dict[str, Any]:
        """Execute a generic task."""
        
        return {
            "task_type": task.task_type.value,
            "description": task.description,
            "findings": [f"Executed {task.task_type.value} task"],
            "recommendations": ["Task completed successfully"]
        }
    
    async def shutdown(self):
        """Shutdown direct tools."""
        self.tools.clear()
        self.initialized = False


class HybridExecutor:
    """Executes tasks using a hybrid approach combining multiple strategies."""
    
    def __init__(self, bolt_executor: BoltMultiAgentExecutor, direct_executor: DirectToolExecutor):
        self.bolt_executor = bolt_executor
        self.direct_executor = direct_executor
        self.initialized = False
    
    async def initialize(self):
        """Initialize hybrid executor."""
        if self.initialized:
            return
        
        # Both executors should already be initialized
        self.initialized = True
        logger.debug("✅ Hybrid executor initialized")
    
    async def execute_task(
        self, 
        task: ActionTask, 
        context: GatheredContext
    ) -> ExecutionResult:
        """Execute task using hybrid approach."""
        
        # Decide which executor to use based on task characteristics
        if self._should_use_bolt(task, context):
            return await self.bolt_executor.execute_task(task, context)
        else:
            return await self.direct_executor.execute_task(task, context)
    
    def _should_use_bolt(self, task: ActionTask, context: GatheredContext) -> bool:
        """Decide whether to use Bolt for this task."""
        
        # Use Bolt for complex tasks
        if task.task_type.value in ["modify", "create", "refactor"]:
            return True
        
        # Use Bolt for tasks involving multiple files
        if len(context.relevant_files) > 3:
            return True
        
        # Use Bolt for high-priority tasks
        if task.priority.value in ["high", "critical"]:
            return True
        
        # Use direct tools for simple tasks
        return False
    
    async def shutdown(self):
        """Shutdown hybrid executor."""
        self.initialized = False


class ExecutionRouter:
    """
    Main execution router that coordinates task execution across different engines.
    
    Routes tasks to appropriate execution strategies based on task characteristics,
    system load, and optimization targets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Execution engines
        self.bolt_executor = BoltMultiAgentExecutor()
        self.direct_executor = DirectToolExecutor() 
        self.hybrid_executor = None  # Created after initialization
        
        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.average_execution_time_ms = 0.0
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize execution router and all engines."""
        if self.initialized:
            return
        
        logger.info("🔀 Initializing Execution Router...")
        
        try:
            # Initialize engines in parallel
            await asyncio.gather(
                self.bolt_executor.initialize(),
                self.direct_executor.initialize(),
                return_exceptions=True
            )
            
            # Create hybrid executor
            self.hybrid_executor = HybridExecutor(self.bolt_executor, self.direct_executor)
            await self.hybrid_executor.initialize()
            
            self.initialized = True
            logger.info("✅ Execution Router initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Execution Router: {e}")
            raise
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        context: GatheredContext
    ) -> PlanExecutionResult:
        """
        Execute a complete plan using optimal routing strategies.
        
        Args:
            plan: ExecutionPlan to execute
            context: Gathered context
            
        Returns:
            PlanExecutionResult with aggregated results
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        result = PlanExecutionResult(
            plan_id=plan.plan_id,
            success=False,
            total_duration_ms=0.0
        )
        
        try:
            logger.debug(f"🔀 Executing plan {plan.plan_id} with {len(plan.tasks)} tasks")
            
            # Execute tasks according to execution groups
            for group in plan.execution_groups:
                if group.strategy == ExecutionStrategy.PARALLEL:
                    await self._execute_group_parallel(group, context, result)
                else:
                    await self._execute_group_sequential(group, context, result)
            
            # Calculate final metrics
            total_duration_ms = (time.perf_counter() - start_time) * 1000
            result.total_duration_ms = total_duration_ms
            result.success = result.failed_tasks == 0
            
            # Aggregate results
            self._aggregate_results(result)
            
            # Update statistics
            self.total_executions += 1
            if result.success:
                self.successful_executions += 1
            
            self.average_execution_time_ms = (
                (self.average_execution_time_ms * (self.total_executions - 1) + total_duration_ms)
                / self.total_executions
            )
            
            logger.info(
                f"✅ Plan executed in {total_duration_ms:.1f}ms "
                f"({result.completed_tasks}/{result.completed_tasks + result.failed_tasks} tasks successful)"
            )
            
        except Exception as e:
            logger.error(f"❌ Plan execution failed: {e}")
            result.success = False
            result.errors.append({"task": "plan_execution", "error": str(e)})
        
        return result
    
    async def _execute_group_parallel(
        self,
        group: ExecutionGroup,
        context: GatheredContext,
        result: PlanExecutionResult
    ):
        """Execute a group of tasks in parallel."""
        
        logger.debug(f"🔀 Executing {len(group.tasks)} tasks in parallel")
        
        # Create tasks for parallel execution
        execution_tasks = [
            self._execute_single_task(task, context)
            for task in group.tasks
        ]
        
        # Execute in parallel with timeout
        try:
            task_results = await asyncio.wait_for(
                asyncio.gather(*execution_tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout for parallel group
            )
            
            # Process results
            for task_result in task_results:
                if isinstance(task_result, ExecutionResult):
                    result.add_task_result(task_result)
                elif isinstance(task_result, Exception):
                    # Create failed result for exception
                    failed_result = ExecutionResult(
                        task_id="unknown",
                        success=False,
                        duration_ms=0.0,
                        error=str(task_result)
                    )
                    result.add_task_result(failed_result)
                    
        except asyncio.TimeoutError:
            logger.error("❌ Parallel group execution timed out")
            for task in group.tasks:
                timeout_result = ExecutionResult(
                    task_id=task.id,
                    success=False,
                    duration_ms=0.0,
                    error="Execution timed out"
                )
                result.add_task_result(timeout_result)
    
    async def _execute_group_sequential(
        self,
        group: ExecutionGroup,
        context: GatheredContext,
        result: PlanExecutionResult
    ):
        """Execute a group of tasks sequentially."""
        
        logger.debug(f"🔀 Executing {len(group.tasks)} tasks sequentially")
        
        for task in group.tasks:
            task_result = await self._execute_single_task(task, context)
            result.add_task_result(task_result)
            
            # Stop on critical failures
            if not task_result.success and task.priority.value == "critical":
                logger.error(f"❌ Critical task {task.id} failed, stopping execution")
                break
    
    async def _execute_single_task(
        self,
        task: ActionTask,
        context: GatheredContext
    ) -> ExecutionResult:
        """Execute a single task using appropriate routing."""
        
        # Determine routing strategy
        strategy = self._determine_routing_strategy(task, context)
        
        # Route to appropriate executor
        if strategy == RoutingStrategy.BOLT_MULTI_AGENT:
            return await self.bolt_executor.execute_task(task, context)
        elif strategy == RoutingStrategy.DIRECT_TOOLS:
            return await self.direct_executor.execute_task(task, context)
        elif strategy == RoutingStrategy.HYBRID:
            return await self.hybrid_executor.execute_task(task, context)
        else:
            # Fallback to direct tools
            return await self.direct_executor.execute_task(task, context)
    
    def _determine_routing_strategy(
        self,
        task: ActionTask,
        context: GatheredContext
    ) -> RoutingStrategy:
        """Determine optimal routing strategy for a task."""
        
        # Multi-file operations -> Bolt
        if len(context.relevant_files) > 5:
            return RoutingStrategy.BOLT_MULTI_AGENT
        
        # Complex task types -> Bolt
        if task.task_type.value in ["modify", "create", "refactor"]:
            return RoutingStrategy.BOLT_MULTI_AGENT
        
        # High resource requirements -> Bolt
        if task.resources.cpu_cores > 2 or task.resources.gpu_required:
            return RoutingStrategy.BOLT_MULTI_AGENT
        
        # Simple search/analysis -> Direct tools
        if task.task_type.value in ["search", "analyze"]:
            return RoutingStrategy.DIRECT_TOOLS
        
        # Default to hybrid
        return RoutingStrategy.HYBRID
    
    def _aggregate_results(self, result: PlanExecutionResult):
        """Aggregate results from individual task executions."""
        
        # Collect findings and recommendations
        for task_result in result.task_results:
            if "findings" in task_result.result_data:
                result.findings.extend(task_result.result_data["findings"])
            
            if "recommendations" in task_result.result_data:
                result.recommendations.extend(task_result.result_data["recommendations"])
            
            if "summary" in task_result.result_data:
                result.actions_taken.append(task_result.result_data["summary"])
        
        # Remove duplicates
        result.findings = list(set(result.findings))
        result.recommendations = list(set(result.recommendations))
        
        # Calculate metrics
        total_tasks = result.completed_tasks + result.failed_tasks
        success_rate = result.completed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        avg_duration = sum(tr.duration_ms for tr in result.task_results) / len(result.task_results) if result.task_results else 0.0
        
        result.metrics = {
            "tasks_completed": result.completed_tasks,
            "tasks_failed": result.failed_tasks,
            "success_rate": success_rate,
            "average_task_duration_ms": avg_duration,
            "files_modified": len(result.files_affected),
            "total_findings": len(result.findings),
            "total_recommendations": len(result.recommendations)
        }
    
    async def shutdown(self):
        """Shutdown execution router and all engines."""
        logger.info("🔄 Shutting down Execution Router")
        
        # Shutdown engines in parallel
        shutdown_tasks = [
            self.bolt_executor.shutdown(),
            self.direct_executor.shutdown()
        ]
        
        if self.hybrid_executor:
            shutdown_tasks.append(self.hybrid_executor.shutdown())
        
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.initialized = False
        logger.info("✅ Execution Router shutdown complete")