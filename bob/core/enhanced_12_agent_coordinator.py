"""Enhanced 12-agent coordinator optimized for M4 Pro (8P+4E cores) with Claude Code integration.

This coordinator maximizes M4 Pro utilization by:
1. Deploying 8 primary agents on P-cores for compute-intensive tasks
2. Using 4 coordination agents on E-cores for I/O and management
3. HTTP/2 session pooling for Claude requests
4. Dynamic workload balancing based on core performance characteristics
5. Specialized agent roles optimized for different task types
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ProcessPoolExecutor

from ..optimization.m4_claude_optimizer import M4ClaudeOptimizer, create_m4_optimizer
from ..agents.agent_pool import WorkStealingAgentPool, WorkStealingTask, TaskPriority
from ..agents.orchestrator import AgentOrchestrator
from ..utils.logging import get_component_logger


class AgentRole(Enum):
    """Specialized agent roles optimized for M4 Pro architecture."""
    
    # P-core agents (8 agents for compute-intensive tasks)
    ANALYZER = "analyzer"          # Code analysis, parsing, AST operations
    ARCHITECT = "architect"        # System design, architecture planning
    OPTIMIZER = "optimizer"        # Performance optimization, code improvement
    GENERATOR = "generator"        # Code generation, template processing
    VALIDATOR = "validator"        # Testing, validation, quality checks
    INTEGRATOR = "integrator"      # Component integration, dependency resolution
    RESEARCHER = "researcher"      # Information gathering, documentation analysis
    SYNTHESIZER = "synthesizer"    # Result combination, report generation
    
    # E-core agents (4 agents for coordination and I/O)
    COORDINATOR = "coordinator"    # Task coordination, workflow management
    DOCUMENTER = "documenter"      # Documentation, logging, reporting
    MONITOR = "monitor"           # System monitoring, health checks
    REPORTER = "reporter"         # Result reporting, user communication


@dataclass
class AgentCapability:
    """Defines an agent's capabilities and performance characteristics."""
    
    role: AgentRole
    cpu_intensive: bool = True      # Should run on P-cores
    io_intensive: bool = False      # Benefits from E-core efficiency
    memory_intensive: bool = False  # Requires memory optimization
    claude_requests: bool = True    # Makes Claude API requests
    specialized_tools: List[str] = field(default_factory=list)
    estimated_task_duration: float = 1.0  # Average task duration in seconds


@dataclass
class Enhanced12AgentTask:
    """Enhanced task with agent role preferences and performance hints."""
    
    task_id: str
    description: str
    task_type: str
    data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Agent role preferences (ordered by preference)
    preferred_roles: List[AgentRole] = field(default_factory=list)
    
    # Performance hints
    cpu_intensive: bool = True
    io_intensive: bool = False
    memory_intensive: bool = False
    estimated_duration: float = 1.0
    requires_claude: bool = True
    
    # Parallelization
    parallelizable: bool = True
    max_parallel_subtasks: int = 4


class Enhanced12AgentCoordinator:
    """Enhanced coordinator managing 12 specialized agents on M4 Pro."""
    
    def __init__(self):
        self.logger = get_component_logger("enhanced_12_agent_coordinator")
        
        # M4 optimization
        self.m4_optimizer: Optional[M4ClaudeOptimizer] = None
        
        # Agent architecture
        self.agent_capabilities = self._define_agent_capabilities()
        self.p_core_agents: List[AgentRole] = []  # 8 P-core agents
        self.e_core_agents: List[AgentRole] = []  # 4 E-core agents
        
        # Core components
        self.agent_pool: Optional[WorkStealingAgentPool] = None
        self.orchestrator: Optional[AgentOrchestrator] = None
        
        # Performance tracking
        self.agent_performance: Dict[AgentRole, Dict[str, float]] = {}
        self.task_routing_stats: Dict[str, int] = {}
        
        # Dynamic optimization
        self.workload_balancer = WorkloadBalancer()
        self.task_router = IntelligentTaskRouter(self.agent_capabilities)
        
        # State
        self._initialized = False
        self._running = False
        self._metrics = {
            "total_tasks": 0,
            "p_core_utilization": 0.0,
            "e_core_utilization": 0.0,
            "avg_task_latency": 0.0,
            "claude_requests_per_second": 0.0,
            "agent_specialization_efficiency": 0.0
        }
    
    def _define_agent_capabilities(self) -> Dict[AgentRole, AgentCapability]:
        """Define capabilities for each agent role."""
        return {
            # P-core agents (compute-intensive)
            AgentRole.ANALYZER: AgentCapability(
                role=AgentRole.ANALYZER,
                cpu_intensive=True,
                specialized_tools=["python_analysis", "ripgrep", "ast_parser"],
                estimated_task_duration=2.0
            ),
            AgentRole.ARCHITECT: AgentCapability(
                role=AgentRole.ARCHITECT,
                cpu_intensive=True,
                specialized_tools=["dependency_graph", "design_patterns"],
                estimated_task_duration=3.0
            ),
            AgentRole.OPTIMIZER: AgentCapability(
                role=AgentRole.OPTIMIZER,
                cpu_intensive=True,
                memory_intensive=True,
                specialized_tools=["performance_analyzer", "code_optimizer"],
                estimated_task_duration=2.5
            ),
            AgentRole.GENERATOR: AgentCapability(
                role=AgentRole.GENERATOR,
                cpu_intensive=True,
                claude_requests=True,
                specialized_tools=["code_generator", "template_engine"],
                estimated_task_duration=1.5
            ),
            AgentRole.VALIDATOR: AgentCapability(
                role=AgentRole.VALIDATOR,
                cpu_intensive=True,
                io_intensive=True,
                specialized_tools=["test_runner", "lint_checker", "validator"],
                estimated_task_duration=1.8
            ),
            AgentRole.INTEGRATOR: AgentCapability(
                role=AgentRole.INTEGRATOR,
                cpu_intensive=True,
                specialized_tools=["integration_tools", "dependency_resolver"],
                estimated_task_duration=2.2
            ),
            AgentRole.RESEARCHER: AgentCapability(
                role=AgentRole.RESEARCHER,
                io_intensive=True,
                claude_requests=True,
                specialized_tools=["web_search", "documentation_parser"],
                estimated_task_duration=1.0
            ),
            AgentRole.SYNTHESIZER: AgentCapability(
                role=AgentRole.SYNTHESIZER,
                cpu_intensive=True,
                memory_intensive=True,
                specialized_tools=["result_aggregator", "report_generator"],
                estimated_task_duration=1.5
            ),
            
            # E-core agents (coordination and I/O)
            AgentRole.COORDINATOR: AgentCapability(
                role=AgentRole.COORDINATOR,
                cpu_intensive=False,
                io_intensive=True,
                specialized_tools=["workflow_manager", "task_scheduler"],
                estimated_task_duration=0.5
            ),
            AgentRole.DOCUMENTER: AgentCapability(
                role=AgentRole.DOCUMENTER,
                cpu_intensive=False,
                io_intensive=True,
                claude_requests=True,
                specialized_tools=["markdown_generator", "documentation_tools"],
                estimated_task_duration=0.8
            ),
            AgentRole.MONITOR: AgentCapability(
                role=AgentRole.MONITOR,
                cpu_intensive=False,
                io_intensive=True,
                specialized_tools=["system_monitor", "health_checker"],
                estimated_task_duration=0.3
            ),
            AgentRole.REPORTER: AgentCapability(
                role=AgentRole.REPORTER,
                cpu_intensive=False,
                io_intensive=True,
                claude_requests=True,
                specialized_tools=["report_formatter", "user_interface"],
                estimated_task_duration=0.6
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the enhanced 12-agent system."""
        if self._initialized:
            return
        
        start_time = time.time()
        self.logger.info("Initializing Enhanced 12-Agent Coordinator for M4 Pro")
        
        # Initialize M4 optimizer
        self.m4_optimizer = create_m4_optimizer(
            max_concurrent_requests=6,  # Higher limit for 12 agents
            p_cores_only=True,
            high_priority=True
        )
        await self.m4_optimizer.initialize()
        
        # Assign agents to cores based on capabilities
        self._assign_agents_to_cores()
        
        # Initialize agent pool with 12 agents
        self.agent_pool = WorkStealingAgentPool(
            num_agents=12,
            enable_work_stealing=True
        )
        await self.agent_pool.initialize()
        
        # Initialize orchestrator
        self.orchestrator = AgentOrchestrator(num_agents=12)
        await self.orchestrator.initialize()
        
        # Initialize performance tracking
        for role in AgentRole:
            self.agent_performance[role] = {
                "tasks_completed": 0,
                "avg_duration": 0.0,
                "success_rate": 1.0,
                "specialization_score": 1.0
            }
        
        # Start background optimization
        self._running = True
        asyncio.create_task(self._continuous_optimization())
        
        self._initialized = True
        init_time = time.time() - start_time
        self.logger.info(f"Enhanced 12-Agent Coordinator initialized in {init_time:.3f}s")
        self.logger.info(f"P-core agents: {[role.value for role in self.p_core_agents]}")
        self.logger.info(f"E-core agents: {[role.value for role in self.e_core_agents]}")
    
    def _assign_agents_to_cores(self) -> None:
        """Assign agents to P-cores or E-cores based on capabilities."""
        p_core_roles = []
        e_core_roles = []
        
        for role, capability in self.agent_capabilities.items():
            if capability.cpu_intensive:
                p_core_roles.append(role)
            else:
                e_core_roles.append(role)
        
        # Ensure we have exactly 8 P-core and 4 E-core agents
        self.p_core_agents = p_core_roles[:8]
        self.e_core_agents = e_core_roles[:4]
        
        # Add any remaining to appropriate cores
        remaining = len(p_core_roles) + len(e_core_roles) - 12
        if remaining > 0:
            self.logger.warning(f"Too many agent roles defined ({remaining} extra), truncating")
    
    async def execute_enhanced_task(self, task: Enhanced12AgentTask) -> Dict[str, Any]:
        """Execute a task using the enhanced 12-agent system."""
        if not self._initialized:
            raise RuntimeError("Coordinator not initialized")
        
        start_time = time.time()
        
        # Route task to optimal agent
        selected_role = self.task_router.select_optimal_agent(task)
        
        # Convert to work-stealing task
        ws_task = self._convert_to_ws_task(task, selected_role)
        
        try:
            # Submit task with role preference
            await self.agent_pool.submit_task(ws_task)
            
            # Wait for completion with timeout
            timeout = max(task.estimated_duration + 2.0, 5.0)
            result = await asyncio.wait_for(
                self.agent_pool.wait_for_task_completion(ws_task.id),
                timeout=timeout
            )
            
            # Update performance metrics
            duration = time.time() - start_time
            self._update_agent_performance(selected_role, duration, True)
            
            self.logger.debug(f"Task {task.task_id} completed by {selected_role.value} in {duration:.3f}s")
            
            return {
                "task_id": task.task_id,
                "success": True,
                "result": result,
                "agent_role": selected_role.value,
                "duration": duration
            }
            
        except asyncio.TimeoutError:
            self._update_agent_performance(selected_role, task.estimated_duration, False)
            return {
                "task_id": task.task_id,
                "success": False,
                "error": f"Task timed out after {timeout}s",
                "agent_role": selected_role.value,
                "duration": time.time() - start_time
            }
        except Exception as e:
            self._update_agent_performance(selected_role, time.time() - start_time, False)
            return {
                "task_id": task.task_id,
                "success": False,
                "error": str(e),
                "agent_role": selected_role.value,
                "duration": time.time() - start_time
            }
    
    async def execute_batch_enhanced(self, tasks: List[Enhanced12AgentTask]) -> List[Dict[str, Any]]:
        """Execute a batch of tasks with intelligent distribution."""
        if not self._initialized:
            raise RuntimeError("Coordinator not initialized")
        
        batch_start = time.time()
        self.logger.info(f"Executing batch of {len(tasks)} tasks across 12 agents")
        
        # Analyze batch for optimization opportunities
        batch_analysis = self._analyze_task_batch(tasks)
        
        # Execute tasks based on analysis
        if batch_analysis["parallelizable_ratio"] > 0.7:
            # High parallelization - use concurrent execution
            results = await self._execute_concurrent_batch(tasks)
        else:
            # Sequential dependencies - use orchestrator
            results = await self._execute_orchestrated_batch(tasks)
        
        batch_time = time.time() - batch_start
        throughput = len(tasks) / batch_time if batch_time > 0 else 0
        
        self.logger.info(f"Batch complete: {len(tasks)} tasks in {batch_time:.3f}s ({throughput:.1f} tasks/sec)")
        
        # Update metrics
        self._metrics["total_tasks"] += len(tasks)
        
        return results
    
    def _convert_to_ws_task(self, task: Enhanced12AgentTask, selected_role: AgentRole) -> WorkStealingTask:
        """Convert enhanced task to work-stealing task."""
        return WorkStealingTask(
            id=task.task_id,
            description=task.description,
            priority=task.priority,
            subdividable=task.parallelizable,
            estimated_duration=task.estimated_duration,
            remaining_work=task.estimated_duration,
            metadata={
                **task.data,
                "task_type": task.task_type,
                "selected_role": selected_role.value,
                "cpu_intensive": task.cpu_intensive,
                "requires_claude": task.requires_claude
            }
        )
    
    def _analyze_task_batch(self, tasks: List[Enhanced12AgentTask]) -> Dict[str, Any]:
        """Analyze a batch of tasks for optimization."""
        total_tasks = len(tasks)
        parallelizable = sum(1 for task in tasks if task.parallelizable)
        cpu_intensive = sum(1 for task in tasks if task.cpu_intensive)
        claude_requests = sum(1 for task in tasks if task.requires_claude)
        
        return {
            "total_tasks": total_tasks,
            "parallelizable_ratio": parallelizable / total_tasks,
            "cpu_intensive_ratio": cpu_intensive / total_tasks,
            "claude_request_ratio": claude_requests / total_tasks,
            "estimated_total_duration": sum(task.estimated_duration for task in tasks)
        }
    
    async def _execute_concurrent_batch(self, tasks: List[Enhanced12AgentTask]) -> List[Dict[str, Any]]:
        """Execute highly parallelizable batch concurrently."""
        concurrent_tasks = []
        for task in tasks:
            concurrent_tasks.append(self.execute_enhanced_task(task))
        
        return await asyncio.gather(*concurrent_tasks, return_exceptions=True)
    
    async def _execute_orchestrated_batch(self, tasks: List[Enhanced12AgentTask]) -> List[Dict[str, Any]]:
        """Execute batch with orchestrator for dependency handling."""
        # Convert to orchestrator tasks
        orchestrator_tasks = []
        for task in tasks:
            orchestrator_task = self._convert_to_orchestrator_task(task)
            orchestrator_tasks.append(orchestrator_task)
        
        # Execute via orchestrator
        results = await self.orchestrator.execute_tasks(orchestrator_tasks)
        
        # Convert results back
        enhanced_results = []
        for result in results:
            enhanced_results.append({
                "task_id": result.task_id,
                "success": result.success,
                "result": result.result,
                "error": result.error,
                "duration": result.duration or 0.0,
                "agent_role": result.agent_id or "unknown"
            })
        
        return enhanced_results
    
    def _convert_to_orchestrator_task(self, task: Enhanced12AgentTask):
        """Convert enhanced task to orchestrator task."""
        from ..agents.types import Task, TaskPriority as OrchestratorPriority
        
        # Map priority
        priority_map = {
            TaskPriority.CRITICAL: OrchestratorPriority.HIGH,
            TaskPriority.HIGH: OrchestratorPriority.HIGH,
            TaskPriority.NORMAL: OrchestratorPriority.MEDIUM,
            TaskPriority.LOW: OrchestratorPriority.LOW
        }
        
        return Task(
            id=task.task_id,
            description=task.description,
            priority=priority_map.get(task.priority, OrchestratorPriority.MEDIUM),
            data=task.data,
            dependencies=[],
            estimated_duration=task.estimated_duration
        )
    
    def _update_agent_performance(self, role: AgentRole, duration: float, success: bool) -> None:
        """Update performance metrics for an agent role."""
        perf = self.agent_performance[role]
        
        # Update task count
        perf["tasks_completed"] += 1
        
        # Update average duration
        prev_avg = perf["avg_duration"]
        count = perf["tasks_completed"]
        perf["avg_duration"] = (prev_avg * (count - 1) + duration) / count
        
        # Update success rate
        prev_success_count = perf["success_rate"] * (count - 1)
        new_success_count = prev_success_count + (1 if success else 0)
        perf["success_rate"] = new_success_count / count
    
    async def _continuous_optimization(self) -> None:
        """Continuous optimization loop."""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Optimize every 5 seconds
                
                # Update workload balancing
                await self.workload_balancer.optimize_workload(self.agent_performance)
                
                # Update task routing
                self.task_router.update_performance_data(self.agent_performance)
                
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
                await asyncio.sleep(10.0)
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        metrics = self._metrics.copy()
        
        # Add agent-specific metrics
        metrics["agent_performance"] = self.agent_performance.copy()
        
        # Add M4 optimizer metrics
        if self.m4_optimizer:
            metrics["m4_optimization"] = self.m4_optimizer.get_optimization_stats()
        
        # Add core utilization
        p_core_utilization = sum(
            perf["avg_duration"] for role in self.p_core_agents
            for perf in [self.agent_performance.get(role, {})]
            if perf
        ) / len(self.p_core_agents) if self.p_core_agents else 0
        
        e_core_utilization = sum(
            perf["avg_duration"] for role in self.e_core_agents
            for perf in [self.agent_performance.get(role, {})]
            if perf
        ) / len(self.e_core_agents) if self.e_core_agents else 0
        
        metrics["p_core_utilization"] = p_core_utilization
        metrics["e_core_utilization"] = e_core_utilization
        
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown the enhanced coordinator."""
        self.logger.info("Shutting down Enhanced 12-Agent Coordinator")
        
        self._running = False
        
        # Shutdown components
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        if self.agent_pool:
            await self.agent_pool.shutdown()
        
        if self.m4_optimizer:
            await self.m4_optimizer.shutdown()
        
        self.logger.info("Enhanced 12-Agent Coordinator shutdown complete")


class WorkloadBalancer:
    """Balances workload across agents based on performance characteristics."""
    
    def __init__(self):
        self.logger = get_component_logger("workload_balancer")
    
    async def optimize_workload(self, agent_performance: Dict[AgentRole, Dict[str, float]]) -> None:
        """Optimize workload distribution based on agent performance."""
        # Identify overloaded agents
        overloaded = []
        underutilized = []
        
        for role, perf in agent_performance.items():
            avg_duration = perf.get("avg_duration", 0.0)
            success_rate = perf.get("success_rate", 1.0)
            
            if avg_duration > 3.0 or success_rate < 0.8:  # Overloaded criteria
                overloaded.append(role)
            elif avg_duration < 0.5 and success_rate > 0.95:  # Underutilized criteria
                underutilized.append(role)
        
        if overloaded:
            self.logger.info(f"Overloaded agents detected: {[r.value for r in overloaded]}")
        
        if underutilized:
            self.logger.debug(f"Underutilized agents: {[r.value for r in underutilized]}")


class IntelligentTaskRouter:
    """Routes tasks to optimal agents based on capabilities and performance."""
    
    def __init__(self, agent_capabilities: Dict[AgentRole, AgentCapability]):
        self.agent_capabilities = agent_capabilities
        self.performance_data: Dict[AgentRole, Dict[str, float]] = {}
        self.logger = get_component_logger("intelligent_task_router")
    
    def select_optimal_agent(self, task: Enhanced12AgentTask) -> AgentRole:
        """Select the optimal agent role for a task."""
        # Start with preferred roles if specified
        if task.preferred_roles:
            for role in task.preferred_roles:
                if self._is_agent_suitable(role, task):
                    return role
        
        # Find best match based on capabilities
        best_role = None
        best_score = -1.0
        
        for role, capability in self.agent_capabilities.items():
            score = self._calculate_suitability_score(role, capability, task)
            if score > best_score:
                best_score = score
                best_role = role
        
        return best_role or AgentRole.ANALYZER  # Fallback
    
    def _is_agent_suitable(self, role: AgentRole, task: Enhanced12AgentTask) -> bool:
        """Check if an agent role is suitable for a task."""
        capability = self.agent_capabilities.get(role)
        if not capability:
            return False
        
        # Check basic compatibility
        if task.cpu_intensive and not capability.cpu_intensive:
            return False
        
        if task.requires_claude and not capability.claude_requests:
            return False
        
        return True
    
    def _calculate_suitability_score(self, role: AgentRole, capability: AgentCapability, task: Enhanced12AgentTask) -> float:
        """Calculate suitability score for agent-task pairing."""
        score = 0.0
        
        # Capability matching
        if task.cpu_intensive == capability.cpu_intensive:
            score += 2.0
        if task.io_intensive == capability.io_intensive:
            score += 1.5
        if task.memory_intensive == capability.memory_intensive:
            score += 1.0
        if task.requires_claude == capability.claude_requests:
            score += 1.5
        
        # Performance history
        perf = self.performance_data.get(role, {})
        success_rate = perf.get("success_rate", 1.0)
        avg_duration = perf.get("avg_duration", capability.estimated_task_duration)
        
        # Prefer agents with higher success rates and reasonable durations
        score += success_rate * 2.0
        if avg_duration < task.estimated_duration * 1.2:  # Within 20% of estimate
            score += 1.0
        
        return score
    
    def update_performance_data(self, performance_data: Dict[AgentRole, Dict[str, float]]) -> None:
        """Update performance data for routing decisions."""
        self.performance_data = performance_data.copy()


# Factory function
def create_enhanced_12_agent_coordinator() -> Enhanced12AgentCoordinator:
    """Create an enhanced 12-agent coordinator."""
    return Enhanced12AgentCoordinator()