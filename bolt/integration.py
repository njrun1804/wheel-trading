#!/usr/bin/env python3
"""
Bolt System Integration Layer

This module integrates all 8 agents with hardware acceleration, Einstein,
Metal performance shaders, and comprehensive monitoring for M4 Pro.

Architecture:
- Agent Orchestration: 8 parallel Claude Code agents
- Hardware State: Real-time M4 Pro monitoring (CPU/GPU/Memory)
- GPU Acceleration: MLX/PyTorch routing with Metal backend
- Memory Management: Unified memory with safety limits
- Einstein Integration: Semantic search and intelligent context
- Performance Monitoring: Real-time metrics and tracing
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import psutil
import os
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jarvis2.core.device_router import DeviceRouter, Backend, OperationType
from jarvis2.hardware.metal_monitor import MetalMonitor
from einstein.unified_index import get_einstein_hub
from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
from src.unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph
from src.unity_wheel.accelerated_tools.python_helpers_turbo import get_code_helper
from src.unity_wheel.accelerated_tools.trace_simple import get_trace_turbo


@dataclass
class SystemState:
    """Real-time system state for M4 Pro hardware."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    cpu_per_core: List[float] = field(default_factory=list)
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_limit_gb: float = 18.0  # M4 Pro limit
    active_agents: int = 0
    operations_per_second: float = 0.0
    
    @property
    def is_healthy(self) -> bool:
        """Check if system is in healthy state."""
        return (
            self.cpu_percent < 90.0 and
            self.memory_percent < 85.0 and
            self.gpu_memory_used_gb < (self.gpu_memory_limit_gb * 0.9)
        )
    
    @property
    def can_spawn_agent(self) -> bool:
        """Check if we can spawn another agent safely."""
        return (
            self.active_agents < 8 and  # Max 8 agents
            self.cpu_percent < 80.0 and
            self.memory_percent < 80.0 and
            self.is_healthy
        )


@dataclass 
class AgentTask:
    """Task assigned to an agent."""
    id: str
    type: str
    description: str
    priority: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, running, completed, failed
    assigned_agent: Optional[int] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Task execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class Agent:
    """Individual agent with hardware acceleration capabilities."""
    
    def __init__(self, agent_id: int, device_router: DeviceRouter, tracer):
        self.id = agent_id
        self.device_router = device_router
        self.tracer = tracer
        self.current_task: Optional[AgentTask] = None
        self.completed_tasks: List[AgentTask] = []
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize accelerated tools for this agent."""
        return {
            "ripgrep": get_ripgrep_turbo(),
            "dependency_graph": get_dependency_graph(),
            "python_helper": get_code_helper(),
            "einstein": None,  # Will be set by orchestrator
        }
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a task with hardware acceleration."""
        async with self.tracer.trace_span(f"agent_{self.id}_task_{task.id}") as span:
            span.set_attribute("agent.id", self.id)
            span.set_attribute("task.type", task.type)
            
            self.current_task = task
            task.status = "running"
            task.assigned_agent = self.id
            task.started_at = time.time()
            
            try:
                # Route to optimal backend based on task type
                backend = self._select_backend_for_task(task)
                span.set_attribute("backend", backend.value)
                
                # Execute task based on type
                if task.type == "code_analysis":
                    result = await self._execute_code_analysis(task, backend)
                elif task.type == "pattern_search":
                    result = await self._execute_pattern_search(task, backend)
                elif task.type == "dependency_check":
                    result = await self._execute_dependency_check(task, backend)
                elif task.type == "optimization":
                    result = await self._execute_optimization(task, backend)
                elif task.type == "semantic_search":
                    result = await self._execute_semantic_search(task, backend)
                else:
                    result = await self._execute_generic(task, backend)
                
                task.result = result
                task.status = "completed"
                task.completed_at = time.time()
                
                span.set_attribute("task.duration", task.duration)
                span.set_attribute("task.success", True)
                
                return result
                
            except Exception as e:
                task.status = "failed"
                task.result = {"error": str(e)}
                task.completed_at = time.time()
                
                span.set_attribute("task.success", False)
                span.set_attribute("error", str(e))
                
                raise
            
            finally:
                self.completed_tasks.append(task)
                self.current_task = None
    
    def _select_backend_for_task(self, task: AgentTask) -> Backend:
        """Select optimal backend for task type."""
        if task.type in ["optimization", "neural_analysis"]:
            return self.device_router.route(OperationType.NEURAL_FORWARD)
        elif task.type in ["pattern_search", "semantic_search"]:
            return self.device_router.route(OperationType.EMBEDDING_SEARCH)
        else:
            return self.device_router.route(OperationType.TREE_SEARCH)
    
    async def _execute_code_analysis(self, task: AgentTask, backend: Backend) -> Dict[str, Any]:
        """Execute code analysis task."""
        analyzer = self.tools["python_helper"]
        files = task.context.get("files", [])
        
        results = []
        for file_path in files:
            analysis = await analyzer.analyze_file(file_path)
            results.append(analysis)
        
        return {
            "agent_id": self.id,
            "task_id": task.id,
            "files_analyzed": len(results),
            "analysis": results,
            "backend": backend.value
        }
    
    async def _execute_pattern_search(self, task: AgentTask, backend: Backend) -> Dict[str, Any]:
        """Execute pattern search using ripgrep."""
        ripgrep = self.tools["ripgrep"]
        patterns = task.context.get("patterns", [])
        path = task.context.get("path", ".")
        
        all_results = []
        for pattern in patterns:
            results = await ripgrep.search(pattern, path)
            all_results.extend(results)
        
        return {
            "agent_id": self.id,
            "task_id": task.id,
            "matches_found": len(all_results),
            "results": all_results,
            "backend": backend.value
        }
    
    async def _execute_dependency_check(self, task: AgentTask, backend: Backend) -> Dict[str, Any]:
        """Execute dependency analysis."""
        dep_graph = self.tools["dependency_graph"]
        
        # Build or update graph
        await dep_graph.build_graph()
        
        # Check for cycles
        cycles = await dep_graph.detect_cycles()
        
        # Find specific symbols if requested
        symbols = task.context.get("symbols", [])
        symbol_results = {}
        for symbol in symbols:
            locations = await dep_graph.find_symbol(symbol)
            symbol_results[symbol] = locations
        
        return {
            "agent_id": self.id,
            "task_id": task.id,
            "cycles_found": len(cycles),
            "cycles": cycles,
            "symbols": symbol_results,
            "backend": backend.value
        }
    
    async def _execute_optimization(self, task: AgentTask, backend: Backend) -> Dict[str, Any]:
        """Execute optimization task using GPU acceleration."""
        # This would use MLX/PyTorch for actual optimization
        # For now, simulate optimization analysis
        target = task.context.get("target", "unknown")
        
        return {
            "agent_id": self.id,
            "task_id": task.id,
            "optimization_target": target,
            "backend": backend.value,
            "gpu_accelerated": backend in [Backend.MLX_GPU, Backend.TORCH_MPS],
            "recommendations": [
                "Use batch processing for better GPU utilization",
                "Consider caching frequently accessed data",
                "Parallelize independent operations"
            ]
        }
    
    async def _execute_semantic_search(self, task: AgentTask, backend: Backend) -> Dict[str, Any]:
        """Execute semantic search using Einstein."""
        query = task.context.get("query", "")
        einstein = self.tools.get("einstein")
        
        if einstein:
            context = await einstein.get_intelligent_context(query)
            return {
                "agent_id": self.id,
                "task_id": task.id,
                "query": query,
                "results": context.get("search_results", []),
                "backend": backend.value
            }
        else:
            # Fallback to pattern search
            return await self._execute_pattern_search(task, backend)
    
    async def _execute_generic(self, task: AgentTask, backend: Backend) -> Dict[str, Any]:
        """Execute generic task."""
        await asyncio.sleep(0.1)  # Simulate work
        
        return {
            "agent_id": self.id,
            "task_id": task.id,
            "description": task.description,
            "backend": backend.value,
            "completed": True
        }


class BoltIntegration:
    """Main integration layer for the Bolt system."""
    
    def __init__(self):
        self.device_router = DeviceRouter()
        self.metal_monitor = MetalMonitor()
        self.tracer = get_trace_turbo()
        self.einstein_hub = None
        self.agents: List[Agent] = []
        self.system_state = SystemState()
        self.task_queue: asyncio.Queue[AgentTask] = asyncio.Queue()
        self.completed_tasks: List[AgentTask] = []
        self._running = False
        self._monitor_task = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize all system components."""
        if self._initialized:
            return
            
        async with self.tracer.trace_span("bolt_initialization") as span:
            # Initialize Einstein
            try:
                self.einstein_hub = get_einstein_hub()
                await self.einstein_hub.initialize()
                span.set_attribute("einstein.initialized", True)
            except Exception as e:
                span.set_attribute("einstein.error", str(e))
                print(f"Warning: Einstein initialization failed: {e}")
            
            # Initialize agents
            for i in range(8):
                agent = Agent(i, self.device_router, self.tracer)
                if self.einstein_hub:
                    agent.tools["einstein"] = self.einstein_hub
                self.agents.append(agent)
            
            span.set_attribute("agents.count", len(self.agents))
            
            # Start monitoring
            self._monitor_task = asyncio.create_task(self._monitor_system())
            
            self._initialized = True
            self._running = True
    
    async def shutdown(self):
        """Shutdown system cleanly."""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Wait for agents to complete current tasks
        await self._wait_for_agents()
        
        # Shutdown Einstein
        if self.einstein_hub:
            # Einstein doesn't have explicit shutdown, but we can clear references
            self.einstein_hub = None
    
    async def solve(self, query: str, analyze_only: bool = False) -> Dict[str, Any]:
        """Main entry point for solving problems."""
        async with self.tracer.trace_span("bolt_solve") as span:
            span.set_attribute("query", query)
            span.set_attribute("analyze_only", analyze_only)
            
            # Get initial context from Einstein
            context = {}
            if self.einstein_hub:
                try:
                    context = await self.einstein_hub.get_intelligent_context(query)
                    span.set_attribute("einstein.context_files", len(context.get("search_results", [])))
                except Exception as e:
                    span.set_attribute("einstein.error", str(e))
            
            # Decompose problem into tasks
            tasks = await self._decompose_problem(query, context)
            span.set_attribute("tasks.count", len(tasks))
            
            if analyze_only:
                return {
                    "query": query,
                    "tasks": [t.__dict__ for t in tasks],
                    "context_files": len(context.get("search_results", [])),
                    "system_state": self.system_state.__dict__,
                    "action": "analysis_only"
                }
            
            # Execute tasks with agents
            results = await self._execute_tasks(tasks)
            
            return {
                "query": query,
                "results": results,
                "tasks_completed": len([r for r in results if r.get("status") == "completed"]),
                "total_duration": sum(t.duration or 0 for t in self.completed_tasks),
                "system_state": self.system_state.__dict__,
                "action": "completed"
            }
    
    async def _decompose_problem(self, query: str, context: Dict[str, Any]) -> List[AgentTask]:
        """Decompose problem into executable tasks."""
        tasks = []
        query_lower = query.lower()
        
        # Always start with context gathering
        tasks.append(AgentTask(
            id="context_0",
            type="semantic_search",
            description="Gather relevant context",
            priority=10,
            context={"query": query}
        ))
        
        # Add specific tasks based on query
        if "optimize" in query_lower:
            tasks.extend([
                AgentTask(
                    id="opt_1",
                    type="pattern_search",
                    description="Find optimization targets",
                    priority=9,
                    context={"patterns": ["TODO.*perf", "FIXME.*slow", "bottleneck"], "path": "."}
                ),
                AgentTask(
                    id="opt_2",
                    type="code_analysis",
                    description="Analyze hot paths",
                    priority=8,
                    context={"files": context.get("search_results", [])}
                ),
                AgentTask(
                    id="opt_3",
                    type="optimization",
                    description="Apply optimizations",
                    priority=7,
                    context={"target": query},
                    dependencies=["opt_1", "opt_2"]
                )
            ])
        
        elif "debug" in query_lower or "fix" in query_lower:
            tasks.extend([
                AgentTask(
                    id="debug_1",
                    type="pattern_search",
                    description="Find error patterns",
                    priority=9,
                    context={"patterns": ["error", "exception", "traceback", "fail"], "path": "."}
                ),
                AgentTask(
                    id="debug_2",
                    type="dependency_check",
                    description="Check dependencies",
                    priority=8,
                    context={"symbols": []}
                ),
                AgentTask(
                    id="debug_3",
                    type="code_analysis",
                    description="Analyze problem areas",
                    priority=7,
                    context={"files": []},
                    dependencies=["debug_1"]
                )
            ])
        
        elif "refactor" in query_lower:
            tasks.extend([
                AgentTask(
                    id="refactor_1",
                    type="dependency_check",
                    description="Map dependencies",
                    priority=9,
                    context={}
                ),
                AgentTask(
                    id="refactor_2",
                    type="pattern_search",
                    description="Find refactoring targets",
                    priority=8,
                    context={"patterns": ["class", "def", "import"], "path": "."}
                ),
                AgentTask(
                    id="refactor_3",
                    type="code_analysis",
                    description="Analyze code structure",
                    priority=7,
                    context={"files": []},
                    dependencies=["refactor_2"]
                )
            ])
        
        else:
            # Generic tasks
            tasks.extend([
                AgentTask(
                    id="generic_1",
                    type="pattern_search",
                    description="Search for relevant code",
                    priority=8,
                    context={"patterns": query.split()[:3], "path": "."}
                ),
                AgentTask(
                    id="generic_2",
                    type="code_analysis",
                    description="Analyze relevant files",
                    priority=7,
                    context={"files": []}
                )
            ])
        
        return tasks
    
    async def _execute_tasks(self, tasks: List[AgentTask]) -> List[Dict[str, Any]]:
        """Execute tasks using available agents."""
        # Add tasks to queue
        for task in sorted(tasks, key=lambda t: t.priority, reverse=True):
            await self.task_queue.put(task)
        
        # Start agent workers
        agent_tasks = []
        for agent in self.agents:
            if self.system_state.can_spawn_agent:
                agent_task = asyncio.create_task(self._agent_worker(agent))
                agent_tasks.append(agent_task)
                self.system_state.active_agents += 1
        
        # Wait for all tasks to complete
        await self.task_queue.join()
        
        # Cancel workers
        for task in agent_tasks:
            task.cancel()
        
        await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Collect results
        results = []
        for task in tasks:
            if task.result:
                results.append({
                    "task_id": task.id,
                    "type": task.type,
                    "status": task.status,
                    "agent_id": task.assigned_agent,
                    "duration": task.duration,
                    "result": task.result
                })
        
        return results
    
    async def _agent_worker(self, agent: Agent):
        """Worker coroutine for an agent."""
        try:
            while True:
                # Check system health before taking task
                if not self.system_state.is_healthy:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Get task with timeout to allow checking system state
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                
                try:
                    # Check dependencies
                    if await self._check_dependencies(task):
                        await agent.execute_task(task)
                        self.completed_tasks.append(task)
                    else:
                        # Put back in queue if dependencies not met
                        await self.task_queue.put(task)
                        await asyncio.sleep(0.1)
                
                finally:
                    self.task_queue.task_done()
                    
        except asyncio.CancelledError:
            pass
        finally:
            self.system_state.active_agents -= 1
    
    async def _check_dependencies(self, task: AgentTask) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        completed_ids = {t.id for t in self.completed_tasks if t.status == "completed"}
        return all(dep_id in completed_ids for dep_id in task.dependencies)
    
    async def _monitor_system(self):
        """Monitor system health continuously."""
        while self._running:
            try:
                # Update CPU metrics
                self.system_state.cpu_percent = psutil.cpu_percent(interval=0.1)
                self.system_state.cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
                
                # Update memory metrics
                mem = psutil.virtual_memory()
                self.system_state.memory_percent = mem.percent
                self.system_state.memory_available_gb = mem.available / (1024**3)
                
                # Update GPU metrics if available
                if self.metal_monitor:
                    gpu_stats = self.metal_monitor.get_current_stats()
                    if gpu_stats:
                        self.system_state.gpu_utilization = gpu_stats.get("utilization", 0)
                        self.system_state.gpu_memory_used_gb = gpu_stats.get("memory_used_gb", 0)
                
                # Calculate operations per second
                if self.completed_tasks:
                    recent_tasks = [t for t in self.completed_tasks[-100:] if t.completed_at]
                    if recent_tasks:
                        time_window = time.time() - min(t.started_at for t in recent_tasks if t.started_at)
                        self.system_state.operations_per_second = len(recent_tasks) / max(time_window, 1)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _wait_for_agents(self):
        """Wait for all agents to complete current tasks."""
        max_wait = 30  # Maximum 30 seconds
        start = time.time()
        
        while any(agent.current_task for agent in self.agents):
            if time.time() - start > max_wait:
                print("Warning: Timeout waiting for agents to complete")
                break
            await asyncio.sleep(0.1)


# CLI Integration
async def bolt_solve_cli(query: str, analyze_only: bool = False) -> int:
    """CLI interface for bolt solve command."""
    integration = BoltIntegration()
    
    try:
        print(f"🚀 Initializing Bolt system for M4 Pro...")
        await integration.initialize()
        
        print(f"🔍 Processing: {query}")
        result = await integration.solve(query, analyze_only)
        
        if result["action"] == "analysis_only":
            print(f"\n📊 Analysis Results:")
            print(f"  - Context files: {result.get('context_files', 0)}")
            print(f"  - Tasks identified: {len(result.get('tasks', []))}")
            print(f"  - System state: {result['system_state']['active_agents']} active agents")
            print(f"  - CPU: {result['system_state']['cpu_percent']:.1f}%")
            print(f"  - Memory: {result['system_state']['memory_percent']:.1f}%")
            
            print(f"\n📋 Tasks:")
            for task in result.get("tasks", []):
                print(f"  [{task['id']}] {task['type']}: {task['description']}")
                if task.get('dependencies'):
                    print(f"    Dependencies: {', '.join(task['dependencies'])}")
        
        else:
            print(f"\n✅ Solution Complete:")
            print(f"  - Tasks completed: {result['tasks_completed']}")
            print(f"  - Total duration: {result['total_duration']:.2f}s")
            print(f"  - Final CPU: {result['system_state']['cpu_percent']:.1f}%")
            print(f"  - Final Memory: {result['system_state']['memory_percent']:.1f}%")
            
            print(f"\n📝 Results:")
            for res in result.get("results", []):
                print(f"  [{res['task_id']}] Agent {res['agent_id']}: {res['status']}")
                if res['duration']:
                    print(f"    Duration: {res['duration']:.2f}s")
                if res.get('result', {}).get('backend'):
                    print(f"    Backend: {res['result']['backend']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        print(f"\n🛑 Shutting down...")
        await integration.shutdown()


if __name__ == "__main__":
    # Test CLI interface
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "optimize all functions"
    sys.exit(asyncio.run(bolt_solve_cli(query)))