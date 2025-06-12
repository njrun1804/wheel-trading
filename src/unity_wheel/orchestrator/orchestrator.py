"""MCP Orchestrator - Coordinates multi-phase code transformations."""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .pressure import MemoryPressureMonitor
from .slice_cache import SliceCache


class Phase(Enum):
    """Orchestrator execution phases."""
    MAP = "map"
    LOGIC = "logic"
    MONTE_CARLO = "monte_carlo"
    PLAN = "plan"
    OPTIMIZE = "optimize"
    EXECUTE = "execute"
    REVIEW = "review"


@dataclass
class ExecutionPlan:
    """Validated execution plan from JSON DAG."""
    command: str
    phases: list[dict[str, Any]]
    token_budget: int = 4096
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_json(cls, plan_path: str) -> 'ExecutionPlan':
        """Load and validate plan from JSON file."""
        with open(plan_path) as f:
            data = json.load(f)
        
        # Validate required fields
        if "command" not in data:
            raise ValueError("Plan missing 'command' field")
        if "phases" not in data or not isinstance(data["phases"], list):
            raise ValueError("Plan missing or invalid 'phases' field")
            
        return cls(
            command=data["command"],
            phases=data["phases"],
            token_budget=data.get("token_budget", 4096),
            max_retries=data.get("max_retries", 3)
        )


@dataclass 
class PhaseResult:
    """Result from executing a phase."""
    phase: Phase
    success: bool
    duration_ms: int
    token_count: int
    data: dict[str, Any]
    error: str | None = None
    retries: int = 0


class MCPOrchestrator:
    """Orchestrates MCP servers for coordinated code transformations."""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.slice_cache = SliceCache(workspace_root)
        self.memory_monitor = MemoryPressureMonitor()
        self.mcp_pool: dict[str, Any] = {}
        self.execution_history: list[PhaseResult] = []
        self.current_plan: ExecutionPlan | None = None
        
    async def initialize(self):
        """Initialize orchestrator and MCP connections."""
        await self.slice_cache.initialize()
        self.memory_monitor.start()
        
        # Initialize essential MCP connections
        essential_mcps = [
            "filesystem", "ripgrep", "dependency_graph",
            "memory", "sequential-thinking", "python_analysis"
        ]
        
        for mcp_name in essential_mcps:
            # In real implementation, establish MCP connections
            self.mcp_pool[mcp_name] = f"mock_{mcp_name}_connection"
            
    async def execute_command(self, command: str) -> dict[str, Any]:
        """Execute natural language command through orchestrated phases."""
        start_time = time.time()
        
        # Generate execution plan
        plan = await self._generate_plan(command)
        self.current_plan = plan
        
        # Execute phases with retry logic
        results = []
        for phase_config in plan.phases:
            phase = Phase(phase_config["phase"])
            result = await self._execute_phase_with_retry(phase, phase_config)
            results.append(result)
            
            if not result.success:
                break
                
        # Generate summary
        total_duration = int((time.time() - start_time) * 1000)
        total_tokens = sum(r.token_count for r in results)
        
        return {
            "command": command,
            "success": all(r.success for r in results),
            "duration_ms": total_duration,
            "total_tokens": total_tokens,
            "phases": [self._phase_to_dict(r) for r in results],
            "memory_peak_mb": self.memory_monitor.peak_memory_mb
        }
        
    async def _generate_plan(self, command: str) -> ExecutionPlan:
        """Generate execution plan from natural language command."""
        # In real implementation, use LLM to generate plan
        # For now, return a mock plan
        return ExecutionPlan(
            command=command,
            phases=[
                {"phase": "map", "targets": ["ripgrep", "dependency_graph"]},
                {"phase": "logic", "pruning": True},
                {"phase": "monte_carlo", "duration_limit_ms": 15000},
                {"phase": "plan", "output": "plan.json"},
                {"phase": "optimize", "engines": ["duckdb", "pyrepl"]},
                {"phase": "execute", "validate": True},
                {"phase": "review", "trace": True}
            ]
        )
        
    async def _execute_phase_with_retry(self, phase: Phase, config: dict[str, Any]) -> PhaseResult:
        """Execute phase with retry logic."""
        retries = 0
        last_error = None
        
        while retries <= self.current_plan.max_retries:
            try:
                # Check memory pressure
                if self.memory_monitor.is_pressure_high():
                    await asyncio.sleep(1)  # Back off
                    
                result = await self._execute_phase(phase, config)
                result.retries = retries
                return result
                
            except Exception as e:
                last_error = str(e)
                retries += 1
                if retries <= self.current_plan.max_retries:
                    await asyncio.sleep(2 ** retries)  # Exponential backoff
                    
        # All retries exhausted
        return PhaseResult(
            phase=phase,
            success=False,
            duration_ms=0,
            token_count=0,
            data={},
            error=last_error,
            retries=retries
        )
        
    async def _execute_phase(self, phase: Phase, config: dict[str, Any]) -> PhaseResult:
        """Execute a single phase."""
        start_time = time.time()
        token_count = 0
        
        try:
            if phase == Phase.MAP:
                data = await self._phase_map(config)
            elif phase == Phase.LOGIC:
                data = await self._phase_logic(config)
            elif phase == Phase.MONTE_CARLO:
                data = await self._phase_monte_carlo(config)
            elif phase == Phase.PLAN:
                data = await self._phase_plan(config)
            elif phase == Phase.OPTIMIZE:
                data = await self._phase_optimize(config)
            elif phase == Phase.EXECUTE:
                data = await self._phase_execute(config)
            elif phase == Phase.REVIEW:
                data = await self._phase_review(config)
            else:
                raise ValueError(f"Unknown phase: {phase}")
                
            # Estimate token usage (mock)
            token_count = len(json.dumps(data)) // 4
            
            duration_ms = int((time.time() - start_time) * 1000)
            return PhaseResult(
                phase=phase,
                success=True,
                duration_ms=duration_ms,
                token_count=token_count,
                data=data
            )
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return PhaseResult(
                phase=phase,
                success=False,
                duration_ms=duration_ms,
                token_count=token_count,
                data={},
                error=str(e)
            )
            
    async def _phase_map(self, config: dict[str, Any]) -> dict[str, Any]:
        """Map phase - discover relevant code slices."""
        targets = config.get("targets", ["ripgrep", "dependency_graph"])
        
        # Parallel search across MCP servers
        tasks = []
        if "ripgrep" in targets:
            tasks.append(self._search_ripgrep())
        if "dependency_graph" in targets:
            tasks.append(self._search_dependency_graph())
            
        results = await asyncio.gather(*tasks)
        
        # Cache results
        candidate_slices = []
        for result in results:
            for file_path, content in result.items():
                slice_hash = hashlib.sha1(content.encode()).hexdigest()
                await self.slice_cache.store(slice_hash, {
                    "file": file_path,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                })
                candidate_slices.append(slice_hash)
                
        return {
            "candidate_slices": candidate_slices,
            "search_results": len(candidate_slices)
        }
        
    async def _phase_logic(self, config: dict[str, Any]) -> dict[str, Any]:
        """Logic phase - prune and analyze code graph."""
        # Mock implementation
        return {
            "pruned_slices": ["hash1", "hash2", "hash3"],
            "call_graph_depth": 3
        }
        
    async def _phase_monte_carlo(self, config: dict[str, Any]) -> dict[str, Any]:
        """Monte Carlo phase - run simulations."""
        # Duration limit available: config.get("duration_limit_ms", 15000)
        
        # Mock Monte Carlo simulation
        await asyncio.sleep(0.1)  # Simulate work
        
        return {
            "simulations_run": 1000,
            "confidence_interval": 0.95,
            "risk_metrics": {"var_95": 0.05, "max_drawdown": 0.10}
        }
        
    async def _phase_plan(self, config: dict[str, Any]) -> dict[str, Any]:
        """Plan phase - generate execution plan."""
        output_file = config.get("output", "plan.json")
        
        plan_data = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "dag": {
                "nodes": ["scan_1", "scan_2", "mutate_1"],
                "edges": [["scan_1", "mutate_1"], ["scan_2", "mutate_1"]]
            }
        }
        
        # Write plan
        plan_path = self.workspace_root / output_file
        with open(plan_path, 'w') as f:
            json.dump(plan_data, f, indent=2)
            
        return {"plan_path": str(plan_path), "nodes": 3}
        
    async def _phase_optimize(self, config: dict[str, Any]) -> dict[str, Any]:
        """Optimize phase - parameter optimization."""
        # Engines available: config.get("engines", ["duckdb"])
        
        # Mock optimization
        return {
            "optimal_params": {"batch_size": 100, "parallelism": 8},
            "improvement": 0.23
        }
        
    async def _phase_execute(self, config: dict[str, Any]) -> dict[str, Any]:
        """Execute phase - apply transformations."""
        validate = config.get("validate", True)
        
        # Mock execution
        return {
            "files_modified": 5,
            "tests_passed": validate,
            "lint_clean": True
        }
        
    async def _phase_review(self, config: dict[str, Any]) -> dict[str, Any]:
        """Review phase - trace and validate results."""
        use_trace = config.get("trace", True)
        
        # Mock review
        return {
            "trace_url": "http://localhost:6006/trace/123" if use_trace else None,
            "post_deploy_check": "passed",
            "ready_for_pr": True
        }
        
    async def _search_ripgrep(self) -> dict[str, str]:
        """Mock ripgrep search."""
        return {
            "src/example.py": "def example_function():\n    pass"
        }
        
    async def _search_dependency_graph(self) -> dict[str, str]:
        """Mock dependency graph search."""
        return {
            "src/utils.py": "class UtilityClass:\n    pass"
        }
        
    def _phase_to_dict(self, result: PhaseResult) -> dict[str, Any]:
        """Convert phase result to dictionary."""
        return {
            "phase": result.phase.value,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "token_count": result.token_count,
            "data": result.data,
            "error": result.error,
            "retries": result.retries
        }
        
    async def shutdown(self):
        """Clean shutdown."""
        self.memory_monitor.stop()
        await self.slice_cache.close()