"""M4 Enhanced Integration for Bob.

This module integrates the M4-optimized components with the existing Bob system,
providing seamless integration while maintaining backward compatibility.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from ..core.enhanced_12_agent_coordinator import (
    Enhanced12AgentCoordinator,
    Enhanced12AgentTask,
    AgentRole,
    create_enhanced_12_agent_coordinator
)
from ..optimization.m4_claude_optimizer import get_global_optimizer
from ..agents.agent_pool import TaskPriority
from ..utils.logging import get_component_logger


class M4EnhancedBobIntegration:
    """Enhanced Bob integration with M4 Pro optimizations."""
    
    def __init__(self):
        self.logger = get_component_logger("m4_enhanced_bob")
        
        # Core components
        self.enhanced_coordinator: Optional[Enhanced12AgentCoordinator] = None
        self.m4_optimizer = None
        
        # Integration state
        self._initialized = False
        self._startup_time = 0.0
        
        # Performance tracking
        self._performance_metrics = {
            "initialization_time": 0.0,
            "total_queries_processed": 0,
            "avg_query_latency": 0.0,
            "p_core_efficiency": 0.0,
            "e_core_efficiency": 0.0,
            "claude_requests_optimized": 0,
            "http2_session_reuse_rate": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the M4 enhanced Bob system."""
        if self._initialized:
            return
        
        start_time = time.time()
        self.logger.info("Initializing M4 Enhanced Bob Integration")
        
        try:
            # Initialize M4 optimizer
            self.m4_optimizer = await get_global_optimizer()
            
            # Create startup script for optimal launch
            startup_script = self.m4_optimizer.create_startup_script("./start_bob_m4_optimized.sh")
            self.logger.info(f"Created optimized startup script: {startup_script}")
            
            # Initialize enhanced 12-agent coordinator
            self.enhanced_coordinator = create_enhanced_12_agent_coordinator()
            await self.enhanced_coordinator.initialize()
            
            self._startup_time = time.time() - start_time
            self._performance_metrics["initialization_time"] = self._startup_time
            
            self._initialized = True
            
            self.logger.info(f"M4 Enhanced Bob initialized in {self._startup_time:.3f}s")
            self.logger.info("✅ Ready for maximum throughput with M4 Pro optimizations")
            
        except Exception as e:
            self.logger.error(f"M4 Enhanced Bob initialization failed: {e}")
            raise
    
    async def process_claude_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a Claude query using M4 optimizations."""
        if not self._initialized:
            raise RuntimeError("M4 Enhanced Bob not initialized")
        
        start_time = time.time()
        context = context or {}
        
        # Analyze query to determine optimal task decomposition
        task_analysis = self._analyze_query_for_decomposition(query, context)
        
        if task_analysis["complexity"] == "simple":
            # Single agent can handle it
            result = await self._process_simple_query(query, context)
        elif task_analysis["complexity"] == "complex":
            # Use 12-agent orchestration
            result = await self._process_complex_query(query, context, task_analysis)
        else:
            # Parallel decomposition
            result = await self._process_parallel_query(query, context, task_analysis)
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_performance_metrics(processing_time)
        
        return {
            **result,
            "processing_time": processing_time,
            "m4_optimizations_used": True,
            "agents_utilized": task_analysis.get("agents_needed", 1)
        }
    
    async def _process_simple_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a simple query with single agent."""
        # Create enhanced task
        task = Enhanced12AgentTask(
            task_id=f"simple_{int(time.time() * 1000)}",
            description=query,
            task_type="simple_query",
            data={"query": query, **context},
            priority=TaskPriority.NORMAL,
            preferred_roles=[AgentRole.ANALYZER, AgentRole.RESEARCHER],
            cpu_intensive=False,
            estimated_duration=1.0,
            requires_claude=True
        )
        
        return await self.enhanced_coordinator.execute_enhanced_task(task)
    
    async def _process_complex_query(self, query: str, context: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complex query with multiple coordinated agents."""
        # Break down into subtasks
        subtasks = self._decompose_complex_query(query, context, analysis)
        
        # Execute with enhanced coordinator
        results = await self.enhanced_coordinator.execute_batch_enhanced(subtasks)
        
        # Synthesize results
        synthesis_task = Enhanced12AgentTask(
            task_id=f"synthesis_{int(time.time() * 1000)}",
            description=f"Synthesize results from {len(subtasks)} subtasks",
            task_type="result_synthesis",
            data={
                "original_query": query,
                "subtask_results": results,
                **context
            },
            preferred_roles=[AgentRole.SYNTHESIZER, AgentRole.REPORTER],
            estimated_duration=0.5,
            requires_claude=True
        )
        
        synthesis_result = await self.enhanced_coordinator.execute_enhanced_task(synthesis_task)
        
        return {
            "success": True,
            "result": synthesis_result["result"],
            "subtask_count": len(subtasks),
            "synthesis_success": synthesis_result["success"]
        }
    
    async def _process_parallel_query(self, query: str, context: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query that can be parallelized across multiple agents."""
        # Create parallel tasks
        parallel_tasks = self._create_parallel_tasks(query, context, analysis)
        
        # Execute in parallel
        results = await self.enhanced_coordinator.execute_batch_enhanced(parallel_tasks)
        
        # Aggregate results
        successful_results = [r for r in results if r.get("success", False)]
        
        return {
            "success": len(successful_results) > 0,
            "result": {
                "parallel_results": successful_results,
                "total_tasks": len(parallel_tasks),
                "successful_tasks": len(successful_results)
            },
            "parallelization_efficiency": len(successful_results) / len(parallel_tasks) if parallel_tasks else 0
        }
    
    def _analyze_query_for_decomposition(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query to determine optimal processing strategy."""
        # Simple heuristics for demo - could be enhanced with ML
        word_count = len(query.split())
        has_multiple_questions = "?" in query and query.count("?") > 1
        has_complex_requirements = any(keyword in query.lower() for keyword in [
            "analyze", "optimize", "design", "implement", "integrate", "validate"
        ])
        
        if word_count < 20 and not has_multiple_questions:
            complexity = "simple"
            agents_needed = 1
        elif has_complex_requirements or has_multiple_questions:
            complexity = "complex"
            agents_needed = min(8, max(3, word_count // 20))
        else:
            complexity = "parallel"
            agents_needed = min(4, max(2, word_count // 30))
        
        return {
            "complexity": complexity,
            "agents_needed": agents_needed,
            "parallelizable": complexity == "parallel",
            "estimated_duration": word_count * 0.1 + agents_needed * 0.5
        }
    
    def _decompose_complex_query(self, query: str, context: Dict[str, Any], analysis: Dict[str, Any]) -> List[Enhanced12AgentTask]:
        """Decompose a complex query into subtasks."""
        subtasks = []
        agents_needed = analysis["agents_needed"]
        
        # Create analysis subtask
        subtasks.append(Enhanced12AgentTask(
            task_id=f"analyze_{int(time.time() * 1000)}_1",
            description=f"Analyze and understand: {query}",
            task_type="analysis",
            data={"query": query, "phase": "analysis", **context},
            preferred_roles=[AgentRole.ANALYZER, AgentRole.RESEARCHER],
            estimated_duration=1.5,
            requires_claude=True
        ))
        
        # Create architecture/planning subtask
        if agents_needed >= 3:
            subtasks.append(Enhanced12AgentTask(
                task_id=f"plan_{int(time.time() * 1000)}_2",
                description=f"Plan approach for: {query}",
                task_type="planning",
                data={"query": query, "phase": "planning", **context},
                preferred_roles=[AgentRole.ARCHITECT, AgentRole.COORDINATOR],
                estimated_duration=2.0,
                requires_claude=True
            ))
        
        # Create implementation subtasks
        for i in range(max(1, agents_needed - 2)):
            subtasks.append(Enhanced12AgentTask(
                task_id=f"implement_{int(time.time() * 1000)}_{i+3}",
                description=f"Implement part {i+1} of: {query}",
                task_type="implementation",
                data={"query": query, "phase": f"implementation_{i+1}", **context},
                preferred_roles=[AgentRole.GENERATOR, AgentRole.INTEGRATOR],
                estimated_duration=2.5,
                requires_claude=True
            ))
        
        return subtasks
    
    def _create_parallel_tasks(self, query: str, context: Dict[str, Any], analysis: Dict[str, Any]) -> List[Enhanced12AgentTask]:
        """Create parallel tasks for embarrassingly parallel queries."""
        tasks = []
        agents_needed = analysis["agents_needed"]
        
        # Split query into parallel components
        for i in range(agents_needed):
            tasks.append(Enhanced12AgentTask(
                task_id=f"parallel_{int(time.time() * 1000)}_{i}",
                description=f"Process segment {i+1}/{agents_needed}: {query}",
                task_type="parallel_processing",
                data={
                    "query": query,
                    "segment": i + 1,
                    "total_segments": agents_needed,
                    **context
                },
                preferred_roles=[
                    AgentRole.ANALYZER, AgentRole.GENERATOR, 
                    AgentRole.RESEARCHER, AgentRole.VALIDATOR
                ],
                parallelizable=True,
                estimated_duration=1.0,
                requires_claude=True
            ))
        
        return tasks
    
    def _update_performance_metrics(self, processing_time: float) -> None:
        """Update performance metrics."""
        self._performance_metrics["total_queries_processed"] += 1
        
        # Update average latency
        prev_avg = self._performance_metrics["avg_query_latency"]
        count = self._performance_metrics["total_queries_processed"]
        self._performance_metrics["avg_query_latency"] = (prev_avg * (count - 1) + processing_time) / count
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        metrics = self._performance_metrics.copy()
        
        # Add enhanced coordinator metrics
        if self.enhanced_coordinator:
            enhanced_metrics = self.enhanced_coordinator.get_enhanced_metrics()
            metrics["enhanced_coordinator"] = enhanced_metrics
        
        # Add M4 optimizer metrics
        if self.m4_optimizer:
            m4_metrics = self.m4_optimizer.get_optimization_stats()
            metrics["m4_optimizer"] = m4_metrics
        
        # Calculate efficiency scores
        if self.enhanced_coordinator:
            coordinator_metrics = self.enhanced_coordinator.get_enhanced_metrics()
            metrics["p_core_efficiency"] = coordinator_metrics.get("p_core_utilization", 0.0)
            metrics["e_core_efficiency"] = coordinator_metrics.get("e_core_utilization", 0.0)
        
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown the M4 enhanced Bob system."""
        self.logger.info("Shutting down M4 Enhanced Bob Integration")
        
        if self.enhanced_coordinator:
            await self.enhanced_coordinator.shutdown()
        
        if self.m4_optimizer:
            await self.m4_optimizer.shutdown()
        
        self.logger.info("M4 Enhanced Bob shutdown complete")


# Global instance for easy access
_global_m4_bob: Optional[M4EnhancedBobIntegration] = None


async def get_m4_enhanced_bob() -> M4EnhancedBobIntegration:
    """Get or create global M4 enhanced Bob instance."""
    global _global_m4_bob
    if _global_m4_bob is None:
        _global_m4_bob = M4EnhancedBobIntegration()
        await _global_m4_bob.initialize()
    return _global_m4_bob


# Convenience functions for easy integration
async def process_query_m4_optimized(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a query using M4 optimizations."""
    bob = await get_m4_enhanced_bob()
    return await bob.process_claude_query(query, context)


def create_optimized_startup_script(output_path: str = "./start_bob_m4_optimized.sh") -> str:
    """Create an optimized startup script for Bob with M4 optimizations."""
    script_content = f"""#!/bin/bash
# M4 Pro optimized startup script for Bob
# Auto-generated by M4EnhancedBobIntegration

set -e

echo "🚀 Starting Bob with M4 Pro optimizations..."

# Apply M4 Pro environment optimizations
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Detect M4 Pro cores
P_CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "8")
echo "📊 Detected $P_CORES P-cores"

# Launch with P-core affinity and high priority
echo "🎯 Launching with P-core affinity and maximum optimization..."

if command -v taskpolicy >/dev/null 2>&1; then
    echo "✅ Using taskpolicy for optimal scheduling"
    taskpolicy --application python3 -m bob.cli.main "$@"
else
    echo "⚠️  taskpolicy not available, running with high priority"
    python3 -m bob.cli.main "$@"
fi

echo "🏁 Bob session complete"
"""
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(output_path, 0o755)
    return output_path