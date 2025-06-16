"""Jarvis - The streamlined meta-coder for Claude Code."""

# Import our accelerated tools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[2]))

from unity_wheel.accelerated_tools import (
    get_code_helper,
    get_dependency_graph,
    get_duckdb_turbo,
    get_python_analyzer,
    get_ripgrep_turbo,
    get_trace_turbo,
)

from ..analysis.mcts_lite import MCTSLite
from ..strategies.strategy_selector import StrategySelector
from .phases import Phase, PhaseExecutor


@dataclass
class JarvisConfig:
    """Configuration for Jarvis."""

    workspace_root: str = "."
    use_mcts: bool = True
    max_mcts_simulations: int = 1000
    hardware_mode: str = "maximum"  # maximum, balanced, eco
    trace_enabled: bool = True
    verbose: bool = True


class Jarvis:
    """Meta-coder assistant optimized for M4 Pro Mac."""

    def __init__(self, config: JarvisConfig | None = None):
        self.config = config or JarvisConfig()
        self.workspace_root = Path(self.config.workspace_root)

        # Initialize accelerated tools
        self._init_tools()

        # Initialize components
        self.phase_executor = PhaseExecutor(self.tools)
        self.strategy_selector = StrategySelector()

        self.mcts = MCTSLite() if self.config.use_mcts else None

        # Results storage
        self.last_result = None

    def _init_tools(self):
        """Initialize all accelerated tools."""
        self.tools = type(
            "Tools",
            (),
            {
                "ripgrep": get_ripgrep_turbo(),
                "dependency_graph": get_dependency_graph(),
                "python_analyzer": get_python_analyzer(),
                "duckdb": get_duckdb_turbo(),
                "tracer": get_trace_turbo() if self.config.trace_enabled else None,
                "code_helper": get_code_helper(),
            },
        )()

        if self.config.verbose:
            print(
                f"🤖 Jarvis initialized with hardware mode: {self.config.hardware_mode}"
            )
            print("   • CPU cores: 12 (8 performance + 4 efficiency)")
            print("   • GPU: Metal (20 cores)")
            print("   • RAM: 24GB (19.2GB available)")

    async def assist(self, query: str) -> dict[str, Any]:
        """Main entry point - assist Claude Code with a complex task."""
        start_time = time.perf_counter()

        if self.config.verbose:
            print(f"\n🎯 Jarvis assisting with: {query}")
            print("=" * 60)

        # Create context
        context = {
            "query": query,
            "workspace_root": str(self.workspace_root),
            "timestamp": time.time(),
        }

        # Execute phases
        results = {}
        phases = [Phase.DISCOVER, Phase.ANALYZE, Phase.IMPLEMENT, Phase.VERIFY]

        for phase in phases:
            if self.config.verbose:
                print(f"\n📍 Phase: {phase.value.upper()}")

            # Execute phase with tracing
            if self.tools.tracer:
                async with self.tools.tracer.trace_span(
                    f"jarvis.{phase.value}"
                ) as span:
                    result = await self.phase_executor.execute_phase(phase, context)
                    span["success"] = result.success
                    span["duration_ms"] = result.duration_ms
            else:
                result = await self.phase_executor.execute_phase(phase, context)

            results[phase] = result
            self.phase_executor.results[phase] = result

            if self.config.verbose:
                status = "✅" if result.success else "❌"
                print(f"   {status} Completed in {result.duration_ms:.1f}ms")

                # Show key metrics
                if phase == Phase.DISCOVER:
                    print(f"   • Files found: {result.data.get('files_found', 0)}")
                elif phase == Phase.ANALYZE:
                    print(f"   • Strategy: {result.data.get('strategy', 'unknown')}")
                    print(f"   • Complexity: {result.data.get('complexity', 0)}")

            # Stop if phase failed
            if not result.success:
                break

        # Calculate total time
        total_time = (time.perf_counter() - start_time) * 1000

        # Build final result
        final_result = {
            "query": query,
            "success": all(r.success for r in results.values()),
            "phases": {
                phase.value: {
                    "success": result.success,
                    "duration_ms": result.duration_ms,
                    "data": result.data,
                    "errors": result.errors,
                }
                for phase, result in results.items()
            },
            "total_duration_ms": total_time,
            "hardware_utilized": {
                "cpu_cores": 12,
                "gpu": "Metal" if self.config.hardware_mode != "eco" else "None",
                "memory_gb": 19.2,
            },
        }

        # Show summary
        if self.config.verbose:
            print("\n" + "=" * 60)
            print(f"✅ COMPLETE in {total_time:.1f}ms")

            # Strategy used
            strategy = results[Phase.ANALYZE].data.get("strategy", "unknown")
            print(f"\n📋 Strategy: {strategy}")

            # Implementation summary
            impl_data = results[Phase.IMPLEMENT].data
            if "files_modified" in impl_data:
                print(f"📝 Files modified: {impl_data['files_modified']}")

        self.last_result = final_result
        return final_result

    async def explain_approach(self, query: str) -> str:
        """Explain how Jarvis would approach a task without executing it."""
        # Quick analysis
        context = {"query": query, "explain_only": True}

        # Just run discovery and analysis phases
        discover_result = await self.phase_executor.execute_phase(
            Phase.DISCOVER, context
        )
        self.phase_executor.results[Phase.DISCOVER] = discover_result

        analyze_result = await self.phase_executor.execute_phase(Phase.ANALYZE, context)

        # Build explanation
        explanation = f"""
🤖 Jarvis Analysis for: "{query}"

1️⃣ **Discovery Phase**
   - Search terms: {discover_result.data.get('search_terms', [])}
   - Files found: {discover_result.data.get('files_found', 0)}
   
2️⃣ **Analysis Phase**  
   - Strategy: {analyze_result.data.get('strategy', 'unknown')}
   - Complexity: {analyze_result.data.get('complexity', 0)}
   - Needs MCTS: {analyze_result.data.get('needs_mcts', False)}
   
3️⃣ **Implementation Approach**
   Based on the analysis, I would:
"""

        strategy = analyze_result.data.get("strategy", "general")

        if strategy == "optimization":
            explanation += """   - Use MCTS to explore optimization paths
   - Profile code to find bottlenecks
   - Apply parallelization where possible
   - Leverage GPU for compute-intensive operations"""
        elif strategy == "refactoring":
            explanation += """   - Build complete dependency graph
   - Identify all symbol usages
   - Perform safe rename/restructure operations
   - Verify no breaking changes"""
        elif strategy == "testing":
            explanation += """   - Analyze code coverage
   - Generate test cases for uncovered paths
   - Set up proper test fixtures
   - Ensure all edge cases are handled"""
        else:
            explanation += """   - Perform targeted code modifications
   - Maintain backward compatibility
   - Document changes made
   - Run verification tests"""

        explanation += f"""

4️⃣ **Hardware Utilization**
   - CPU: All 12 cores for parallel operations
   - GPU: Metal acceleration for {strategy} tasks
   - Memory: 19.2GB available for large-scale analysis
   
⏱️  Estimated time: {self._estimate_time(analyze_result.data)}ms
"""

        return explanation

    def _estimate_time(self, analysis_data: dict[str, Any]) -> int:
        """Estimate execution time based on analysis."""
        base_time = 100  # Base overhead

        # Add time based on complexity
        complexity = analysis_data.get("complexity", 0)
        base_time += complexity * 10

        # Add time for MCTS if needed
        if analysis_data.get("needs_mcts", False):
            base_time += 500

        # Add time based on files
        files = analysis_data.get("files_analyzed", 0)
        base_time += files * 50

        return base_time

    async def cleanup(self):
        """Cleanup resources."""
        self.tools.ripgrep.cleanup()
        self.tools.dependency_graph.cleanup()
        self.tools.python_analyzer.cleanup()
        self.tools.duckdb.cleanup()
        if self.tools.tracer:
            await self.tools.tracer.cleanup()
        self.tools.code_helper.cleanup()
