"""Use Jarvis2 to analyze and plan test completion strategy."""
import asyncio
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

from jarvis2.core.orchestrator import CodeRequest, Jarvis2Orchestrator


async def analyze_test_situation():
    """Use Jarvis2 to analyze the test situation."""
    jarvis = Jarvis2Orchestrator()
    await jarvis.initialize()

    try:
        # 1. High-level analysis
        print("=== Phase 1: High-Level Test Analysis ===\n")

        analysis_request = CodeRequest(
            """
        Analyze this test situation for Jarvis2 on M4 Pro:
        
        Current State:
        - Basic functionality works (code generation in ~50ms)
        - Hardware detection correct (M4 Pro, 24GB RAM, 16 GPU cores)
        - Process isolation works with spawn method
        - Memory management within 18GB Metal limit
        
        Issues:
        - Parallel request handling hangs (possible queue blocking)
        - Some tests timeout with spawn method overhead
        - PyTorch MPS requires spawn to avoid deadlocks
        - Test expectations don't match M4 Pro unified memory performance
        
        Question: Which issues are test artifacts vs real problems?
        """
        )

        analysis = await jarvis.generate_code(analysis_request)
        print("Jarvis2 Analysis:")
        print(analysis.code)
        print(f"\nConfidence: {analysis.confidence:.0%}\n")

        # 2. Sequential reasoning about fixes
        print("=== Phase 2: Sequential Fix Strategy ===\n")

        strategy_request = CodeRequest(
            """
        Create a sequential plan to efficiently complete all remaining tests:
        
        Test Categories:
        1. Device routing tests (mostly passing, need performance expectation updates)
        2. Process isolation tests (timeout issues with spawn method)
        3. Memory management tests (need to verify shared memory works)
        4. Performance benchmarks (need M4 Pro specific baselines)
        5. MCTS correctness tests (need to verify search improves solutions)
        6. Integration tests (parallel requests hanging)
        
        Constraints:
        - Must use spawn method for PyTorch MPS
        - Must stay within 18GB Metal memory limit
        - Tests should complete in reasonable time
        - No dummy implementations allowed
        
        Create an efficient sequential plan considering dependencies.
        """
        )

        strategy = await jarvis.generate_code(strategy_request)
        print("Sequential Strategy:")
        print(strategy.code)
        print(f"\nConfidence: {strategy.confidence:.0%}\n")

        # 3. Identify test artifacts vs real issues
        print("=== Phase 3: Test Artifacts vs Real Issues ===\n")

        artifacts_request = CodeRequest(
            """
        Categorize these issues as test artifacts or real problems:
        
        1. Spawn method makes worker initialization take 1-2 seconds
        2. GPU speedup only 1.1x-1.9x instead of expected 3x-5x
        3. Parallel requests hang in asyncio.gather
        4. "Task was destroyed but pending" warnings
        5. Duplicate experience IDs causing database errors
        6. Tests expecting fork() behavior with spawn()
        7. Memory pressure detection showing false positives
        
        For each, explain why it's a test artifact or real problem.
        """
        )

        artifacts = await jarvis.generate_code(artifacts_request)
        print("Test Artifacts Analysis:")
        print(artifacts.code)
        print(f"\nConfidence: {artifacts.confidence:.0%}\n")

        # 4. Optimization opportunities
        print("=== Phase 4: Test Optimization Opportunities ===\n")

        optimization_request = CodeRequest(
            """
        Suggest optimizations to make tests run efficiently on M4 Pro:
        
        Current pain points:
        - Worker initialization overhead (spawn method)
        - Large tensor operations in tests (4096x768)
        - Repeated initialization/shutdown cycles
        - Synchronous queue operations in async context
        
        Suggest specific optimizations without compromising test validity.
        """
        )

        optimizations = await jarvis.generate_code(optimization_request)
        print("Optimization Suggestions:")
        print(optimizations.code)
        print(f"\nConfidence: {optimizations.confidence:.0%}\n")

        # 5. Final execution plan
        print("=== Phase 5: Execution Plan ===\n")

        plan_request = CodeRequest(
            """
        Create a concrete execution plan with specific commands and file edits:
        
        Priority order:
        1. Fix hanging parallel requests (highest impact)
        2. Adjust test timeouts for spawn method
        3. Update performance expectations for M4 Pro
        4. Add test fixtures for worker reuse
        5. Implement health checks for workers
        
        Include specific code changes and test commands.
        """
        )

        plan = await jarvis.generate_code(plan_request)
        print("Execution Plan:")
        print(plan.code)
        print(f"\nConfidence: {plan.confidence:.0%}\n")

        # Save all analyses
        results = {
            "analysis": analysis.code,
            "strategy": strategy.code,
            "artifacts": artifacts.code,
            "optimizations": optimizations.code,
            "plan": plan.code,
        }

        with open("jarvis2_test_analysis.json", "w") as f:
            json.dump(results, f, indent=2)

        print("✅ Analysis saved to jarvis2_test_analysis.json")

    finally:
        await jarvis.shutdown()


if __name__ == "__main__":
    print("Using Jarvis2 to analyze test completion strategy...\n")
    asyncio.run(analyze_test_situation())
