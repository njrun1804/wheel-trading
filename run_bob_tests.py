#!/usr/bin/env python3
"""
Real-world test runner for Bob M4 optimizations.

This script executes the comprehensive test suite to validate the enhanced
12-agent coordinator and M4 Pro optimizations.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bob.testing.real_world_test_suite import RealWorldTestSuite


async def main():
    """Main test execution function."""
    print("🚀 Bob M4 Pro Real-World Test Suite")
    print("=" * 50)
    print("Testing enhanced 12-agent coordinator with M4 optimizations")
    print("=" * 50)
    
    # Initialize test suite
    test_suite = RealWorldTestSuite()
    
    try:
        print("🔧 Setting up test environment...")
        await test_suite.setup()
        
        print(f"📋 Executing {len(test_suite.tests)} real-world tests in parallel")
        print("🎯 This will validate:")
        print("  - 12-agent coordination")
        print("  - M4 Pro P-core/E-core utilization")
        print("  - HTTP/2 Claude request optimization")
        print("  - Intelligent task routing")
        print("  - Performance against baselines")
        print()
        
        # Execute tests
        start_time = time.time()
        results = await test_suite.run_parallel_tests()
        total_time = time.time() - start_time
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        summary = results["execution_summary"]
        performance = results["performance_metrics"]
        
        print(f"✅ Passed: {summary['passed_tests']}/{summary['total_tests']} tests")
        print(f"📈 Success Rate: {summary['success_rate']:.1%}")
        print(f"⏱️  Total Execution Time: {summary['total_execution_time']:.2f}s")
        print(f"🚀 Parallel Efficiency: {summary['parallel_efficiency']:.1f}x speedup")
        print(f"🤖 Avg Agents per Test: {performance['avg_agents_per_test']:.1f}")
        print(f"⚡ Avg Test Duration: {performance['avg_test_duration']:.2f}s")
        
        # Complexity breakdown
        print(f"\n📋 COMPLEXITY BREAKDOWN:")
        for complexity, stats in results["complexity_breakdown"].items():
            print(f"  {complexity.upper()}: {stats['passed']}/{stats['total']} passed (avg: {stats['avg_duration']:.2f}s)")
        
        # Category breakdown
        print(f"\n🎯 CATEGORY BREAKDOWN:")
        for category, stats in results["category_breakdown"].items():
            print(f"  {category.replace('_', ' ').title()}: {stats['success_rate']:.1%} success rate")
        
        # Baseline analysis
        print(f"\n📊 PERFORMANCE VS BASELINES:")
        baseline_analysis = results["baseline_analysis"]
        for metric, analysis in baseline_analysis.items():
            if isinstance(analysis, dict) and "meets_baseline" in analysis:
                status = "✅" if analysis["meets_baseline"] else "❌"
                print(f"  {metric}: {status} {analysis['actual']:.3f} vs {analysis['baseline']:.3f}")
        
        # Failed tests details
        failed_tests = [r for r in results["detailed_results"] if not r["success"]]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"  - {test['test_id']}: {test.get('error', 'Unknown error')}")
        
        # Performance insights
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        
        # Check if M4 optimizations are working
        if summary['parallel_efficiency'] > 2.0:
            print("  ✅ Excellent parallel efficiency - M4 optimizations working well")
        elif summary['parallel_efficiency'] > 1.5:
            print("  ✅ Good parallel efficiency - M4 optimizations active")
        else:
            print("  ⚠️  Low parallel efficiency - may need optimization tuning")
        
        if performance['avg_agents_per_test'] > 1.5:
            print("  ✅ Multi-agent coordination is active")
        else:
            print("  ⚠️  Most tests using single agents - check task complexity")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if summary['success_rate'] >= 0.8:
            print("  ✅ Bob system is performing well with M4 optimizations")
        elif summary['success_rate'] >= 0.6:
            print("  ⚠️  Bob system needs some optimization adjustments")
        else:
            print("  ❌ Bob system has significant issues requiring attention")
        
        # Save detailed results
        output_file = project_root / "bob_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        print("\n🧹 Cleaning up test environment...")
        await test_suite.cleanup()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    results = asyncio.run(main())