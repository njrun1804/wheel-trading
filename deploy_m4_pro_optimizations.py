#!/usr/bin/env python3
"""
M4 Pro Optimization Deployment Script

Deploys and validates all M4 Pro optimizations for Einstein/Bolt system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main deployment function"""
    logger.info("🚀 Starting M4 Pro optimization deployment...")

    try:
        # Import after setting up logging
        from bolt.m4_pro_integration import deploy_einstein_with_m4_pro_optimizations

        # Deploy the system
        deployment_report = await deploy_einstein_with_m4_pro_optimizations()

        # Print deployment summary
        print("\n" + "=" * 80)
        print("M4 PRO OPTIMIZATION DEPLOYMENT REPORT")
        print("=" * 80)

        # System status
        status = deployment_report["system_status"]
        print("\n📊 SYSTEM STATUS:")
        print(f"   Active Components: {status['active_components']}/5")
        print(f"   Initialization Time: {status['initialization_time_ms']:.1f}ms")
        print(
            f"   Benchmark Validation: {'✅ PASSED' if status['benchmark_validation_passed'] else '❌ FAILED'}"
        )

        # Component status
        print("\n🔧 COMPONENT STATUS:")
        components = [
            ("Unified Memory", status["unified_memory_active"]),
            ("Metal Search", status["metal_search_active"]),
            ("Adaptive Concurrency", status["adaptive_concurrency_active"]),
            ("Memory Pools", status["memory_pools_active"]),
            ("ANE Acceleration", status["ane_acceleration_active"]),
        ]

        for name, active in components:
            status_icon = "✅" if active else "❌"
            print(f"   {status_icon} {name}")

        # Performance improvements
        if "latest_benchmarks" in deployment_report:
            benchmarks = deployment_report["latest_benchmarks"]
            if "performance_improvements" in benchmarks:
                improvements = benchmarks["performance_improvements"]
                print("\n📈 PERFORMANCE IMPROVEMENTS:")
                for component, improvement in improvements.items():
                    print(f"   {component}: {improvement:+.1f}%")

        # Component statistics
        if "component_stats" in deployment_report:
            stats = deployment_report["component_stats"]
            print("\n📊 COMPONENT STATISTICS:")

            # Memory usage
            if "unified_memory" in stats:
                mem_stats = stats["unified_memory"]
                print(f"   Memory Usage: {mem_stats.get('total_memory_mb', 0):.1f}MB")

            # Search performance
            if "metal_search" in stats:
                search_stats = stats["metal_search"]
                print(
                    f"   Search Latency: {search_stats.get('average_latency_ms', 0):.1f}ms"
                )
                print(
                    f"   GPU Utilization: {search_stats.get('gpu_usage_percent', 0):.1f}%"
                )

            # Concurrency
            if "adaptive_concurrency" in stats:
                conc_stats = stats["adaptive_concurrency"]
                print(
                    f"   Adaptive Adjustments: {conc_stats.get('adaptive_adjustments', 0)}"
                )

        # Save detailed report
        report_file = Path("m4_pro_deployment_report.json")
        with open(report_file, "w") as f:
            json.dump(deployment_report, f, indent=2, default=str)

        print(f"\n📄 Detailed report saved to: {report_file}")

        # Success message
        if status["active_components"] >= 4:  # At least 4/5 components active
            print("\n🎉 DEPLOYMENT SUCCESSFUL!")
            print(
                f"   Einstein is now optimized for M4 Pro with {status['active_components']}/5 optimizations active."
            )
            print("   Expected performance improvements:")
            print("   • Search latency: <10ms (target: 2.3x improvement)")
            print("   • Concurrent operations: <30ms (target: 2.7x improvement)")
            print("   • Memory usage: <400MB (target: 48% reduction)")
            print("   • Overall throughput: >50 ops/sec (target: 4x improvement)")
        else:
            print("\n⚠️  PARTIAL DEPLOYMENT")
            print(f"   Only {status['active_components']}/5 optimizations are active.")
            print("   System will work but with reduced performance benefits.")

        print("\n🔍 USAGE:")
        print("   # Use the optimized system")
        print("   from bolt.m4_pro_integration import get_m4_pro_system")
        print("   system = get_m4_pro_system()")
        print("   report = await system.get_system_performance_report()")

        return 0 if status["active_components"] >= 4 else 1

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("   Make sure all dependencies are installed:")
        logger.error("   • mlx-python (for Metal GPU acceleration)")
        logger.error("   • coremltools (for ANE acceleration)")
        return 1

    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        logger.error("   Check the logs above for detailed error information")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
