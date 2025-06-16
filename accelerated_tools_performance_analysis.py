#!/usr/bin/env python3
"""
Comprehensive Performance Analysis for Accelerated Tools
Analyzes actual performance vs claimed improvements
"""

import asyncio
import json
from pathlib import Path
from typing import Any


class PerformanceAnalyzer:
    """Analyzes accelerated tools performance."""

    def __init__(self):
        self.claimed_improvements = {
            "ripgrep_turbo": {
                "improvement_factor": 30,
                "baseline_ms": 150,
                "description": "30x faster search with all CPU cores",
            },
            "python_analysis_turbo": {
                "improvement_factor": 173,
                "baseline_ms": 2600,
                "description": "173x faster with MLX GPU acceleration",
            },
            "dependency_graph_turbo": {
                "improvement_factor": 12,
                "baseline_ms": 6000,
                "description": "12x faster with parallel AST parsing",
            },
            "duckdb_turbo": {
                "improvement_factor": 7,
                "baseline_ms": 100,
                "description": "7x faster with connection pool and M4 optimizations",
            },
            "trace_turbo": {
                "improvement_factor": 5,
                "baseline_ms": 50,
                "description": "5x faster tracing with multiple backends",
            },
        }

        # Load test results
        self.test_results = self._load_test_results()

    def _load_test_results(self) -> dict:
        """Load test results from JSON file."""
        results_file = Path("quick_accelerated_tools_test_results.json")
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
        return {}

    def analyze_performance(self) -> dict[str, Any]:
        """Analyze performance of each tool."""
        analysis = {
            "summary": {
                "total_tools": len(self.claimed_improvements),
                "tested_tools": 0,
                "successful_tests": 0,
                "performance_achieved": {},
                "recommendations": [],
            },
            "detailed_analysis": {},
        }

        for tool_name, claims in self.claimed_improvements.items():
            tool_analysis = self._analyze_tool(tool_name, claims)
            analysis["detailed_analysis"][tool_name] = tool_analysis

            if tool_analysis["test_status"] == "success":
                analysis["summary"]["successful_tests"] += 1
                analysis["summary"]["performance_achieved"][tool_name] = tool_analysis[
                    "performance_ratio"
                ]

            analysis["summary"]["tested_tools"] += 1

        # Generate recommendations
        analysis["summary"]["recommendations"] = self._generate_recommendations(
            analysis
        )

        return analysis

    def _analyze_tool(self, tool_name: str, claims: dict) -> dict[str, Any]:
        """Analyze individual tool performance."""
        test_data = self.test_results.get("tests", {}).get(tool_name, {})

        if not test_data or test_data.get("status") != "success":
            return {
                "test_status": "failed",
                "error": test_data.get("error", "No test data available"),
                "claimed_improvement": claims["improvement_factor"],
                "claimed_description": claims["description"],
            }

        # Get the primary performance metric for each tool
        actual_time_ms = self._get_primary_metric(tool_name, test_data)

        if actual_time_ms is None:
            return {
                "test_status": "no_metric",
                "error": "Could not extract performance metric",
                "claimed_improvement": claims["improvement_factor"],
                "claimed_description": claims["description"],
            }

        # Calculate actual improvement
        baseline_ms = claims["baseline_ms"]
        actual_improvement = baseline_ms / actual_time_ms if actual_time_ms > 0 else 0
        performance_ratio = actual_improvement / claims["improvement_factor"]

        return {
            "test_status": "success",
            "claimed_improvement": claims["improvement_factor"],
            "actual_improvement": round(actual_improvement, 1),
            "performance_ratio": round(performance_ratio, 2),
            "baseline_ms": baseline_ms,
            "actual_ms": round(actual_time_ms, 2),
            "claimed_description": claims["description"],
            "meets_target": performance_ratio >= 0.5,  # At least 50% of claimed
            "exceeds_target": performance_ratio >= 1.0,
            "hardware_utilization": self._get_hardware_utilization(
                tool_name, test_data
            ),
        }

    def _get_primary_metric(self, tool_name: str, test_data: dict) -> float:
        """Get primary performance metric for each tool."""
        metric_map = {
            "ripgrep_turbo": "search_time_ms",
            "python_analysis_turbo": "single_file_time_ms",
            "dependency_graph_turbo": "build_time_ms",
            "duckdb_turbo": "query_time_ms",
            "trace_turbo": "span_time_ms",
        }

        metric_key = metric_map.get(tool_name)
        if metric_key and metric_key in test_data:
            return test_data[metric_key]

        # Fallback: look for any time-related metric
        for key, value in test_data.items():
            if "time_ms" in key and isinstance(value, int | float):
                return value

        return None

    def _get_hardware_utilization(
        self, tool_name: str, test_data: dict
    ) -> dict[str, Any]:
        """Extract hardware utilization info."""
        utilization = {}

        # CPU cores
        if "cpu_cores" in test_data:
            utilization["cpu_cores"] = test_data["cpu_cores"]
        elif "cpu_workers" in test_data:
            utilization["cpu_cores"] = test_data["cpu_workers"]

        # Memory usage
        if "cache_size_mb" in test_data:
            utilization["cache_size_mb"] = test_data["cache_size_mb"]

        # Connection pools
        if "pool_size" in test_data:
            utilization["pool_size"] = test_data["pool_size"]

        # GPU acceleration
        hardware = self.test_results.get("hardware", {})
        if hardware.get("mlx_available"):
            utilization["gpu_available"] = True

        return utilization

    def _generate_recommendations(self, analysis: dict) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []

        successful_tests = analysis["summary"]["successful_tests"]
        total_tools = analysis["summary"]["total_tools"]

        # Overall success rate
        success_rate = successful_tests / total_tools
        if success_rate < 0.8:
            recommendations.append(
                f"Only {successful_tests}/{total_tools} tools tested successfully. "
                f"Consider investigating failed tools."
            )

        # Performance analysis
        performance_data = analysis["summary"]["performance_achieved"]
        if performance_data:
            excellent_performers = [k for k, v in performance_data.items() if v >= 1.0]
            good_performers = [k for k, v in performance_data.items() if 0.5 <= v < 1.0]
            poor_performers = [k for k, v in performance_data.items() if v < 0.5]

            if excellent_performers:
                recommendations.append(
                    f"🎉 Excellent performance achieved: {', '.join(excellent_performers)} "
                    f"(meeting or exceeding claimed improvements)"
                )

            if good_performers:
                recommendations.append(
                    f"✅ Good performance: {', '.join(good_performers)} "
                    f"(achieving 50-99% of claimed improvements)"
                )

            if poor_performers:
                recommendations.append(
                    f"⚠️ Underperforming tools: {', '.join(poor_performers)} "
                    f"(achieving <50% of claimed improvements)"
                )

        # Hardware-specific recommendations
        hardware = self.test_results.get("hardware", {})
        if hardware.get("is_m4_pro"):
            recommendations.append("✅ Running on M4 Pro - optimal hardware detected")
        else:
            recommendations.append(
                "⚠️ Not running on M4 Pro - performance may be suboptimal"
            )

        if hardware.get("mlx_available"):
            recommendations.append("✅ MLX GPU acceleration available")
        else:
            recommendations.append(
                "❌ MLX GPU acceleration not available - Python analysis may be slower"
            )

        return recommendations

    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        analysis = self.analyze_performance()

        report = []
        report.append("=" * 80)
        report.append("ACCELERATED TOOLS PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Executive Summary
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("-" * 40)

        summary = analysis["summary"]
        report.append(
            f"Tools Tested: {summary['successful_tests']}/{summary['total_tools']}"
        )

        if summary["performance_achieved"]:
            avg_performance = sum(summary["performance_achieved"].values()) / len(
                summary["performance_achieved"]
            )
            report.append(
                f"Average Performance Ratio: {avg_performance:.1%} of claimed improvements"
            )

        report.append("")

        # Hardware Configuration
        hardware = self.test_results.get("hardware", {})
        report.append("💻 HARDWARE CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Platform: {hardware.get('platform', 'Unknown')}")
        report.append(
            f"CPU: {hardware.get('cpu_brand', 'Unknown')} ({hardware.get('cpu_count', 'Unknown')} cores)"
        )
        report.append(f"Memory: {hardware.get('memory_gb', 'Unknown')}GB")
        report.append(f"M4 Pro: {'✅' if hardware.get('is_m4_pro') else '❌'}")
        report.append(f"MLX GPU: {'✅' if hardware.get('mlx_available') else '❌'}")
        report.append("")

        # Detailed Tool Analysis
        report.append("🔧 DETAILED TOOL ANALYSIS")
        report.append("-" * 40)

        for tool_name, tool_analysis in analysis["detailed_analysis"].items():
            report.append(f"\n{tool_name.upper().replace('_', ' ')}")
            report.append("  " + tool_analysis["claimed_description"])

            if tool_analysis["test_status"] == "success":
                status = "✅" if tool_analysis["meets_target"] else "⚠️"
                report.append(f"  {status} Status: {tool_analysis['test_status']}")
                report.append(
                    f"  📈 Claimed: {tool_analysis['claimed_improvement']}x faster"
                )
                report.append(
                    f"  📊 Actual: {tool_analysis['actual_improvement']}x faster"
                )
                report.append(
                    f"  🎯 Performance: {tool_analysis['performance_ratio']:.1%} of target"
                )
                report.append(
                    f"  ⏱️  Time: {tool_analysis['actual_ms']}ms (baseline: {tool_analysis['baseline_ms']}ms)"
                )

                # Hardware utilization
                hw_util = tool_analysis["hardware_utilization"]
                if hw_util:
                    utilization_details = []
                    if "cpu_cores" in hw_util:
                        utilization_details.append(f"{hw_util['cpu_cores']} CPU cores")
                    if "cache_size_mb" in hw_util:
                        utilization_details.append(
                            f"{hw_util['cache_size_mb']}MB cache"
                        )
                    if "pool_size" in hw_util:
                        utilization_details.append(f"Pool size: {hw_util['pool_size']}")
                    if utilization_details:
                        report.append(
                            f"  🔧 Utilization: {', '.join(utilization_details)}"
                        )
            else:
                report.append(f"  ❌ Status: {tool_analysis['test_status']}")
                if "error" in tool_analysis:
                    report.append(f"  🚫 Error: {tool_analysis['error']}")

        report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        for rec in summary["recommendations"]:
            report.append(f"• {rec}")

        report.append("")

        # Performance Summary Table
        report.append("📈 PERFORMANCE SUMMARY TABLE")
        report.append("-" * 40)
        report.append(
            f"{'Tool':<25} {'Claimed':<10} {'Actual':<10} {'Ratio':<10} {'Status'}"
        )
        report.append("-" * 65)

        for tool_name, tool_analysis in analysis["detailed_analysis"].items():
            if tool_analysis["test_status"] == "success":
                status = "✅ Pass" if tool_analysis["meets_target"] else "⚠️ Below"
                if tool_analysis["exceeds_target"]:
                    status = "🎉 Exceed"

                report.append(
                    f"{tool_name:<25} "
                    f"{tool_analysis['claimed_improvement']}x{'':<7} "
                    f"{tool_analysis['actual_improvement']}x{'':<6} "
                    f"{tool_analysis['performance_ratio']:.1%}{'':<6} "
                    f"{status}"
                )
            else:
                report.append(
                    f"{tool_name:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} ❌ Failed"
                )

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_report(self, filename: str = "accelerated_tools_performance_report.md"):
        """Save the performance report."""
        report = self.generate_report()
        with open(filename, "w") as f:
            f.write(report)
        print(f"📝 Performance report saved to {filename}")


async def main():
    """Generate performance analysis report."""
    analyzer = PerformanceAnalyzer()

    # Print the report
    report = analyzer.generate_report()
    print(report)

    # Save to file
    analyzer.save_report()

    # Also save analysis data as JSON
    analysis = analyzer.analyze_performance()
    with open("accelerated_tools_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print("\n💾 Analysis data saved to accelerated_tools_analysis.json")


if __name__ == "__main__":
    asyncio.run(main())
