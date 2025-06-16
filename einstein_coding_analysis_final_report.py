#!/usr/bin/env python3
"""
Final comprehensive analysis of Einstein's semantic search coding capabilities.

Combines all test results and provides detailed analysis of performance.
"""

import json
import time
from pathlib import Path


def analyze_einstein_coding_capabilities():
    """Analyze Einstein's coding analysis capabilities based on test results."""

    print("=" * 80)
    print("EINSTEIN SEMANTIC SEARCH - CODING ANALYSIS FINAL REPORT")
    print("=" * 80)

    # Load test results
    project_root = Path.cwd()
    quick_results_file = project_root / "einstein_coding_quick_results.json"

    if not quick_results_file.exists():
        print("❌ Test results file not found. Please run the tests first.")
        return

    with open(quick_results_file) as f:
        results = json.load(f)

    print(
        f"Test timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))}"
    )
    print(f"Project root: {project_root}")

    # Analyze test results
    tests = results.get("tests", [])
    faiss_status = results.get("faiss_status", {})

    print("\n" + "=" * 40)
    print("1. SYSTEM CAPABILITIES")
    print("=" * 40)

    print(f"FAISS Available: {faiss_status.get('available', False)}")

    # Basic search performance
    basic_tests = [t for t in tests if not t["query_name"].startswith("semantic:")]
    semantic_tests = [t for t in tests if t["query_name"].startswith("semantic:")]

    print("\n" + "=" * 40)
    print("2. BASIC SEARCH PERFORMANCE")
    print("=" * 40)

    successful_basic = [t for t in basic_tests if t["success"]]

    if successful_basic:
        avg_time = sum(t["execution_time_ms"] for t in successful_basic) / len(
            successful_basic
        )
        total_results = sum(t["result_count"] for t in successful_basic)

        print(
            f"Successful Tests: {len(successful_basic)}/{len(basic_tests)} ({len(successful_basic)/len(basic_tests):.1%})"
        )
        print(f"Average Response Time: {avg_time:.1f}ms")
        print(f"Total Results Found: {total_results}")

        print("\nDetailed Performance:")
        for test in successful_basic:
            print(
                f"  {test['query_name']}: {test['result_count']} results in {test['execution_time_ms']:.1f}ms"
            )

    print("\n" + "=" * 40)
    print("3. SEMANTIC SEARCH ANALYSIS")
    print("=" * 40)

    successful_semantic = [t for t in semantic_tests if t["success"]]

    if successful_semantic:
        avg_semantic_time = sum(
            t["execution_time_ms"] for t in successful_semantic
        ) / len(successful_semantic)
        semantic_understanding_count = sum(
            1 for t in successful_semantic if t.get("semantic_understanding", False)
        )
        understanding_rate = semantic_understanding_count / len(successful_semantic)

        print(
            f"Semantic Tests: {len(successful_semantic)}/{len(semantic_tests)} successful"
        )
        print(f"Average Semantic Search Time: {avg_semantic_time:.1f}ms")
        print(f"Semantic Understanding Rate: {understanding_rate:.1%}")

        print("\nSemantic Test Details:")
        for test in successful_semantic:
            understanding = "✅" if test.get("semantic_understanding", False) else "❌"
            print(f"  {understanding} {test['query']}")
            print(
                f"     Results: {test['result_count']}, Time: {test['execution_time_ms']:.1f}ms"
            )

    # Coding-specific analysis
    print("\n" + "=" * 40)
    print("4. CODING QUERY EFFECTIVENESS")
    print("=" * 40)

    # Analyze how well Einstein handles different coding concepts
    coding_effectiveness = {
        "class definitions": [],
        "async functions": [],
        "import statements": [],
        "error handling": [],
        "dataclass definitions": [],
        "inheritance patterns": [],
        "mathematical functions": [],
    }

    for test in tests:
        query_name = test["query_name"].lower()
        if "class" in query_name:
            coding_effectiveness["class definitions"].append(test)
        elif "async" in query_name:
            coding_effectiveness["async functions"].append(test)
        elif "import" in query_name:
            coding_effectiveness["import statements"].append(test)
        elif "error" in query_name or "try" in query_name:
            coding_effectiveness["error handling"].append(test)
        elif "dataclass" in query_name:
            coding_effectiveness["dataclass definitions"].append(test)
        elif "inheritance" in query_name:
            coding_effectiveness["inheritance patterns"].append(test)
        elif "mathematical" in query_name:
            coding_effectiveness["mathematical functions"].append(test)

    for concept, concept_tests in coding_effectiveness.items():
        if concept_tests:
            successful = [t for t in concept_tests if t["success"]]
            if successful:
                avg_results = sum(t["result_count"] for t in successful) / len(
                    successful
                )
                avg_time = sum(t["execution_time_ms"] for t in successful) / len(
                    successful
                )

                effectiveness_score = (
                    "High"
                    if avg_results >= 5
                    else "Medium"
                    if avg_results >= 2
                    else "Low"
                )
                speed_score = (
                    "Fast"
                    if avg_time <= 50
                    else "Medium"
                    if avg_time <= 100
                    else "Slow"
                )

                print(
                    f"  {concept.title()}: {effectiveness_score} effectiveness, {speed_score} speed"
                )
                print(f"    Avg results: {avg_results:.1f}, Avg time: {avg_time:.1f}ms")

    # Performance comparison
    print("\n" + "=" * 40)
    print("5. PERFORMANCE COMPARISON")
    print("=" * 40)

    if successful_basic and successful_semantic:
        basic_avg_time = sum(t["execution_time_ms"] for t in successful_basic) / len(
            successful_basic
        )
        semantic_avg_time = sum(
            t["execution_time_ms"] for t in successful_semantic
        ) / len(successful_semantic)

        performance_ratio = (
            semantic_avg_time / basic_avg_time if basic_avg_time > 0 else 0
        )

        print(f"Basic Search Avg: {basic_avg_time:.1f}ms")
        print(f"Semantic Search Avg: {semantic_avg_time:.1f}ms")
        print(f"Semantic Overhead: {performance_ratio:.1f}x")

        if performance_ratio < 2:
            print("✅ Semantic search overhead is acceptable")
        elif performance_ratio < 5:
            print("⚠️  Semantic search has moderate overhead")
        else:
            print("❌ Semantic search has high overhead")

    # Quality assessment
    print("\n" + "=" * 40)
    print("6. SEMANTIC UNDERSTANDING QUALITY")
    print("=" * 40)

    # Analyze the quality of semantic understanding
    understanding_indicators = []

    for test in successful_semantic:
        if test.get("semantic_understanding", False):
            understanding_indicators.append(
                {
                    "query": test["query"],
                    "results": test["result_count"],
                    "time": test["execution_time_ms"],
                }
            )

    if understanding_indicators:
        print(f"Queries with semantic understanding: {len(understanding_indicators)}")
        print("Evidence of semantic understanding:")
        for indicator in understanding_indicators:
            print(
                f"  ✅ '{indicator['query']}' - {indicator['results']} relevant results"
            )

    # Recommendations
    print("\n" + "=" * 40)
    print("7. RECOMMENDATIONS")
    print("=" * 40)

    recommendations = []

    if faiss_status.get("available", False):
        recommendations.append("✅ FAISS is available and functional")
    else:
        recommendations.append(
            "❌ FAISS not available - install for better semantic search"
        )

    if successful_basic:
        avg_basic_time = sum(t["execution_time_ms"] for t in successful_basic) / len(
            successful_basic
        )
        if avg_basic_time < 50:
            recommendations.append("✅ Basic search performance is excellent")
        elif avg_basic_time < 100:
            recommendations.append("⚠️  Basic search performance is acceptable")
        else:
            recommendations.append("❌ Basic search performance needs optimization")

    if successful_semantic:
        understanding_rate = sum(
            1 for t in successful_semantic if t.get("semantic_understanding", False)
        ) / len(successful_semantic)
        if understanding_rate >= 0.7:
            recommendations.append("✅ High semantic understanding rate")
        elif understanding_rate >= 0.4:
            recommendations.append("⚠️  Moderate semantic understanding rate")
        else:
            recommendations.append(
                "❌ Low semantic understanding rate - needs improvement"
            )

    for rec in recommendations:
        print(f"  {rec}")

    # Overall assessment
    print("\n" + "=" * 40)
    print("8. OVERALL ASSESSMENT")
    print("=" * 40)

    total_successful = len([t for t in tests if t["success"]])
    total_tests = len(tests)
    success_rate = total_successful / total_tests if total_tests > 0 else 0

    semantic_capability = len(understanding_indicators) > 0
    performance_acceptable = True

    if successful_basic:
        avg_basic_time = sum(t["execution_time_ms"] for t in successful_basic) / len(
            successful_basic
        )
        performance_acceptable = avg_basic_time < 100

    print(f"Test Success Rate: {success_rate:.1%} ({total_successful}/{total_tests})")
    print(f"Semantic Capability: {'✅ Present' if semantic_capability else '❌ Limited'}")
    print(
        f"Performance: {'✅ Acceptable' if performance_acceptable else '❌ Needs optimization'}"
    )
    print(
        f"FAISS Integration: {'✅ Working' if faiss_status.get('available', False) else '❌ Not available'}"
    )

    # Overall grade
    if success_rate >= 0.8 and semantic_capability and performance_acceptable:
        grade = "A - Excellent"
    elif success_rate >= 0.6 and (semantic_capability or performance_acceptable):
        grade = "B - Good"
    elif success_rate >= 0.4:
        grade = "C - Fair"
    else:
        grade = "D - Needs improvement"

    print(f"\n🎯 OVERALL GRADE: {grade}")

    # Save final report
    final_report = {
        "timestamp": time.time(),
        "overall_grade": grade,
        "success_rate": success_rate,
        "semantic_capability": semantic_capability,
        "performance_acceptable": performance_acceptable,
        "faiss_available": faiss_status.get("available", False),
        "recommendations": recommendations,
        "detailed_results": results,
    }

    report_file = project_root / "einstein_coding_analysis_final_report.json"
    with open(report_file, "w") as f:
        json.dump(final_report, f, indent=2, default=str)

    print(f"\n📊 Final report saved to: {report_file}")

    return final_report


if __name__ == "__main__":
    analyze_einstein_coding_capabilities()
