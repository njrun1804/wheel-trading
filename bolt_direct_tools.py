#!/usr/bin/env python3
"""
Direct Tool Integration - Bypass broken layers for immediate functionality.

This module provides direct access to tools and functionality, bypassing
the broken bolt integration layers. Use these tools when bolt's orchestration
system fails but you need immediate access to core functionality.

Key Features:
- Direct database access without connection issues
- Immediate search functionality without asyncio problems
- Simple task execution without complex agent coordination
- Trading-specific analysis tools
"""

import contextlib
import logging
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DirectSearchResult:
    """Simple search result structure."""

    file: str
    line: int
    content: str
    type: str = "direct"


@dataclass
class DirectAnalysisResult:
    """Analysis result from direct tools."""

    analysis_type: str
    findings: list[str]
    recommendations: list[str]
    execution_time: float
    files_analyzed: int


class DirectRipgrep:
    """Direct ripgrep access bypassing all integration layers."""

    @staticmethod
    def search(
        pattern: str, path: str = ".", include_types: list[str] | None = None
    ) -> list[DirectSearchResult]:
        """Direct ripgrep search with no dependencies."""
        try:
            cmd = ["rg", "--line-number", "--no-heading", "--with-filename"]

            if include_types:
                for file_type in include_types:
                    cmd.extend(["-t", file_type])

            cmd.extend([pattern, path])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            results = []
            for line in result.stdout.strip().split("\n"):
                if ":" in line and line.strip():
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        file_path, line_num, content = parts
                        try:
                            results.append(
                                DirectSearchResult(
                                    file=file_path,
                                    line=int(line_num),
                                    content=content.strip(),
                                )
                            )
                        except ValueError:
                            continue

            return results

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Ripgrep failed: {e}")
            return DirectRipgrep._fallback_search(pattern, path)

    @staticmethod
    def _fallback_search(pattern: str, path: str) -> list[DirectSearchResult]:
        """Fallback to grep if ripgrep fails."""
        try:
            cmd = ["grep", "-rn", "--include=*.py", pattern, path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            results = []
            for line in result.stdout.strip().split("\n"):
                if ":" in line and line.strip():
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        results.append(
                            DirectSearchResult(
                                file=parts[0],
                                line=int(parts[1]) if parts[1].isdigit() else 1,
                                content=parts[2].strip(),
                                type="grep",
                            )
                        )
            return results
        except Exception:
            return []

    @staticmethod
    def count_pattern(pattern: str, path: str = ".") -> dict[str, int]:
        """Count pattern occurrences by file."""
        try:
            cmd = ["rg", "--count", pattern, path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

            counts = {}
            for line in result.stdout.strip().split("\n"):
                if ":" in line:
                    file_path, count = line.rsplit(":", 1)
                    try:
                        counts[file_path] = int(count)
                    except ValueError:
                        continue

            return counts
        except Exception:
            return {}


class DirectDatabase:
    """Direct database access without connection pooling complexity."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def query(self, sql: str, params: tuple | None = None) -> list[dict[str, Any]]:
        """Execute query directly."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            if sql.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                conn.commit()
                return []

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return []
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    def table_info(self, table_name: str) -> list[dict[str, Any]]:
        """Get table schema information."""
        return self.query(f"PRAGMA table_info({table_name})")

    def list_tables(self) -> list[str]:
        """List all tables in database."""
        rows = self.query("SELECT name FROM sqlite_master WHERE type='table'")
        return [row["name"] for row in rows]


class DirectFileAnalyzer:
    """Direct file analysis without complex dependencies."""

    @staticmethod
    def analyze_python_file(file_path: str) -> dict[str, Any]:
        """Basic Python file analysis."""
        try:
            with open(file_path) as f:
                content = f.read()

            lines = content.split("\n")

            # Count basic metrics
            imports = [
                line
                for line in lines
                if line.strip().startswith("import ")
                or line.strip().startswith("from ")
            ]
            classes = [line for line in lines if line.strip().startswith("class ")]
            functions = [line for line in lines if line.strip().startswith("def ")]
            todos = [line for line in lines if "TODO" in line or "FIXME" in line]

            return {
                "file": file_path,
                "total_lines": len(lines),
                "imports": len(imports),
                "classes": len(classes),
                "functions": len(functions),
                "todos": len(todos),
                "class_names": [
                    cls.split("class ")[1].split("(")[0].split(":")[0].strip()
                    for cls in classes
                ],
                "function_names": [
                    func.split("def ")[1].split("(")[0].strip() for func in functions
                ],
            }

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return {"file": file_path, "error": str(e)}

    @staticmethod
    def find_complex_functions(path: str = ".") -> list[dict[str, Any]]:
        """Find potentially complex functions."""
        search_results = DirectRipgrep.search(r"def .+:", path, ["py"])

        complex_indicators = []
        for result in search_results:
            # Simple heuristics for complexity
            content = result.content.lower()

            complexity_score = 0
            if "for " in content or "while " in content:
                complexity_score += 1
            if "if " in content:
                complexity_score += 1
            if len(content) > 80:  # Long line
                complexity_score += 1

            if complexity_score >= 2:
                complex_indicators.append(
                    {
                        "file": result.file,
                        "line": result.line,
                        "function": result.content,
                        "complexity_score": complexity_score,
                    }
                )

        return sorted(
            complex_indicators, key=lambda x: x["complexity_score"], reverse=True
        )


class DirectTradingAnalyzer:
    """Trading-specific analysis tools."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.search = DirectRipgrep()

    def analyze_wheel_strategy(self) -> DirectAnalysisResult:
        """Direct analysis of wheel strategy implementation."""
        start_time = time.time()

        # Look for wheel strategy patterns
        wheel_files = self.search.search("class.*Wheel", str(self.project_root), ["py"])
        strategy_files = self.search.search(
            "def.*wheel", str(self.project_root), ["py"]
        )

        findings = []
        recommendations = []

        if wheel_files:
            findings.append(f"Found {len(wheel_files)} wheel strategy classes")
            for result in wheel_files[:3]:
                findings.append(f"  • {result.file}:{result.line} - {result.content}")

        if strategy_files:
            findings.append(f"Found {len(strategy_files)} wheel strategy functions")

        # Look for performance issues
        perf_issues = self.search.count_pattern(
            "TODO.*performance|SLOW|bottleneck", str(self.project_root)
        )
        if perf_issues:
            findings.append(f"Found performance TODOs in {len(perf_issues)} files")
            recommendations.append("Review and address performance TODOs")

        # Check for error handling
        error_handling = self.search.search(
            "except|raise|try:",
            str(self.project_root / "src/unity_wheel/strategy"),
            ["py"],
        )
        if error_handling:
            findings.append(f"Found {len(error_handling)} error handling patterns")
        else:
            recommendations.append("Add comprehensive error handling to strategy code")

        return DirectAnalysisResult(
            analysis_type="wheel_strategy",
            findings=findings,
            recommendations=recommendations,
            execution_time=time.time() - start_time,
            files_analyzed=len(wheel_files) + len(strategy_files),
        )

    def analyze_options_pricing(self) -> DirectAnalysisResult:
        """Direct analysis of options pricing components."""
        start_time = time.time()

        # Look for options pricing code
        pricing_files = self.search.search(
            "price|pricing|option.*price", str(self.project_root / "src"), ["py"]
        )
        greeks_files = self.search.search(
            "delta|gamma|theta|vega", str(self.project_root), ["py"]
        )

        findings = []
        recommendations = []

        if pricing_files:
            findings.append(
                f"Found {len(pricing_files)} pricing-related code locations"
            )

        if greeks_files:
            findings.append(f"Found {len(greeks_files)} Greeks calculations")

        # Check for mathematical accuracy concerns
        math_issues = self.search.search(
            "math|numpy|calculation",
            str(self.project_root / "src/unity_wheel/math"),
            ["py"],
        )
        if math_issues:
            findings.append(f"Found {len(math_issues)} mathematical operations")
            recommendations.append(
                "Verify mathematical accuracy and numerical stability"
            )

        return DirectAnalysisResult(
            analysis_type="options_pricing",
            findings=findings,
            recommendations=recommendations,
            execution_time=time.time() - start_time,
            files_analyzed=len(pricing_files) + len(greeks_files),
        )

    def analyze_risk_management(self) -> DirectAnalysisResult:
        """Direct analysis of risk management components."""
        start_time = time.time()

        # Look for risk management code
        risk_files = self.search.search(
            "risk|Risk", str(self.project_root / "src/unity_wheel/risk"), ["py"]
        )
        limit_files = self.search.search(
            "limit|Limit|threshold", str(self.project_root), ["py"]
        )

        findings = []
        recommendations = []

        if risk_files:
            findings.append(f"Found {len(risk_files)} risk management components")

        if limit_files:
            findings.append(f"Found {len(limit_files)} limit/threshold implementations")

        # Check for risk monitoring
        monitoring = self.search.search(
            "monitor|alert|warning",
            str(self.project_root / "src/unity_wheel/risk"),
            ["py"],
        )
        if monitoring:
            findings.append(f"Found {len(monitoring)} monitoring mechanisms")
        else:
            recommendations.append("Implement real-time risk monitoring")

        return DirectAnalysisResult(
            analysis_type="risk_management",
            findings=findings,
            recommendations=recommendations,
            execution_time=time.time() - start_time,
            files_analyzed=len(risk_files) + len(limit_files),
        )


class DirectToolsRunner:
    """Simple runner for direct tools without complex orchestration."""

    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.trading_analyzer = DirectTradingAnalyzer(project_root)

    def run_quick_analysis(self, analysis_type: str) -> dict[str, Any]:
        """Run quick analysis based on type."""
        start_time = time.time()

        try:
            if analysis_type == "wheel_strategy":
                result = self.trading_analyzer.analyze_wheel_strategy()
            elif analysis_type == "options_pricing":
                result = self.trading_analyzer.analyze_options_pricing()
            elif analysis_type == "risk_management":
                result = self.trading_analyzer.analyze_risk_management()
            elif analysis_type == "performance":
                result = self._analyze_performance()
            elif analysis_type == "database":
                result = self._analyze_database()
            else:
                result = self._generic_analysis(analysis_type)

            return {
                "success": True,
                "analysis_type": analysis_type,
                "result": result,
                "total_time": time.time() - start_time,
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type,
                "total_time": time.time() - start_time,
            }

    def _analyze_performance(self) -> DirectAnalysisResult:
        """Performance analysis using direct tools."""
        start_time = time.time()

        # Look for performance patterns
        slow_patterns = DirectRipgrep.search(
            "slow|performance|bottleneck|optimize", self.project_root, ["py"]
        )
        complex_functions = DirectFileAnalyzer.find_complex_functions(self.project_root)

        findings = []
        recommendations = []

        if slow_patterns:
            findings.append(f"Found {len(slow_patterns)} performance-related comments")

        if complex_functions:
            findings.append(
                f"Found {len(complex_functions)} potentially complex functions"
            )
            for func in complex_functions[:3]:
                findings.append(
                    f"  • {func['file']}:{func['line']} (score: {func['complexity_score']})"
                )

        recommendations.extend(
            [
                "Profile identified complex functions",
                "Consider algorithmic optimizations",
                "Add performance monitoring",
            ]
        )

        return DirectAnalysisResult(
            analysis_type="performance",
            findings=findings,
            recommendations=recommendations,
            execution_time=time.time() - start_time,
            files_analyzed=len(slow_patterns) + len(complex_functions),
        )

    def _analyze_database(self) -> DirectAnalysisResult:
        """Database analysis using direct tools."""
        start_time = time.time()

        findings = []
        recommendations = []

        # Look for database files
        db_files = list(Path(self.project_root).glob("**/*.db"))
        duckdb_files = list(Path(self.project_root).glob("**/*.duckdb"))

        if db_files or duckdb_files:
            findings.append(f"Found {len(db_files + duckdb_files)} database files")

        # Try to analyze main database
        main_db = Path(self.project_root) / "data" / "wheel_trading_master.duckdb"
        if main_db.exists():
            try:
                db = DirectDatabase(str(main_db))
                tables = db.list_tables()
                findings.append(f"Main database has {len(tables)} tables")

                if tables:
                    findings.append(f"Tables: {', '.join(tables[:5])}")
                    recommendations.append("Verify database schema integrity")

            except Exception as e:
                findings.append(f"Database access failed: {e}")
                recommendations.append("Fix database connectivity issues")

        return DirectAnalysisResult(
            analysis_type="database",
            findings=findings,
            recommendations=recommendations,
            execution_time=time.time() - start_time,
            files_analyzed=len(db_files + duckdb_files),
        )

    def _generic_analysis(self, query: str) -> DirectAnalysisResult:
        """Generic analysis for unknown query types."""
        start_time = time.time()

        # Simple keyword search
        results = DirectRipgrep.search(query, self.project_root, ["py", "md"])

        findings = [f"Found {len(results)} matches for '{query}'"]

        if results:
            file_counts = {}
            for result in results:
                file_counts[result.file] = file_counts.get(result.file, 0) + 1

            top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            findings.append(
                f"Top files: {', '.join([f'{file}({count})' for file, count in top_files])}"
            )

        return DirectAnalysisResult(
            analysis_type="generic",
            findings=findings,
            recommendations=["Review search results for relevant patterns"],
            execution_time=time.time() - start_time,
            files_analyzed=len(set(r.file for r in results)),
        )

    def quick_solve(self, query: str) -> dict[str, Any]:
        """Quick solve for common query patterns."""
        query_lower = query.lower()

        # Route to appropriate analysis
        if "wheel" in query_lower and "strategy" in query_lower:
            return self.run_quick_analysis("wheel_strategy")
        elif "options" in query_lower and (
            "pricing" in query_lower or "price" in query_lower
        ):
            return self.run_quick_analysis("options_pricing")
        elif "risk" in query_lower:
            return self.run_quick_analysis("risk_management")
        elif "performance" in query_lower or "optimize" in query_lower:
            return self.run_quick_analysis("performance")
        elif "database" in query_lower or "db" in query_lower:
            return self.run_quick_analysis("database")
        else:
            # Extract key terms for generic analysis
            key_terms = [word for word in query_lower.split() if len(word) > 3]
            if key_terms:
                return self.run_quick_analysis(key_terms[0])
            else:
                return {"success": False, "error": "Could not understand query"}


# Convenience functions
def direct_search(pattern: str, path: str = ".") -> list[DirectSearchResult]:
    """Direct search bypassing all broken layers."""
    return DirectRipgrep.search(pattern, path)


def direct_analyze(analysis_type: str, project_root: str = ".") -> dict[str, Any]:
    """Direct analysis bypassing bolt orchestration."""
    runner = DirectToolsRunner(project_root)
    return runner.run_quick_analysis(analysis_type)


def direct_solve(query: str, project_root: str = ".") -> dict[str, Any]:
    """Direct query resolution bypassing bolt system."""
    runner = DirectToolsRunner(project_root)
    return runner.quick_solve(query)


# CLI entry point
if __name__ == "__main__":
    import sys

    def main():
        if len(sys.argv) < 2:
            print("Usage: python bolt_direct_tools.py 'query' [project_root]")
            print("Examples:")
            print("  python bolt_direct_tools.py 'wheel strategy analysis'")
            print("  python bolt_direct_tools.py 'performance issues' /path/to/project")
            return

        query = sys.argv[1]
        project_root = sys.argv[2] if len(sys.argv) > 2 else "."

        print("🔧 Direct Tools - Bypassing Bolt Integration")
        print("=" * 50)

        result = direct_solve(query, project_root)

        print(f"Query: {query}")
        print(f"Success: {result['success']}")

        if result["success"]:
            analysis_result = result["result"]
            print(f"Analysis Type: {analysis_result.analysis_type}")
            print(f"Execution Time: {analysis_result.execution_time:.2f}s")
            print(f"Files Analyzed: {analysis_result.files_analyzed}")

            print("\nFindings:")
            for finding in analysis_result.findings:
                print(f"  • {finding}")

            print("\nRecommendations:")
            for rec in analysis_result.recommendations:
                print(f"  • {rec}")
        else:
            print(f"Error: {result['error']}")

    main()
