#!/usr/bin/env python3
"""AI Development Assistant for Unity Wheel Trading Bot.

Provides structured interfaces and automation tools specifically designed
for autonomous AI developers like Claude Code and Codex.
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.unity_wheel.utils.logging import StructuredLogger
from src.config.loader import get_config
import logging

logger = StructuredLogger(logging.getLogger(__name__))


class AIDevAssistant:
    """Development assistant optimized for AI autonomous developers."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.src_path = self.project_root / "src"
        self.tests_path = self.project_root / "tests"
        
        logger.info("ai_dev_assistant_initialized")
    
    def execute_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Execute command and return structured result for AI consumption."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                check=False
            )
            
            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "exit_code": -1,
                "error": str(e),
                "command": " ".join(cmd),
                "timestamp": datetime.now().isoformat()
            }
    
    def validate_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check with structured output."""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {},
            "issues": [],
            "recommendations": []
        }
        
        # Python version check
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        health_report["checks"]["python_version"] = {
            "status": "pass" if sys.version_info >= (3, 8) else "fail",
            "value": python_version,
            "requirement": ">=3.8"
        }
        
        # Project structure validation
        required_paths = [
            "src/unity_wheel",
            "tests",
            "config.yaml",
            "pyproject.toml"
        ]
        
        structure_issues = []
        for path in required_paths:
            full_path = self.project_root / path
            if not full_path.exists():
                structure_issues.append(f"Missing: {path}")
        
        health_report["checks"]["project_structure"] = {
            "status": "pass" if not structure_issues else "fail",
            "issues": structure_issues
        }
        
        # Configuration validation
        try:
            config = get_config()
            health_report["checks"]["configuration"] = {
                "status": "pass",
                "environment": getattr(config.metadata, 'environment', 'unknown'),
                "unity_ticker": getattr(config.unity, 'ticker', 'unknown')
            }
        except Exception as e:
            health_report["checks"]["configuration"] = {
                "status": "fail",
                "error": str(e)
            }
            health_report["issues"].append("Configuration loading failed")
        
        # Import validation
        critical_imports = [
            "src.unity_wheel.api.advisor",
            "src.unity_wheel.math.options",
            "src.unity_wheel.strategy.wheel"
        ]
        
        import_issues = []
        for module in critical_imports:
            try:
                __import__(module)
            except ImportError as e:
                import_issues.append(f"Cannot import {module}: {e}")
        
        health_report["checks"]["imports"] = {
            "status": "pass" if not import_issues else "fail",
            "issues": import_issues
        }
        
        # Determine overall status
        failed_checks = [k for k, v in health_report["checks"].items() if v["status"] == "fail"]
        if failed_checks:
            health_report["overall_status"] = "unhealthy"
            health_report["issues"].extend([f"Failed check: {check}" for check in failed_checks])
        
        logger.info("system_health_validated", extra={"status": health_report["overall_status"]})
        return health_report
    
    def run_test_suite(self, filter_pattern: Optional[str] = None, 
                      include_slow: bool = False) -> Dict[str, Any]:
        """Run test suite with structured output for AI analysis."""
        cmd = ["python", "-m", "pytest", "--tb=short", "--json-report", "--json-report-file=test-results.json"]
        
        if filter_pattern:
            cmd.extend(["-k", filter_pattern])
        
        if not include_slow:
            cmd.extend(["-m", "not slow"])
        
        result = self.execute_command(cmd)
        
        # Parse JSON test results if available
        test_report_path = self.project_root / "test-results.json"
        test_details = {}
        
        if test_report_path.exists():
            try:
                with open(test_report_path) as f:
                    test_details = json.load(f)
            except Exception as e:
                test_details = {"parse_error": str(e)}
        
        return {
            "execution": result,
            "test_details": test_details,
            "summary": {
                "passed": result["success"],
                "filter_used": filter_pattern,
                "included_slow_tests": include_slow
            }
        }
    
    def analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality with structured metrics for AI optimization."""
        quality_report = {
            "timestamp": datetime.now().isoformat(),
            "tools": {},
            "overall_score": 0,
            "actionable_items": []
        }
        
        # Black formatting check
        black_result = self.execute_command(["python", "-m", "black", "--check", "--diff", "src", "tests"])
        quality_report["tools"]["black"] = {
            "status": "pass" if black_result["success"] else "fail",
            "issues_found": not black_result["success"],
            "auto_fixable": True,
            "fix_command": "python -m black src tests"
        }
        
        # Flake8 linting
        flake8_result = self.execute_command(["python", "-m", "flake8", "--format=json", "src", "tests"])
        flake8_issues = []
        if flake8_result["stderr"]:
            try:
                # Parse flake8 JSON output if available
                lines = flake8_result["stderr"].split('\n')
                for line in lines:
                    if line.strip():
                        flake8_issues.append(line)
            except:
                flake8_issues = ["Parse error in flake8 output"]
        
        quality_report["tools"]["flake8"] = {
            "status": "pass" if flake8_result["success"] else "fail",
            "issues": flake8_issues,
            "issue_count": len(flake8_issues)
        }
        
        # MyPy type checking
        mypy_result = self.execute_command(["python", "-m", "mypy", "--json-report", "mypy-report", "src"])
        quality_report["tools"]["mypy"] = {
            "status": "pass" if mypy_result["success"] else "fail",
            "output": mypy_result["stderr"]
        }
        
        # Calculate overall score (0-100)
        tool_scores = []
        for tool, data in quality_report["tools"].items():
            tool_scores.append(100 if data["status"] == "pass" else 0)
        
        quality_report["overall_score"] = sum(tool_scores) / len(tool_scores) if tool_scores else 0
        
        # Generate actionable items
        if quality_report["tools"]["black"]["status"] == "fail":
            quality_report["actionable_items"].append({
                "priority": "high",
                "action": "format_code",
                "command": "python -m black src tests",
                "description": "Auto-format code with Black"
            })
        
        if quality_report["tools"]["flake8"]["issue_count"] > 0:
            quality_report["actionable_items"].append({
                "priority": "medium",
                "action": "fix_linting",
                "description": f"Fix {quality_report['tools']['flake8']['issue_count']} linting issues"
            })
        
        return quality_report
    
    def generate_performance_profile(self) -> Dict[str, Any]:
        """Generate performance profile for critical system components."""
        profile_report = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Run performance benchmarks
        benchmark_result = self.execute_command([
            "python", "-m", "pytest", 
            "tests/test_performance_benchmarks.py", 
            "-v", "--tb=short"
        ])
        
        profile_report["components"]["benchmarks"] = {
            "status": "pass" if benchmark_result["success"] else "fail",
            "output": benchmark_result["stdout"]
        }
        
        # Analyze benchmark output for timing data
        if benchmark_result["stdout"]:
            lines = benchmark_result["stdout"].split('\n')
            timing_data = []
            
            for line in lines:
                if 'ms' in line and 'test_' in line:
                    timing_data.append(line.strip())
            
            profile_report["components"]["timing_analysis"] = timing_data
        
        # Memory usage analysis
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            profile_report["components"]["memory"] = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "status": "healthy" if memory_info.rss / 1024 / 1024 < 200 else "warning"
            }
        except ImportError:
            profile_report["components"]["memory"] = {"error": "psutil not available"}
        
        return profile_report
    
    def analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage with AI-actionable insights."""
        coverage_result = self.execute_command([
            "python", "-m", "pytest", 
            "--cov=src", 
            "--cov-report=json",
            "--cov-report=term-missing"
        ])
        
        coverage_report = {
            "timestamp": datetime.now().isoformat(),
            "execution": coverage_result,
            "metrics": {},
            "gaps": [],
            "recommendations": []
        }
        
        # Parse coverage.json if available
        coverage_json_path = self.project_root / "coverage.json"
        if coverage_json_path.exists():
            try:
                with open(coverage_json_path) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                coverage_report["metrics"]["total_coverage"] = total_coverage
                
                # Identify files with low coverage
                files = coverage_data.get("files", {})
                low_coverage_files = []
                
                for filepath, file_data in files.items():
                    file_coverage = file_data.get("summary", {}).get("percent_covered", 0)
                    if file_coverage < 80:  # Threshold for concern
                        low_coverage_files.append({
                            "file": filepath,
                            "coverage": file_coverage,
                            "missing_lines": file_data.get("missing_lines", [])
                        })
                
                coverage_report["gaps"] = low_coverage_files
                
                # Generate recommendations
                if total_coverage < 80:
                    coverage_report["recommendations"].append({
                        "priority": "high",
                        "action": "increase_coverage",
                        "target": "80%",
                        "current": f"{total_coverage:.1f}%"
                    })
                
            except Exception as e:
                coverage_report["metrics"]["parse_error"] = str(e)
        
        return coverage_report
    
    def create_optimization_plan(self) -> Dict[str, Any]:
        """Create a structured optimization plan based on current system analysis."""
        # Gather all analysis data
        health = self.validate_system_health()
        quality = self.analyze_code_quality()
        performance = self.generate_performance_profile()
        coverage = self.analyze_test_coverage()
        
        optimization_plan = {
            "timestamp": datetime.now().isoformat(),
            "current_state": {
                "health_status": health["overall_status"],
                "quality_score": quality["overall_score"],
                "test_coverage": coverage.get("metrics", {}).get("total_coverage", 0)
            },
            "immediate_actions": [],
            "medium_term_goals": [],
            "long_term_objectives": [],
            "automation_opportunities": []
        }
        
        # Immediate actions (can be automated)
        if health["overall_status"] == "unhealthy":
            optimization_plan["immediate_actions"].append({
                "priority": "critical",
                "action": "fix_health_issues",
                "issues": health["issues"],
                "automated": False
            })
        
        for item in quality.get("actionable_items", []):
            if item.get("command"):
                optimization_plan["immediate_actions"].append({
                    "priority": item["priority"],
                    "action": item["action"],
                    "command": item["command"],
                    "automated": True
                })
        
        # Medium-term goals
        if coverage.get("metrics", {}).get("total_coverage", 0) < 80:
            optimization_plan["medium_term_goals"].append({
                "goal": "achieve_80_percent_coverage",
                "current": coverage.get("metrics", {}).get("total_coverage", 0),
                "target": 80,
                "files_needing_attention": len(coverage.get("gaps", []))
            })
        
        # Automation opportunities
        optimization_plan["automation_opportunities"].extend([
            {
                "area": "pre_commit_hooks",
                "description": "Automate code formatting and linting before commits",
                "impact": "high"
            },
            {
                "area": "continuous_testing",
                "description": "Automated test execution on code changes",
                "impact": "medium"
            },
            {
                "area": "performance_monitoring",
                "description": "Automated performance regression detection",
                "impact": "medium"
            }
        ])
        
        logger.info("optimization_plan_created", extra={
            "immediate_actions": len(optimization_plan["immediate_actions"]),
            "automation_opportunities": len(optimization_plan["automation_opportunities"])
        })
        
        return optimization_plan
    
    def execute_automated_fixes(self) -> Dict[str, Any]:
        """Execute all automated fixes that can be safely applied."""
        fix_results = {
            "timestamp": datetime.now().isoformat(),
            "fixes_applied": [],
            "errors": [],
            "summary": {}
        }
        
        # Auto-format code
        black_result = self.execute_command(["python", "-m", "black", "src", "tests"])
        fix_results["fixes_applied"].append({
            "fix": "code_formatting",
            "success": black_result["success"],
            "command": "python -m black src tests"
        })
        
        if not black_result["success"]:
            fix_results["errors"].append(f"Black formatting failed: {black_result['stderr']}")
        
        # Sort imports
        isort_result = self.execute_command(["python", "-m", "isort", "src", "tests"])
        fix_results["fixes_applied"].append({
            "fix": "import_sorting",
            "success": isort_result["success"],
            "command": "python -m isort src tests"
        })
        
        if not isort_result["success"]:
            fix_results["errors"].append(f"Import sorting failed: {isort_result['stderr']}")
        
        # Summary
        successful_fixes = sum(1 for fix in fix_results["fixes_applied"] if fix["success"])
        fix_results["summary"] = {
            "total_attempted": len(fix_results["fixes_applied"]),
            "successful": successful_fixes,
            "failed": len(fix_results["fixes_applied"]) - successful_fixes
        }
        
        logger.info("automated_fixes_completed", extra=fix_results["summary"])
        return fix_results
    
    def export_system_state(self) -> Dict[str, Any]:
        """Export complete system state for AI analysis and decision-making."""
        system_state = {
            "timestamp": datetime.now().isoformat(),
            "health": self.validate_system_health(),
            "quality": self.analyze_code_quality(),
            "performance": self.generate_performance_profile(),
            "coverage": self.analyze_test_coverage(),
            "optimization_plan": self.create_optimization_plan()
        }
        
        # Save to file for AI consumption
        export_path = self.project_root / "ai-system-state.json"
        with open(export_path, 'w') as f:
            json.dump(system_state, f, indent=2)
        
        logger.info("system_state_exported", extra={"path": str(export_path)})
        return system_state


def main():
    """Main entry point for AI development assistant."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Development Assistant for Unity Wheel Trading Bot")
    parser.add_argument('action', choices=[
        'health', 'quality', 'performance', 'coverage', 'optimize', 'fix', 'export'
    ], help='Action to perform')
    
    parser.add_argument('--output', '-o', type=str, help='Output file for results')
    parser.add_argument('--filter', '-f', type=str, help='Filter for tests')
    
    args = parser.parse_args()
    
    assistant = AIDevAssistant()
    
    # Execute requested action
    if args.action == 'health':
        result = assistant.validate_system_health()
    elif args.action == 'quality':
        result = assistant.analyze_code_quality()
    elif args.action == 'performance':
        result = assistant.generate_performance_profile()
    elif args.action == 'coverage':
        result = assistant.analyze_test_coverage()
    elif args.action == 'optimize':
        result = assistant.create_optimization_plan()
    elif args.action == 'fix':
        result = assistant.execute_automated_fixes()
    elif args.action == 'export':
        result = assistant.export_system_state()
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()