#!/usr/bin/env python3
"""
Execution Monitor - Actually monitors code execution and captures failures
This watches when code runs, captures errors from logs, and triggers self-correction
"""

import subprocess
import time
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from meta_prime import MetaPrime


@dataclass
class ExecutionResult:
    """Result of code execution monitoring"""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    timestamp: float
    success: bool


class ExecutionMonitor:
    """Monitors actual code execution and captures failures"""
    
    def __init__(self):
        self.meta_prime = MetaPrime()
        self.execution_history = []
        self.error_patterns = {}
        
        print("🔍 Execution Monitor Active - Watching code execution...")
        
    def monitor_command_execution(self, command: str, cwd: str = ".") -> ExecutionResult:
        """Monitor execution of a command and capture all output"""
        
        print(f"⚡ Executing: {command}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=30
            )
            
            execution_time = time.time() - start_time
            success = result.returncode == 0
            
            exec_result = ExecutionResult(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                timestamp=time.time(),
                success=success
            )
            
            # Record execution
            self._record_execution(exec_result)
            
            # Analyze for errors
            if not success:
                self._analyze_execution_failure(exec_result)
                
            return exec_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            exec_result = ExecutionResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="Command timed out",
                execution_time=execution_time,
                timestamp=time.time(),
                success=False
            )
            
            self._record_execution(exec_result)
            return exec_result
            
    def _record_execution(self, result: ExecutionResult):
        """Record execution result in meta system"""
        
        self.execution_history.append(result)
        
        # Record observation
        self.meta_prime.observe("code_execution", {
            "command": result.command,
            "success": result.success,
            "exit_code": result.exit_code,
            "execution_time": result.execution_time,
            "has_stdout": len(result.stdout) > 0,
            "has_stderr": len(result.stderr) > 0,
            "stderr_preview": result.stderr[:200] if result.stderr else "",
            "timestamp": result.timestamp
        })
        
        if result.success:
            print(f"✅ Success: {result.command} ({result.execution_time:.1f}s)")
        else:
            print(f"❌ Failed: {result.command} (exit {result.exit_code})")
            if result.stderr:
                print(f"   Error: {result.stderr[:100]}...")
                
    def _analyze_execution_failure(self, result: ExecutionResult):
        """Analyze execution failure and identify patterns"""
        
        error_analysis = {
            "command": result.command,
            "exit_code": result.exit_code,
            "error_type": self._classify_error(result.stderr),
            "potential_fix": self._suggest_fix(result.stderr),
            "timestamp": result.timestamp
        }
        
        # Record error analysis
        self.meta_prime.observe("execution_failure_analysis", error_analysis)
        
        # Update error patterns
        error_type = error_analysis["error_type"]
        if error_type not in self.error_patterns:
            self.error_patterns[error_type] = {
                "count": 0,
                "examples": [],
                "fixes_tried": []
            }
            
        pattern = self.error_patterns[error_type]
        pattern["count"] += 1
        pattern["examples"].append({
            "command": result.command,
            "error": result.stderr[:200],
            "timestamp": result.timestamp
        })
        
        # Keep only recent examples
        if len(pattern["examples"]) > 5:
            pattern["examples"] = pattern["examples"][-5:]
            
        print(f"🔍 Error pattern identified: {error_type} (seen {pattern['count']} times)")
        
    def _classify_error(self, stderr: str) -> str:
        """Classify the type of error from stderr"""
        
        error_classifications = [
            ("import_error", ["ModuleNotFoundError", "ImportError", "No module named"]),
            ("syntax_error", ["SyntaxError", "invalid syntax", "IndentationError"]),
            ("type_error", ["TypeError", "AttributeError", "'NoneType'"]),
            ("file_error", ["FileNotFoundError", "PermissionError", "No such file"]),
            ("network_error", ["ConnectionError", "timeout", "Unable to connect"]),
            ("database_error", ["sqlite3.OperationalError", "database", "SQL"]),
            ("api_error", ["401", "403", "404", "500", "API"]),
            ("dependency_error", ["pip", "install", "requirements"]),
        ]
        
        stderr_lower = stderr.lower()
        
        for error_type, keywords in error_classifications:
            if any(keyword.lower() in stderr_lower for keyword in keywords):
                return error_type
                
        return "unknown_error"
        
    def _suggest_fix(self, stderr: str) -> str:
        """Suggest a potential fix based on the error"""
        
        error_type = self._classify_error(stderr)
        
        fix_suggestions = {
            "import_error": "Install missing dependency: pip install <module>",
            "syntax_error": "Check Python syntax, indentation, and brackets",
            "type_error": "Check variable types and method calls",
            "file_error": "Verify file paths exist and permissions are correct",
            "network_error": "Check internet connection and API endpoints",
            "database_error": "Verify database file exists and permissions",
            "api_error": "Check API keys, endpoints, and rate limits",
            "dependency_error": "Run: pip install -r requirements.txt"
        }
        
        return fix_suggestions.get(error_type, "Review error message and check documentation")
        
    def monitor_python_script(self, script_path: str) -> ExecutionResult:
        """Monitor execution of a Python script"""
        
        return self.monitor_command_execution(f"python {script_path}")
        
    def monitor_test_execution(self, test_command: str = "pytest") -> ExecutionResult:
        """Monitor test execution"""
        
        return self.monitor_command_execution(test_command)
        
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of execution failures"""
        
        failed_executions = [exec for exec in self.execution_history if not exec.success]
        
        if not failed_executions:
            return {"total_failures": 0, "error_patterns": {}}
            
        # Group by error type
        by_error_type = {}
        for exec_result in failed_executions:
            error_type = self._classify_error(exec_result.stderr)
            if error_type not in by_error_type:
                by_error_type[error_type] = []
            by_error_type[error_type].append(exec_result)
            
        return {
            "total_failures": len(failed_executions),
            "unique_error_types": len(by_error_type),
            "error_patterns": {
                error_type: {
                    "count": len(results),
                    "most_recent": results[-1].timestamp,
                    "commands": list(set(r.command for r in results))
                }
                for error_type, results in by_error_type.items()
            },
            "failure_rate": len(failed_executions) / len(self.execution_history) if self.execution_history else 0
        }


class AutoCorrector:
    """Automatically attempts to fix execution failures"""
    
    def __init__(self, execution_monitor: ExecutionMonitor):
        self.execution_monitor = execution_monitor
        self.meta_prime = MetaPrime()
        self.correction_attempts = []
        
    def attempt_auto_correction(self, failed_result: ExecutionResult) -> bool:
        """Attempt to automatically correct a failed execution"""
        
        error_type = self.execution_monitor._classify_error(failed_result.stderr)
        
        print(f"🔧 Attempting auto-correction for {error_type}...")
        
        correction_made = False
        
        if error_type == "import_error":
            correction_made = self._fix_import_error(failed_result)
        elif error_type == "dependency_error":
            correction_made = self._fix_dependency_error(failed_result)
        elif error_type == "file_error":
            correction_made = self._fix_file_error(failed_result)
        elif error_type == "syntax_error":
            correction_made = self._fix_syntax_error(failed_result)
        else:
            print(f"⏳ No auto-correction available for {error_type}")
            
        # Record correction attempt
        self.correction_attempts.append({
            "original_error": error_type,
            "command": failed_result.command,
            "correction_made": correction_made,
            "timestamp": time.time()
        })
        
        self.meta_prime.observe("auto_correction_attempt", {
            "error_type": error_type,
            "command": failed_result.command,
            "correction_successful": correction_made,
            "correction_method": f"fix_{error_type}",
            "timestamp": time.time()
        })
        
        return correction_made
        
    def _fix_import_error(self, failed_result: ExecutionResult) -> bool:
        """Try to fix import errors by installing missing packages"""
        
        # Extract module name from error
        import_match = re.search(r"No module named '([^']+)'", failed_result.stderr)
        if not import_match:
            return False
            
        module_name = import_match.group(1)
        
        print(f"🔧 Installing missing module: {module_name}")
        
        # Try to install the module
        install_result = self.execution_monitor.monitor_command_execution(f"pip install {module_name}")
        
        if install_result.success:
            print(f"✅ Successfully installed {module_name}")
            
            # Try running the original command again
            print(f"🔄 Retrying original command: {failed_result.command}")
            retry_result = self.execution_monitor.monitor_command_execution(failed_result.command)
            
            return retry_result.success
        else:
            print(f"❌ Failed to install {module_name}")
            return False
            
    def _fix_dependency_error(self, failed_result: ExecutionResult) -> bool:
        """Try to fix dependency errors"""
        
        # Check if requirements.txt exists
        if Path("requirements.txt").exists():
            print("🔧 Installing requirements.txt...")
            install_result = self.execution_monitor.monitor_command_execution("pip install -r requirements.txt")
            
            if install_result.success:
                # Retry original command
                retry_result = self.execution_monitor.monitor_command_execution(failed_result.command)
                return retry_result.success
                
        return False
        
    def _fix_file_error(self, failed_result: ExecutionResult) -> bool:
        """Try to fix file errors by creating missing directories"""
        
        # Extract file path from error
        file_match = re.search(r"No such file or directory: '([^']+)'", failed_result.stderr)
        if not file_match:
            return False
            
        file_path = Path(file_match.group(1))
        
        # Try to create parent directory
        if not file_path.parent.exists():
            print(f"🔧 Creating missing directory: {file_path.parent}")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Retry original command
            retry_result = self.execution_monitor.monitor_command_execution(failed_result.command)
            return retry_result.success
            
        return False
        
    def _fix_syntax_error(self, failed_result: ExecutionResult) -> bool:
        """Try to fix basic syntax errors"""
        
        # For now, just log - actual syntax fixing would need AST manipulation
        print("⏳ Syntax error auto-correction not implemented yet")
        print("   Consider using an IDE with syntax checking")
        
        return False


def monitor_project_execution():
    """Monitor execution of common project commands"""
    
    monitor = ExecutionMonitor()
    corrector = AutoCorrector(monitor)
    
    print("🔄 Monitoring Project Execution...")
    
    # Common commands to monitor
    commands_to_test = [
        "python -c 'print(\"Hello World\")'",  # Basic Python
        "python -m pip list",  # Check pip
        "python run.py --help",  # Main script
        "pytest --version"  # Test framework
    ]
    
    results = []
    corrections_made = 0
    
    for command in commands_to_test:
        print(f"\n📋 Testing: {command}")
        
        result = monitor.monitor_command_execution(command)
        results.append(result)
        
        if not result.success:
            # Attempt auto-correction
            corrected = corrector.attempt_auto_correction(result)
            if corrected:
                corrections_made += 1
                print("✅ Auto-correction successful!")
            else:
                print("❌ Auto-correction failed")
                
    # Summary
    successful = len([r for r in results if r.success])
    print(f"\n📊 Execution Summary:")
    print(f"   Commands tested: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(results) - successful}")
    print(f"   Auto-corrections made: {corrections_made}")
    
    # Error patterns
    failure_summary = monitor.get_failure_summary()
    if failure_summary["total_failures"] > 0:
        print(f"\n🔍 Error Patterns:")
        for error_type, info in failure_summary["error_patterns"].items():
            print(f"   • {error_type}: {info['count']} occurrences")
            
    return {
        "results": results,
        "corrections_made": corrections_made,
        "failure_summary": failure_summary
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Monitor specific command
        command = " ".join(sys.argv[1:])
        monitor = ExecutionMonitor()
        result = monitor.monitor_command_execution(command)
        
        if not result.success:
            corrector = AutoCorrector(monitor)
            corrector.attempt_auto_correction(result)
    else:
        # Monitor project
        monitor_project_execution()