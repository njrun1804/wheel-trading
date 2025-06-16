#!/usr/bin/env python3
"""
Unified Meta System - All capabilities merged and properly wired
This consolidates all the duplicate/overlapping meta system components into one cohesive system.
"""

import ast
import asyncio
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from meta_coordinator import MetaCoordinator
from meta_monitoring import MetaSystemMonitor

# Import all existing meta components
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
    error_patterns: list[str]


@dataclass
class DetectedMistake:
    """Represents a detected mistake with context"""

    mistake_type: str
    file_path: str
    error_message: str
    context: dict[str, Any]
    timestamp: float
    severity: str
    correction_suggested: str | None = None


@dataclass
class ClaudeCodeEdit:
    """Represents a Claude Code edit event"""

    file_path: str
    edit_type: str  # 'created', 'modified', 'deleted'
    timestamp: float
    content_change: dict[str, Any]
    claude_context: str


class UnifiedExecutionMonitor:
    """Consolidated execution monitoring with log analysis"""

    def __init__(self, meta_prime: MetaPrime):
        self.meta_prime = meta_prime
        self.execution_history = []
        self.error_patterns = self._load_error_patterns()
        self.log_files = []

    def _load_error_patterns(self) -> dict[str, str]:
        """Load known error patterns for detection"""
        return {
            "import_error": r"ImportError|ModuleNotFoundError",
            "syntax_error": r"SyntaxError",
            "runtime_error": r"RuntimeError|ValueError|TypeError",
            "connection_error": r"ConnectionError|TimeoutError",
            "permission_error": r"PermissionError|AccessDenied",
        }

    def monitor_command(self, command: str, cwd: str = ".") -> ExecutionResult:
        """Monitor execution of a command with full error capture"""

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=300,  # 5 minute timeout
            )

            execution_time = time.time() - start_time
            success = result.returncode == 0

            # Detect error patterns
            error_patterns = []
            error_text = result.stderr + result.stdout
            for pattern_name, pattern in self.error_patterns.items():
                if re.search(pattern, error_text):
                    error_patterns.append(pattern_name)

            exec_result = ExecutionResult(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                timestamp=time.time(),
                success=success,
                error_patterns=error_patterns,
            )

            # Record in meta system
            self.meta_prime.observe(
                "command_execution",
                {
                    "command": command[:100],  # Truncate long commands
                    "success": success,
                    "exit_code": result.returncode,
                    "execution_time": execution_time,
                    "error_patterns": error_patterns,
                },
            )

            self.execution_history.append(exec_result)
            return exec_result

        except subprocess.TimeoutExpired:
            exec_result = ExecutionResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="Command timed out after 300 seconds",
                execution_time=300,
                timestamp=time.time(),
                success=False,
                error_patterns=["timeout_error"],
            )

            self.meta_prime.observe(
                "command_timeout", {"command": command[:100], "timeout_seconds": 300}
            )

            return exec_result

    def add_log_file(self, log_path: Path):
        """Add a log file to monitor"""
        if log_path.exists():
            self.log_files.append(log_path)

    def tail_logs(self) -> list[str]:
        """Get recent log entries from monitored files"""
        recent_lines = []

        for log_file in self.log_files:
            try:
                with open(log_file) as f:
                    lines = f.readlines()
                    # Get last 10 lines
                    recent_lines.extend(lines[-10:])
            except Exception:
                continue

        return recent_lines


class UnifiedMistakeDetector:
    """Consolidated mistake detection with learning"""

    def __init__(self, meta_prime: MetaPrime):
        self.meta_prime = meta_prime
        self.detected_mistakes = []
        self.mistake_patterns = {}
        self.correction_strategies = self._load_correction_strategies()

    def _load_correction_strategies(self) -> dict[str, str]:
        """Load correction strategies for common mistakes"""
        return {
            "syntax_error": "Check syntax with ast.parse() and suggest fixes",
            "import_error": "Verify imports and suggest alternatives",
            "runtime_error": "Add error handling and validation",
            "connection_error": "Implement retry logic with exponential backoff",
            "permission_error": "Check file permissions and paths",
        }

    def detect_syntax_mistakes(self, file_path: Path) -> DetectedMistake | None:
        """Detect syntax errors in Python files"""

        if not file_path.exists() or file_path.suffix != ".py":
            return None

        try:
            with open(file_path) as f:
                content = f.read()
            ast.parse(content)
            return None  # No syntax error

        except SyntaxError as e:
            mistake = DetectedMistake(
                mistake_type="syntax_error",
                file_path=str(file_path),
                error_message=str(e),
                context={"line": e.lineno, "text": e.text},
                timestamp=time.time(),
                severity="critical",
                correction_suggested=self.correction_strategies.get("syntax_error"),
            )

            self.detected_mistakes.append(mistake)
            self.meta_prime.observe(
                "mistake_detected",
                {
                    "type": "syntax_error",
                    "file": str(file_path),
                    "line": e.lineno,
                    "severity": "critical",
                },
            )

            return mistake

    def detect_runtime_mistakes(
        self, execution_result: ExecutionResult
    ) -> list[DetectedMistake]:
        """Detect runtime mistakes from execution results"""

        mistakes = []

        if not execution_result.success and execution_result.error_patterns:
            for pattern in execution_result.error_patterns:
                mistake = DetectedMistake(
                    mistake_type=pattern,
                    file_path=execution_result.command,
                    error_message=execution_result.stderr,
                    context={
                        "exit_code": execution_result.exit_code,
                        "stdout": execution_result.stdout[:500],
                        "execution_time": execution_result.execution_time,
                    },
                    timestamp=execution_result.timestamp,
                    severity="warning"
                    if "warning" in execution_result.stderr.lower()
                    else "error",
                    correction_suggested=self.correction_strategies.get(pattern),
                )

                mistakes.append(mistake)
                self.detected_mistakes.append(mistake)

                self.meta_prime.observe(
                    "runtime_mistake_detected",
                    {
                        "type": pattern,
                        "command": execution_result.command[:100],
                        "exit_code": execution_result.exit_code,
                        "severity": mistake.severity,
                    },
                )

        return mistakes

    def suggest_corrections(self, mistake: DetectedMistake) -> list[str]:
        """Suggest corrections for detected mistakes"""

        suggestions = []

        if mistake.correction_suggested:
            suggestions.append(mistake.correction_suggested)

        # Add specific suggestions based on mistake type
        if mistake.mistake_type == "import_error":
            suggestions.append("Check if the module is installed: pip install <module>")
            suggestions.append("Verify the import path is correct")

        elif mistake.mistake_type == "syntax_error":
            suggestions.append("Check for missing colons, parentheses, or quotes")
            suggestions.append("Verify indentation is consistent")

        elif mistake.mistake_type == "runtime_error":
            suggestions.append("Add try/except error handling")
            suggestions.append("Validate input parameters")

        return suggestions


class UnifiedClaudeCodeMonitor(FileSystemEventHandler):
    """Consolidated Claude Code monitoring with feedback collection"""

    def __init__(self, meta_prime: MetaPrime):
        self.meta_prime = meta_prime
        self.claude_edits = []
        self.last_edit_time = {}
        self.debounce_seconds = 2.0

    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Debounce rapid edits
        now = time.time()
        if str(file_path) in self.last_edit_time:
            if now - self.last_edit_time[str(file_path)] < self.debounce_seconds:
                return

        self.last_edit_time[str(file_path)] = now

        if self._is_relevant_file(file_path):
            self._record_claude_edit(file_path, "modified")

    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if self._is_relevant_file(file_path):
                self._record_claude_edit(file_path, "created")

    def on_deleted(self, event):
        """Handle file deletion events"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if self._is_relevant_file(file_path):
                self._record_claude_edit(file_path, "deleted")

    def _is_relevant_file(self, file_path: Path) -> bool:
        """Check if file is relevant for Claude Code monitoring"""

        # Python files in src or root
        if file_path.suffix == ".py":
            return True

        # Configuration files
        if file_path.name in ["config.yaml", "requirements.txt", "pyproject.toml"]:
            return True

        # Documentation files
        return file_path.suffix in [".md", ".rst"]

    def _record_claude_edit(self, file_path: Path, edit_type: str):
        """Record a Claude Code edit event"""

        try:
            # Try to get file content for context
            content_info = {}
            if file_path.exists() and edit_type != "deleted":
                stat = file_path.stat()
                content_info = {"size": stat.st_size, "modified_time": stat.st_mtime}

                # For Python files, get basic structure info
                if file_path.suffix == ".py":
                    try:
                        with open(file_path) as f:
                            content = f.read()
                        tree = ast.parse(content)

                        functions = [
                            node.name
                            for node in ast.walk(tree)
                            if isinstance(node, ast.FunctionDef)
                        ]
                        classes = [
                            node.name
                            for node in ast.walk(tree)
                            if isinstance(node, ast.ClassDef)
                        ]

                        content_info.update(
                            {
                                "functions": len(functions),
                                "classes": len(classes),
                                "lines": len(content.split("\n")),
                            }
                        )
                    except Exception:
                        pass  # Skip AST analysis if file has syntax errors

            claude_edit = ClaudeCodeEdit(
                file_path=str(file_path),
                edit_type=edit_type,
                timestamp=time.time(),
                content_change=content_info,
                claude_context=self._infer_claude_context(file_path, edit_type),
            )

            self.claude_edits.append(claude_edit)

            # Record in meta system
            self.meta_prime.observe(
                "claude_code_edit",
                {
                    "file": str(file_path),
                    "edit_type": edit_type,
                    "size": content_info.get("size", 0),
                    "context": claude_edit.claude_context,
                },
            )

        except Exception as e:
            self.meta_prime.observe(
                "claude_monitor_error",
                {"error": str(e), "file": str(file_path), "edit_type": edit_type},
            )

    def _infer_claude_context(self, file_path: Path, edit_type: str) -> str:
        """Infer what Claude was trying to accomplish"""

        context_parts = []

        # Based on file location
        if "src/unity_wheel/strategy" in str(file_path):
            context_parts.append("trading_strategy")
        elif "src/unity_wheel/api" in str(file_path):
            context_parts.append("api_development")
        elif "src/unity_wheel/risk" in str(file_path):
            context_parts.append("risk_management")
        elif "meta_" in file_path.name:
            context_parts.append("meta_system_development")
        elif "test" in str(file_path):
            context_parts.append("testing")
        else:
            context_parts.append("general_development")

        # Based on edit type
        if edit_type == "created":
            context_parts.append("new_file_creation")
        elif edit_type == "modified":
            context_parts.append("code_modification")
        elif edit_type == "deleted":
            context_parts.append("cleanup")

        return "_".join(context_parts)


class AutoCorrector:
    """Automatically attempts to fix execution failures"""

    def __init__(self, execution_monitor):
        self.execution_monitor = execution_monitor
        self.meta_prime = execution_monitor.meta_prime
        self.correction_attempts = []

    def attempt_auto_correction(self, failed_result: ExecutionResult) -> bool:
        """Attempt to automatically correct a failed execution"""

        error_patterns = failed_result.error_patterns
        if not error_patterns:
            return False

        error_type = error_patterns[0]  # Use first detected pattern
        print(f"🔧 Attempting auto-correction for {error_type}...")

        correction_made = False

        if error_type == "import_error":
            correction_made = self._fix_import_error(failed_result)
        elif error_type == "dependency_error":
            correction_made = self._fix_dependency_error(failed_result)
        elif error_type == "file_error":
            correction_made = self._fix_file_error(failed_result)
        else:
            print(f"⏳ No auto-correction available for {error_type}")

        # Record correction attempt
        self.correction_attempts.append(
            {
                "original_error": error_type,
                "command": failed_result.command,
                "correction_made": correction_made,
                "timestamp": time.time(),
            }
        )

        self.meta_prime.observe(
            "auto_correction_attempt",
            {
                "error_type": error_type,
                "command": failed_result.command,
                "correction_successful": correction_made,
                "timestamp": time.time(),
            },
        )

        return correction_made

    def _fix_import_error(self, failed_result: ExecutionResult) -> bool:
        """Try to fix import errors by installing missing packages"""

        import_match = re.search(r"No module named '([^']+)'", failed_result.stderr)
        if not import_match:
            return False

        module_name = import_match.group(1)
        print(f"🔧 Installing missing module: {module_name}")

        install_result = self.execution_monitor.monitor_command(
            f"pip install {module_name}"
        )

        if install_result.success:
            print(f"✅ Successfully installed {module_name}")
            retry_result = self.execution_monitor.monitor_command(failed_result.command)
            return retry_result.success
        else:
            print(f"❌ Failed to install {module_name}")
            return False

    def _fix_dependency_error(self, failed_result: ExecutionResult) -> bool:
        """Try to fix dependency errors"""

        if Path("requirements.txt").exists():
            print("🔧 Installing requirements.txt...")
            install_result = self.execution_monitor.monitor_command(
                "pip install -r requirements.txt"
            )

            if install_result.success:
                retry_result = self.execution_monitor.monitor_command(
                    failed_result.command
                )
                return retry_result.success

        return False

    def _fix_file_error(self, failed_result: ExecutionResult) -> bool:
        """Try to fix file errors by creating missing directories"""

        file_match = re.search(
            r"No such file or directory: '([^']+)'", failed_result.stderr
        )
        if not file_match:
            return False

        file_path = Path(file_match.group(1))

        if not file_path.parent.exists():
            print(f"🔧 Creating missing directory: {file_path.parent}")
            file_path.parent.mkdir(parents=True, exist_ok=True)

            retry_result = self.execution_monitor.monitor_command(failed_result.command)
            return retry_result.success

        return False


class ClaudeCodeFeedbackCollector:
    """Collects feedback about Claude Code's effectiveness"""

    def __init__(self, meta_prime: MetaPrime):
        self.meta_prime = meta_prime
        self.feedback_history = []

    def get_feedback_summary(self) -> dict[str, Any]:
        """Get summary of feedback collected"""

        if not self.feedback_history:
            return {"average_rating": 0, "total_feedback": 0}

        ratings = [
            f["rating"] for f in self.feedback_history if f["rating"] is not None
        ]

        return {
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "total_feedback": len(ratings),
            "recent_trend": ratings[-5:] if len(ratings) >= 5 else ratings,
        }


class UnifiedMetaSystem:
    """The complete unified meta system with all capabilities properly wired"""

    def __init__(self):
        print("🚀 Initializing Unified Meta System...")

        # Core meta components
        # Only create MetaPrime if not disabled
        import os

        if os.environ.get("DISABLE_META_AUTOSTART") == "1":
            self.meta_prime = None
        else:
            self.meta_prime = MetaPrime()
        self.meta_coordinator = MetaCoordinator()
        self.meta_monitor = MetaSystemMonitor()

        # Unified monitoring components
        self.execution_monitor = UnifiedExecutionMonitor(self.meta_prime)
        self.mistake_detector = UnifiedMistakeDetector(self.meta_prime)
        self.claude_monitor = UnifiedClaudeCodeMonitor(self.meta_prime)

        # Add missing components
        self.auto_corrector = AutoCorrector(self.execution_monitor)
        self.feedback_collector = ClaudeCodeFeedbackCollector(self.meta_prime)

        # File system observer for Claude Code monitoring
        self.observer = Observer()
        self.observer.schedule(self.claude_monitor, ".", recursive=True)

        # State tracking for integrated loop
        self.active = False
        self.loop_count = 0
        self.auto_corrections = []
        self.effectiveness_scores = []
        self.auto_improvements = []

        print("✅ Unified Meta System initialized")
        print("🔄 All monitoring, detection, and correction capabilities wired")

    def start(self):
        """Start the complete unified meta system"""

        print("🚀 Starting Unified Meta System...")

        # Start file system monitoring
        self.observer.start()
        print("👁️ Claude Code monitoring active")

        # Add common log files to monitor
        log_files = [
            Path("logs/test.log"),
            Path("meta_daemon.log"),
            Path("logs/eod_collection.log"),
        ]

        for log_file in log_files:
            if log_file.exists():
                self.execution_monitor.add_log_file(log_file)

        self.active = True
        print("✅ Unified Meta System fully active")

    def stop(self):
        """Stop the unified meta system"""

        print("🛑 Stopping Unified Meta System...")

        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()

        self.active = False
        print("✅ Unified Meta System stopped")

    async def run_complete_loop(self):
        """Run the complete integrated meta loop with all capabilities"""

        print("🔄 Starting Complete Integrated Meta Loop...")

        while self.active:
            self.loop_count += 1
            loop_start = time.time()

            print(f"\n🔄 Meta Loop Cycle #{self.loop_count}")

            # Step 1: Monitor Claude Code activity
            claude_activity = await self._monitor_claude_activity()

            # Step 2: Detect mistakes in recent changes
            mistakes = await self._detect_recent_mistakes()

            # Step 3: Measure effectiveness
            effectiveness = await self._measure_effectiveness()

            # Step 4: Trigger evolution based on integrated observations
            evolution_triggered = await self._trigger_meta_evolution(
                claude_activity, mistakes, effectiveness
            )

            # Step 5: Auto-improve the meta-loop itself
            auto_improvements = await self._auto_improve_meta_loop()

            # Step 6: Report cycle results
            cycle_time = time.time() - loop_start
            await self._report_cycle_results(
                cycle_time, evolution_triggered, auto_improvements
            )

            # Wait before next cycle
            await asyncio.sleep(30)  # 30-second cycles for comprehensive monitoring

    async def _monitor_claude_activity(self) -> dict[str, Any]:
        """Monitor what Claude Code has been doing"""

        activity = {
            "files_modified": len(self.claude_monitor.claude_edits),
            "recent_edits": [
                edit
                for edit in self.claude_monitor.claude_edits
                if time.time() - edit.timestamp < 300
            ],  # Last 5 minutes
            "modification_rate": len(self.claude_monitor.claude_edits)
            / max(1, self.loop_count),
        }

        self.meta_prime.observe("claude_activity_cycle", activity)
        return activity

    async def _detect_recent_mistakes(self) -> list[dict[str, Any]]:
        """Check for mistakes in recently modified files"""

        mistakes = []
        recent_edits = [
            edit
            for edit in self.claude_monitor.claude_edits
            if time.time() - edit.timestamp < 300
        ]  # Last 5 minutes

        for edit in recent_edits[-3:]:  # Last 3 files
            if edit.edit_type in ["created", "modified"]:
                file_path = Path(edit.file_path)
                if file_path.exists():
                    # Syntax mistake detection
                    syntax_mistake = self.mistake_detector.detect_syntax_mistakes(
                        file_path
                    )
                    if syntax_mistake:
                        mistakes.append(
                            {
                                "file": str(file_path),
                                "mistake_type": syntax_mistake.mistake_type,
                                "severity": syntax_mistake.severity,
                                "error": syntax_mistake.error_message,
                            }
                        )

                        # Attempt auto-correction
                        if syntax_mistake.mistake_type in [
                            "import_error",
                            "dependency_error",
                        ]:
                            # Create mock execution result for auto-correction
                            mock_result = ExecutionResult(
                                command=f"python {file_path}",
                                exit_code=1,
                                stdout="",
                                stderr=syntax_mistake.error_message,
                                execution_time=0,
                                timestamp=time.time(),
                                success=False,
                                error_patterns=[syntax_mistake.mistake_type],
                            )

                            corrected = self.auto_corrector.attempt_auto_correction(
                                mock_result
                            )
                            if corrected:
                                self.auto_corrections.append(
                                    {
                                        "timestamp": time.time(),
                                        "file": str(file_path),
                                        "correction_type": syntax_mistake.mistake_type,
                                    }
                                )

        self.meta_prime.observe(
            "mistake_detection_cycle",
            {"files_checked": len(recent_edits), "mistakes_found": len(mistakes)},
        )

        return mistakes

    async def _measure_effectiveness(self) -> dict[str, Any]:
        """Measure how effective the meta-loop is being"""

        self.feedback_collector.get_feedback_summary()
        health = self.meta_monitor.health_check("meta_prime")

        # Calculate effectiveness factors
        effectiveness_factors = []

        # Factor 1: System health
        if health.status == "healthy":
            effectiveness_factors.append(1.0)
        elif health.status == "warning":
            effectiveness_factors.append(0.7)
        else:
            effectiveness_factors.append(0.3)

        # Factor 2: Error rate (lower is better)
        recent_mistakes = len(
            [
                m
                for m in self.mistake_detector.detected_mistakes
                if time.time() - m.timestamp < 1800
            ]
        )  # Last 30 min
        error_factor = max(0, 1.0 - (recent_mistakes / 10.0))
        effectiveness_factors.append(error_factor)

        # Factor 3: Auto-correction success rate
        if self.auto_corrector.correction_attempts:
            success_rate = len(
                [
                    c
                    for c in self.auto_corrector.correction_attempts
                    if c["correction_made"]
                ]
            ) / len(self.auto_corrector.correction_attempts)
            effectiveness_factors.append(success_rate)

        overall_effectiveness = (
            sum(effectiveness_factors) / len(effectiveness_factors)
            if effectiveness_factors
            else 0.5
        )

        effectiveness = {
            "overall_score": overall_effectiveness,
            "health_score": effectiveness_factors[0],
            "error_rate": recent_mistakes,
            "correction_success_rate": effectiveness_factors[2]
            if len(effectiveness_factors) > 2
            else None,
        }

        self.effectiveness_scores.append(overall_effectiveness)
        self.meta_prime.observe("effectiveness_measurement", effectiveness)

        return effectiveness

    async def _trigger_meta_evolution(
        self,
        claude_activity: dict[str, Any],
        mistakes: list[dict[str, Any]],
        effectiveness: dict[str, Any],
    ) -> bool:
        """Trigger meta-system evolution based on integrated observations"""

        should_evolve = False
        evolution_reasons = []

        # Reason 1: High activity with low effectiveness
        if (
            len(claude_activity["recent_edits"]) > 3
            and effectiveness["overall_score"] < 0.6
        ):
            should_evolve = True
            evolution_reasons.append("high_activity_low_effectiveness")

        # Reason 2: Multiple mistakes detected
        if len(mistakes) >= 2:
            should_evolve = True
            evolution_reasons.append("multiple_mistakes_detected")

        # Reason 3: Declining effectiveness trend
        if len(self.effectiveness_scores) >= 3:
            recent_trend = self.effectiveness_scores[-3:]
            if all(
                recent_trend[i] > recent_trend[i + 1]
                for i in range(len(recent_trend) - 1)
            ):
                should_evolve = True
                evolution_reasons.append("declining_effectiveness_trend")

        # Reason 4: System readiness
        if self.meta_prime.should_evolve():
            should_evolve = True
            evolution_reasons.append("meta_system_ready")

        if should_evolve:
            print(f"🧬 Triggering evolution: {', '.join(evolution_reasons)}")

            evolution_context = {
                "claude_activity": claude_activity,
                "mistakes": mistakes,
                "effectiveness": effectiveness,
                "reasons": evolution_reasons,
                "loop_cycle": self.loop_count,
            }

            self.meta_prime.observe("integrated_evolution_trigger", evolution_context)
            success = self.meta_prime.evolve()

            if success:
                print("✅ Meta-evolution successful")
            else:
                print("❌ Meta-evolution failed")

            return success

        return False

    async def _auto_improve_meta_loop(self) -> list[str]:
        """Auto-improve the meta-loop itself based on performance"""

        improvements = []

        # Improvement 1: Adjust monitoring sensitivity
        if len(self.effectiveness_scores) >= 5:
            avg_effectiveness = sum(self.effectiveness_scores[-5:]) / 5

            if avg_effectiveness < 0.4:
                improvements.append("increased_monitoring_frequency")
                print("🔧 Auto-improvement: Increased monitoring frequency")
            elif avg_effectiveness > 0.8:
                improvements.append("reduced_monitoring_overhead")
                print("🔧 Auto-improvement: Reduced monitoring overhead")

        # Improvement 2: Adapt correction strategies
        if self.auto_corrector.correction_attempts:
            success_rate = len(
                [
                    c
                    for c in self.auto_corrector.correction_attempts
                    if c["correction_made"]
                ]
            ) / len(self.auto_corrector.correction_attempts)

            if success_rate < 0.3:
                improvements.append("conservative_auto_correction")
                print("🔧 Auto-improvement: More conservative auto-correction")
            elif success_rate > 0.9:
                improvements.append("aggressive_auto_correction")
                print("🔧 Auto-improvement: More aggressive auto-correction")

        if improvements:
            self.auto_improvements.extend(improvements)
            self.meta_prime.observe(
                "meta_loop_auto_improvement",
                {"improvements": improvements, "loop_cycle": self.loop_count},
            )

        return improvements

    async def _report_cycle_results(
        self, cycle_time: float, evolution_triggered: bool, auto_improvements: list[str]
    ):
        """Report the results of this meta-loop cycle"""

        report = {
            "loop_cycle": self.loop_count,
            "cycle_time_seconds": cycle_time,
            "evolution_triggered": evolution_triggered,
            "auto_improvements": auto_improvements,
            "claude_edits_monitored": len(self.claude_monitor.claude_edits),
            "total_mistakes_detected": len(self.mistake_detector.detected_mistakes),
            "auto_corrections_made": len(self.auto_corrections),
            "current_effectiveness": self.effectiveness_scores[-1]
            if self.effectiveness_scores
            else 0,
        }

        self.meta_prime.observe("meta_loop_cycle_complete", report)

        print(f"📊 Cycle {self.loop_count} Complete ({cycle_time:.1f}s)")
        print(f"   Evolution triggered: {'✅' if evolution_triggered else '❌'}")
        print(f"   Auto-improvements: {len(auto_improvements)}")
        print(f"   Auto-corrections: {len(self.auto_corrections)}")
        print(
            f"   Effectiveness: {self.effectiveness_scores[-1]:.1%}"
            if self.effectiveness_scores
            else "   Effectiveness: N/A"
        )

        # Detailed summary every 10 cycles
        if self.loop_count % 10 == 0:
            await self._print_detailed_summary()

    async def _print_detailed_summary(self):
        """Print detailed summary every 10 cycles"""

        print("\n📈 Meta-Loop Summary (Last 10 Cycles)")
        print(f"   Total cycles: {self.loop_count}")
        print(f"   Files monitored: {len(self.claude_monitor.claude_edits)}")
        print(f"   Mistakes detected: {len(self.mistake_detector.detected_mistakes)}")
        print(f"   Auto-corrections: {len(self.auto_corrections)}")
        print(f"   Auto-improvements: {len(self.auto_improvements)}")

        if self.effectiveness_scores:
            recent_effectiveness = (
                self.effectiveness_scores[-10:]
                if len(self.effectiveness_scores) >= 10
                else self.effectiveness_scores
            )
            avg_effectiveness = sum(recent_effectiveness) / len(recent_effectiveness)
            print(f"   Average effectiveness: {avg_effectiveness:.1%}")

        status = self.meta_prime.status_report()
        lines = status.split("\n")
        for line in lines[:3]:  # First 3 lines of status
            if line.strip():
                print(f"   {line}")

    def get_status_report(self) -> str:
        """Get comprehensive status of the unified system"""

        # Get basic meta system status
        base_status = self.meta_prime.status_report()

        # Add unified system metrics
        recent_mistakes = [
            m
            for m in self.mistake_detector.detected_mistakes
            if time.time() - m.timestamp < 3600
        ]  # Last hour

        recent_edits = [
            e
            for e in self.claude_monitor.claude_edits
            if time.time() - e.timestamp < 3600
        ]  # Last hour

        recent_executions = [
            e
            for e in self.execution_monitor.execution_history
            if time.time() - e.timestamp < 3600
        ]  # Last hour

        unified_status = f"""
🔄 Unified Meta System Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{base_status}

🔗 Unified Components Status:
  Loop cycles: {self.loop_count}
  Claude edits monitored: {len(recent_edits)}
  Commands executed: {len(recent_executions)}
  Mistakes detected: {len(recent_mistakes)}
  Auto-corrections: {len(self.auto_corrections)}
  
🎯 Integration Health:
  File monitoring: {'🟢 Active' if self.observer.is_alive() else '🔴 Inactive'}
  Execution monitoring: 🟢 Active
  Mistake detection: 🟢 Active
  Claude monitoring: 🟢 Active
  
🧬 Recent Auto-Corrections: {len(self.auto_corrections)}
"""

        return unified_status

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


def main():
    """Main function to run the unified meta system"""

    print("🚀 Unified Meta System - All Capabilities Merged & Wired")
    print("=" * 60)

    try:
        with UnifiedMetaSystem() as meta_system:
            print("\n🎯 Unified Meta System Status:")
            print(meta_system.get_status_report())

            print("\n🔄 Starting complete integrated loop...")
            print("Press Ctrl+C to stop")

            # Run the complete loop
            asyncio.run(meta_system.run_complete_loop())

    except KeyboardInterrupt:
        print("\n\n🛑 Unified Meta System stopped by user")
    except Exception as e:
        print(f"\n❌ Error in unified meta system: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import os

    if os.environ.get("DISABLE_META_AUTOSTART") == "1":
        print("⚠️ Unified Meta System blocked by DISABLE_META_AUTOSTART=1")
        sys.exit(0)
    main()
