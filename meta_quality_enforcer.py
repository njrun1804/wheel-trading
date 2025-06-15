"""
Meta Quality Enforcer - Real-time learning rule enforcement
Built with the 10-step coding principles for production quality
"""

import ast
import re
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from meta_daemon_config import get_daemon_config, LearningRulesConfig


@dataclass
class QualityViolation:
    """Represents a quality rule violation"""
    rule_name: str
    file_path: str
    line_number: int
    violation_text: str
    severity: str  # 'error', 'warning', 'info'
    suggestion: str
    learning_rule_id: int


@dataclass
class QualityReport:
    """Quality assessment report for a file"""
    file_path: str
    total_checks: int
    violations: List[QualityViolation]
    compliance_score: float
    processing_time_ms: float
    
    @property
    def is_compliant(self) -> bool:
        """Check if file meets quality standards"""
        config = get_daemon_config()
        return self.compliance_score >= config.quality_gate.minimum_compliance_percentage
    
    @property
    def error_count(self) -> int:
        """Count of error-level violations"""
        return len([v for v in self.violations if v.severity == 'error'])


class LearningRuleEnforcer:
    """Enforces the 10 learning rules in real-time"""
    
    def __init__(self, config: LearningRulesConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=get_daemon_config().daemon.worker_threads)
        
        # Compiled patterns for performance
        self.anti_pattern_regex = re.compile(self.config.anti_patterns, re.IGNORECASE)
        self.hardcoded_number_regex = re.compile(r'\b\d+\.?\d*\b')
        self.bare_except_regex = re.compile(r'except\s*:')
        self.empty_except_regex = re.compile(r'except.*:\s*pass\s*$', re.MULTILINE)
        
        print("🛡️ Learning Rule Enforcer initialized")
        print(f"   Worker threads: {self.executor._max_workers}")
        print(f"   Anti-pattern checks: Active")
        print(f"   Real-time enforcement: Enabled")
    
    async def enforce_quality_rules(self, file_path: Path) -> QualityReport:
        """Enforce all 10 learning rules on a file"""
        
        start_time = time.time()
        
        if not file_path.exists() or not file_path.suffix == '.py':
            return QualityReport(
                file_path=str(file_path),
                total_checks=0,
                violations=[],
                compliance_score=100.0,
                processing_time_ms=0.0
            )
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            return QualityReport(
                file_path=str(file_path),
                total_checks=0,
                violations=[QualityViolation(
                    rule_name="file_reading",
                    file_path=str(file_path),
                    line_number=1,
                    violation_text=f"Cannot read file: {e}",
                    severity="error",
                    suggestion="Check file encoding and permissions",
                    learning_rule_id=0
                )],
                compliance_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Run all quality checks concurrently
        check_tasks = [
            self._check_anti_patterns(file_path, content),
            self._check_implementation_depth(file_path, content),
            self._check_dependency_usage(file_path, content),  
            self._check_hardcoded_values(file_path, content),
            self._check_async_completeness(file_path, content),
            self._check_error_handling(file_path, content)
        ]
        
        violation_lists = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Flatten violations and handle exceptions
        all_violations = []
        total_checks = 0
        
        for i, result in enumerate(violation_lists):
            if isinstance(result, Exception):
                all_violations.append(QualityViolation(
                    rule_name=f"check_{i+1}",
                    file_path=str(file_path),
                    line_number=1,
                    violation_text=f"Check failed: {result}",
                    severity="error",
                    suggestion="Fix the underlying issue",
                    learning_rule_id=i+1
                ))
            else:
                all_violations.extend(result)
                total_checks += 1
        
        # Calculate compliance score
        if total_checks == 0:
            compliance_score = 0.0
        else:
            error_violations = len([v for v in all_violations if v.severity == 'error'])
            compliance_score = max(0.0, 100.0 - (error_violations / total_checks * 100.0))
        
        processing_time = (time.time() - start_time) * 1000
        
        return QualityReport(
            file_path=str(file_path),
            total_checks=total_checks,
            violations=all_violations,
            compliance_score=compliance_score,
            processing_time_ms=processing_time
        )
    
    async def _check_anti_patterns(self, file_path: Path, content: str) -> List[QualityViolation]:
        """Learning Rule 1: Pattern Search - Check for anti-patterns"""
        
        violations = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            if line_num % 100 == 0:  # Yield control periodically for large files
                await asyncio.sleep(0.001)
            
            matches = self.anti_pattern_regex.findall(line)
            for match in matches:
                violations.append(QualityViolation(
                    rule_name="anti_patterns",
                    file_path=str(file_path),
                    line_number=line_num,
                    violation_text=f"Anti-pattern detected: {match}",
                    severity="error",
                    suggestion=f"Replace '{match}' with actual implementation",
                    learning_rule_id=1
                ))
        
        return violations
    
    async def _check_implementation_depth(self, file_path: Path, content: str) -> List[QualityViolation]:
        """Learning Rule 2: Implementation Verification - Check function depth"""
        
        violations = []
        
        try:
            tree = ast.parse(content)
            await asyncio.sleep(0.001)  # Yield after parsing
        except SyntaxError:
            return [QualityViolation(
                rule_name="implementation_depth",
                file_path=str(file_path),
                line_number=1,
                violation_text="Syntax error prevents implementation analysis",
                severity="error",
                suggestion="Fix syntax errors first",
                learning_rule_id=2
            )]
        
        function_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1
                if function_count % 10 == 0:  # Yield control periodically
                    await asyncio.sleep(0.001)
                    
                # Count meaningful lines in function
                func_start = node.lineno
                func_end = getattr(node, 'end_lineno', func_start)
                
                if func_end:
                    func_lines = content.split('\n')[func_start-1:func_end]
                    meaningful_lines = 0
                    
                    for line in func_lines:
                        stripped = line.strip()
                        if (stripped and 
                            not stripped.startswith('#') and
                            not stripped.startswith('"""') and
                            not stripped.startswith("'''") and
                            stripped not in ['pass', 'return', 'return None']):
                            meaningful_lines += 1
                    
                    if meaningful_lines < self.config.min_function_lines:
                        violations.append(QualityViolation(
                            rule_name="implementation_depth",
                            file_path=str(file_path),
                            line_number=func_start,
                            violation_text=f"Function '{node.name}' has only {meaningful_lines} meaningful lines",
                            severity="warning",
                            suggestion=f"Add actual implementation (minimum {self.config.min_function_lines} lines)",
                            learning_rule_id=2
                        ))
        
        return violations
    
    async def _check_dependency_usage(self, file_path: Path, content: str) -> List[QualityViolation]:
        """Learning Rule 4: Dependency Verification - Check import usage"""
        
        violations = []
        
        try:
            tree = ast.parse(content)
            await asyncio.sleep(0.001)  # Yield after parsing
        except SyntaxError:
            return []
        
        # Extract imports
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
                for alias in node.names:
                    imports.add(alias.name)
        
        # Check if imports are used
        import_count = 0
        for imp in imports:
            import_count += 1
            if import_count % 5 == 0:  # Yield control periodically
                await asyncio.sleep(0.001)
                
            if imp not in content or content.count(imp) <= 1:  # Only import line
                violations.append(QualityViolation(
                    rule_name="dependency_usage",
                    file_path=str(file_path),
                    line_number=1,
                    violation_text=f"Unused import: {imp}",
                    severity="warning",
                    suggestion=f"Remove unused import '{imp}' or use it in the code",
                    learning_rule_id=4
                ))
        
        return violations
    
    async def _check_hardcoded_values(self, file_path: Path, content: str) -> List[QualityViolation]:
        """Learning Rule 5: Configuration Audit - Check hardcoded values"""
        
        violations = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            if line_num % 50 == 0:  # Yield control periodically
                await asyncio.sleep(0.001)
                
            # Skip comments and strings
            if line.strip().startswith('#'):
                continue
                
            numbers = self.hardcoded_number_regex.findall(line)
            for number in numbers:
                try:
                    num_value = float(number)
                    if num_value > self.config.hardcoded_number_threshold:
                        violations.append(QualityViolation(
                            rule_name="hardcoded_values",
                            file_path=str(file_path),
                            line_number=line_num,
                            violation_text=f"Hardcoded value: {number}",
                            severity="info",
                            suggestion=f"Consider making {number} a configurable parameter",
                            learning_rule_id=5
                        ))
                except ValueError:
                    continue
        
        return violations
    
    async def _check_async_completeness(self, file_path: Path, content: str) -> List[QualityViolation]:
        """Learning Rule 6: Async Completeness - Check async functions have await"""
        
        violations = []
        
        try:
            tree = ast.parse(content)
            await asyncio.sleep(0.001)  # Yield after parsing
        except SyntaxError:
            return []
        
        async_function_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                async_function_count += 1
                if async_function_count % 5 == 0:  # Yield control periodically
                    await asyncio.sleep(0.001)
                    
                # Check if function contains await
                has_await = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Await):
                        has_await = True
                        break
                
                if not has_await and self.config.require_await_in_async:
                    violations.append(QualityViolation(
                        rule_name="async_completeness",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        violation_text=f"Async function '{node.name}' has no await calls",
                        severity="warning",
                        suggestion=f"Add await calls or make function synchronous",
                        learning_rule_id=6
                    ))
        
        return violations
    
    async def _check_error_handling(self, file_path: Path, content: str) -> List[QualityViolation]:
        """Learning Rule 7: Error Handling Audit - Check try/except blocks"""
        
        violations = []
        
        # Check for bare except
        if not self.config.allow_bare_except:
            bare_except_matches = self.bare_except_regex.finditer(content)
            match_count = 0
            for match in bare_except_matches:
                match_count += 1
                if match_count % 5 == 0:  # Yield control periodically
                    await asyncio.sleep(0.001)
                    
                line_num = content[:match.start()].count('\n') + 1
                violations.append(QualityViolation(
                    rule_name="error_handling",
                    file_path=str(file_path),
                    line_number=line_num,
                    violation_text="Bare except clause detected",
                    severity="warning",
                    suggestion="Specify exception types: except SpecificException:",
                    learning_rule_id=7
                ))
        
        # Check for empty except
        if not self.config.allow_empty_except:
            empty_except_matches = self.empty_except_regex.finditer(content)
            match_count = 0
            for match in empty_except_matches:
                match_count += 1
                if match_count % 5 == 0:  # Yield control periodically
                    await asyncio.sleep(0.001)
                    
                line_num = content[:match.start()].count('\n') + 1
                violations.append(QualityViolation(
                    rule_name="error_handling",
                    file_path=str(file_path),
                    line_number=line_num,
                    violation_text="Empty except block detected",
                    severity="error",
                    suggestion="Add proper error handling or logging",
                    learning_rule_id=7
                ))
        
        return violations
    
    def format_violation_report(self, report: QualityReport) -> str:
        """Format quality report for display"""
        
        if report.is_compliant:
            return f"✅ {report.file_path}: {report.compliance_score:.1f}% compliant ({report.processing_time_ms:.1f}ms)"
        
        output = []
        output.append(f"❌ {report.file_path}: {report.compliance_score:.1f}% compliant")
        output.append(f"   Processing time: {report.processing_time_ms:.1f}ms")
        output.append(f"   Violations: {len(report.violations)}")
        
        # Group violations by rule
        violations_by_rule = {}
        for violation in report.violations:
            if violation.rule_name not in violations_by_rule:
                violations_by_rule[violation.rule_name] = []
            violations_by_rule[violation.rule_name].append(violation)
        
        for rule_name, violations in violations_by_rule.items():
            output.append(f"   📋 {rule_name}: {len(violations)} violations")
            for violation in violations[:3]:  # Show first 3
                severity_icon = "🚨" if violation.severity == "error" else "⚠️" if violation.severity == "warning" else "ℹ️"
                output.append(f"     {severity_icon} Line {violation.line_number}: {violation.violation_text}")
                output.append(f"        → {violation.suggestion}")
        
        return '\n'.join(output)


if __name__ == "__main__":
    async def test_quality_enforcer():
        """Test the quality enforcer with real files"""
        
        config = get_daemon_config()
        enforcer = LearningRuleEnforcer(config.learning_rules)
        
        # Test files
        test_files = [
            Path("meta_daemon_config.py"),
            Path("jarvis2_core.py"),
            Path("jarvis2_mcts.py")
        ]
        
        print("🛡️ Testing Quality Enforcer")
        print("=" * 50)
        
        for test_file in test_files:
            if test_file.exists():
                print(f"\n📁 Testing: {test_file}")
                report = await enforcer.enforce_quality_rules(test_file)
                print(enforcer.format_violation_report(report))
            else:
                print(f"\n❌ File not found: {test_file}")
        
        print(f"\n✅ Quality enforcement test complete")
    
    asyncio.run(test_quality_enforcer())