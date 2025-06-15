#!/usr/bin/env python3
"""
Automatic Mistake Detection - Learn from Claude Code failures
Detects when Claude makes mistakes and learns patterns to avoid them
"""

import ast
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from meta_prime import MetaPrime


@dataclass
class Mistake:
    """Represents a detected mistake"""
    mistake_type: str
    file_path: str
    error_message: str
    context: Dict[str, Any]
    timestamp: float
    severity: str  # 'critical', 'warning', 'minor'


class MistakeDetector:
    """Automatically detects Claude Code mistakes and learns from them"""
    
    def __init__(self):
        self.meta_prime = MetaPrime()
        self.detected_mistakes = []
        self.mistake_patterns = {}
        
        print("🕵️ Mistake Detection System Active")
        
    def check_for_syntax_errors(self, file_path: Path) -> Optional[Mistake]:
        """Check if file has syntax errors"""
        
        if not file_path.exists() or file_path.suffix != '.py':
            return None
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            ast.parse(content)
            return None  # No syntax error
            
        except SyntaxError as e:
            mistake = Mistake(
                mistake_type="syntax_error",
                file_path=str(file_path),
                error_message=str(e),
                context={
                    "line_number": e.lineno,
                    "text": e.text,
                    "filename": e.filename
                },
                timestamp=time.time(),
                severity="critical"
            )
            
            self._record_mistake(mistake)
            return mistake
            
    def check_for_import_errors(self, file_path: Path) -> List[Mistake]:
        """Check for import-related mistakes"""
        
        mistakes = []
        
        if not file_path.exists() or file_path.suffix != '.py':
            return mistakes
            
        try:
            # Try to import the module to check for import errors
            result = subprocess.run([
                'python', '-c', f'import {file_path.stem}'
            ], capture_output=True, text=True, timeout=10, cwd=file_path.parent)
            
            if result.returncode != 0:
                error_output = result.stderr
                
                if "ModuleNotFoundError" in error_output:
                    mistake = Mistake(
                        mistake_type="missing_dependency",
                        file_path=str(file_path),
                        error_message=error_output,
                        context={"import_check": True},
                        timestamp=time.time(),
                        severity="warning"
                    )
                    mistakes.append(mistake)
                    self._record_mistake(mistake)
                    
                elif "ImportError" in error_output:
                    mistake = Mistake(
                        mistake_type="import_error",
                        file_path=str(file_path),
                        error_message=error_output,
                        context={"import_check": True},
                        timestamp=time.time(),
                        severity="warning"
                    )
                    mistakes.append(mistake)
                    self._record_mistake(mistake)
                    
        except subprocess.TimeoutExpired:
            mistake = Mistake(
                mistake_type="import_timeout",
                file_path=str(file_path),
                error_message="Import check timed out",
                context={"timeout": True},
                timestamp=time.time(),
                severity="minor"
            )
            mistakes.append(mistake)
            self._record_mistake(mistake)
            
        except Exception as e:
            # Don't fail the detection, just note it
            pass
            
        return mistakes
        
    def check_for_logical_errors(self, file_path: Path) -> List[Mistake]:
        """Check for logical mistakes in code"""
        
        mistakes = []
        
        if not file_path.exists() or file_path.suffix != '.py':
            return mistakes
            
        try:
            content = file_path.read_text()
            
            # Check for common logical mistakes
            logical_checks = [
                self._check_unused_variables,
                self._check_unreachable_code,
                self._check_infinite_loops,
                self._check_hardcoded_values
            ]
            
            for check in logical_checks:
                check_mistakes = check(file_path, content)
                mistakes.extend(check_mistakes)
                
        except Exception as e:
            print(f"Warning: Could not check logical errors in {file_path}: {e}")
            
        return mistakes
        
    def _check_unused_variables(self, file_path: Path, content: str) -> List[Mistake]:
        """Check for unused variables"""
        mistakes = []
        
        try:
            tree = ast.parse(content)
            
            # Simple unused variable detection
            assigned_vars = set()
            used_vars = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assigned_vars.add(target.id)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)
                    
            unused = assigned_vars - used_vars - {'_'}  # Ignore underscore
            
            if len(unused) > 3:  # Only flag if many unused vars
                mistake = Mistake(
                    mistake_type="many_unused_variables",
                    file_path=str(file_path),
                    error_message=f"Found {len(unused)} unused variables: {list(unused)[:5]}",
                    context={"unused_count": len(unused), "variables": list(unused)[:10]},
                    timestamp=time.time(),
                    severity="minor"
                )
                mistakes.append(mistake)
                self._record_mistake(mistake)
                
        except Exception:
            pass  # Don't fail detection on parsing errors
            
        return mistakes
        
    def _check_unreachable_code(self, file_path: Path, content: str) -> List[Mistake]:
        """Check for unreachable code"""
        mistakes = []
        
        # Simple check: code after return statements
        lines = content.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('return ') and i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                if next_line and not next_line.startswith(('#', '"""', "'''")):
                    # Check if it's not part of a different function/class
                    if not any(keyword in next_line for keyword in ['def ', 'class ', 'if ', 'else:', 'elif ', 'except', 'finally']):
                        mistake = Mistake(
                            mistake_type="unreachable_code",
                            file_path=str(file_path),
                            error_message=f"Code after return on line {i+2}: {next_line}",
                            context={"line_number": i+2, "code": next_line},
                            timestamp=time.time(),
                            severity="minor"
                        )
                        mistakes.append(mistake)
                        self._record_mistake(mistake)
                        break  # Only flag first occurrence
                        
        return mistakes
        
    def _check_infinite_loops(self, file_path: Path, content: str) -> List[Mistake]:
        """Check for potential infinite loops"""
        mistakes = []
        
        # Simple check: while True without break
        if 'while True:' in content and 'break' not in content:
            mistake = Mistake(
                mistake_type="potential_infinite_loop",
                file_path=str(file_path),
                error_message="Found 'while True:' without 'break' statement",
                context={"pattern": "while_true_no_break"},
                timestamp=time.time(),
                severity="warning"
            )
            mistakes.append(mistake)
            self._record_mistake(mistake)
            
        return mistakes
        
    def _check_hardcoded_values(self, file_path: Path, content: str) -> List[Mistake]:
        """Check for excessive hardcoded values"""
        mistakes = []
        
        # Count numeric literals
        import re
        numbers = re.findall(r'\b\d{3,}\b', content)  # 3+ digit numbers
        
        if len(numbers) > 10:  # Many hardcoded numbers
            mistake = Mistake(
                mistake_type="excessive_hardcoded_values",
                file_path=str(file_path),
                error_message=f"Found {len(numbers)} hardcoded numeric values",
                context={"hardcoded_count": len(numbers), "examples": numbers[:5]},
                timestamp=time.time(),
                severity="minor"
            )
            mistakes.append(mistake)
            self._record_mistake(mistake)
            
        return mistakes
        
    def _record_mistake(self, mistake: Mistake):
        """Record a detected mistake"""
        
        self.detected_mistakes.append(mistake)
        
        # Record in meta system
        self.meta_prime.observe("mistake_detected", {
            "mistake_type": mistake.mistake_type,
            "file_path": mistake.file_path,
            "error_message": mistake.error_message,
            "severity": mistake.severity,
            "context": mistake.context,
            "timestamp": mistake.timestamp
        })
        
        # Update mistake patterns
        if mistake.mistake_type not in self.mistake_patterns:
            self.mistake_patterns[mistake.mistake_type] = {
                "count": 0,
                "files": [],
                "recent_examples": []
            }
            
        pattern = self.mistake_patterns[mistake.mistake_type]
        pattern["count"] += 1
        pattern["files"].append(mistake.file_path)
        pattern["recent_examples"].append({
            "error": mistake.error_message,
            "timestamp": mistake.timestamp,
            "file": mistake.file_path
        })
        
        # Keep only recent examples
        if len(pattern["recent_examples"]) > 5:
            pattern["recent_examples"] = pattern["recent_examples"][-5:]
            
        print(f"🚨 Mistake detected: {mistake.mistake_type} in {Path(mistake.file_path).name}")
        
    def analyze_mistake_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in detected mistakes"""
        
        if not self.detected_mistakes:
            return {"total_mistakes": 0, "patterns": {}}
            
        # Group by type
        by_type = {}
        for mistake in self.detected_mistakes:
            if mistake.mistake_type not in by_type:
                by_type[mistake.mistake_type] = []
            by_type[mistake.mistake_type].append(mistake)
            
        # Analyze trends
        analysis = {
            "total_mistakes": len(self.detected_mistakes),
            "unique_types": len(by_type),
            "most_common": max(by_type.keys(), key=lambda k: len(by_type[k])) if by_type else None,
            "patterns": {}
        }
        
        for mistake_type, mistakes in by_type.items():
            analysis["patterns"][mistake_type] = {
                "count": len(mistakes),
                "severity_distribution": {
                    severity: len([m for m in mistakes if m.severity == severity])
                    for severity in ['critical', 'warning', 'minor']
                },
                "affected_files": len(set(m.file_path for m in mistakes)),
                "recent_trend": len([m for m in mistakes if time.time() - m.timestamp < 3600])  # Last hour
            }
            
        return analysis
        
    def get_mistake_prevention_suggestions(self) -> List[str]:
        """Get suggestions to prevent common mistakes"""
        
        analysis = self.analyze_mistake_patterns()
        suggestions = []
        
        if not analysis["patterns"]:
            return ["No mistakes detected yet. Keep up the good work!"]
            
        for mistake_type, pattern_info in analysis["patterns"].items():
            if pattern_info["count"] >= 3:  # Recurring issue
                if mistake_type == "syntax_error":
                    suggestions.append("Consider using an IDE with syntax highlighting")
                elif mistake_type == "missing_dependency":
                    suggestions.append("Check imports before running code")
                elif mistake_type == "excessive_hardcoded_values":
                    suggestions.append("Consider using configuration files for constants")
                elif mistake_type == "unused_variables":
                    suggestions.append("Clean up unused variables regularly")
                    
        if not suggestions:
            suggestions.append("Mistakes detected but no specific patterns identified yet")
            
        return suggestions
        
    def check_file_thoroughly(self, file_path: Path) -> Dict[str, Any]:
        """Run all mistake checks on a file"""
        
        all_mistakes = []
        
        # Run all checks
        syntax_mistake = self.check_for_syntax_errors(file_path)
        if syntax_mistake:
            all_mistakes.append(syntax_mistake)
            
        import_mistakes = self.check_for_import_errors(file_path)
        all_mistakes.extend(import_mistakes)
        
        logical_mistakes = self.check_for_logical_errors(file_path)
        all_mistakes.extend(logical_mistakes)
        
        # Summary
        summary = {
            "file_path": str(file_path),
            "total_mistakes": len(all_mistakes),
            "mistakes_by_severity": {
                "critical": len([m for m in all_mistakes if m.severity == "critical"]),
                "warning": len([m for m in all_mistakes if m.severity == "warning"]),
                "minor": len([m for m in all_mistakes if m.severity == "minor"])
            },
            "mistake_types": list(set(m.mistake_type for m in all_mistakes)),
            "clean_file": len(all_mistakes) == 0
        }
        
        if all_mistakes:
            print(f"🔍 Found {len(all_mistakes)} potential issues in {file_path.name}")
        else:
            print(f"✅ No issues detected in {file_path.name}")
            
        return summary


def run_mistake_detection_on_project():
    """Run mistake detection on the entire project"""
    
    detector = MistakeDetector()
    
    print("🕵️ Running Comprehensive Mistake Detection...")
    
    # Find Python files to check
    python_files = list(Path(".").glob("*.py"))
    src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
    all_files = python_files + src_files
    
    results = []
    
    for file_path in all_files:
        if file_path.name.startswith('.'):  # Skip hidden files
            continue
            
        result = detector.check_file_thoroughly(file_path)
        results.append(result)
        
    # Overall analysis
    total_mistakes = sum(r["total_mistakes"] for r in results)
    clean_files = sum(1 for r in results if r["clean_file"])
    
    print(f"\n📊 Project Mistake Analysis:")
    print(f"   Files checked: {len(results)}")
    print(f"   Clean files: {clean_files}")
    print(f"   Files with issues: {len(results) - clean_files}")
    print(f"   Total potential issues: {total_mistakes}")
    
    # Pattern analysis
    pattern_analysis = detector.analyze_mistake_patterns()
    if pattern_analysis["patterns"]:
        print(f"\n🔍 Most Common Issue Types:")
        for mistake_type, info in pattern_analysis["patterns"].items():
            print(f"   • {mistake_type}: {info['count']} occurrences")
            
    # Suggestions
    suggestions = detector.get_mistake_prevention_suggestions()
    print(f"\n💡 Prevention Suggestions:")
    for suggestion in suggestions:
        print(f"   • {suggestion}")
        
    return {
        "files_checked": len(results),
        "total_mistakes": total_mistakes,
        "pattern_analysis": pattern_analysis,
        "suggestions": suggestions
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Check specific file
        file_path = Path(sys.argv[1])
        detector = MistakeDetector()
        detector.check_file_thoroughly(file_path)
    else:
        # Check entire project
        run_mistake_detection_on_project()