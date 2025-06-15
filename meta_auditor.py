"""
MetaAuditor - Claude's Self-Audit Integration

This component integrates Claude's proven audit checklist into the meta system's 
self-evaluation and code generation processes. It learns from Claude's established
patterns for high-quality code assessment.

Design Decision: Incorporate proven audit patterns vs organic discovery
Rationale: Claude has already optimized these patterns through experience - leverage them
Alternative: Let system discover audit patterns organically
Prediction: Will accelerate quality improvement and reduce learning time
"""

import ast
import re
import time
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AuditResult:
    """Result of a code audit"""
    file_path: str
    step_name: str
    status: str  # 'PASS', 'NEEDS_WORK', 'N/A'
    findings: List[str]
    concerns: List[str]
    confidence: float


class MetaAuditor:
    """Implements Claude's 12-step audit process for meta system self-evaluation"""
    
    def __init__(self, meta_db_path: str = None):
        if meta_db_path is None:
            from meta_config import get_meta_config
            config = get_meta_config()
            meta_db_path = config.database.evolution_db
        self.db = sqlite3.connect(meta_db_path)
        self.birth_time = time.time()
        
        # Claude's proven audit patterns
        self.audit_patterns = self._load_claude_audit_patterns()
        self.mac_compatibility_checks = self._load_mac_m4_compatibility()
        
        print(f"🔍 MetaAuditor initialized at {time.ctime(self.birth_time)}")
        print(f"📋 Loaded {len(self.audit_patterns)} audit patterns from Claude's experience")
        
        self._init_auditor_schema()
        self._record_birth()
        
    def _init_auditor_schema(self):
        """Initialize auditor database schema"""
        
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS audit_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                file_path TEXT NOT NULL,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL,
                findings_json TEXT NOT NULL,
                concerns_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                audit_generation INTEGER NOT NULL
            )
        """)
        
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS audit_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                pattern_name TEXT NOT NULL,
                pattern_regex TEXT NOT NULL,
                severity TEXT NOT NULL,
                success_rate REAL NOT NULL,
                claude_verified BOOLEAN DEFAULT TRUE
            )
        """)
        
        self.db.commit()
        
    def _record_birth(self):
        """Record auditor initialization"""
        
        # Check if observations table exists, if not, this auditor is standalone
        try:
            cursor = self.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='observations'")
            if cursor.fetchone():
                # Table exists, use it
                self.db.execute("""
                    INSERT INTO observations (timestamp, event_type, details, context)
                    VALUES (?, ?, ?, ?)
                """, (
                    time.time(),
                    "meta_auditor_birth",
                    '{"component": "MetaAuditor", "audit_steps": 12, "claude_patterns": "integrated"}',
                    "MetaAuditor"
                ))
            else:
                # Create our own events table for standalone operation
                self.db.execute("""
                    CREATE TABLE IF NOT EXISTS auditor_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        event_type TEXT NOT NULL,
                        details TEXT NOT NULL,
                        context TEXT
                    )
                """)
                
                self.db.execute("""
                    INSERT INTO auditor_events (timestamp, event_type, details, context)
                    VALUES (?, ?, ?, ?)
                """, (
                    time.time(),
                    "meta_auditor_birth",
                    '{"component": "MetaAuditor", "audit_steps": 12, "claude_patterns": "integrated"}',
                    "MetaAuditor"
                ))
        except Exception as e:
            print(f"Warning: Could not record auditor birth: {e}")
        
        self.db.commit()
        
    def _load_claude_audit_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load Claude's proven audit patterns"""
        
        return {
            "anti_stub_patterns": {
                "regex": r"TODO|FIXME|dummy|mock|fake|placeholder|stub|simplified|hardcoded|NotImplemented|pass\s*#?",
                "severity": "high",
                "description": "Incomplete implementation indicators",
                "claude_verified": True
            },
            "shallow_implementation": {
                "min_lines": 3,
                "constant_return_check": True,
                "description": "Functions with insufficient implementation",
                "claude_verified": True
            },
            "unused_imports": {
                "check_usage": True,
                "description": "Import statements without usage",
                "claude_verified": True
            },
            "hardcoded_values": {
                "numeric_literals": r"\b\d+\.?\d*\b",
                "url_patterns": r"https?://|localhost|\w+\.\w+",
                "path_patterns": r"['\"][/\\].*['\"]",
                "description": "Values that should be configurable",
                "claude_verified": True
            },
            "async_completeness": {
                "check_await_usage": True,
                "description": "Async functions without await calls",
                "claude_verified": True
            },
            "error_handling": {
                "empty_except": r"except.*:\s*pass",
                "bare_except": r"except\s*:",
                "description": "Inadequate error handling patterns",
                "claude_verified": True
            }
        }
        
    def _load_mac_m4_compatibility(self) -> Dict[str, List[str]]:
        """Load M4 Mac specific compatibility checks"""
        
        return {
            "incompatible_packages": [
                "tensorflow-gpu", "torch-cuda", "cupy", "pycuda",
                "nvidia-ml-py", "tensorflow-gpu", "paddle-gpu"
            ],
            "mac_optimized_alternatives": {
                "tensorflow-gpu": "tensorflow-metal",
                "torch-cuda": "torch (with MPS)",
                "sentence-transformers": "lightweight custom embeddings",
                "heavy_ml_frameworks": "mlx for Metal acceleration"
            },
            "m4_specific_optimizations": [
                "8 P-cores + 4 E-cores utilization",
                "20 GPU cores (Metal)",
                "24GB unified memory",
                "Serial: KXQ93HN7DP"
            ]
        }
        
    def audit_file(self, file_path: Path, generation: int = 0) -> List[AuditResult]:
        """Run complete 12-step audit on a file"""
        
        if not file_path.exists():
            return []
            
        print(f"🔍 Auditing: {file_path}")
        
        results = []
        
        # Step 1: Pattern Search (anti-stub)
        results.append(self._step_1_pattern_search(file_path, generation))
        
        # Step 2: Shallow Implementation Scan  
        results.append(self._step_2_shallow_implementation(file_path, generation))
        
        # Step 3: Dependency Usage
        results.append(self._step_3_dependency_usage(file_path, generation))
        
        # Step 4: Config vs Hard-Code Audit
        results.append(self._step_4_hardcode_audit(file_path, generation))
        
        # Step 5: Async Completeness
        results.append(self._step_5_async_completeness(file_path, generation))
        
        # Step 6: Error Handling Review
        results.append(self._step_6_error_handling(file_path, generation))
        
        # Step 7: Lint/Type Check (simplified)
        results.append(self._step_7_lint_check(file_path, generation))
        
        # Step 8: Mac M4 Compatibility
        results.append(self._step_8_mac_compatibility(file_path, generation))
        
        # Step 9: Proof of Functionality (skip for now)
        results.append(AuditResult(str(file_path), "proof_of_functionality", "N/A", 
                                  ["Requires runtime execution"], [], 0.5))
        
        # Step 10: Integration Trace (skip for now)
        results.append(AuditResult(str(file_path), "integration_trace", "N/A",
                                  ["Requires runtime tracing"], [], 0.5))
        
        # Step 11: Test Coverage (skip for now)
        results.append(AuditResult(str(file_path), "test_coverage", "N/A",
                                  ["No test suite detected"], [], 0.5))
        
        # Step 12: Performance Check (skip for now)
        results.append(AuditResult(str(file_path), "performance_check", "N/A",
                                  ["Requires runtime benchmarking"], [], 0.5))
        
        # Store results
        for result in results:
            self._store_audit_result(result, generation)
            
        return results
        
    def _step_1_pattern_search(self, file_path: Path, generation: int) -> AuditResult:
        """Step 1: Search for incomplete implementation patterns"""
        
        content = file_path.read_text()
        pattern = self.audit_patterns["anti_stub_patterns"]["regex"]
        
        matches = re.findall(pattern, content, re.IGNORECASE)
        findings = []
        
        if matches:
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(f"Line {i}: {line.strip()}")
                    
        status = "PASS" if not findings else "NEEDS_WORK"
        concerns = [f"Found {len(findings)} incomplete implementation indicators"] if findings else []
        
        return AuditResult(
            file_path=str(file_path),
            step_name="pattern_search_anti_stub",
            status=status,
            findings=findings,
            concerns=concerns,
            confidence=0.9
        )
        
    def _step_2_shallow_implementation(self, file_path: Path, generation: int) -> AuditResult:
        """Step 2: Check for shallow function implementations"""
        
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
        except Exception as e:
            return AuditResult(str(file_path), "shallow_implementation", "N/A",
                             [f"Could not parse Python file: {e}"], [], 0.1)
            
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count non-comment lines
                func_lines = []
                start_line = node.lineno
                end_line = node.end_lineno or start_line
                
                content_lines = content.split('\n')[start_line-1:end_line]
                for line in content_lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                        func_lines.append(line)
                        
                if len(func_lines) < 3:
                    findings.append(f"Function '{node.name}' has only {len(func_lines)} implementation lines")
                    
        status = "PASS" if len(findings) < 3 else "NEEDS_WORK"
        concerns = findings[:3] if findings else []
        
        return AuditResult(
            file_path=str(file_path),
            step_name="shallow_implementation",
            status=status,
            findings=findings,
            concerns=concerns,
            confidence=0.8
        )
        
    def _step_3_dependency_usage(self, file_path: Path, generation: int) -> AuditResult:
        """Step 3: Check import usage"""
        
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
        except Exception as e:
            return AuditResult(str(file_path), "dependency_usage", "N/A",
                             [f"Could not parse Python file: {e}"], [], 0.1)
            
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
                    
        # Simple usage check
        findings = []
        for imp in imports:
            if imp not in content:
                findings.append(f"Unused import: {imp}")
                
        status = "PASS" if len(findings) < 2 else "NEEDS_WORK"
        concerns = findings[:3] if findings else []
        
        return AuditResult(
            file_path=str(file_path),
            step_name="dependency_usage",
            status=status,
            findings=findings,
            concerns=concerns,
            confidence=0.7
        )
        
    def _step_4_hardcode_audit(self, file_path: Path, generation: int) -> AuditResult:
        """Step 4: Check for hardcoded values"""
        
        content = file_path.read_text()
        findings = []
        
        # Check for numeric literals
        numeric_pattern = r'\b\d+\.?\d*\b'
        numbers = re.findall(numeric_pattern, content)
        
        # Filter out line numbers, small constants, etc.  
        from meta_config import get_meta_config
        config = get_meta_config()
        threshold = config.quality.hardcoded_number_threshold if hasattr(config, 'quality') else 10
        significant_numbers = [n for n in numbers if float(n) > threshold and '.' not in n[:2]]
        
        for num in significant_numbers[:5]:  # Top 5
            findings.append(f"Hardcoded number: {num}")
            
        # Check for URLs/paths
        url_pattern = r'https?://\S+|localhost:\d+'
        urls = re.findall(url_pattern, content)
        for url in urls:
            findings.append(f"Hardcoded URL: {url}")
            
        status = "PASS" if len(findings) < 3 else "NEEDS_WORK"
        concerns = findings[:3] if findings else []
        
        return AuditResult(
            file_path=str(file_path),
            step_name="hardcode_audit",
            status=status,
            findings=findings,
            concerns=concerns,
            confidence=0.6
        )
        
    def _step_5_async_completeness(self, file_path: Path, generation: int) -> AuditResult:
        """Step 5: Check async function completeness"""
        
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
        except Exception as e:
            return AuditResult(str(file_path), "async_completeness", "N/A",
                             [f"Could not parse Python file: {e}"], [], 0.1)
            
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                # Check if function contains await
                has_await = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Await):
                        has_await = True
                        break
                        
                if not has_await:
                    findings.append(f"Async function '{node.name}' has no await calls")
                    
        status = "PASS" if not findings else "NEEDS_WORK"
        concerns = findings if findings else []
        
        return AuditResult(
            file_path=str(file_path),
            step_name="async_completeness",
            status=status,
            findings=findings,
            concerns=concerns,
            confidence=0.9
        )
        
    def _step_6_error_handling(self, file_path: Path, generation: int) -> AuditResult:
        """Step 6: Review error handling"""
        
        content = file_path.read_text()
        findings = []
        
        # Check for empty except blocks
        empty_except_pattern = r'except.*?:\s*pass'
        empty_excepts = re.findall(empty_except_pattern, content, re.DOTALL)
        for match in empty_excepts:
            findings.append(f"Empty except block: {match.strip()}")
            
        # Check for bare except
        bare_except_pattern = r'except\s*:'
        bare_excepts = re.findall(bare_except_pattern, content)
        for match in bare_excepts:
            findings.append(f"Bare except clause: {match}")
            
        status = "PASS" if len(findings) < 2 else "NEEDS_WORK"
        concerns = findings if findings else []
        
        return AuditResult(
            file_path=str(file_path),
            step_name="error_handling",
            status=status,
            findings=findings,
            concerns=concerns,
            confidence=0.8
        )
        
    def _step_7_lint_check(self, file_path: Path, generation: int) -> AuditResult:
        """Step 7: Basic lint check"""
        
        # Simple syntax check
        try:
            content = file_path.read_text()
            ast.parse(content)
            findings = ["Syntax check: PASS"]
            status = "PASS"
        except SyntaxError as e:
            findings = [f"Syntax error: {e}"]
            status = "NEEDS_WORK"
            
        return AuditResult(
            file_path=str(file_path),
            step_name="lint_check",
            status=status,
            findings=findings,
            concerns=findings if status == "NEEDS_WORK" else [],
            confidence=0.9
        )
        
    def _step_8_mac_compatibility(self, file_path: Path, generation: int) -> AuditResult:
        """Step 8: Check M4 Mac compatibility"""
        
        content = file_path.read_text()
        findings = []
        concerns = []
        
        # Check for incompatible packages
        incompatible = self.mac_compatibility_checks["incompatible_packages"]
        for package in incompatible:
            if package in content:
                findings.append(f"Incompatible package detected: {package}")
                concerns.append(f"Replace {package} with Mac-compatible alternative")
                
        # Check for CUDA references
        cuda_patterns = ['cuda', 'gpu', 'nvidia', 'torch.cuda']
        for pattern in cuda_patterns:
            if pattern in content.lower():
                findings.append(f"CUDA reference detected: {pattern}")
                
        status = "PASS" if not concerns else "NEEDS_WORK"
        
        return AuditResult(
            file_path=str(file_path),
            step_name="mac_compatibility",
            status=status,
            findings=findings,
            concerns=concerns,
            confidence=0.8
        )
        
    def _store_audit_result(self, result: AuditResult, generation: int):
        """Store audit result in database"""
        
        import json
        
        self.db.execute("""
            INSERT INTO audit_results 
            (timestamp, file_path, step_name, status, findings_json, 
             concerns_json, confidence, audit_generation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), result.file_path, result.step_name, result.status,
            json.dumps(result.findings), json.dumps(result.concerns),
            result.confidence, generation
        ))
        
        self.db.commit()
        
    def audit_meta_system(self, generation: int = 0) -> Dict[str, Any]:
        """Audit the entire meta system"""
        
        meta_files = [
            Path("meta_prime.py"),
            Path("meta_watcher.py"), 
            Path("meta_coordinator.py"),
            Path("meta_generator.py"),
            Path("meta_auditor.py")
        ]
        
        all_results = []
        overall_status = "PASS"
        major_concerns = []
        
        for file_path in meta_files:
            if file_path.exists():
                results = self.audit_file(file_path, generation)
                all_results.extend(results)
                
                # Collect major concerns
                for result in results:
                    if result.status == "NEEDS_WORK":
                        major_concerns.extend(result.concerns)
                        
        # Determine overall status
        needs_work_count = sum(1 for r in all_results if r.status == "NEEDS_WORK")
        if needs_work_count > 5:
            overall_status = "NEEDS_WORK"
            
        # Top 5 concerns
        top_concerns = major_concerns[:5]
        
        summary = {
            "overall_status": overall_status,
            "total_audits": len(all_results),
            "needs_work_count": needs_work_count,
            "top_concerns": top_concerns,
            "files_audited": len([f for f in meta_files if f.exists()]),
            "generation": generation
        }
        
        print(f"🎯 Meta System Audit Complete")
        print(f"   Overall Status: {overall_status}")
        print(f"   Files Audited: {summary['files_audited']}")
        print(f"   Issues Found: {needs_work_count}")
        if top_concerns:
            print(f"   Top Concerns:")
            for concern in top_concerns[:3]:
                print(f"     • {concern}")
                
        return summary
        
    def get_audit_report(self, generation: int = None) -> str:
        """Generate audit report"""
        
        if generation is not None:
            where_clause = "WHERE audit_generation = ?"
            params = (generation,)
        else:
            where_clause = ""
            params = ()
            
        cursor = self.db.execute(f"""
            SELECT step_name, status, COUNT(*) as count
            FROM audit_results 
            {where_clause}
            GROUP BY step_name, status
            ORDER BY step_name
        """, params)
        
        audit_stats = cursor.fetchall()
        
        report = f"""
🔍 MetaAuditor Report
━━━━━━━━━━━━━━━━━━━━━━━━

Audit Statistics by Step:
"""
        
        current_step = ""
        for step_name, status, count in audit_stats:
            if step_name != current_step:
                report += f"\n{step_name}:\n"
                current_step = step_name
            report += f"  {status}: {count}\n"
            
        return report


# Test the auditor
if __name__ == "__main__":
    auditor = MetaAuditor()
    
    print(auditor.get_audit_report())
    
    # Audit the current meta system
    summary = auditor.audit_meta_system(generation=0)
    
    print(f"\n{auditor.get_audit_report()}")
    print(f"\nOVERALL STATUS: {summary['overall_status']} ({summary['needs_work_count']} issues)")
    if summary['top_concerns']:
        print(f"TOP CONCERNS: {', '.join(summary['top_concerns'][:3])}")