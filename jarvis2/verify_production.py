#!/usr/bin/env python3
"""Comprehensive verification of Jarvis2 production readiness."""
import ast
import json
import re
import subprocess
from pathlib import Path


class ProductionVerifier:
    """Verify Jarvis2 is truly production ready."""
    
    def __init__(self):
        self.issues = []
        self.jarvis_path = Path(__file__).parent
        
    def run_verification(self):
        """Run all verification checks."""
        print("=== JARVIS2 PRODUCTION VERIFICATION ===\n")
        
        # 1. Pattern search
        print("1. Searching for anti-patterns...")
        self.search_anti_patterns()
        
        # 2. Function analysis
        print("\n2. Analyzing function implementations...")
        self.analyze_functions()
        
        # 3. Async verification
        print("\n3. Verifying async functions...")
        self.verify_async_functions()
        
        # 4. Configuration audit
        print("\n4. Auditing hardcoded values...")
        self.audit_hardcoded_values()
        
        # 5. Exception handling
        print("\n5. Checking exception handlers...")
        self.check_exception_handlers()
        
        # 6. Import verification
        print("\n6. Verifying imports...")
        self.verify_imports()
        
        # Report
        self.generate_report()
    
    def search_anti_patterns(self):
        """Search for TODO, dummy, etc."""
        patterns = [
            'TODO', 'FIXME', 'XXX', 'HACK',
            'dummy', 'mock', 'fake', 'placeholder',
            'stub', 'simplified', 'return 0\\.1',
            'return 0$', 'pass\\s*#', '\\.\\.\\.\\s*$'
        ]
        
        for pattern in patterns:
            try:
                result = subprocess.run(
                    ['grep', '-r', '-n', '-i', pattern, str(self.jarvis_path)],
                    capture_output=True, text=True
                )
                
                if result.stdout:
                    matches = result.stdout.strip().split('\n')
                    for match in matches:
                        # Skip this verification script and docs
                        if 'verify_production.py' in match or '.md' in match:
                            continue
                        # Skip __pycache__
                        if '__pycache__' in match:
                            continue
                        self.issues.append({
                            'type': 'anti-pattern',
                            'pattern': pattern,
                            'location': match
                        })
            except Exception as e:
                print(f"  Error searching for {pattern}: {e}")
    
    def analyze_functions(self):
        """Find functions with trivial implementations."""
        for py_file in self.jarvis_path.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check for trivial functions
                        if len(node.body) == 1:
                            stmt = node.body[0]
                            # Single pass
                            if isinstance(stmt, ast.Pass):
                                self.issues.append({
                                    'type': 'trivial_function',
                                    'name': node.name,
                                    'file': str(py_file.relative_to(self.jarvis_path)),
                                    'line': node.lineno,
                                    'reason': 'single pass statement'
                                })
                            # Single return constant
                            elif isinstance(stmt, ast.Return):
                                if isinstance(stmt.value, ast.Constant):
                                    self.issues.append({
                                        'type': 'trivial_function',
                                        'name': node.name,
                                        'file': str(py_file.relative_to(self.jarvis_path)),
                                        'line': node.lineno,
                                        'reason': f'returns constant: {stmt.value.value}'
                                    })
                        # Very short functions (< 3 meaningful lines)
                        elif len([s for s in node.body if not isinstance(s, (ast.Pass, ast.Expr))]) < 2:
                            if node.name not in ['__init__', '__str__', '__repr__']:
                                self.issues.append({
                                    'type': 'trivial_function',
                                    'name': node.name,
                                    'file': str(py_file.relative_to(self.jarvis_path)),
                                    'line': node.lineno,
                                    'reason': 'less than 2 meaningful statements'
                                })
                                
            except Exception as e:
                print(f"  Error analyzing {py_file}: {e}")
    
    def verify_async_functions(self):
        """Find async functions without await."""
        for py_file in self.jarvis_path.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.AsyncFunctionDef):
                        # Check if function has any await
                        has_await = False
                        for child in ast.walk(node):
                            if isinstance(child, (ast.Await, ast.AsyncWith, ast.AsyncFor)):
                                has_await = True
                                break
                        
                        if not has_await:
                            self.issues.append({
                                'type': 'async_without_await',
                                'name': node.name,
                                'file': str(py_file.relative_to(self.jarvis_path)),
                                'line': node.lineno
                            })
                            
            except Exception as e:
                print(f"  Error verifying async in {py_file}: {e}")
    
    def audit_hardcoded_values(self):
        """Find hardcoded configuration values."""
        # Common patterns for hardcoded values
        patterns = [
            (r'\b\d{3,4}\b', 'dimension'),  # 768, 512, etc
            (r'0\.\d+', 'float_constant'),  # 0.1, 0.5, etc
            (r'batch_size\s*=\s*\d+', 'batch_size'),
            (r'learning_rate\s*=\s*[\d.e-]+', 'learning_rate'),
            (r'dropout\s*=\s*[\d.]+', 'dropout'),
            (r'max_\w+\s*=\s*\d+', 'max_limit'),
        ]
        
        for py_file in self.jarvis_path.rglob("*.py"):
            if '__pycache__' in str(py_file) or 'config' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern, value_type in patterns:
                        if re.search(pattern, line):
                            # Skip comments
                            if '#' in line and line.index('#') < line.index(re.search(pattern, line).group()):
                                continue
                            self.issues.append({
                                'type': 'hardcoded_value',
                                'value_type': value_type,
                                'file': str(py_file.relative_to(self.jarvis_path)),
                                'line': line_num,
                                'content': line.strip()
                            })
                            
            except Exception as e:
                print(f"  Error auditing {py_file}: {e}")
    
    def check_exception_handlers(self):
        """Find empty or overly broad exception handlers."""
        for py_file in self.jarvis_path.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ExceptHandler):
                        # Check for bare except
                        if node.type is None:
                            self.issues.append({
                                'type': 'bare_except',
                                'file': str(py_file.relative_to(self.jarvis_path)),
                                'line': node.lineno
                            })
                        # Check for empty handler
                        elif len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                            self.issues.append({
                                'type': 'empty_except',
                                'file': str(py_file.relative_to(self.jarvis_path)),
                                'line': node.lineno
                            })
                            
            except Exception as e:
                print(f"  Error checking exceptions in {py_file}: {e}")
    
    def verify_imports(self):
        """Verify all imports are used."""
        for py_file in self.jarvis_path.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                imports = []
                
                # Collect imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module.split('.')[0])
                
                # Check if used (simple check)
                for imp in set(imports):
                    # Skip common always-used imports
                    if imp in ['__future__', 'typing', 'dataclasses', 'abc']:
                        continue
                    
                    # Remove import lines and check if still referenced
                    content_no_imports = '\n'.join(
                        line for line in content.split('\n')
                        if not line.strip().startswith(('import ', 'from '))
                    )
                    
                    if imp not in content_no_imports:
                        self.issues.append({
                            'type': 'unused_import',
                            'import': imp,
                            'file': str(py_file.relative_to(self.jarvis_path))
                        })
                        
            except Exception as e:
                print(f"  Error verifying imports in {py_file}: {e}")
    
    def generate_report(self):
        """Generate final verification report."""
        print("\n" + "="*60)
        print("VERIFICATION REPORT")
        print("="*60)
        
        if not self.issues:
            print("\n✅ NO ISSUES FOUND! Jarvis2 appears to be production ready.")
        else:
            print(f"\n❌ FOUND {len(self.issues)} ISSUES:\n")
            
            # Group by type
            issues_by_type = {}
            for issue in self.issues:
                issue_type = issue['type']
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append(issue)
            
            # Report by type
            for issue_type, issues in issues_by_type.items():
                print(f"\n{issue_type.upper()} ({len(issues)} found):")
                print("-" * 40)
                
                for issue in issues[:5]:  # Show first 5
                    if issue_type == 'anti-pattern':
                        print(f"  Pattern '{issue['pattern']}': {issue['location']}")
                    elif issue_type == 'trivial_function':
                        print(f"  {issue['file']}:{issue['line']} - {issue['name']}() - {issue['reason']}")
                    elif issue_type == 'async_without_await':
                        print(f"  {issue['file']}:{issue['line']} - async {issue['name']}() has no await")
                    elif issue_type == 'hardcoded_value':
                        print(f"  {issue['file']}:{issue['line']} - {issue['value_type']}: {issue['content']}")
                    elif issue_type == 'unused_import':
                        print(f"  {issue['file']} - unused import: {issue['import']}")
                    else:
                        print(f"  {issue}")
                
                if len(issues) > 5:
                    print(f"  ... and {len(issues) - 5} more")
        
        # Save detailed report
        with open('jarvis2_verification_report.json', 'w') as f:
            json.dump({
                'total_issues': len(self.issues),
                'issues_by_type': {k: len(v) for k, v in issues_by_type.items()},
                'issues': self.issues
            }, f, indent=2)
        
        print(f"\nDetailed report saved to: jarvis2_verification_report.json")


if __name__ == "__main__":
    verifier = ProductionVerifier()
    verifier.run_verification()