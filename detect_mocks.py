#!/usr/bin/env python3
"""Detect mock usage in the codebase and suggest replacements."""

import ast
import os
from typing import Dict, List, Set, Tuple
import re


class MockDetector(ast.NodeVisitor):
    """AST visitor to detect mock usage patterns."""
    
    def __init__(self):
        self.mock_imports = []
        self.patch_decorators = []
        self.mock_calls = []
        self.fake_data = []
        
    def visit_Import(self, node):
        """Detect import mock statements."""
        for alias in node.names:
            if 'mock' in alias.name.lower():
                self.mock_imports.append({
                    'line': node.lineno,
                    'import': alias.name,
                    'alias': alias.asname
                })
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Detect from x import mock statements."""
        if node.module and 'mock' in node.module.lower():
            for alias in node.names:
                self.mock_imports.append({
                    'line': node.lineno,
                    'module': node.module,
                    'import': alias.name,
                    'alias': alias.asname
                })
        # Also check for unittest.mock
        elif node.module == 'unittest':
            for alias in node.names:
                if alias.name == 'mock':
                    self.mock_imports.append({
                        'line': node.lineno,
                        'module': 'unittest',
                        'import': 'mock',
                        'alias': alias.asname
                    })
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Detect @patch decorators."""
        for decorator in node.decorator_list:
            decorator_str = ast.unparse(decorator) if hasattr(ast, 'unparse') else str(decorator)
            if 'patch' in decorator_str or 'mock' in decorator_str:
                self.patch_decorators.append({
                    'line': node.lineno,
                    'function': node.name,
                    'decorator': decorator_str
                })
        self.generic_visit(node)
        
    def visit_Call(self, node):
        """Detect Mock() and MagicMock() calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id in ['Mock', 'MagicMock', 'AsyncMock', 'PropertyMock']:
                self.mock_calls.append({
                    'line': node.lineno,
                    'type': node.func.id
                })
        self.generic_visit(node)
        
    def visit_Assign(self, node):
        """Detect fake API keys and test data."""
        # Check for api_key assignments
        if hasattr(node.value, 'value'):
            value_str = str(node.value.value) if hasattr(node.value, 'value') else ''
            # Check targets
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if 'api_key' in target.id.lower():
                        if value_str.lower() in ['test', 'dummy', 'fake', 'mock']:
                            self.fake_data.append({
                                'line': node.lineno,
                                'variable': target.id,
                                'value': value_str
                            })
        self.generic_visit(node)


def analyze_file(filepath: str) -> Dict[str, List]:
    """Analyze a single Python file for mock usage."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        detector = MockDetector()
        detector.visit(tree)
        
        # Also do regex searches for common patterns
        regex_patterns = {
            'patch_decorator': r'@(mock\.)?patch',
            'with_patch': r'with\s+(mock\.)?patch',
            'mock_open': r'mock_open\(',
            'return_value': r'\.return_value\s*=',
            'side_effect': r'\.side_effect\s*=',
            'assert_called': r'\.assert_called',
            'fake_api_key': r'api_key\s*=\s*["\'](?:test|dummy|fake|mock)',
        }
        
        regex_matches = {}
        for pattern_name, pattern in regex_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            regex_matches[pattern_name] = [m.start() for m in matches]
        
        return {
            'mock_imports': detector.mock_imports,
            'patch_decorators': detector.patch_decorators,
            'mock_calls': detector.mock_calls,
            'fake_data': detector.fake_data,
            'regex_matches': regex_matches
        }
    except Exception as e:
        return {'error': str(e)}


def scan_directory(directory: str, exclude_dirs: List[str] = None) -> Dict[str, Dict]:
    """Scan directory for mock usage."""
    if exclude_dirs is None:
        exclude_dirs = ['venv', '.venv', '__pycache__', '.git', 'node_modules']
    
    results = {}
    
    for root, dirs, files in os.walk(directory):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                relative_path = os.path.relpath(filepath, directory)
                
                analysis = analyze_file(filepath)
                if any(analysis.get(k) for k in ['mock_imports', 'patch_decorators', 'mock_calls', 'fake_data']):
                    results[relative_path] = analysis
                elif analysis.get('regex_matches'):
                    # Check if regex found anything
                    if any(matches for matches in analysis['regex_matches'].values()):
                        results[relative_path] = analysis
    
    return results


def generate_report(results: Dict[str, Dict], output_file: str = None):
    """Generate a detailed report of mock usage."""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("MOCK USAGE DETECTION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Total files with mocks: {len(results)}\n")
    
    # Summary statistics
    total_imports = sum(len(r.get('mock_imports', [])) for r in results.values())
    total_patches = sum(len(r.get('patch_decorators', [])) for r in results.values())
    total_calls = sum(len(r.get('mock_calls', [])) for r in results.values())
    total_fake = sum(len(r.get('fake_data', [])) for r in results.values())
    
    report_lines.append("SUMMARY:")
    report_lines.append(f"  Mock imports: {total_imports}")
    report_lines.append(f"  Patch decorators: {total_patches}")
    report_lines.append(f"  Mock() calls: {total_calls}")
    report_lines.append(f"  Fake API keys/data: {total_fake}")
    report_lines.append("")
    
    # Group by test vs non-test files
    test_files = {k: v for k, v in results.items() if 'test' in k}
    non_test_files = {k: v for k, v in results.items() if 'test' not in k}
    
    if non_test_files:
        report_lines.append("\n❌ CRITICAL: Mock usage in non-test files:")
        report_lines.append("="*60)
        for filepath, analysis in sorted(non_test_files.items()):
            report_lines.append(f"\n{filepath}:")
            _add_file_details(report_lines, analysis)
    
    if test_files:
        report_lines.append("\n\n⚠️  Mock usage in test files:")
        report_lines.append("="*60)
        
        # Group by directory
        by_dir = {}
        for filepath in sorted(test_files.keys()):
            dir_name = os.path.dirname(filepath)
            if dir_name not in by_dir:
                by_dir[dir_name] = []
            by_dir[dir_name].append(filepath)
        
        for dir_name, files in sorted(by_dir.items()):
            report_lines.append(f"\n{dir_name}/")
            for filepath in files:
                basename = os.path.basename(filepath)
                report_lines.append(f"  - {basename}")
                analysis = test_files[filepath]
                
                # Just show counts
                mock_count = len(analysis.get('mock_imports', []))
                patch_count = len(analysis.get('patch_decorators', []))
                if mock_count or patch_count:
                    report_lines.append(f"    Imports: {mock_count}, Patches: {patch_count}")
    
    # Recommendations
    report_lines.append("\n\n" + "="*80)
    report_lines.append("RECOMMENDATIONS:")
    report_lines.append("="*80)
    
    report_lines.append("\n1. Replace mock API keys with environment variables:")
    report_lines.append("   - Use get_databento_api_key() from secrets.integration")
    report_lines.append("   - Use get_fred_api_key() from secrets.integration")
    
    report_lines.append("\n2. For integration tests, use real APIs with:")
    report_lines.append("   - Test-specific API keys")
    report_lines.append("   - Rate limiting to avoid quota issues")
    report_lines.append("   - Cached responses for repeatability")
    
    report_lines.append("\n3. For unit tests that must mock:")
    report_lines.append("   - Use dependency injection")
    report_lines.append("   - Create test fixtures with real data structure")
    report_lines.append("   - Document why mocking is necessary")
    
    report_lines.append("\n4. Priority fixes:")
    if non_test_files:
        report_lines.append("   ❌ Remove ALL mocks from non-test files immediately")
    report_lines.append("   - Start with API client tests")
    report_lines.append("   - Then data provider tests")
    report_lines.append("   - Finally strategy and analytics tests")
    
    # Output report
    report_text = '\n'.join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    else:
        print(report_text)
    
    return report_text


def _add_file_details(lines: List[str], analysis: Dict):
    """Add detailed analysis for a file."""
    if analysis.get('mock_imports'):
        lines.append("  Mock imports:")
        for imp in analysis['mock_imports']:
            lines.append(f"    Line {imp['line']}: {imp.get('module', '')}.{imp['import']}")
    
    if analysis.get('patch_decorators'):
        lines.append("  Patch decorators:")
        for patch in analysis['patch_decorators']:
            lines.append(f"    Line {patch['line']}: {patch['function']}()")
    
    if analysis.get('fake_data'):
        lines.append("  Fake data:")
        for fake in analysis['fake_data']:
            lines.append(f"    Line {fake['line']}: {fake['variable']} = '{fake['value']}'")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect mock usage in Python code')
    parser.add_argument('directory', nargs='?', default='.', help='Directory to scan')
    parser.add_argument('-o', '--output', help='Output file for report')
    parser.add_argument('--exclude', nargs='*', help='Additional directories to exclude')
    
    args = parser.parse_args()
    
    exclude_dirs = ['venv', '.venv', '__pycache__', '.git', 'node_modules', 'archive']
    if args.exclude:
        exclude_dirs.extend(args.exclude)
    
    print(f"Scanning {args.directory} for mock usage...")
    results = scan_directory(args.directory, exclude_dirs)
    
    generate_report(results, args.output)


if __name__ == "__main__":
    main()