#!/usr/bin/env python3
"""
Import Analysis Tool for Python 3.13 Standard Library vs Third-Party Dependencies

This tool analyzes Python imports to categorize them as:
1. Standard library modules (built-in to Python 3.13)
2. Third-party packages (external dependencies)
3. Local/project modules (internal to this codebase)
"""

import ast
import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

# Comprehensive Python 3.13 Standard Library modules
# Based on official Python 3.13 documentation
PYTHON_313_STDLIB = {
    # Built-in modules (always available)
    '__future__', '__main__', '_thread', 'builtins', 'sys',
    
    # Core language support
    'abc', 'ast', 'codecs', 'collections', 'copy', 'enum', 'functools',
    'gc', 'inspect', 'io', 'itertools', 'operator', 'pickle', 'types',
    'typing', 'weakref', 'contextlib', 'reprlib', 'dataclasses',
    
    # Text processing
    're', 'string', 'difflib', 'textwrap', 'unicodedata', 'stringprep',
    'readline', 'rlcompleter',
    
    # Binary data
    'struct', 'codecs', 'binascii', 'base64', 'binhex', 'uu', 'quopri',
    
    # Data types
    'datetime', 'calendar', 'collections', 'heapq', 'bisect', 'array',
    'copy', 'pprint', 'reprlib', 'enum', 'graphlib',
    
    # Numeric and mathematical
    'numbers', 'math', 'cmath', 'decimal', 'fractions', 'random',
    'statistics',
    
    # Functional programming
    'itertools', 'functools', 'operator',
    
    # File and directory access
    'pathlib', 'os', 'os.path', 'fileinput', 'stat', 'filecmp',
    'tempfile', 'glob', 'fnmatch', 'linecache', 'shutil',
    
    # Data persistence
    'pickle', 'copyreg', 'shelve', 'marshal', 'dbm', 'sqlite3',
    
    # Data compression and archiving
    'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
    
    # File formats
    'csv', 'configparser', 'tomllib', 'netrc', 'plistlib',
    
    # Cryptographic services
    'hashlib', 'hmac', 'secrets', 'ssl',
    
    # Generic OS services
    'os', 'io', 'time', 'argparse', 'getopt', 'logging', 'getpass',
    'curses', 'platform', 'errno', 'ctypes',
    
    # Concurrent execution
    'threading', 'multiprocessing', 'concurrent', 'subprocess', 'sched',
    'queue', '_thread', 'asyncio', 'asynchat', 'asyncore',
    
    # Networking and interprocess communication
    'socket', 'ssl', 'select', 'selectors', 'signal', 'mmap',
    
    # Internet data handling
    'email', 'json', 'mailbox', 'mimetypes', 'base64', 'binhex',
    'binascii', 'quopri', 'uu',
    
    # Structured markup processing
    'html', 'xml',
    
    # Internet protocols and support
    'webbrowser', 'cgi', 'cgitb', 'wsgiref', 'urllib', 'http',
    'ftplib', 'poplib', 'imaplib', 'smtplib', 'uuid', 'socketserver',
    'xmlrpc',
    
    # Multimedia services
    'audioop', 'aifc', 'sunau', 'wave', 'chunk', 'colorsys', 'imghdr',
    'sndhdr', 'ossaudiodev',
    
    # Internationalization
    'gettext', 'locale',
    
    # Program frameworks
    'turtle', 'cmd', 'shlex',
    
    # Graphical user interfaces
    'tkinter',
    
    # Development tools
    'typing', 'pydoc', 'doctest', 'unittest', 'test', 'lib2to3',
    
    # Debugging and profiling
    'bdb', 'faulthandler', 'pdb', 'profile', 'cProfile', 'pstats',
    'timeit', 'trace', 'tracemalloc',
    
    # Software packaging and distribution
    'distutils', 'ensurepip', 'venv', 'zipapp',
    
    # Python runtime services
    'sys', 'sysconfig', 'builtins', '__main__', 'warnings', 'dataclasses',
    'contextlib', 'abc', 'atexit', 'traceback', 'importlib', 'keyword',
    'pkgutil', 'modulefinder', 'runpy', 'importlib',
    
    # Custom Python interpreters
    'code', 'codeop',
    
    # Importing modules
    'zipimport', 'pkgutil', 'modulefinder', 'runpy', 'importlib',
    
    # Python language services
    'ast', 'symtable', 'symbol', 'token', 'keyword', 'tokenize',
    'tabnanny', 'pyclbr', 'py_compile', 'compileall', 'dis', 'pickletools',
    
    # MS Windows specific
    'msilib', 'msvcrt', 'winreg', 'winsound',
    
    # Unix specific
    'posix', 'pwd', 'grp', 'crypt', 'termios', 'tty', 'pty', 'fcntl',
    'resource', 'nis', 'syslog',
    
    # Superseded modules (deprecated but still available in 3.13)
    'optparse', 'imp',
    
    # Python 3.13 specific additions
    'pathlib', 'tomllib', 'graphlib',
}

# Modules that were removed or deprecated in recent Python versions
DEPRECATED_MODULES = {
    'imp': ('3.4', 'Use importlib instead'),
    'formatter': ('3.4', 'No replacement'),
    'optparse': ('3.2', 'Use argparse instead'),
    'platform': ('Partial deprecation in 3.8', 'Some functions deprecated'),
}

# Common third-party packages that might conflict with stdlib names
POTENTIAL_CONFLICTS = {
    'collections': 'collections-extended',
    'queue': 'Queue (Python 2)',
    'urllib': 'urllib3',
    'json': 'ujson, orjson',
    'ssl': 'pyOpenSSL',
    'sqlite3': 'sqlite3-python',
    'uuid': 'uuid-python',
    'logging': 'loguru',
    'datetime': 'pendulum, arrow',
    'pathlib': 'pathlib2',
    'statistics': 'numpy, scipy',
    'math': 'numpy, scipy',
    'random': 'numpy.random',
}

class ImportAnalyzer:
    """Analyzes Python imports to categorize them."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.stdlib_imports = set()
        self.third_party_imports = set()
        self.local_imports = set()
        self.import_locations = defaultdict(list)
        self.import_errors = []
        
    def is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is part of Python standard library."""
        # Handle submodules (e.g., os.path -> os)
        base_module = module_name.split('.')[0]
        return base_module in PYTHON_313_STDLIB
    
    def is_local_module(self, module_name: str, file_path: Path) -> bool:
        """Check if a module is local to the project."""
        # Handle relative imports
        if module_name.startswith('.'):
            return True
            
        # Check if module exists in project structure
        base_module = module_name.split('.')[0]
        
        # Common local module patterns
        local_patterns = ['src', 'unity_wheel', 'jarvis', 'jarvis2', 'einstein', 'bolt', 'meta']
        if base_module in local_patterns:
            return True
            
        # Check if file exists in project
        possible_paths = [
            self.project_root / f"{base_module}.py",
            self.project_root / base_module / "__init__.py",
            self.project_root / "src" / f"{base_module}.py",
            self.project_root / "src" / base_module / "__init__.py",
        ]
        
        return any(path.exists() for path in possible_paths)
    
    def extract_imports_from_file(self, file_path: Path) -> List[Tuple[str, int, str]]:
        """Extract all imports from a Python file."""
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, node.lineno, 'import'))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append((node.module, node.lineno, 'from'))
                        
        except Exception as e:
            self.import_errors.append(f"{file_path}: {str(e)}")
            
        return imports
    
    def analyze_file(self, file_path: Path):
        """Analyze imports in a single Python file."""
        imports = self.extract_imports_from_file(file_path)
        
        for module_name, line_no, import_type in imports:
            location = f"{file_path.relative_to(self.project_root)}:{line_no}"
            self.import_locations[module_name].append(location)
            
            if self.is_stdlib_module(module_name):
                self.stdlib_imports.add(module_name)
            elif self.is_local_module(module_name, file_path):
                self.local_imports.add(module_name)
            else:
                self.third_party_imports.add(module_name)
    
    def analyze_project(self) -> Dict[str, Any]:
        """Analyze all Python files in the project."""
        python_files = list(self.project_root.rglob("*.py"))
        
        print(f"Analyzing {len(python_files)} Python files...")
        
        for file_path in python_files:
            # Skip test files, cache, and build directories
            relative_path = file_path.relative_to(self.project_root)
            if any(part.startswith('.') or part in ['__pycache__', 'build', 'dist', 'node_modules'] 
                   for part in relative_path.parts):
                continue
                
            self.analyze_file(file_path)
        
        # Identify potential conflicts
        potential_conflicts = {}
        for module in self.third_party_imports:
            if module.split('.')[0] in POTENTIAL_CONFLICTS:
                potential_conflicts[module] = POTENTIAL_CONFLICTS[module.split('.')[0]]
        
        # Check for deprecated modules
        deprecated_found = {}
        for module in self.stdlib_imports:
            if module in DEPRECATED_MODULES:
                deprecated_found[module] = DEPRECATED_MODULES[module]
        
        return {
            'summary': {
                'total_stdlib_modules': len(self.stdlib_imports),
                'total_third_party_modules': len(self.third_party_imports),
                'total_local_modules': len(self.local_imports),
                'total_unique_imports': len(self.stdlib_imports) + len(self.third_party_imports) + len(self.local_imports),
                'files_analyzed': len(python_files),
                'import_errors': len(self.import_errors)
            },
            'stdlib_modules': sorted(list(self.stdlib_imports)),
            'third_party_modules': sorted(list(self.third_party_imports)),
            'local_modules': sorted(list(self.local_imports)),
            'potential_conflicts': potential_conflicts,
            'deprecated_modules': deprecated_found,
            'import_locations': dict(self.import_locations),
            'errors': self.import_errors
        }

def main():
    """Run the import analysis."""
    project_root = os.getcwd()
    analyzer = ImportAnalyzer(project_root)
    
    print("Starting import analysis...")
    print(f"Project root: {project_root}")
    print(f"Python 3.13 stdlib modules: {len(PYTHON_313_STDLIB)}")
    print()
    
    results = analyzer.analyze_project()
    
    # Print summary
    print("=" * 80)
    print("IMPORT ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Files analyzed: {results['summary']['files_analyzed']}")
    print(f"Total unique imports: {results['summary']['total_unique_imports']}")
    print(f"Standard library modules: {results['summary']['total_stdlib_modules']}")
    print(f"Third-party modules: {results['summary']['total_third_party_modules']}")
    print(f"Local/project modules: {results['summary']['total_local_modules']}")
    print(f"Import errors: {results['summary']['import_errors']}")
    print()
    
    # Standard library modules
    print("STANDARD LIBRARY IMPORTS:")
    print("-" * 40)
    for module in results['stdlib_modules']:
        print(f"  {module}")
    print()
    
    # Third-party modules
    print("THIRD-PARTY IMPORTS:")
    print("-" * 40)
    for module in results['third_party_modules']:
        print(f"  {module}")
    print()
    
    # Local modules
    print("LOCAL/PROJECT IMPORTS:")
    print("-" * 40)
    for module in results['local_modules']:
        print(f"  {module}")
    print()
    
    # Potential conflicts
    if results['potential_conflicts']:
        print("POTENTIAL STDLIB/THIRD-PARTY CONFLICTS:")
        print("-" * 40)
        for module, alternatives in results['potential_conflicts'].items():
            print(f"  {module} -> {alternatives}")
        print()
    
    # Deprecated modules
    if results['deprecated_modules']:
        print("DEPRECATED STDLIB MODULES FOUND:")
        print("-" * 40)
        for module, (version, replacement) in results['deprecated_modules'].items():
            print(f"  {module} (deprecated in {version}): {replacement}")
        print()
    
    # Errors
    if results['errors']:
        print("IMPORT ANALYSIS ERRORS:")
        print("-" * 40)
        for error in results['errors']:
            print(f"  {error}")
        print()
    
    # Save detailed results
    output_file = "import_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()