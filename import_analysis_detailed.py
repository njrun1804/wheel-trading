#!/usr/bin/env python3
"""
Detailed Import Analysis - Focus on Conflicts and Categorization Issues
"""

import json
from collections import defaultdict
from typing import Dict, List, Set, Any

def analyze_import_conflicts():
    """Analyze the import results for detailed conflicts and issues."""
    
    # Load the analysis results
    with open('import_analysis_results.json', 'r') as f:
        results = json.load(f)
    
    # Extract data
    stdlib_modules = set(results['stdlib_modules'])
    third_party_modules = set(results['third_party_modules'])
    local_modules = set(results['local_modules'])
    
    # Analysis categories
    analysis = {
        'summary': results['summary'],
        'categorization_issues': [],
        'potential_stdlib_conflicts': [],
        'ambiguous_modules': [],
        'local_module_patterns': defaultdict(int),
        'third_party_analysis': {
            'core_dependencies': [],
            'ml_ai_libraries': [],
            'data_processing': [],
            'web_frameworks': [],
            'system_libraries': [],
            'other': []
        },
        'stdlib_usage_analysis': {
            'async_concurrent': [],
            'data_structures': [],
            'file_io': [],
            'networking': [],
            'math_science': [],
            'system_os': [],
            'testing_debug': [],
            'other': []
        }
    }
    
    # Check for modules that might be misclassified
    suspicious_stdlib = []
    for module in stdlib_modules:
        if '.' in module and module not in ['collections.abc', 'concurrent.futures', 'email.mime.multipart', 
                                             'email.mime.text', 'importlib.util', 'logging.config', 
                                             'logging.handlers', 'multiprocessing.shared_memory', 
                                             'unittest.mock', 'urllib.parse']:
            # These might be local modules misclassified as stdlib
            if not module.startswith(('math.', 'logging.', 'email.', 'xml.', 'html.')):
                suspicious_stdlib.append(module)
    
    analysis['categorization_issues'] = suspicious_stdlib
    
    # Find potential conflicts between stdlib and third-party names
    stdlib_base_names = {m.split('.')[0] for m in stdlib_modules}
    third_party_base_names = {m.split('.')[0] for m in third_party_modules}
    
    conflicts = []
    for name in stdlib_base_names.intersection(third_party_base_names):
        stdlib_versions = [m for m in stdlib_modules if m.startswith(name)]
        third_party_versions = [m for m in third_party_modules if m.startswith(name)]
        conflicts.append({
            'module_name': name,
            'stdlib_imports': stdlib_versions,
            'third_party_imports': third_party_versions
        })
    
    analysis['potential_stdlib_conflicts'] = conflicts
    
    # Analyze local module patterns
    for module in local_modules:
        base = module.split('.')[0]
        analysis['local_module_patterns'][base] += 1
    
    # Categorize third-party modules
    ml_ai_keywords = ['torch', 'tensorflow', 'sklearn', 'numpy', 'scipy', 'pandas', 'mlx', 
                      'faiss', 'sentence_transformers', 'transformers', 'neural', 'gpu']
    data_keywords = ['duckdb', 'sqlite', 'polars', 'pyarrow', 'databento', 'redis', 'database']
    web_keywords = ['flask', 'fastapi', 'httpx', 'requests', 'aiohttp', 'uvloop', 'websocket']
    system_keywords = ['psutil', 'watchdog', 'watchfiles', 'ctypes', 'metal', 'cuda', 'opencl']
    
    for module in third_party_modules:
        base_module = module.split('.')[0].lower()
        
        if any(keyword in base_module for keyword in ml_ai_keywords):
            analysis['third_party_analysis']['ml_ai_libraries'].append(module)
        elif any(keyword in base_module for keyword in data_keywords):
            analysis['third_party_analysis']['data_processing'].append(module)
        elif any(keyword in base_module for keyword in web_keywords):
            analysis['third_party_analysis']['web_frameworks'].append(module)
        elif any(keyword in base_module for keyword in system_keywords):
            analysis['third_party_analysis']['system_libraries'].append(module)
        elif base_module in ['click', 'pydantic', 'rich', 'black', 'pytest', 'hypothesis', 
                            'yaml', 'toml', 'orjson', 'cryptography', 'packaging']:
            analysis['third_party_analysis']['core_dependencies'].append(module)
        else:
            analysis['third_party_analysis']['other'].append(module)
    
    # Categorize stdlib usage
    async_modules = [m for m in stdlib_modules if 'async' in m or m in ['threading', 'multiprocessing', 'concurrent.futures', 'queue']]
    data_modules = [m for m in stdlib_modules if m in ['collections', 'collections.abc', 'heapq', 'bisect', 'array']]
    file_modules = [m for m in stdlib_modules if m in ['os', 'pathlib', 'shutil', 'tempfile', 'glob', 'io']]
    network_modules = [m for m in stdlib_modules if m in ['socket', 'ssl', 'urllib.parse', 'smtplib', 'webbrowser']]
    math_modules = [m for m in stdlib_modules if m.startswith('math') or m in ['statistics', 'decimal', 'random']]
    system_modules = [m for m in stdlib_modules if m in ['sys', 'os', 'platform', 'signal', 'resource', 'ctypes']]
    test_modules = [m for m in stdlib_modules if m.startswith('unittest') or m in ['pstats', 'cProfile', 'traceback', 'tracemalloc']]
    
    analysis['stdlib_usage_analysis']['async_concurrent'] = async_modules
    analysis['stdlib_usage_analysis']['data_structures'] = data_modules
    analysis['stdlib_usage_analysis']['file_io'] = file_modules
    analysis['stdlib_usage_analysis']['networking'] = network_modules
    analysis['stdlib_usage_analysis']['math_science'] = math_modules
    analysis['stdlib_usage_analysis']['system_os'] = system_modules
    analysis['stdlib_usage_analysis']['testing_debug'] = test_modules
    
    remaining_stdlib = stdlib_modules - set(async_modules + data_modules + file_modules + 
                                           network_modules + math_modules + system_modules + test_modules)
    analysis['stdlib_usage_analysis']['other'] = list(remaining_stdlib)
    
    return analysis

def print_detailed_analysis(analysis: Dict[str, Any]):
    """Print the detailed analysis results."""
    
    print("=" * 80)
    print("DETAILED IMPORT ANALYSIS REPORT")
    print("=" * 80)
    
    # Summary
    summary = analysis['summary']
    print(f"Files Analyzed: {summary['files_analyzed']:,}")
    print(f"Total Unique Imports: {summary['total_unique_imports']:,}")
    print(f"Standard Library: {summary['total_stdlib_modules']}")
    print(f"Third-Party: {summary['total_third_party_modules']}")
    print(f"Local/Project: {summary['total_local_modules']}")
    print(f"Import Errors: {summary['import_errors']}")
    print()
    
    # Categorization Issues
    if analysis['categorization_issues']:
        print("POTENTIAL CATEGORIZATION ISSUES:")
        print("-" * 50)
        print("These modules are classified as stdlib but might be local:")
        for module in analysis['categorization_issues']:
            print(f"  {module}")
        print()
    
    # Conflicts
    if analysis['potential_stdlib_conflicts']:
        print("STDLIB/THIRD-PARTY NAME CONFLICTS:")
        print("-" * 50)
        for conflict in analysis['potential_stdlib_conflicts']:
            print(f"Module name: {conflict['module_name']}")
            print(f"  Stdlib: {conflict['stdlib_imports']}")
            print(f"  Third-party: {conflict['third_party_imports']}")
            print()
    
    # Local module patterns
    print("LOCAL MODULE PATTERNS:")
    print("-" * 50)
    sorted_patterns = sorted(analysis['local_module_patterns'].items(), key=lambda x: x[1], reverse=True)
    for pattern, count in sorted_patterns[:15]:  # Top 15
        print(f"  {pattern}: {count} imports")
    print()
    
    # Third-party analysis
    print("THIRD-PARTY LIBRARY CATEGORIES:")
    print("-" * 50)
    tp_analysis = analysis['third_party_analysis']
    
    categories = [
        ('Core Dependencies', tp_analysis['core_dependencies']),
        ('ML/AI Libraries', tp_analysis['ml_ai_libraries']),
        ('Data Processing', tp_analysis['data_processing']),
        ('Web Frameworks', tp_analysis['web_frameworks']),
        ('System Libraries', tp_analysis['system_libraries']),
        ('Other', tp_analysis['other'])
    ]
    
    for category, modules in categories:
        print(f"{category} ({len(modules)}):")
        for module in sorted(modules)[:10]:  # Show first 10
            print(f"  {module}")
        if len(modules) > 10:
            print(f"  ... and {len(modules) - 10} more")
        print()
    
    # Stdlib usage analysis
    print("STANDARD LIBRARY USAGE PATTERNS:")
    print("-" * 50)
    stdlib_analysis = analysis['stdlib_usage_analysis']
    
    stdlib_categories = [
        ('Async/Concurrent', stdlib_analysis['async_concurrent']),
        ('Data Structures', stdlib_analysis['data_structures']),
        ('File I/O', stdlib_analysis['file_io']),
        ('Networking', stdlib_analysis['networking']),
        ('Math/Science', stdlib_analysis['math_science']),
        ('System/OS', stdlib_analysis['system_os']),
        ('Testing/Debug', stdlib_analysis['testing_debug']),
        ('Other', stdlib_analysis['other'])
    ]
    
    for category, modules in stdlib_categories:
        if modules:
            print(f"{category} ({len(modules)}): {', '.join(sorted(modules))}")
    print()

def main():
    """Run the detailed analysis."""
    try:
        analysis = analyze_import_conflicts()
        print_detailed_analysis(analysis)
        
        # Save detailed analysis
        with open('import_analysis_detailed.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print("Detailed analysis saved to: import_analysis_detailed.json")
        
    except FileNotFoundError:
        print("Error: import_analysis_results.json not found. Run import_analyzer.py first.")
        return
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        return

if __name__ == "__main__":
    main()