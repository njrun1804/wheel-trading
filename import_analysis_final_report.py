#!/usr/bin/env python3
"""
Final Import Analysis Report - Complete Analysis with Corrections

This script provides the definitive analysis of Python imports in the wheel-trading codebase,
identifying actual dependencies vs built-in functionality with corrections for misclassifications.
"""

import json
from typing import Dict, List, Set, Any
from collections import defaultdict

def generate_final_report():
    """Generate the final corrected import analysis report."""
    
    # Load the detailed analysis
    with open('import_analysis_detailed.json', 'r') as f:
        detailed_analysis = json.load(f)
    
    # Load the original analysis
    with open('import_analysis_results.json', 'r') as f:
        original_analysis = json.load(f)
    
    # Manual corrections for misclassified modules
    # These are clearly project-local modules that were misclassified as third-party
    local_modules_corrections = {
        'account', 'advisor', 'advisor_simple', 'agent_pool', 'alerts', 'allocator',
        'analytics', 'anomaly_detector', 'api', 'analyzer', 'cache', 'cache_manager',
        'client', 'collector', 'commands', 'confidence', 'context', 'core',
        'dashboard', 'decision', 'dependencies', 'diagnostics', 'display', 'engine',
        'engines', 'exceptions', 'formatters', 'health', 'loader', 'main', 'manager',
        'models', 'monitor', 'monitoring', 'oauth', 'optimizer', 'options', 'performance',
        'phases', 'planner', 'planning', 'pools', 'portfolio', 'position', 'pressure',
        'processor', 'resources', 'routing', 'schema', 'server', 'solution', 'storage',
        'strategies', 'strategy', 'system', 'tasks', 'utils', 'validation', 'wheel'
    }
    
    # Modules that are actually third-party but might be misclassified
    third_party_corrections = {
        'click', 'requests', 'aiohttp', 'pandas', 'numpy', 'scipy', 'matplotlib',
        'flask', 'fastapi', 'pydantic', 'sqlalchemy', 'duckdb', 'polars', 'pyarrow',
        'mlx', 'torch', 'tensorflow', 'sklearn', 'faiss', 'transformers', 'rich',
        'pytest', 'hypothesis', 'black', 'mypy', 'ruff', 'uvloop', 'httpx', 'orjson',
        'cryptography', 'psutil', 'watchdog', 'watchfiles', 'redis', 'lmdb', 'yaml',
        'toml', 'tomli', 'dotenv', 'tenacity', 'packaging', 'setuptools', 'wheel',
        'databento', 'databento_dbn', 'gurobipy', 'pulp', 'ortools', 'numba', 'numexpr',
        'pytz', 'dateutil', 'opik', 'mlflow', 'opentelemetry', 'phoenix', 'tiktoken',
        'sentence_transformers', 'libcst', 'astor', 'blessed', 'jinja2', 'networkx',
        'coremltools', 'pkg_resources', 'psycopg2', 'sqlparse', 'statsmodels',
        'google', 'lz4', 'hnswlib', 'pynvml', 'lru', 'memory_profiler'
    }
    
    # Correct the categorization
    stdlib_modules = set(original_analysis['stdlib_modules'])
    third_party_modules = set(original_analysis['third_party_modules'])
    local_modules = set(original_analysis['local_modules'])
    
    # Move misclassified modules
    for module in list(third_party_modules):
        base_module = module.split('.')[0]
        if (base_module in local_modules_corrections or 
            any(module.startswith(f"{local}.") for local in local_modules_corrections)):
            third_party_modules.discard(module)
            local_modules.add(module)
    
    # Identify actual stdlib vs third-party conflicts
    stdlib_base_names = {m.split('.')[0] for m in stdlib_modules}
    third_party_base_names = {m.split('.')[0] for m in third_party_modules}
    
    actual_conflicts = []
    for name in stdlib_base_names.intersection(third_party_base_names):
        if name in third_party_corrections:  # Only report if it's actually a real third-party package
            stdlib_versions = [m for m in stdlib_modules if m.startswith(name)]
            third_party_versions = [m for m in third_party_modules if m.startswith(name)]
            actual_conflicts.append({
                'module_name': name,
                'stdlib_imports': stdlib_versions,
                'third_party_imports': third_party_versions,
                'risk_level': 'HIGH' if name in ['json', 'sqlite3', 'logging', 'email'] else 'MEDIUM'
            })
    
    # Categorize corrected third-party modules
    ml_ai_packages = []
    data_packages = []
    web_packages = []
    dev_tools = []
    system_packages = []
    core_packages = []
    
    for module in third_party_modules:
        base_module = module.split('.')[0].lower()
        
        # ML/AI packages
        if any(keyword in base_module for keyword in ['torch', 'tensorflow', 'sklearn', 'numpy', 
                                                      'scipy', 'pandas', 'mlx', 'faiss', 'transformers',
                                                      'sentence', 'neural', 'gpu', 'metal', 'cuda']):
            ml_ai_packages.append(module)
        # Data processing packages
        elif any(keyword in base_module for keyword in ['duckdb', 'polars', 'pyarrow', 'databento', 
                                                        'redis', 'lmdb', 'database', 'sqlite']):
            data_packages.append(module)
        # Web frameworks
        elif any(keyword in base_module for keyword in ['flask', 'fastapi', 'aiohttp', 'httpx', 
                                                        'requests', 'uvloop', 'websocket']):
            web_packages.append(module)
        # Development tools
        elif any(keyword in base_module for keyword in ['pytest', 'hypothesis', 'black', 'mypy', 
                                                        'ruff', 'setuptools', 'wheel', 'packaging']):
            dev_tools.append(module)
        # System packages
        elif any(keyword in base_module for keyword in ['psutil', 'watchdog', 'watchfiles', 'ctypes']):
            system_packages.append(module)
        # Core packages
        elif base_module in ['click', 'rich', 'pydantic', 'cryptography', 'tenacity', 'dotenv',
                            'yaml', 'toml', 'tomli', 'orjson', 'dateutil', 'pytz']:
            core_packages.append(module)
    
    # Identify problematic imports that need attention
    problematic_imports = []
    
    # Check for imports that might be incorrect
    for module in stdlib_modules:
        if '.' in module and not module.startswith(('collections.', 'concurrent.', 'email.', 
                                                   'importlib.', 'logging.', 'multiprocessing.', 
                                                   'unittest.', 'urllib.', 'xml.', 'html.')):
            if module not in ['math.options', 'math.options_enhanced', 'math.options_gpu']:
                problematic_imports.append({
                    'module': module,
                    'issue': 'Possibly misclassified as stdlib',
                    'suggested_category': 'local'
                })
    
    # Generate final report
    final_report = {
        'analysis_metadata': {
            'total_files_analyzed': original_analysis['summary']['files_analyzed'],
            'total_unique_imports': len(stdlib_modules) + len(third_party_modules) + len(local_modules),
            'analysis_date': '2025-06-16',
            'python_version': '3.13',
            'corrections_applied': True
        },
        'corrected_categorization': {
            'stdlib_modules': {
                'count': len(stdlib_modules),
                'modules': sorted(list(stdlib_modules))
            },
            'third_party_modules': {
                'count': len(third_party_modules),
                'modules': sorted(list(third_party_modules))
            },
            'local_modules': {
                'count': len(local_modules),
                'modules': sorted(list(local_modules))
            }
        },
        'third_party_categorization': {
            'ml_ai_packages': {
                'count': len(ml_ai_packages),
                'modules': sorted(ml_ai_packages)
            },
            'data_processing_packages': {
                'count': len(data_packages),
                'modules': sorted(data_packages)
            },
            'web_frameworks': {
                'count': len(web_packages),
                'modules': sorted(web_packages)
            },
            'development_tools': {
                'count': len(dev_tools),
                'modules': sorted(dev_tools)
            },
            'system_packages': {
                'count': len(system_packages),
                'modules': sorted(system_packages)
            },
            'core_packages': {
                'count': len(core_packages),
                'modules': sorted(core_packages)
            }
        },
        'potential_conflicts': actual_conflicts,
        'problematic_imports': problematic_imports,
        'stdlib_usage_analysis': {
            'async_concurrent': [m for m in stdlib_modules if m in ['asyncio', 'threading', 'multiprocessing', 'concurrent.futures', 'queue']],
            'data_structures': [m for m in stdlib_modules if m in ['collections', 'collections.abc', 'heapq', 'bisect', 'array']],
            'file_io': [m for m in stdlib_modules if m in ['os', 'pathlib', 'shutil', 'tempfile', 'glob', 'io']],
            'networking': [m for m in stdlib_modules if m in ['socket', 'ssl', 'urllib.parse', 'smtplib', 'webbrowser']],
            'math_science': [m for m in stdlib_modules if m.startswith('math') or m in ['statistics', 'decimal', 'random']],
            'system_os': [m for m in stdlib_modules if m in ['sys', 'os', 'platform', 'signal', 'resource', 'ctypes']],
            'testing_debug': [m for m in stdlib_modules if m.startswith('unittest') or m in ['pstats', 'cProfile', 'traceback', 'tracemalloc']],
            'text_processing': [m for m in stdlib_modules if m in ['re', 'string', 'textwrap', 'difflib']],
            'serialization': [m for m in stdlib_modules if m in ['json', 'pickle', 'base64', 'binascii']],
            'crypto_security': [m for m in stdlib_modules if m in ['hashlib', 'hmac', 'secrets', 'ssl']],
            'other': [m for m in stdlib_modules if m not in [
                'asyncio', 'threading', 'multiprocessing', 'concurrent.futures', 'queue',
                'collections', 'collections.abc', 'heapq', 'bisect', 'array',
                'os', 'pathlib', 'shutil', 'tempfile', 'glob', 'io',
                'socket', 'ssl', 'urllib.parse', 'smtplib', 'webbrowser',
                'sys', 'platform', 'signal', 'resource', 'ctypes',
                're', 'string', 'textwrap', 'difflib',
                'json', 'pickle', 'base64', 'binascii',
                'hashlib', 'hmac', 'secrets', 'ssl'
            ] and not m.startswith(('math', 'unittest', 'statistics', 'decimal', 'random', 'pstats', 'cProfile', 'traceback', 'tracemalloc'))]
        },
        'recommendations': {
            'dependency_management': [
                'Consider using requirements.txt or pyproject.toml to clearly define third-party dependencies',
                'Pin versions of critical dependencies like numpy, pandas, and ML libraries',
                'Consider using dependency groups for different environments (dev, test, prod)'
            ],
            'code_organization': [
                'Ensure all local modules are properly organized under src/ directory',
                'Consider creating __init__.py files for better package structure',
                'Use absolute imports for better clarity'
            ],
            'performance_optimization': [
                'Consider using faster alternatives like orjson instead of json for performance-critical paths',
                'Use uvloop for async operations if not already implemented',
                'Consider caching frequently imported modules'
            ],
            'security_considerations': [
                'Audit cryptography and security-related packages regularly',
                'Use secrets module instead of random for cryptographic operations',
                'Consider using virtual environments to isolate dependencies'
            ]
        }
    }
    
    return final_report

def print_final_report(report: Dict[str, Any]):
    """Print the final analysis report."""
    
    print("=" * 100)
    print("FINAL IMPORT ANALYSIS REPORT - WHEEL TRADING CODEBASE")
    print("=" * 100)
    
    # Metadata
    metadata = report['analysis_metadata']
    print(f"Analysis Date: {metadata['analysis_date']}")
    print(f"Python Version: {metadata['python_version']}")
    print(f"Files Analyzed: {metadata['total_files_analyzed']:,}")
    print(f"Total Unique Imports: {metadata['total_unique_imports']:,}")
    print(f"Corrections Applied: {metadata['corrections_applied']}")
    print()
    
    # Corrected categorization summary
    cat = report['corrected_categorization']
    print("CORRECTED IMPORT CATEGORIZATION:")
    print("-" * 60)
    print(f"Standard Library Modules: {cat['stdlib_modules']['count']}")
    print(f"Third-Party Packages: {cat['third_party_modules']['count']}")
    print(f"Local/Project Modules: {cat['local_modules']['count']}")
    print()
    
    # Third-party categorization
    tp_cat = report['third_party_categorization']
    print("THIRD-PARTY PACKAGE CATEGORIES:")
    print("-" * 60)
    for category, data in tp_cat.items():
        category_name = category.replace('_', ' ').title()
        print(f"{category_name}: {data['count']} packages")
        if data['modules']:
            # Show first 5 modules
            sample = data['modules'][:5]
            print(f"  Examples: {', '.join(sample)}")
            if len(data['modules']) > 5:
                print(f"  ... and {len(data['modules']) - 5} more")
    print()
    
    # Potential conflicts
    conflicts = report['potential_conflicts']
    if conflicts:
        print("POTENTIAL STDLIB/THIRD-PARTY CONFLICTS:")
        print("-" * 60)
        for conflict in conflicts:
            risk = conflict['risk_level']
            print(f"⚠️  {conflict['module_name']} [{risk} RISK]")
            print(f"   Stdlib: {conflict['stdlib_imports']}")
            print(f"   Third-party: {conflict['third_party_imports']}")
            print()
    
    # Problematic imports
    problematic = report['problematic_imports']
    if problematic:
        print("PROBLEMATIC IMPORTS REQUIRING ATTENTION:")
        print("-" * 60)
        for item in problematic[:10]:  # Show first 10
            print(f"❌ {item['module']}: {item['issue']} -> {item['suggested_category']}")
        if len(problematic) > 10:
            print(f"... and {len(problematic) - 10} more issues")
        print()
    
    # Standard library usage
    stdlib_usage = report['stdlib_usage_analysis']
    print("STANDARD LIBRARY USAGE PATTERNS:")
    print("-" * 60)
    for category, modules in stdlib_usage.items():
        if modules:
            category_name = category.replace('_', ' ').title()
            print(f"{category_name} ({len(modules)}): {', '.join(sorted(modules)[:5])}")
            if len(modules) > 5:
                print(f"  ... and {len(modules) - 5} more")
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 60)
    recommendations = report['recommendations']
    for category, items in recommendations.items():
        category_name = category.replace('_', ' ').title()
        print(f"{category_name}:")
        for item in items:
            print(f"  • {item}")
        print()

def main():
    """Generate and display the final report."""
    
    try:
        print("Generating final corrected import analysis report...")
        report = generate_final_report()
        
        print_final_report(report)
        
        # Save the final report
        with open('final_import_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Final report saved to: final_import_analysis_report.json")
        
    except FileNotFoundError as e:
        print(f"Error: Required analysis file not found: {e}")
        print("Please run import_analyzer.py and import_analysis_detailed.py first.")
    except Exception as e:
        print(f"Error generating final report: {e}")
        raise

if __name__ == "__main__":
    main()