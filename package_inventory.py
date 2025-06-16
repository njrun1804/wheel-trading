#!/usr/bin/env python3
"""
Comprehensive Python Package Inventory for Wheel Trading System
Creates detailed inventory of all installed packages with metadata
"""

import json
import sys
import subprocess
import os
from datetime import datetime

def get_pip_list():
    """Get all packages from pip list"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'], 
                              capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error getting pip list: {e}")
        return []

def get_package_info(package_name):
    """Get detailed info for a specific package"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name], 
                              capture_output=True, text=True, check=True)
        info = {}
        for line in result.stdout.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()
        return info
    except Exception:
        return {}

def get_editable_packages():
    """Get editable packages"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--editable'], 
                              capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) < 3:
            return []
        
        editable = []
        for line in lines[2:]:  # Skip header
            if line.strip() and not line.startswith('-'):
                parts = line.split()
                if len(parts) >= 3:
                    editable.append({
                        'name': parts[0],
                        'version': parts[1],
                        'location': ' '.join(parts[2:])
                    })
        return editable
    except Exception as e:
        print(f"Error getting editable packages: {e}")
        return []

def categorize_installation_source(location):
    """Categorize installation source based on location"""
    if not location:
        return 'unknown'
    
    location_lower = location.lower()
    if '.pyenv' in location_lower:
        return 'pyenv'
    elif '.local' in location_lower:
        return 'local'
    elif 'homebrew' in location_lower or '/opt/homebrew' in location_lower:
        return 'homebrew'
    elif 'conda' in location_lower or 'anaconda' in location_lower:
        return 'conda'
    elif 'site-packages' in location_lower:
        return 'system'
    else:
        return 'other'

def main():
    print("Creating comprehensive Python package inventory...")
    print(f"Using Python: {sys.executable}")
    
    # Get all packages
    packages = get_pip_list()
    editable_packages = get_editable_packages()
    
    print(f"Found {len(packages)} packages")
    print(f"Found {len(editable_packages)} editable packages")
    
    # Initialize inventory
    inventory = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'created_by': 'wheel-trading package inventory script',
            'python_executable': sys.executable
        },
        'python_info': {
            'version': sys.version,
            'version_info': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
            'executable': sys.executable,
            'platform': sys.platform,
            'prefix': sys.prefix,
            'path': sys.path[:5]  # First 5 paths
        },
        'packages': {},
        'editable_packages': editable_packages,
        'installation_sources': {
            'pyenv': 0,
            'local': 0,
            'homebrew': 0,
            'conda': 0,
            'system': 0,
            'unknown': 0,
            'other': 0
        },
        'key_packages': {},
        'package_categories': {
            'data_science': [],
            'web_frameworks': [],
            'testing': [],
            'development_tools': [],
            'ai_ml': [],
            'trading_finance': [],
            'databases': [],
            'mcp_servers': [],
            'observability': []
        },
        'dependency_analysis': {
            'most_required': [],
            'most_dependencies': [],
            'circular_dependencies': []
        },
        'summary': {
            'total_packages': len(packages),
            'editable_count': len(editable_packages),
            'installation_locations': []
        }
    }
    
    # Key packages to highlight
    key_package_names = [
        'numpy', 'pandas', 'torch', 'tensorflow', 'mlx', 'anthropic', 
        'databento', 'duckdb', 'fastapi', 'uvicorn', 'pytest', 'scipy',
        'matplotlib', 'scikit-learn', 'jupyter', 'black', 'mypy', 'langchain',
        'alpaca-py', 'quantlib', 'cvxpy', 'polars', 'plotly', 'mlflow',
        'arize-phoenix', 'logfire', 'mcp', 'pydantic', 'sqlalchemy'
    ]
    
    # Package categories
    categories = {
        'data_science': ['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'polars', 'pyarrow'],
        'web_frameworks': ['fastapi', 'uvicorn', 'flask', 'starlette', 'gunicorn', 'httpx', 'aiohttp'],
        'testing': ['pytest', 'pytest-cov', 'pytest-mock', 'pytest-benchmark', 'coverage', 'hypothesis'],
        'development_tools': ['black', 'mypy', 'ruff', 'pre-commit', 'isort', 'vulture', 'bandit'],
        'ai_ml': ['torch', 'tensorflow', 'mlx', 'scikit-learn', 'anthropic', 'openai', 'langchain', 'sentence-transformers'],
        'trading_finance': ['databento', 'alpaca-py', 'quantlib', 'cvxpy', 'fredapi', 'yfinance', 'exchange-calendars'],
        'databases': ['duckdb', 'sqlalchemy', 'alembic', 'aiosqlite', 'asyncpg', 'psycopg2-binary'],
        'mcp_servers': ['mcp', 'mcp-py-repl', 'mcp-server-duckdb', 'mcp-server-scikit-learn', 'mcp-server-stats'],
        'observability': ['arize-phoenix', 'logfire', 'mlflow', 'opentelemetry-api', 'prometheus-client', 'structlog']
    }
    
    # Track dependencies for analysis
    dependency_counts = {}
    dependent_counts = {}
    
    # Process packages in batches to show progress
    batch_size = 50
    total_batches = (len(packages) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(packages))
        batch_packages = packages[start_idx:end_idx]
        
        print(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_packages)} packages)...")
        
        for pkg in batch_packages:
            name = pkg['name']
            version = pkg['version']
            
            # Get detailed info
            pkg_info = get_package_info(name)
            
            location = pkg_info.get('Location', '')
            requires = pkg_info.get('Requires', '').split(', ') if pkg_info.get('Requires') else []
            required_by = pkg_info.get('Required-by', '').split(', ') if pkg_info.get('Required-by') else []
            
            # Clean up empty strings
            requires = [r.strip() for r in requires if r.strip()]
            required_by = [r.strip() for r in required_by if r.strip()]
            
            # Track for dependency analysis
            dependency_counts[name] = len(requires)
            dependent_counts[name] = len(required_by)
            
            # Categorize installation source
            install_source = categorize_installation_source(location)
            inventory['installation_sources'][install_source] += 1
            
            # Create package entry
            package_entry = {
                'version': version,
                'location': location,
                'installation_method': install_source,
                'requires': requires,
                'required_by': required_by,
                'dependency_count': len(requires),
                'dependents_count': len(required_by),
                'summary': pkg_info.get('Summary', ''),
                'homepage': pkg_info.get('Home-page', ''),
                'license': pkg_info.get('License', ''),
                'is_key_package': name.lower() in [k.lower() for k in key_package_names]
            }
            
            inventory['packages'][name] = package_entry
            
            # Add to key packages if relevant
            if package_entry['is_key_package']:
                inventory['key_packages'][name] = package_entry
                
            # Categorize package
            for category, pkg_list in categories.items():
                if name.lower() in [p.lower() for p in pkg_list]:
                    inventory['package_categories'][category].append(name)
    
    # Dependency analysis
    print("Analyzing dependencies...")
    
    # Most required packages (dependencies)
    most_required = sorted(dependent_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    inventory['dependency_analysis']['most_required'] = [
        {'package': pkg, 'required_by_count': count} for pkg, count in most_required
    ]
    
    # Packages with most dependencies
    most_dependencies = sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    inventory['dependency_analysis']['most_dependencies'] = [
        {'package': pkg, 'dependency_count': count} for pkg, count in most_dependencies
    ]
    
    # Get installation locations
    locations = set()
    for pkg_name, pkg_info in inventory['packages'].items():
        if pkg_info['location']:
            # Get the site-packages directory
            parts = pkg_info['location'].split('/')
            if 'site-packages' in parts:
                idx = parts.index('site-packages')
                location = '/'.join(parts[:idx+1])
                locations.add(location)
    
    inventory['summary']['installation_locations'] = sorted(list(locations))
    
    # Add disk usage information
    print("Calculating disk usage...")
    disk_usage = []
    for location in inventory['summary']['installation_locations']:
        if os.path.exists(location):
            try:
                size = subprocess.run(['du', '-sh', location], 
                                    capture_output=True, text=True, check=True)
                disk_usage.append({
                    'location': location,
                    'size': size.stdout.split()[0] if size.stdout else 'unknown'
                })
            except Exception:
                disk_usage.append({
                    'location': location,
                    'size': 'unknown'
                })
    
    inventory['summary']['disk_usage'] = disk_usage
    
    # Save to file
    output_file = 'comprehensive_package_inventory.json'
    with open(output_file, 'w') as f:
        json.dump(inventory, f, indent=2)
    
    print(f"\n" + "="*60)
    print("COMPREHENSIVE PACKAGE INVENTORY COMPLETE")
    print("="*60)
    print(f"Total packages: {inventory['summary']['total_packages']}")
    print(f"Editable packages: {inventory['summary']['editable_count']}")
    print(f"Key packages: {len(inventory['key_packages'])}")
    print(f"Installation sources: {inventory['installation_sources']}")
    print(f"\nPackage categories:")
    for category, pkgs in inventory['package_categories'].items():
        if pkgs:
            print(f"  {category}: {len(pkgs)} packages")
    
    print(f"\nTop 5 most required packages:")
    for item in inventory['dependency_analysis']['most_required'][:5]:
        print(f"  {item['package']}: required by {item['required_by_count']} packages")
    
    print(f"\nDisk usage:")
    for usage in inventory['summary']['disk_usage']:
        print(f"  {usage['location']}: {usage['size']}")
    
    print(f"\nSaved to: {os.path.abspath(output_file)}")
    
    return inventory

if __name__ == '__main__':
    main()