#!/usr/bin/env python3
"""
Check specialized requirements files against installed packages.
Focus on critical dependencies for bolt, jarvis2, and claude integration.
"""

import subprocess
import sys
from pathlib import Path
import re
from typing import Dict, List, Tuple, Set
try:
    import pkg_resources
    HAS_PKG_RESOURCES = True
except ImportError:
    HAS_PKG_RESOURCES = False

try:
    from packaging import requirements
    from packaging.version import Version, parse as parse_version
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False

def get_installed_packages() -> Dict[str, str]:
    """Get all installed packages and their versions."""
    installed = {}
    
    if HAS_PKG_RESOURCES:
        try:
            for dist in pkg_resources.working_set:
                installed[dist.project_name.lower()] = dist.version
            return installed
        except Exception as e:
            print(f"Warning: Could not get installed packages via pkg_resources: {e}")
    
    # Fallback to pip list
    result = subprocess.run(['pip3', 'list'], capture_output=True, text=True)
    if result.returncode != 0:
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines[2:]:  # Skip header
            if line.strip() and not line.startswith('-'):
                parts = line.split()
                if len(parts) >= 2:
                    installed[parts[0].lower().replace('_', '-')] = parts[1]
    return installed

def parse_requirements_file(file_path: Path) -> List[Tuple[str, str, str]]:
    """Parse requirements file and return list of (name, version_spec, comment)."""
    reqs = []
    if not file_path.exists():
        return reqs
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Handle comments at end of line
            if '#' in line:
                req_part, comment = line.split('#', 1)
                req_part = req_part.strip()
                comment = comment.strip()
            else:
                req_part = line
                comment = ""
            
            if not req_part:
                continue
                
            # Handle platform/environment markers
            if ';' in req_part:
                req_part = req_part.split(';')[0].strip()
            
            if HAS_PACKAGING:
                try:
                    req = requirements.Requirement(req_part)
                    reqs.append((req.name.lower(), str(req.specifier) if req.specifier else "", comment))
                except Exception as e:
                    print(f"Warning: Could not parse requirement '{req_part}' at line {line_num}: {e}")
            else:
                # Simple parsing without packaging library
                if '>=' in req_part:
                    name, version = req_part.split('>=', 1)
                    reqs.append((name.strip().lower(), f">={version.strip()}", comment))
                elif '==' in req_part:
                    name, version = req_part.split('==', 1)
                    reqs.append((name.strip().lower(), f"=={version.strip()}", comment))
                elif '<' in req_part:
                    name, version = req_part.split('<', 1)
                    reqs.append((name.strip().lower(), f"<{version.strip()}", comment))
                else:
                    reqs.append((req_part.strip().lower(), "", comment))
    
    return reqs

def check_version_compatibility(installed_version: str, required_spec: str) -> bool:
    """Check if installed version satisfies requirement specification."""
    if not required_spec or not HAS_PACKAGING:
        return True
    
    try:
        installed_ver = parse_version(installed_version)
        spec = requirements.Requirement(f"dummy{required_spec}")
        return installed_ver in spec.specifier
    except Exception:
        return True  # Be permissive if we can't parse

def analyze_requirements():
    """Analyze all specialized requirements files."""
    base_dir = Path(__file__).parent
    installed = get_installed_packages()
    
    requirements_files = {
        'Bolt Multi-Agent System': 'requirements_bolt.txt',
        'Jarvis2 Meta-Coding': 'jarvis2_requirements.txt', 
        'Claude Integration': 'requirements_claude_integration.txt'
    }
    
    print("🔍 SPECIALIZED REQUIREMENTS ANALYSIS")
    print("=" * 60)
    
    all_missing = []
    all_version_issues = []
    critical_missing = []
    
    for system_name, req_file in requirements_files.items():
        print(f"\n📋 {system_name}")
        print("-" * 40)
        
        req_path = base_dir / req_file
        if not req_path.exists():
            print(f"❌ Requirements file not found: {req_file}")
            continue
            
        reqs = parse_requirements_file(req_path)
        if not reqs:
            print(f"⚠️  No requirements found in {req_file}")
            continue
        
        missing = []
        version_issues = []
        installed_count = 0
        
        for name, version_spec, comment in reqs:
            if name in installed:
                installed_version = installed[name]
                installed_count += 1
                
                if version_spec and not check_version_compatibility(installed_version, version_spec):
                    version_issues.append((name, installed_version, version_spec, comment))
                    print(f"⚠️  {name}: {installed_version} (want {version_spec}) - {comment}")
                else:
                    print(f"✅ {name}: {installed_version} - {comment}")
            else:
                missing.append((name, version_spec, comment))
                print(f"❌ {name}: NOT INSTALLED (want {version_spec}) - {comment}")
                
                # Check if this is critical
                if any(critical in comment.lower() for critical in ['critical', 'core', 'essential', 'mlx', 'gpu', 'acceleration']):
                    critical_missing.append((system_name, name, version_spec, comment))
        
        print(f"\n📊 Summary for {system_name}:")
        print(f"   ✅ Installed: {installed_count}")
        print(f"   ❌ Missing: {len(missing)}")
        print(f"   ⚠️  Version issues: {len(version_issues)}")
        
        all_missing.extend([(system_name, name, spec, comment) for name, spec, comment in missing])
        all_version_issues.extend([(system_name, name, installed_ver, spec, comment) for name, installed_ver, spec, comment in version_issues])
    
    # Critical Dependencies Analysis
    print(f"\n🎯 CRITICAL DEPENDENCIES ANALYSIS")
    print("=" * 60)
    
    critical_deps = {
        'MLX (M4 Pro GPU)': ['mlx', 'mlx-lm'],
        'Async Libraries': ['aiofiles', 'asyncio-mqtt', 'httpx'],
        'AI/ML Frameworks': ['transformers', 'sentence-transformers', 'torch'],
        'Performance': ['numpy', 'pandas', 'scipy', 'orjson', 'lmdb'],
        'Code Analysis': ['ast-comments', 'libcst', 'black', 'mypy'],
        'Hardware Monitoring': ['psutil', 'py-spy', 'memory-profiler']
    }
    
    for category, deps in critical_deps.items():
        print(f"\n🔧 {category}:")
        for dep in deps:
            if dep in installed:
                print(f"   ✅ {dep}: {installed[dep]}")
            else:
                print(f"   ❌ {dep}: NOT INSTALLED")
    
    # Final Summary
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total packages checked: {len(set(name for _, name, _, _ in all_missing + [(s, n, '', '', c) for s, n, i, r, c in all_version_issues]))}")
    print(f"Missing packages: {len(all_missing)}")
    print(f"Version conflicts: {len(all_version_issues)}")
    print(f"Critical missing: {len(critical_missing)}")
    
    if critical_missing:
        print(f"\n🚨 CRITICAL MISSING DEPENDENCIES:")
        for system, name, spec, comment in critical_missing:
            print(f"   {system}: {name} {spec} - {comment}")
    
    # Installation Commands
    if all_missing:
        print(f"\n💡 INSTALLATION COMMANDS:")
        print("Run these commands to install missing packages:")
        
        for system_name, req_file in requirements_files.items():
            system_missing = [name for s, name, _, _ in all_missing if s == system_name]
            if system_missing:
                print(f"\n# {system_name}")
                print(f"pip install -r {req_file}")
    
    return len(all_missing) == 0 and len(all_version_issues) == 0

if __name__ == "__main__":
    success = analyze_requirements()
    sys.exit(0 if success else 1)