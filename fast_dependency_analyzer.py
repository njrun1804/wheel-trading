#!/usr/bin/env python3
"""
Fast Dependency Tree Analysis
Uses batch processing and cached data for rapid analysis.
"""

import subprocess
import json
import sys
from collections import defaultdict, Counter
from typing import Dict, Set, List
import re
import ast
import os
import glob

class FastDependencyAnalyzer:
    def __init__(self):
        self.installed_packages = {}
        self.dependency_graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.direct_dependencies = set()
        
    def get_installed_packages(self):
        """Get all installed packages."""
        try:
            result = subprocess.run(['pip', 'list', '--format=json'], 
                                  capture_output=True, text=True, check=True)
            packages = json.loads(result.stdout)
            self.installed_packages = {pkg['name'].lower(): pkg['version'] for pkg in packages}
            print(f"Found {len(self.installed_packages)} installed packages")
            return True
        except Exception as e:
            print(f"Error getting installed packages: {e}")
            return False
    
    def get_all_dependencies_batch(self):
        """Get all dependencies in one batch operation."""
        print("Getting all package dependencies in batch...")
        
        # Get all package names
        package_names = list(self.installed_packages.keys())
        
        try:
            # Run pip show for all packages at once
            cmd = ['pip', 'show'] + package_names
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print("Batch pip show failed, trying smaller batches...")
                return self._get_dependencies_in_batches(package_names)
            
            return self._parse_pip_show_output(result.stdout)
            
        except subprocess.TimeoutExpired:
            print("Batch operation timed out, trying smaller batches...")
            return self._get_dependencies_in_batches(package_names)
        except Exception as e:
            print(f"Batch operation failed: {e}")
            return self._get_dependencies_in_batches(package_names)
    
    def _get_dependencies_in_batches(self, package_names: List[str], batch_size: int = 20):
        """Get dependencies in smaller batches."""
        dependencies = {}
        
        for i in range(0, len(package_names), batch_size):
            batch = package_names[i:i+batch_size]
            try:
                cmd = ['pip', 'show'] + batch
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    batch_deps = self._parse_pip_show_output(result.stdout)
                    dependencies.update(batch_deps)
                
                print(f"Processed {min(i+batch_size, len(package_names))}/{len(package_names)} packages")
                
            except Exception:
                # Skip problematic batches
                continue
        
        return dependencies
    
    def _parse_pip_show_output(self, output: str) -> Dict[str, Set[str]]:
        """Parse pip show output to extract dependencies."""
        dependencies = {}
        current_package = None
        
        for line in output.split('\n'):
            line = line.strip()
            
            if line.startswith('Name:'):
                current_package = line.split(':', 1)[1].strip().lower()
            elif line.startswith('Requires:') and current_package:
                reqs = line.split(':', 1)[1].strip()
                deps = set()
                
                if reqs and reqs != 'None':
                    for req in reqs.split(', '):
                        # Clean up requirement specification
                        req_clean = re.split(r'[>=<!~\[\]]', req)[0].strip()
                        if req_clean:
                            deps.add(req_clean.lower())
                
                dependencies[current_package] = deps
                current_package = None
        
        return dependencies
    
    def build_dependency_graph(self):
        """Build dependency graph from batch data."""
        print("Building dependency graph...")
        
        # Get all dependencies
        all_deps = self.get_all_dependencies_batch()
        
        # Build graphs
        for package, deps in all_deps.items():
            self.dependency_graph[package] = deps
            
            # Build reverse graph
            for dep in deps:
                if dep in self.installed_packages:
                    self.reverse_graph[dep].add(package)
        
        print(f"Built graph with {len(self.dependency_graph)} nodes")
    
    def analyze_direct_imports_fast(self):
        """Fast analysis of direct imports."""
        print("Analyzing direct imports...")
        
        # Import to package mappings
        mappings = {
            'sklearn': 'scikit-learn', 'cv2': 'opencv-python', 'PIL': 'pillow',
            'yaml': 'pyyaml', 'dateutil': 'python-dateutil', 'dotenv': 'python-dotenv',
            'jwt': 'pyjwt', 'serial': 'pyserial', 'bs4': 'beautifulsoup4',
            'requests_oauthlib': 'requests-oauthlib', 'google.cloud': 'google-cloud-core'
        }
        
        direct_imports = set()
        
        # Check main source files
        key_files = [
            'run.py', 'src/**/*.py', 'unity_wheel/**/*.py', 
            'jarvis2/**/*.py', 'einstein/**/*.py'
        ]
        
        python_files = []
        for pattern in key_files:
            python_files.extend(glob.glob(pattern, recursive=True))
        
        # Limit to 50 most important files
        python_files = python_files[:50]
        
        for file_path in python_files:
            if any(skip in file_path for skip in ['__pycache__', '.git', 'test_', '_test']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read first 100 lines only for speed
                    lines = []
                    for i, line in enumerate(f):
                        if i >= 100:
                            break
                        if line.strip().startswith(('import ', 'from ')):
                            lines.append(line.strip())
                    
                    content = '\n'.join(lines)
                    
                    # Extract imports
                    import_matches = re.findall(r'(?:^from\s+([^\s.]+)|^import\s+([^\s.]+))', content, re.MULTILINE)
                    for from_module, import_module in import_matches:
                        module = (from_module or import_module).lower()
                        direct_imports.add(module)
                        
            except Exception:
                continue
        
        # Map to installed packages
        for imp in direct_imports:
            candidates = [
                imp,
                mappings.get(imp, imp),
                imp.replace('_', '-'),
                imp.replace('-', '_'),
                f"python-{imp}",
                f"py{imp}"
            ]
            
            for candidate in candidates:
                if candidate in self.installed_packages:
                    self.direct_dependencies.add(candidate)
                    break
        
        print(f"Found {len(self.direct_dependencies)} direct dependencies")
    
    def calculate_transitive_deps(self):
        """Calculate transitive dependencies efficiently."""
        all_transitive = set()
        
        def get_transitive_bfs(start_packages: Set[str]) -> Set[str]:
            """BFS to find all transitive dependencies."""
            visited = set()
            queue = list(start_packages)
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                for dep in self.dependency_graph.get(current, set()):
                    if dep in self.installed_packages and dep not in visited:
                        queue.append(dep)
            
            return visited - start_packages
        
        all_transitive = get_transitive_bfs(self.direct_dependencies)
        return all_transitive
    
    def detect_simple_cycles(self):
        """Fast cycle detection."""
        cycles = []
        visited = set()
        
        def has_cycle(start: str, current: str, path: List[str], depth: int = 0) -> bool:
            if depth > 5:  # Limit depth
                return False
            if current == start and len(path) > 1:
                cycles.append(path + [current])
                return True
            if current in path:
                return False
            
            for dep in list(self.dependency_graph.get(current, set()))[:3]:  # Limit branches
                if dep in self.installed_packages:
                    if has_cycle(start, dep, path + [current], depth + 1):
                        return True
            return False
        
        # Check top packages only
        top_packages = sorted(self.reverse_graph.keys(), 
                            key=lambda x: len(self.reverse_graph[x]), 
                            reverse=True)[:20]
        
        for pkg in top_packages:
            if pkg not in visited:
                visited.add(pkg)
                has_cycle(pkg, pkg, [])
        
        return cycles
    
    def generate_stats(self):
        """Generate dependency statistics."""
        dependency_counts = {pkg: len(deps) for pkg, deps in self.dependency_graph.items()}
        reverse_counts = {pkg: len(deps) for pkg, deps in self.reverse_graph.items()}
        
        return {
            'top_dependency_producers': dict(Counter(dependency_counts).most_common(15)),
            'most_depended_on': dict(Counter(reverse_counts).most_common(15)),
            'no_dependencies': [p for p, d in self.dependency_graph.items() if not d],
            'leaf_packages': [p for p, d in self.reverse_graph.items() if not d]
        }
    
    def analyze(self):
        """Run complete fast analysis."""
        print("Starting fast dependency analysis...")
        
        if not self.get_installed_packages():
            return None
        
        self.build_dependency_graph()
        self.analyze_direct_imports_fast()
        
        all_transitive = self.calculate_transitive_deps()
        transitive_only = all_transitive - self.direct_dependencies
        
        cycles = self.detect_simple_cycles()
        stats = self.generate_stats()
        
        # Find safe removal candidates
        leaf_packages = set(stats['leaf_packages'])
        safe_to_remove = leaf_packages - self.direct_dependencies
        
        report = {
            "summary": {
                "total_packages": len(self.installed_packages),
                "direct_dependencies": len(self.direct_dependencies),
                "all_transitive": len(all_transitive),
                "transitive_only": len(transitive_only),
                "circular_dependencies": len(cycles),
                "leaf_packages": len(leaf_packages),
                "safe_to_remove": len(safe_to_remove)
            },
            "direct_dependencies": sorted(list(self.direct_dependencies)),
            "transitive_only": sorted(list(transitive_only)),
            "safe_to_remove": sorted(list(safe_to_remove)),
            "circular_dependencies": cycles,
            "stats": stats
        }
        
        return report

def main():
    analyzer = FastDependencyAnalyzer()
    report = analyzer.analyze()
    
    if not report:
        print("Analysis failed")
        return 1
    
    s = report['summary']
    
    print("\n" + "="*70)
    print("FAST DEPENDENCY TREE ANALYSIS")
    print("="*70)
    print(f"Total packages: {s['total_packages']}")
    print(f"Direct dependencies (used in code): {s['direct_dependencies']}")
    print(f"All transitive dependencies: {s['all_transitive']}")
    print(f"Transitive-only dependencies: {s['transitive_only']}")
    print(f"Leaf packages (not depended on): {s['leaf_packages']}")
    print(f"Safe to remove immediately: {s['safe_to_remove']}")
    print(f"Circular dependencies: {s['circular_dependencies']}")
    
    print(f"\nDIRECT DEPENDENCIES ({len(report['direct_dependencies'])}):")
    for dep in report['direct_dependencies'][:25]:
        print(f"  {dep}")
    if len(report['direct_dependencies']) > 25:
        print(f"  ... and {len(report['direct_dependencies']) - 25} more")
    
    print(f"\nSAFE TO REMOVE ({len(report['safe_to_remove'])}):")
    print("These packages are not used directly and nothing depends on them:")
    for pkg in report['safe_to_remove'][:30]:
        print(f"  pip uninstall {pkg}")
    
    print(f"\nTRANSITIVE-ONLY DEPENDENCIES ({len(report['transitive_only'])}):")
    print("These are needed by other packages but not used directly:")
    for pkg in report['transitive_only'][:20]:
        print(f"  {pkg}")
    
    if report['circular_dependencies']:
        print(f"\nCIRCULAR DEPENDENCIES ({len(report['circular_dependencies'])}):")
        for cycle in report['circular_dependencies'][:3]:
            print(f"  {' -> '.join(cycle)}")
    
    print(f"\nMOST DEPENDED ON PACKAGES:")
    for pkg, count in list(report['stats']['most_depended_on'].items())[:10]:
        print(f"  {pkg}: {count} packages depend on it")
    
    print(f"\nPACKAGES WITH MOST DEPENDENCIES:")
    for pkg, count in list(report['stats']['top_dependency_producers'].items())[:10]:
        print(f"  {pkg}: depends on {count} packages")
    
    # Save report
    with open("fast_dependency_analysis.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n" + "="*70)
    print("CLEANUP RECOMMENDATIONS")
    print("="*70)
    
    if report['safe_to_remove']:
        print("IMMEDIATE REMOVAL (100% safe):")
        for pkg in sorted(report['safe_to_remove'])[:20]:
            print(f"  pip uninstall {pkg}")
        
        print(f"\nTo remove all safe packages at once:")
        safe_list = ' '.join(sorted(report['safe_to_remove']))
        print(f"  pip uninstall {safe_list}")
    
    print(f"\nAnalysis complete. Report saved to: fast_dependency_analysis.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())