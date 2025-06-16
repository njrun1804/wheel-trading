#!/usr/bin/env python3
"""
Simple Dependency Tree Analysis
Fast analysis of transitive dependencies and direct dependencies.
"""

import subprocess
import json
import sys
from collections import defaultdict, deque
from typing import Dict, Set, List
import ast
import os
import re

class SimpleDependencyAnalyzer:
    def __init__(self):
        self.installed_packages = {}
        self.dependency_graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.direct_dependencies = set()
        
    def get_installed_packages(self):
        """Get all installed packages with versions."""
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
    
    def get_package_dependencies_batch(self, package_names: List[str]) -> Dict[str, Set[str]]:
        """Get dependencies for multiple packages efficiently."""
        dependencies = {}
        
        for package in package_names:
            deps = set()
            try:
                result = subprocess.run(['pip', 'show', package], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('Requires:'):
                            reqs = line.split(':', 1)[1].strip()
                            if reqs and reqs != 'None':
                                for req in reqs.split(', '):
                                    # Clean up requirement specification
                                    req_clean = re.split(r'[>=<!~]', req)[0].strip()
                                    if req_clean:
                                        deps.add(req_clean.lower())
                            break
            except Exception:
                pass
            dependencies[package] = deps
        
        return dependencies
    
    def build_dependency_graph(self):
        """Build dependency graph efficiently."""
        print("Building dependency graph...")
        
        # Process packages in batches
        package_list = list(self.installed_packages.keys())
        batch_size = 50
        
        for i in range(0, len(package_list), batch_size):
            batch = package_list[i:i+batch_size]
            batch_deps = self.get_package_dependencies_batch(batch)
            
            for package, deps in batch_deps.items():
                self.dependency_graph[package] = deps
                # Build reverse graph
                for dep in deps:
                    if dep in self.installed_packages:
                        self.reverse_graph[dep].add(package)
            
            print(f"Processed {min(i+batch_size, len(package_list))}/{len(package_list)} packages")
        
        print(f"Built graph with {len(self.dependency_graph)} nodes")
    
    def analyze_direct_imports(self):
        """Analyze Python files to find direct imports."""
        print("Analyzing direct imports from code...")
        
        # Common import mappings
        import_mappings = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'pillow',
            'yaml': 'pyyaml',
            'dateutil': 'python-dateutil',
            'dotenv': 'python-dotenv',
            'jwt': 'pyjwt',
            'serial': 'pyserial',
        }
        
        direct_imports = set()
        
        # Find Python files in common locations
        search_paths = [
            'src/**/*.py',
            '*.py',
            'tests/**/*.py',
            'examples/**/*.py',
            'scripts/**/*.py'
        ]
        
        import glob
        python_files = []
        for pattern in search_paths:
            python_files.extend(glob.glob(pattern, recursive=True))
        
        # Limit to first 100 files for speed
        python_files = python_files[:100]
        
        for file_path in python_files:
            if '__pycache__' in file_path or '.git' in file_path:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Simple regex approach for speed
                    import_lines = re.findall(r'^(?:from\s+(\w+)|import\s+(\w+))', content, re.MULTILINE)
                    for from_module, import_module in import_lines:
                        module = (from_module or import_module).lower()
                        direct_imports.add(module)
                        
            except Exception:
                continue
        
        # Map imports to installed packages
        for imp in direct_imports:
            # Direct match
            if imp in self.installed_packages:
                self.direct_dependencies.add(imp)
            # Try mappings
            elif imp in import_mappings and import_mappings[imp] in self.installed_packages:
                self.direct_dependencies.add(import_mappings[imp])
            # Try common transformations
            else:
                candidates = [
                    imp,
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
    
    def find_transitive_dependencies(self):
        """Find all transitive dependencies of direct dependencies."""
        all_transitive = set()
        
        def get_transitive(package: str, visited: Set[str] = None) -> Set[str]:
            if visited is None:
                visited = set()
            if package in visited:
                return set()
            
            visited.add(package)
            transitive = set()
            
            for dep in self.dependency_graph.get(package, set()):
                if dep in self.installed_packages:
                    transitive.add(dep)
                    transitive.update(get_transitive(dep, visited.copy()))
            
            return transitive
        
        for direct_dep in self.direct_dependencies:
            all_transitive.update(get_transitive(direct_dep))
        
        return all_transitive
    
    def detect_simple_cycles(self):
        """Detect simple circular dependencies."""
        cycles = []
        visited = set()
        
        def dfs(node: str, path: List[str], rec_stack: Set[str]):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if len(cycle) <= 5:  # Only report short cycles
                    cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in list(self.dependency_graph.get(node, set()))[:3]:  # Limit for speed
                if neighbor in self.installed_packages:
                    dfs(neighbor, path.copy(), rec_stack.copy())
            
            rec_stack.remove(node)
        
        # Only check top 50 packages for speed
        top_packages = list(self.installed_packages.keys())[:50]
        for package in top_packages:
            if package not in visited:
                dfs(package, [], set())
        
        return cycles
    
    def analyze(self):
        """Run complete dependency analysis."""
        print("Starting dependency analysis...")
        
        # Step 1: Get installed packages
        if not self.get_installed_packages():
            return None
        
        # Step 2: Build dependency graph
        self.build_dependency_graph()
        
        # Step 3: Analyze direct imports
        self.analyze_direct_imports()
        
        # Step 4: Find transitive dependencies
        all_transitive = self.find_transitive_dependencies()
        transitive_only = all_transitive - self.direct_dependencies
        
        # Step 5: Detect cycles
        cycles = self.detect_simple_cycles()
        
        # Step 6: Calculate some basic stats
        dependency_counts = {pkg: len(deps) for pkg, deps in self.dependency_graph.items()}
        reverse_counts = {pkg: len(deps) for pkg, deps in self.reverse_graph.items()}
        
        report = {
            "summary": {
                "total_packages": len(self.installed_packages),
                "direct_dependencies": len(self.direct_dependencies),
                "all_transitive": len(all_transitive),
                "transitive_only": len(transitive_only),
                "circular_dependencies": len(cycles),
                "packages_with_no_deps": len([p for p, d in self.dependency_graph.items() if not d]),
                "packages_not_depended_on": len([p for p, d in self.reverse_graph.items() if not d])
            },
            "direct_dependencies": sorted(list(self.direct_dependencies)),
            "transitive_only": sorted(list(transitive_only)),
            "circular_dependencies": cycles,
            "top_dependency_counts": dict(sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            "most_depended_on": dict(sorted(reverse_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            "packages_with_no_dependencies": [p for p, d in self.dependency_graph.items() if not d][:20],
            "leaf_packages": [p for p, d in self.reverse_graph.items() if not d][:20]
        }
        
        return report

def main():
    analyzer = SimpleDependencyAnalyzer()
    report = analyzer.analyze()
    
    if not report:
        print("Failed to generate dependency analysis")
        return 1
    
    # Print summary
    print("\n" + "="*60)
    print("DEPENDENCY TREE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total installed packages: {report['summary']['total_packages']}")
    print(f"Direct dependencies (used in code): {report['summary']['direct_dependencies']}")
    print(f"All transitive dependencies: {report['summary']['all_transitive']}")
    print(f"Transitive-only dependencies: {report['summary']['transitive_only']}")
    print(f"Packages with no dependencies: {report['summary']['packages_with_no_deps']}")
    print(f"Packages not depended on (potential leaves): {report['summary']['packages_not_depended_on']}")
    print(f"Circular dependencies found: {report['summary']['circular_dependencies']}")
    
    # Print direct dependencies
    print(f"\nDIRECT DEPENDENCIES ({len(report['direct_dependencies'])}):")
    for i, dep in enumerate(report['direct_dependencies']):
        if i < 20:
            print(f"  {dep}")
        elif i == 20:
            print(f"  ... and {len(report['direct_dependencies']) - 20} more")
            break
    
    # Print transitive-only
    print(f"\nTRANSITIVE-ONLY DEPENDENCIES ({len(report['transitive_only'])}) - Removal Candidates:")
    for i, dep in enumerate(report['transitive_only']):
        if i < 30:
            print(f"  {dep}")
        elif i == 30:
            print(f"  ... and {len(report['transitive_only']) - 30} more")
            break
    
    # Print circular dependencies
    if report['circular_dependencies']:
        print(f"\nCIRCULAR DEPENDENCIES ({len(report['circular_dependencies'])}):")
        for cycle in report['circular_dependencies'][:5]:
            print(f"  {' -> '.join(cycle)}")
    
    # Print packages with most dependencies
    print(f"\nPACKAGES WITH MOST DEPENDENCIES:")
    for pkg, count in list(report['top_dependency_counts'].items())[:10]:
        print(f"  {pkg}: {count} dependencies")
    
    # Print most depended on packages
    print(f"\nMOST DEPENDED ON PACKAGES:")
    for pkg, count in list(report['most_depended_on'].items())[:10]:
        print(f"  {pkg}: depended on by {count} packages")
    
    # Print leaf packages (not depended on by anything)
    print(f"\nLEAF PACKAGES (not depended on by anything) - Safe to remove:")
    for pkg in report['leaf_packages'][:15]:
        print(f"  {pkg}")
    
    # Save detailed report
    output_file = "dependency_analysis_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_file}")
    
    # Generate removal recommendations
    print(f"\n" + "="*60)
    print("PACKAGE REMOVAL RECOMMENDATIONS")
    print("="*60)
    
    safe_to_remove = set(report['leaf_packages']) - set(report['direct_dependencies'])
    print(f"DEFINITELY SAFE TO REMOVE ({len(safe_to_remove)}):")
    print("(Leaf packages not used directly in code)")
    for pkg in sorted(safe_to_remove)[:20]:
        print(f"  pip uninstall {pkg}")
    
    transitive_candidates = set(report['transitive_only']) - set(report['leaf_packages'])
    print(f"\nCAREFUL REMOVAL CANDIDATES ({len(transitive_candidates)}):")
    print("(Transitive dependencies - check if needed)")
    for pkg in sorted(transitive_candidates)[:15]:
        print(f"  # Check: pip uninstall {pkg}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())