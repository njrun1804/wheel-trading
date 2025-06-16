#!/usr/bin/env python3
"""
Complete Dependency Tree Analysis
Maps transitive dependencies, identifies circular dependencies, and analyzes dependency chains.
"""

import subprocess
import json
import sys
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Optional
import ast
import os
import glob
from pathlib import Path
try:
    import pkg_resources
except ImportError:
    pkg_resources = None

class DependencyTreeAnalyzer:
    def __init__(self):
        self.installed_packages = {}
        self.dependency_graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.direct_dependencies = set()
        self.transitive_only = set()
        self.dependency_chains = {}
        self.circular_deps = []
        self.package_depths = {}
        
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
    
    def get_package_dependencies(self, package_name: str) -> Set[str]:
        """Get direct dependencies for a package using pip show."""
        dependencies = set()
        
        # Try pkg_resources first if available
        if pkg_resources:
            try:
                dist = pkg_resources.get_distribution(package_name)
                for req in dist.requires():
                    dep_name = req.project_name.lower()
                    dependencies.add(dep_name)
                return dependencies
            except Exception:
                pass
        
        # Fallback to pip show
        try:
            result = subprocess.run(['pip', 'show', package_name], 
                                  capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if line.startswith('Requires:'):
                    reqs = line.split(':', 1)[1].strip()
                    if reqs and reqs != 'None':
                        for req in reqs.split(', '):
                            # Clean up requirement specification
                            req_clean = req.split('>=')[0].split('==')[0].split('<')[0].split('!')[0].strip()
                            if req_clean:
                                dependencies.add(req_clean.lower())
                    break
        except Exception as e:
            print(f"Error getting dependencies for {package_name}: {e}")
        
        return dependencies
    
    def build_dependency_graph(self):
        """Build complete dependency graph for all installed packages."""
        print("Building dependency graph...")
        for package in self.installed_packages:
            deps = self.get_package_dependencies(package)
            self.dependency_graph[package] = deps
            
            # Build reverse graph
            for dep in deps:
                self.reverse_graph[dep].add(package)
        
        print(f"Built graph with {len(self.dependency_graph)} nodes")
    
    def analyze_code_imports(self, root_path: str = "."):
        """Analyze Python code to find directly imported packages."""
        print("Analyzing code imports...")
        direct_imports = set()
        
        # Find all Python files
        python_files = []
        for pattern in ["**/*.py", "*.py"]:
            python_files.extend(glob.glob(os.path.join(root_path, pattern), recursive=True))
        
        for file_path in python_files:
            if '__pycache__' in file_path or '.git' in file_path:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read(), filename=file_path)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    module_name = alias.name.split('.')[0].lower()
                                    direct_imports.add(module_name)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    module_name = node.module.split('.')[0].lower()
                                    direct_imports.add(module_name)
                    except SyntaxError:
                        continue
            except Exception:
                continue
        
        # Map imports to installed packages
        for imp in direct_imports:
            if imp in self.installed_packages:
                self.direct_dependencies.add(imp)
            else:
                # Try to find package that provides this module
                for pkg in self.installed_packages:
                    if imp.replace('_', '-') == pkg or imp.replace('-', '_') == pkg:
                        self.direct_dependencies.add(pkg)
                        break
        
        print(f"Found {len(self.direct_dependencies)} direct dependencies from code analysis")
    
    def find_transitive_only(self):
        """Identify packages that are only transitive dependencies."""
        all_transitive = set()
        
        # Get all transitive dependencies of direct dependencies
        def get_all_transitive(package: str, visited: Set[str] = None) -> Set[str]:
            if visited is None:
                visited = set()
            if package in visited:
                return set()
            
            visited.add(package)
            transitive = set()
            
            for dep in self.dependency_graph.get(package, set()):
                if dep in self.installed_packages:
                    transitive.add(dep)
                    transitive.update(get_all_transitive(dep, visited.copy()))
            
            return transitive
        
        for direct_dep in self.direct_dependencies:
            all_transitive.update(get_all_transitive(direct_dep))
        
        # Transitive-only are those that are transitive but not direct
        self.transitive_only = all_transitive - self.direct_dependencies
        print(f"Found {len(self.transitive_only)} transitive-only dependencies")
    
    def detect_circular_dependencies(self):
        """Detect circular dependencies using DFS."""
        print("Detecting circular dependencies...")
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                if neighbor in self.installed_packages:
                    dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
            return False
        
        for package in self.installed_packages:
            if package not in visited:
                dfs(package, [])
        
        self.circular_deps = cycles
        print(f"Found {len(cycles)} circular dependencies")
    
    def calculate_dependency_depths(self):
        """Calculate the depth of each package in the dependency tree."""
        print("Calculating dependency depths...")
        
        # Start with packages that have no dependencies (depth 0)
        no_deps = {pkg for pkg, deps in self.dependency_graph.items() 
                  if not deps or not any(d in self.installed_packages for d in deps)}
        
        depths = {pkg: 0 for pkg in no_deps}
        queue = deque(no_deps)
        
        while queue:
            current = queue.popleft()
            current_depth = depths[current]
            
            # Update depth of packages that depend on current
            for dependent in self.reverse_graph.get(current, set()):
                if dependent in self.installed_packages:
                    new_depth = current_depth + 1
                    if dependent not in depths or depths[dependent] < new_depth:
                        depths[dependent] = new_depth
                        queue.append(dependent)
        
        self.package_depths = depths
        print(f"Calculated depths for {len(depths)} packages")
    
    def build_dependency_chains(self):
        """Build complete dependency chains from root packages to leaves."""
        print("Building dependency chains...")
        
        def get_chain_to_root(package: str, visited: Set[str] = None) -> List[List[str]]:
            if visited is None:
                visited = set()
            if package in visited:
                return [[package]]  # Circular reference
            
            visited.add(package)
            dependents = self.reverse_graph.get(package, set())
            
            if not dependents or package in self.direct_dependencies:
                return [[package]]
            
            chains = []
            for dependent in dependents:
                if dependent in self.installed_packages:
                    for chain in get_chain_to_root(dependent, visited.copy()):
                        chains.append(chain + [package])
            
            return chains if chains else [[package]]
        
        for package in self.installed_packages:
            self.dependency_chains[package] = get_chain_to_root(package)
    
    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive dependency analysis report."""
        report = {
            "summary": {
                "total_packages": len(self.installed_packages),
                "direct_dependencies": len(self.direct_dependencies),
                "transitive_only": len(self.transitive_only),
                "circular_dependencies": len(self.circular_deps),
                "max_depth": max(self.package_depths.values()) if self.package_depths else 0
            },
            "direct_dependencies": sorted(list(self.direct_dependencies)),
            "transitive_only": sorted(list(self.transitive_only)),
            "circular_dependencies": self.circular_deps,
            "dependency_depths": dict(sorted(self.package_depths.items(), key=lambda x: x[1])),
            "top_level_packages": [pkg for pkg, depth in self.package_depths.items() if depth == 0],
            "deepest_packages": []
        }
        
        # Find deepest packages
        if self.package_depths:
            max_depth = max(self.package_depths.values())
            report["deepest_packages"] = [pkg for pkg, depth in self.package_depths.items() 
                                        if depth == max_depth]
        
        # Add dependency chains for key packages
        report["dependency_chains"] = {}
        for pkg in list(self.direct_dependencies)[:10]:  # Top 10 direct deps
            chains = self.dependency_chains.get(pkg, [])
            if chains:
                # Get shortest chain
                shortest = min(chains, key=len) if chains else []
                report["dependency_chains"][pkg] = shortest
        
        return report
    
    def analyze(self, root_path: str = "."):
        """Run complete dependency analysis."""
        print("Starting complete dependency tree analysis...")
        
        # Step 1: Get installed packages
        if not self.get_installed_packages():
            return None
        
        # Step 2: Build dependency graph
        self.build_dependency_graph()
        
        # Step 3: Analyze code imports
        self.analyze_code_imports(root_path)
        
        # Step 4: Find transitive-only dependencies
        self.find_transitive_only()
        
        # Step 5: Detect circular dependencies
        self.detect_circular_dependencies()
        
        # Step 6: Calculate depths
        self.calculate_dependency_depths()
        
        # Step 7: Build dependency chains
        self.build_dependency_chains()
        
        # Step 8: Generate report
        return self.generate_analysis_report()

def main():
    analyzer = DependencyTreeAnalyzer()
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
    print(f"Transitive-only dependencies: {report['summary']['transitive_only']}")
    print(f"Circular dependencies: {report['summary']['circular_dependencies']}")
    print(f"Maximum dependency depth: {report['summary']['max_depth']}")
    
    # Print direct dependencies
    print("\nDIRECT DEPENDENCIES (used in code):")
    for dep in report['direct_dependencies'][:20]:  # Show first 20
        print(f"  {dep}")
    if len(report['direct_dependencies']) > 20:
        print(f"  ... and {len(report['direct_dependencies']) - 20} more")
    
    # Print transitive-only
    print("\nTRANSITIVE-ONLY DEPENDENCIES (candidates for removal):")
    for dep in report['transitive_only'][:20]:  # Show first 20
        print(f"  {dep}")
    if len(report['transitive_only']) > 20:
        print(f"  ... and {len(report['transitive_only']) - 20} more")
    
    # Print circular dependencies
    if report['circular_dependencies']:
        print("\nCIRCULAR DEPENDENCIES:")
        for cycle in report['circular_dependencies'][:5]:  # Show first 5
            print(f"  {' -> '.join(cycle)}")
    
    # Print top-level packages
    print(f"\nTOP-LEVEL PACKAGES (depth 0): {len(report['top_level_packages'])}")
    for pkg in report['top_level_packages'][:10]:
        print(f"  {pkg}")
    
    # Print deepest packages
    if report['deepest_packages']:
        print(f"\nDEEPEST PACKAGES (depth {report['summary']['max_depth']}):")
        for pkg in report['deepest_packages'][:10]:
            print(f"  {pkg}")
    
    # Save detailed report
    output_file = "dependency_tree_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())