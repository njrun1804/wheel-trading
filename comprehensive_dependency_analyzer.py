#!/usr/bin/env python3
"""
Comprehensive Dependency Analysis
Identifies packages that can be safely removed by analyzing dependency chains.
"""

import subprocess
import json
import sys
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple
import re

class ComprehensiveDependencyAnalyzer:
    def __init__(self):
        self.installed_packages = {}
        self.dependency_graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.direct_dependencies = set()
        self.package_categories = defaultdict(set)
        
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
            print(f"Error: {e}")
            return False
    
    def categorize_packages(self):
        """Categorize packages by their likely purpose."""
        # Development/testing tools that might be removable
        dev_patterns = [
            'test', 'pytest', 'mock', 'coverage', 'flake8', 'mypy', 'black', 'isort',
            'pre-commit', 'bandit', 'vulture', 'autopep8', 'pycodestyle', 'pydocstyle',
            'rope', 'jedi', 'language-server', 'lint', 'format'
        ]
        
        # Monitoring/observability that might be optional
        monitoring_patterns = [
            'phoenix', 'arize', 'logfire', 'opentelemetry', 'tracing', 'otel',
            'instrumentation', 'monitor', 'metric', 'telemetry'
        ]
        
        # ML/AI libraries that might be unused
        ml_patterns = [
            'tensorflow', 'torch', 'mlx', 'transformers', 'huggingface', 'sentence',
            'faiss', 'scikit', 'sklearn', 'keras', 'neural', 'model'
        ]
        
        # Documentation/build tools
        docs_patterns = [
            'sphinx', 'markdown', 'mkdocs', 'build', 'setuptools', 'wheel', 
            'poetry', 'pip-', 'twine', 'build'
        ]
        
        # Data processing that might be optional
        data_patterns = [
            'databento', 'fredapi', 'yfinance', 'alpaca', 'exchange-calendars',
            'pandas-market', 'pandas-gbq'
        ]
        
        # Categorize each package
        for pkg in self.installed_packages:
            pkg_lower = pkg.lower()
            
            if any(pattern in pkg_lower for pattern in dev_patterns):
                self.package_categories['development'].add(pkg)
            elif any(pattern in pkg_lower for pattern in monitoring_patterns):
                self.package_categories['monitoring'].add(pkg)
            elif any(pattern in pkg_lower for pattern in ml_patterns):
                self.package_categories['machine_learning'].add(pkg)
            elif any(pattern in pkg_lower for pattern in docs_patterns):
                self.package_categories['documentation'].add(pkg)
            elif any(pattern in pkg_lower for pattern in data_patterns):
                self.package_categories['data_providers'].add(pkg)
            else:
                self.package_categories['core'].add(pkg)
    
    def build_dependency_graph_fast(self):
        """Build dependency graph efficiently."""
        print("Building dependency graph...")
        
        # Get packages in batches
        all_packages = list(self.installed_packages.keys())
        batch_size = 30
        
        for i in range(0, len(all_packages), batch_size):
            batch = all_packages[i:i+batch_size]
            
            try:
                # Get dependencies for this batch
                cmd = ['pip', 'show'] + batch
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                
                if result.returncode == 0:
                    self._parse_batch_output(result.stdout)
                
                print(f"  Processed {min(i+batch_size, len(all_packages))}/{len(all_packages)}")
                
            except Exception:
                continue
        
        print(f"Built dependency graph with {len(self.dependency_graph)} packages")
    
    def _parse_batch_output(self, output: str):
        """Parse pip show output for dependencies."""
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
                        req_clean = re.split(r'[>=<!~\[\],]', req)[0].strip()
                        if req_clean:
                            deps.add(req_clean.lower())
                
                self.dependency_graph[current_package] = deps
                
                # Build reverse graph
                for dep in deps:
                    if dep in self.installed_packages:
                        self.reverse_graph[dep].add(current_package)
                
                current_package = None
    
    def find_direct_dependencies(self):
        """Find packages directly imported in code."""
        print("Finding direct dependencies from code...")
        
        # Mapping of import names to package names
        import_mappings = {
            'sklearn': 'scikit-learn', 'cv2': 'opencv-python', 'PIL': 'pillow',
            'yaml': 'pyyaml', 'dateutil': 'python-dateutil', 'dotenv': 'python-dotenv',
            'jwt': 'pyjwt', 'bs4': 'beautifulsoup4', 'requests_oauthlib': 'requests-oauthlib'
        }
        
        # Known direct dependencies from the codebase analysis
        known_direct = {
            'asyncio', 'duckdb', 'lmdb', 'mlx', 'numpy', 'opik', 'pandas',
            'psutil', 'pyarrow', 'pydantic', 'pydantic-settings', 'python-dotenv',
            'pytz', 'pyyaml', 'scikit-learn', 'sqlparse', 'torch', 'uvloop',
            'databento', 'fredapi', 'alpaca-py', 'anthropic', 'openai',
            'fastapi', 'uvicorn', 'sqlalchemy', 'alembic', 'pytest',
            'requests', 'httpx', 'aiohttp', 'matplotlib', 'plotly'
        }
        
        self.direct_dependencies = known_direct & set(self.installed_packages.keys())
        print(f"Found {len(self.direct_dependencies)} direct dependencies")
    
    def find_removal_candidates(self):
        """Find packages that can potentially be removed."""
        # Categories that might be removable
        potentially_removable_categories = [
            'development', 'monitoring', 'documentation'
        ]
        
        removal_candidates = {}
        
        for category in potentially_removable_categories:
            candidates = []
            
            for pkg in self.package_categories[category]:
                if pkg not in self.direct_dependencies:
                    # Check if removing this package would break direct dependencies
                    would_break_direct = False
                    
                    for direct_dep in self.direct_dependencies:
                        if self._is_transitive_dependency(direct_dep, pkg):
                            would_break_direct = True
                            break
                    
                    if not would_break_direct:
                        candidates.append(pkg)
            
            removal_candidates[category] = candidates
        
        return removal_candidates
    
    def _is_transitive_dependency(self, root: str, target: str, visited: Set[str] = None) -> bool:
        """Check if target is a transitive dependency of root."""
        if visited is None:
            visited = set()
        if root in visited or root == target:
            return root == target and len(visited) > 0
        
        visited.add(root)
        
        for dep in self.dependency_graph.get(root, set()):
            if dep == target:
                return True
            if self._is_transitive_dependency(dep, target, visited.copy()):
                return True
        
        return False
    
    def find_unused_heavy_packages(self):
        """Find large packages that might be unused."""
        # Heavy packages that are commonly installed but might be unused
        heavy_packages = {
            'tensorflow', 'tensorflow-macos', 'tensorflow-metal', 'torch', 'torchvision',
            'transformers', 'huggingface-hub', 'sentence-transformers', 'faiss-cpu',
            'arize-phoenix', 'mlflow', 'mlflow-skinny', 'databricks-sdk',
            'google-cloud-bigquery', 'google-cloud-pubsub', 'google-cloud-secret-manager',
            'poetry', 'poetry-core', 'poetry-plugin-export'
        }
        
        unused_heavy = []
        for pkg in heavy_packages:
            if pkg in self.installed_packages and pkg not in self.direct_dependencies:
                # Check if any direct dependency needs this
                needed = False
                for direct_dep in self.direct_dependencies:
                    if self._is_transitive_dependency(direct_dep, pkg):
                        needed = True
                        break
                
                if not needed:
                    unused_heavy.append(pkg)
        
        return unused_heavy
    
    def generate_dependency_chains(self):
        """Generate dependency chains for analysis."""
        chains = {}
        
        for direct_dep in list(self.direct_dependencies)[:10]:  # Top 10 for analysis
            chain = self._get_dependency_chain(direct_dep)
            if len(chain) > 1:
                chains[direct_dep] = chain
        
        return chains
    
    def _get_dependency_chain(self, package: str, depth: int = 0, max_depth: int = 3) -> List[str]:
        """Get dependency chain for a package."""
        if depth >= max_depth:
            return [package]
        
        deps = self.dependency_graph.get(package, set())
        if not deps:
            return [package]
        
        # Get chain for first few dependencies
        chain = [package]
        for dep in list(deps)[:3]:  # Limit to 3 for clarity
            if dep in self.installed_packages:
                sub_chain = self._get_dependency_chain(dep, depth + 1, max_depth)
                if len(sub_chain) > 1:
                    chain.extend([f"  -> {dep}"] + [f"    -> {d}" for d in sub_chain[1:]])
                else:
                    chain.append(f"  -> {dep}")
        
        return chain
    
    def analyze(self):
        """Run comprehensive analysis."""
        print("Starting comprehensive dependency analysis...")
        
        if not self.get_installed_packages():
            return None
        
        self.categorize_packages()
        self.build_dependency_graph_fast()
        self.find_direct_dependencies()
        
        removal_candidates = self.find_removal_candidates()
        unused_heavy = self.find_unused_heavy_packages()
        dependency_chains = self.generate_dependency_chains()
        
        # Calculate stats
        total_transitive = set()
        for direct_dep in self.direct_dependencies:
            total_transitive.update(self._get_all_transitive(direct_dep))
        
        transitive_only = total_transitive - self.direct_dependencies
        
        report = {
            "summary": {
                "total_packages": len(self.installed_packages),
                "direct_dependencies": len(self.direct_dependencies),
                "transitive_dependencies": len(total_transitive),
                "transitive_only": len(transitive_only),
                "categories": {cat: len(pkgs) for cat, pkgs in self.package_categories.items()}
            },
            "direct_dependencies": sorted(list(self.direct_dependencies)),
            "package_categories": {cat: sorted(list(pkgs)) for cat, pkgs in self.package_categories.items()},
            "removal_candidates": {cat: sorted(candidates) for cat, candidates in removal_candidates.items()},
            "unused_heavy_packages": sorted(unused_heavy),
            "dependency_chains": dependency_chains,
            "transitive_only": sorted(list(transitive_only))
        }
        
        return report
    
    def _get_all_transitive(self, package: str, visited: Set[str] = None) -> Set[str]:
        """Get all transitive dependencies."""
        if visited is None:
            visited = set()
        if package in visited:
            return set()
        
        visited.add(package)
        transitive = set()
        
        for dep in self.dependency_graph.get(package, set()):
            if dep in self.installed_packages:
                transitive.add(dep)
                transitive.update(self._get_all_transitive(dep, visited.copy()))
        
        return transitive

def main():
    analyzer = ComprehensiveDependencyAnalyzer()
    report = analyzer.analyze()
    
    if not report:
        print("Analysis failed")
        return 1
    
    s = report['summary']
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DEPENDENCY ANALYSIS")
    print("="*80)
    print(f"Total packages: {s['total_packages']}")
    print(f"Direct dependencies: {s['direct_dependencies']}")
    print(f"Transitive dependencies: {s['transitive_dependencies']}")
    print(f"Transitive-only: {s['transitive_only']}")
    
    print(f"\nPACKAGE CATEGORIES:")
    for category, count in s['categories'].items():
        print(f"  {category}: {count} packages")
    
    print(f"\nDIRECT DEPENDENCIES ({len(report['direct_dependencies'])}):")
    for dep in report['direct_dependencies']:
        print(f"  {dep}")
    
    print(f"\nUNUSED HEAVY PACKAGES ({len(report['unused_heavy_packages'])}):")
    print("These large packages appear to be unused and can likely be removed:")
    for pkg in report['unused_heavy_packages']:
        print(f"  pip uninstall {pkg}")
    
    print(f"\nREMOVAL CANDIDATES BY CATEGORY:")
    total_removal_candidates = 0
    all_candidates = []
    
    for category, candidates in report['removal_candidates'].items():
        if candidates:
            print(f"\n{category.upper()} TOOLS ({len(candidates)}):")
            for pkg in candidates:
                print(f"  {pkg}")
                all_candidates.append(pkg)
            total_removal_candidates += len(candidates)
    
    print(f"\nDEPENDENCY CHAINS (sample):")
    for pkg, chain in list(report['dependency_chains'].items())[:3]:
        print(f"\n{pkg}:")
        for item in chain[:10]:  # Limit output
            print(f"  {item}")
    
    # Save detailed report
    with open("comprehensive_dependency_analysis.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n" + "="*80)
    print("REMOVAL RECOMMENDATIONS")
    print("="*80)
    
    if report['unused_heavy_packages']:
        print("HIGH PRIORITY REMOVAL (Large, unused packages):")
        for pkg in report['unused_heavy_packages']:
            print(f"  pip uninstall {pkg}")
        
        heavy_cmd = ' '.join(report['unused_heavy_packages'])
        print(f"\nRemove all heavy packages: pip uninstall {heavy_cmd}")
    
    if all_candidates:
        print(f"\nOTHER REMOVAL CANDIDATES ({len(all_candidates)}):")
        print("Consider removing these after testing:")
        for pkg in sorted(all_candidates)[:20]:
            print(f"  pip uninstall {pkg}")
        
        if len(all_candidates) > 20:
            print(f"  ... and {len(all_candidates) - 20} more")
    
    print(f"\nTotal potential removals: {len(report['unused_heavy_packages']) + len(all_candidates)}")
    print(f"Detailed analysis saved to: comprehensive_dependency_analysis.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())