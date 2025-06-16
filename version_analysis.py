#!/usr/bin/env python3
"""
Version Constraint Analysis Tool
Analyzes version constraints across all requirements files for compatibility and optimization.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

try:
    import toml
    HAS_TOML = True
except ImportError:
    HAS_TOML = False

class VersionAnalyzer:
    def __init__(self):
        self.package_versions = defaultdict(list)  # package -> [(file, constraint)]
        self.all_packages = set()
        self.files_analyzed = []
        
    def parse_version_constraint(self, constraint: str) -> Dict[str, Any]:
        """Parse a version constraint string into components."""
        # Remove environment markers
        if ';' in constraint:
            constraint = constraint.split(';')[0].strip()
            
        # Handle git dependencies
        if constraint.startswith('-e git+') or constraint.startswith('git+'):
            return {
                'type': 'git',
                'constraint': constraint,
                'is_pinned': True,
                'is_flexible': False
            }
            
        # Extract package name and version spec
        match = re.match(r'^([a-zA-Z0-9\-_.]+)(.*)$', constraint)
        if not match:
            return {'type': 'unknown', 'constraint': constraint}
            
        package, version_spec = match.groups()
        version_spec = version_spec.strip()
        
        if not version_spec:
            return {
                'type': 'no_version',
                'package': package,
                'constraint': version_spec,
                'is_pinned': False,
                'is_flexible': True
            }
            
        # Analyze version constraint types
        is_pinned = '==' in version_spec
        is_flexible = '>=' in version_spec and '<' in version_spec
        is_overly_restrictive = '==' in version_spec and not ('~=' in version_spec or '^' in version_spec)
        
        # Check for range constraints
        has_upper_bound = '<' in version_spec
        has_lower_bound = '>=' in version_spec or '>' in version_spec
        
        return {
            'type': 'version',
            'package': package,
            'constraint': version_spec,
            'is_pinned': is_pinned,
            'is_flexible': is_flexible,
            'is_overly_restrictive': is_overly_restrictive,
            'has_upper_bound': has_upper_bound,
            'has_lower_bound': has_lower_bound
        }
        
    def analyze_requirements_file(self, file_path: Path) -> None:
        """Analyze a requirements.txt file."""
        try:
            content = file_path.read_text()
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                
                # Skip comments, empty lines, and includes
                if not line or line.startswith('#') or line.startswith('-r'):
                    continue
                    
                constraint_info = self.parse_version_constraint(line)
                if constraint_info['type'] in ['version', 'no_version']:
                    package = constraint_info['package']
                    self.package_versions[package].append((str(file_path), constraint_info))
                    self.all_packages.add(package)
                    
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            
    def analyze_pyproject_toml(self, file_path: Path) -> None:
        """Analyze pyproject.toml file."""
        if not HAS_TOML:
            print(f"Warning: toml module not available, skipping {file_path}")
            return
            
        try:
            data = toml.load(file_path)
            
            # Check poetry dependencies
            if 'tool' in data and 'poetry' in data['tool'] and 'dependencies' in data['tool']['poetry']:
                deps = data['tool']['poetry']['dependencies']
                for package, constraint in deps.items():
                    if package == 'python':
                        continue
                        
                    if isinstance(constraint, str):
                        constraint_info = self.parse_version_constraint(f"{package}{constraint}")
                    else:
                        # Complex constraint (dict format)
                        constraint_info = {
                            'type': 'complex',
                            'package': package,
                            'constraint': str(constraint),
                            'is_pinned': False,
                            'is_flexible': True
                        }
                        
                    if constraint_info['type'] in ['version', 'no_version', 'complex']:
                        self.package_versions[package].append((str(file_path), constraint_info))
                        self.all_packages.add(package)
                        
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            
    def find_version_conflicts(self) -> Dict[str, List[Tuple[str, str]]]:
        """Find packages with conflicting version requirements."""
        conflicts = {}
        
        for package, versions in self.package_versions.items():
            if len(versions) > 1:
                constraints = [(file, info['constraint']) for file, info in versions]
                
                # Check for actual conflicts (different pinned versions)
                pinned_versions = []
                for file, info in versions:
                    if info.get('is_pinned') and '==' in info['constraint']:
                        version = info['constraint'].split('==')[1].split(',')[0].strip()
                        pinned_versions.append((file, version))
                        
                if len(set(v for _, v in pinned_versions)) > 1:
                    conflicts[package] = constraints
                    
        return conflicts
        
    def find_overly_restrictive(self) -> Dict[str, List[Tuple[str, str]]]:
        """Find packages with overly restrictive version pinning."""
        restrictive = {}
        
        for package, versions in self.package_versions.items():
            for file, info in versions:
                if info.get('is_overly_restrictive'):
                    if package not in restrictive:
                        restrictive[package] = []
                    restrictive[package].append((file, info['constraint']))
                    
        return restrictive
        
    def find_flexible_candidates(self) -> Dict[str, List[Tuple[str, str]]]:
        """Find packages that could use more flexible versioning."""
        candidates = {}
        
        for package, versions in self.package_versions.items():
            for file, info in versions:
                # Packages with only lower bounds could add upper bounds
                if (info.get('has_lower_bound') and 
                    not info.get('has_upper_bound') and 
                    not info.get('is_pinned') and
                    info['type'] == 'version'):
                    
                    if package not in candidates:
                        candidates[package] = []
                    candidates[package].append((file, info['constraint']))
                    
        return candidates
        
    def analyze_pinning_patterns(self) -> Dict[str, Any]:
        """Analyze version constraint patterns."""
        patterns = {
            'exact_pins': 0,
            'range_constraints': 0,
            'lower_bound_only': 0,
            'upper_bound_only': 0,
            'no_constraints': 0,
            'git_dependencies': 0,
            'complex_constraints': 0
        }
        
        constraint_types = []
        
        for package, versions in self.package_versions.items():
            for file, info in versions:
                if info['type'] == 'git':
                    patterns['git_dependencies'] += 1
                    constraint_types.append('git')
                elif info['type'] == 'complex':
                    patterns['complex_constraints'] += 1
                    constraint_types.append('complex')
                elif info['type'] == 'no_version':
                    patterns['no_constraints'] += 1
                    constraint_types.append('no_constraints')
                elif info['type'] == 'version':
                    if info.get('is_pinned'):
                        patterns['exact_pins'] += 1
                        constraint_types.append('exact_pin')
                    elif info.get('is_flexible'):
                        patterns['range_constraints'] += 1
                        constraint_types.append('range')
                    elif info.get('has_lower_bound') and not info.get('has_upper_bound'):
                        patterns['lower_bound_only'] += 1
                        constraint_types.append('lower_only')
                    elif info.get('has_upper_bound') and not info.get('has_lower_bound'):
                        patterns['upper_bound_only'] += 1
                        constraint_types.append('upper_only')
                        
        patterns['distribution'] = {
            constraint_type: constraint_types.count(constraint_type) 
            for constraint_type in set(constraint_types)
        }
        
        return patterns
        
    def generate_recommendations(self) -> Dict[str, List[str]]:
        """Generate recommendations for improving version constraints."""
        recommendations = defaultdict(list)
        
        conflicts = self.find_version_conflicts()
        restrictive = self.find_overly_restrictive()
        flexible_candidates = self.find_flexible_candidates()
        
        # Conflict recommendations
        for package in conflicts:
            recommendations[package].append(
                "CRITICAL: Resolve version conflicts across requirements files"
            )
            
        # Overly restrictive recommendations
        for package in restrictive:
            recommendations[package].append(
                "Consider using compatible release (~=) or range constraints instead of exact pins"
            )
            
        # Flexibility recommendations
        for package in flexible_candidates:
            if len(self.package_versions[package]) == 1:  # Only in one file
                recommendations[package].append(
                    "Consider adding upper bound to prevent breaking changes"
                )
                
        return dict(recommendations)
        
    def run_analysis(self, base_path: Path) -> Dict[str, Any]:
        """Run complete version constraint analysis."""
        # Find all requirements files
        requirements_files = [
            base_path / "requirements.txt",
            base_path / "requirements-dev.txt", 
            base_path / "requirements-ci.txt",
            base_path / "jarvis2_requirements.txt",
            base_path / "requirements-updated.txt",
            base_path / "requirements_claude_integration.txt",
            base_path / "requirements_bolt.txt",
            base_path / "pyproject.toml"
        ]
        
        # Analyze each file
        for file_path in requirements_files:
            if file_path.exists():
                self.files_analyzed.append(str(file_path))
                if file_path.suffix == '.toml':
                    self.analyze_pyproject_toml(file_path)
                else:
                    self.analyze_requirements_file(file_path)
                    
        # Generate analysis results
        return {
            'files_analyzed': self.files_analyzed,
            'total_packages': len(self.all_packages),
            'total_constraints': sum(len(versions) for versions in self.package_versions.values()),
            'overly_restrictive': self.find_overly_restrictive(),
            'version_conflicts': self.find_version_conflicts(),
            'flexible_candidates': self.find_flexible_candidates(),
            'pinning_patterns': self.analyze_pinning_patterns(),
            'recommendations': self.generate_recommendations(),
            'package_summary': {
                package: [info['constraint'] for file, info in versions]
                for package, versions in self.package_versions.items()
            }
        }

def main():
    """Main analysis function."""
    analyzer = VersionAnalyzer()
    base_path = Path('.')
    
    print("🔍 Analyzing version constraints across all requirements files...\n")
    
    results = analyzer.run_analysis(base_path)
    
    print(f"📊 Analysis Summary:")
    print(f"   Files analyzed: {len(results['files_analyzed'])}")
    print(f"   Total packages: {results['total_packages']}")
    print(f"   Total constraints: {results['total_constraints']}")
    print()
    
    # Version conflicts
    conflicts = results['version_conflicts']
    if conflicts:
        print("⚠️  VERSION CONFLICTS FOUND:")
        for package, constraints in conflicts.items():
            print(f"   {package}:")
            for file, constraint in constraints:
                print(f"     {Path(file).name}: {constraint}")
        print()
    else:
        print("✅ No version conflicts detected\n")
        
    # Overly restrictive
    restrictive = results['overly_restrictive']
    if restrictive:
        print("🔒 OVERLY RESTRICTIVE CONSTRAINTS:")
        for package, constraints in list(restrictive.items())[:10]:  # Show top 10
            print(f"   {package}: {constraints[0][1]}")
        if len(restrictive) > 10:
            print(f"   ... and {len(restrictive) - 10} more")
        print()
    else:
        print("✅ No overly restrictive constraints detected\n")
        
    # Flexible candidates
    flexible = results['flexible_candidates']
    if flexible:
        print("📈 CANDIDATES FOR MORE FLEXIBLE VERSIONING:")
        for package, constraints in list(flexible.items())[:10]:  # Show top 10
            print(f"   {package}: {constraints[0][1]} (could add upper bound)")
        if len(flexible) > 10:
            print(f"   ... and {len(flexible) - 10} more")
        print()
    else:
        print("✅ All packages have appropriate version flexibility\n")
        
    # Pinning patterns
    patterns = results['pinning_patterns']
    print("📋 VERSION CONSTRAINT PATTERNS:")
    for pattern, count in patterns.items():
        if pattern != 'distribution' and count > 0:
            print(f"   {pattern.replace('_', ' ').title()}: {count}")
    print()
    
    # Save detailed results
    output_file = base_path / "version_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Detailed analysis saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()