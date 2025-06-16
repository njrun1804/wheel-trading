#!/usr/bin/env python3
import json

with open('version_analysis_results.json') as f:
    data = json.load(f)

print('=== VERSION ANALYSIS SUMMARY ===')
print()

print('📁 Files Analyzed:')
for f in data['files_analyzed']:
    print(f'   - {f}')
print()

print(f'📊 Statistics:')
print(f'   Total packages: {data["total_packages"]}')
print(f'   Total constraints: {data["total_constraints"]}')
print(f'   Overly restrictive: {len(data["overly_restrictive"])}')
print(f'   Version conflicts: {len(data["version_conflicts"])}')
print(f'   Flexible candidates: {len(data["flexible_candidates"])}')
print()

print('⚠️  Key Issues Found:')
if data['version_conflicts']:
    print(f'   🔴 {len(data["version_conflicts"])} packages with version conflicts')
    for pkg in list(data['version_conflicts'].keys())[:5]:
        print(f'      - {pkg}')
else:
    print('   ✅ No version conflicts detected')

print(f'   🔒 {len(data["overly_restrictive"])} packages with exact pins (overly restrictive)')
print(f'   📈 {len(data["flexible_candidates"])} packages could use upper bounds')
print()

print('🔍 Most Common Constraint Patterns:')
patterns = data['pinning_patterns']
for pattern, count in patterns.items():
    if pattern != 'distribution' and count > 0:
        print(f'   {pattern.replace("_", " ").title()}: {count}')
print()

# Show some specific examples
print('💡 Specific Examples:')
print()

print('🔒 Most Restrictive Packages (exact pins):')
restrictive = data['overly_restrictive']
for i, (pkg, constraints) in enumerate(list(restrictive.items())[:10]):
    file, constraint = constraints[0]
    print(f'   {i+1:2d}. {pkg:<20} {constraint:<15} in {file}')
print()

print('📈 Packages Needing Upper Bounds:')
flexible = data['flexible_candidates']
for i, (pkg, constraints) in enumerate(list(flexible.items())[:10]):
    file, constraint = constraints[0]
    print(f'   {i+1:2d}. {pkg:<20} {constraint:<15} in {file}')
print()

print('📊 File-by-File Breakdown:')
file_stats = {}
for pkg, versions in data['package_summary'].items():
    for file, info in data['overly_restrictive'].get(pkg, []):
        if file not in file_stats:
            file_stats[file] = {'restrictive': 0, 'packages': []}
        file_stats[file]['restrictive'] += 1
        file_stats[file]['packages'].append(pkg)

for file, stats in sorted(file_stats.items(), key=lambda x: x[1]['restrictive'], reverse=True):
    print(f'   {file}: {stats["restrictive"]} overly restrictive constraints')