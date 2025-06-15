#!/usr/bin/env python3
"""
Comprehensive Import Dependency Audit for Einstein Python Files
Analyzes each import statement and determines if it's actually used in the code.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ImportInfo:
    """Information about an import statement."""
    file_path: str
    line_number: int
    import_type: str  # 'import' or 'from_import'
    module: str
    names: List[str]  # List of imported names/aliases
    original_statement: str
    is_used: bool = False
    usage_locations: List[int] = None  # Line numbers where used
    
    def __post_init__(self):
        if self.usage_locations is None:
            self.usage_locations = []

class ImportUsageAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze import usage."""
    
    def __init__(self, content_lines: List[str]):
        self.content_lines = content_lines
        self.imports: List[ImportInfo] = []
        self.name_usage: Dict[str, List[int]] = defaultdict(list)  # name -> line numbers
        self.current_line = 0
        
    def visit_Import(self, node):
        """Handle 'import module' statements."""
        for alias in node.names:
            import_name = alias.asname if alias.asname else alias.name.split('.')[-1]
            
            import_info = ImportInfo(
                file_path="",  # Will be set by caller
                line_number=node.lineno,
                import_type='import',
                module=alias.name,
                names=[import_name],
                original_statement=self.content_lines[node.lineno - 1].strip()
            )
            self.imports.append(import_info)
    
    def visit_ImportFrom(self, node):
        """Handle 'from module import name' statements."""
        imported_names = []
        for alias in node.names:
            import_name = alias.asname if alias.asname else alias.name
            imported_names.append(import_name)
        
        import_info = ImportInfo(
            file_path="",  # Will be set by caller
            line_number=node.lineno,
            import_type='from_import',
            module=node.module or '',
            names=imported_names,
            original_statement=self.content_lines[node.lineno - 1].strip()
        )
        self.imports.append(import_info)
        
    def visit_Name(self, node):
        """Track usage of names."""
        self.name_usage[node.id].append(node.lineno)
        
    def visit_Attribute(self, node):
        """Track usage of module.attribute patterns."""
        if isinstance(node.value, ast.Name):
            # Record both the base name and the full attribute access
            self.name_usage[node.value.id].append(node.lineno)
            full_name = f"{node.value.id}.{node.attr}"
            self.name_usage[full_name].append(node.lineno)
        self.generic_visit(node)

def analyze_file_imports(file_path: Path) -> Tuple[List[ImportInfo], Dict[str, Any]]:
    """Analyze imports and their usage in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content_lines = content.split('\n')
        tree = ast.parse(content)
        
        analyzer = ImportUsageAnalyzer(content_lines)
        analyzer.visit(tree)
        
        # Set file path for all imports
        for import_info in analyzer.imports:
            import_info.file_path = str(file_path)
        
        # Check usage for each import
        for import_info in analyzer.imports:
            for name in import_info.names:
                if name in analyzer.name_usage:
                    # Filter out usage in import section itself
                    usage_lines = [line for line in analyzer.name_usage[name] 
                                 if line != import_info.line_number]
                    if usage_lines:
                        import_info.is_used = True
                        import_info.usage_locations = usage_lines
        
        stats = {
            'total_imports': len(analyzer.imports),
            'used_imports': len([i for i in analyzer.imports if i.is_used]),
            'unused_imports': len([i for i in analyzer.imports if not i.is_used]),
            'import_types': {
                'import': len([i for i in analyzer.imports if i.import_type == 'import']),
                'from_import': len([i for i in analyzer.imports if i.import_type == 'from_import'])
            }
        }
        
        return analyzer.imports, stats
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return [], {'error': str(e)}

def generate_audit_report():
    """Generate comprehensive audit report for all Python files in einstein directory."""
    
    einstein_dir = Path(".")
    if not einstein_dir.exists():
        print("Einstein directory not found")
        return
    
    all_imports = []
    file_stats = {}
    
    print("🔍 COMPREHENSIVE IMPORT DEPENDENCY AUDIT - EINSTEIN SYSTEM")
    print("=" * 70)
    
    # Analyze each Python file
    for py_file in sorted(einstein_dir.glob("*.py")):
        if py_file.name.startswith('.'):
            continue
            
        imports, stats = analyze_file_imports(py_file)
        all_imports.extend(imports)
        file_stats[py_file.name] = stats
        
        print(f"\n📁 {py_file.name}")
        print("-" * 50)
        
        if 'error' in stats:
            print(f"❌ Error: {stats['error']}")
            continue
        
        print(f"📊 Statistics:")
        print(f"   Total imports: {stats['total_imports']}")
        print(f"   Used imports: {stats['used_imports']}")
        print(f"   Unused imports: {stats['unused_imports']}")
        print(f"   Import statements: {stats['import_types']['import']}")
        print(f"   From-import statements: {stats['import_types']['from_import']}")
        
        # Show unused imports
        unused = [imp for imp in imports if not imp.is_used]
        if unused:
            print(f"\n🚫 UNUSED IMPORTS ({len(unused)}):")
            for imp in unused:
                names_str = ', '.join(imp.names)
                print(f"   Line {imp.line_number}: {names_str}")
                print(f"      Statement: {imp.original_statement}")
        
        # Show used imports with usage count
        used = [imp for imp in imports if imp.is_used]
        if used:
            print(f"\n✅ USED IMPORTS ({len(used)}):")
            for imp in used:
                names_str = ', '.join(imp.names)
                usage_count = sum(len(imp.usage_locations) for name in imp.names)
                print(f"   Line {imp.line_number}: {names_str} (used {usage_count} times)")
    
    # Overall summary
    print(f"\n\n📈 OVERALL SUMMARY")
    print("=" * 50)
    
    total_files = len(file_stats)
    total_imports = sum(stats.get('total_imports', 0) for stats in file_stats.values())
    total_used = sum(stats.get('used_imports', 0) for stats in file_stats.values())
    total_unused = sum(stats.get('unused_imports', 0) for stats in file_stats.values())
    
    print(f"Files analyzed: {total_files}")
    print(f"Total imports: {total_imports}")
    print(f"Used imports: {total_used} ({total_used/total_imports*100:.1f}%)")
    print(f"Unused imports: {total_unused} ({total_unused/total_imports*100:.1f}%)")
    
    # Most common unused imports
    unused_by_name = defaultdict(int)
    for imp in all_imports:
        if not imp.is_used:
            for name in imp.names:
                unused_by_name[name] += 1
    
    if unused_by_name:
        print(f"\n🔥 MOST COMMONLY UNUSED IMPORTS:")
        for name, count in sorted(unused_by_name.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {name}: {count} occurrences")
    
    # Import source analysis
    import_sources = defaultdict(int) 
    for imp in all_imports:
        if imp.module:
            base_module = imp.module.split('.')[0]
            import_sources[base_module] += 1
    
    print(f"\n📦 IMPORT SOURCES (Top 10):")
    for module, count in sorted(import_sources.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {module}: {count} imports")

if __name__ == "__main__":
    generate_audit_report()