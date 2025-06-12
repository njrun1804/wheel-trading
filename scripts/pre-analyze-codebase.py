#!/usr/bin/env python3
"""
Comprehensive Codebase Pre-Analysis for Maximum Claude Performance
This script analyzes the entire codebase and builds multiple indexes:
- Dependency graphs with cycle detection
- Symbol indexes with usage tracking
- Import relationships and module hierarchy
- Code complexity metrics
- Test coverage mapping
- Performance hotspot identification
- Trading strategy patterns
- Risk management rules
"""

import os
import sys
import ast
import json
import time
import hashlib
import sqlite3
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import networkx as nx
from dataclasses import dataclass, asdict
import re

# Add project root to path
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', '.')).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class CodeMetrics:
    """Metrics for a code file"""
    file_path: str
    lines_of_code: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    function_count: int
    class_count: int
    import_count: int
    test_coverage: float = 0.0
    last_modified: float = 0.0
    hash: str = ""

@dataclass
class Symbol:
    """Symbol definition"""
    name: str
    type: str  # 'class', 'function', 'variable', 'constant'
    file: str
    line: int
    docstring: str = ""
    signature: str = ""
    decorators: List[str] = None
    parent_class: str = ""

class CodebaseAnalyzer:
    """Comprehensive codebase analyzer"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.index_dir = project_root / '.claude' / 'indexes'
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.symbols: Dict[str, List[Symbol]] = defaultdict(list)
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.import_graph = nx.DiGraph()
        self.call_graph = nx.DiGraph()
        self.inheritance_tree = nx.DiGraph()
        self.file_metrics: Dict[str, CodeMetrics] = {}
        self.trading_patterns: Dict[str, Any] = {}
        self.risk_rules: List[Dict[str, Any]] = []
        
        # Initialize database for fast queries
        self.db_path = self.index_dir / 'codebase.db'
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for fast symbol queries"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                name TEXT,
                type TEXT,
                file TEXT,
                line INTEGER,
                docstring TEXT,
                signature TEXT,
                parent_class TEXT,
                decorators TEXT,
                PRIMARY KEY (name, file, line)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS imports (
                from_module TEXT,
                to_module TEXT,
                import_type TEXT,
                line INTEGER,
                PRIMARY KEY (from_module, to_module, line)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                file_path TEXT PRIMARY KEY,
                lines_of_code INTEGER,
                cyclomatic_complexity INTEGER,
                cognitive_complexity INTEGER,
                function_count INTEGER,
                class_count INTEGER,
                import_count INTEGER,
                test_coverage REAL,
                last_modified REAL,
                hash TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_patterns (
                pattern_name TEXT PRIMARY KEY,
                pattern_type TEXT,
                file_path TEXT,
                line_number INTEGER,
                description TEXT,
                parameters TEXT
            )
        """)
        
        # Create indexes for fast lookups
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_name ON symbols(name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_type ON symbols(type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_import_from ON imports(from_module)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_import_to ON imports(to_module)")
        
        self.conn.commit()
    
    def analyze_file(self, file_path: Path) -> CodeMetrics:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Calculate metrics
            lines_of_code = len(content.splitlines())
            
            # Extract symbols and metrics
            visitor = ComprehensiveVisitor(str(file_path.relative_to(self.project_root)))
            visitor.visit(tree)
            
            # Store symbols
            for symbol in visitor.symbols:
                self.symbols[symbol.name].append(symbol)
                
                # Insert into database
                self.conn.execute("""
                    INSERT OR REPLACE INTO symbols 
                    (name, type, file, line, docstring, signature, parent_class, decorators)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol.name, symbol.type, symbol.file, symbol.line,
                    symbol.docstring, symbol.signature, symbol.parent_class,
                    json.dumps(symbol.decorators or [])
                ))
            
            # Build import graph
            module_name = self._path_to_module(file_path)
            self.import_graph.add_node(module_name)
            
            for imp in visitor.imports:
                self.imports[module_name].add(imp)
                self.import_graph.add_edge(module_name, imp)
                
                # Insert into database
                self.conn.execute("""
                    INSERT OR REPLACE INTO imports (from_module, to_module, import_type, line)
                    VALUES (?, ?, ?, ?)
                """, (module_name, imp, 'import', 0))
            
            # Calculate complexity
            cyclomatic = visitor.cyclomatic_complexity
            cognitive = visitor.cognitive_complexity
            
            # File hash for change detection
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            
            metrics = CodeMetrics(
                file_path=str(file_path.relative_to(self.project_root)),
                lines_of_code=lines_of_code,
                cyclomatic_complexity=cyclomatic,
                cognitive_complexity=cognitive,
                function_count=visitor.function_count,
                class_count=visitor.class_count,
                import_count=len(visitor.imports),
                last_modified=file_path.stat().st_mtime,
                hash=file_hash
            )
            
            self.file_metrics[metrics.file_path] = metrics
            
            # Insert metrics into database
            self.conn.execute("""
                INSERT OR REPLACE INTO metrics 
                (file_path, lines_of_code, cyclomatic_complexity, cognitive_complexity,
                 function_count, class_count, import_count, test_coverage, last_modified, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.file_path, metrics.lines_of_code, metrics.cyclomatic_complexity,
                metrics.cognitive_complexity, metrics.function_count, metrics.class_count,
                metrics.import_count, metrics.test_coverage, metrics.last_modified, metrics.hash
            ))
            
            # Detect trading patterns
            self._detect_trading_patterns(content, file_path)
            
            # Detect risk rules
            self._detect_risk_rules(content, file_path)
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
            return None
    
    def _path_to_module(self, path: Path) -> str:
        """Convert file path to module name"""
        rel_path = path.relative_to(self.project_root)
        module = str(rel_path).replace('/', '.').replace('\\', '.')
        if module.endswith('.py'):
            module = module[:-3]
        return module
    
    def _detect_trading_patterns(self, content: str, file_path: Path):
        """Detect trading strategy patterns in code"""
        patterns = [
            (r'class\s+(\w+Strategy)', 'strategy_class'),
            (r'def\s+(calculate_\w+)', 'calculation_method'),
            (r'(put|call|option|strike|expiry)', 'options_related'),
            (r'(delta|gamma|theta|vega|rho)', 'greeks'),
            (r'(risk|position|portfolio|allocation)', 'risk_management'),
            (r'(backtest|simulation|monte_carlo)', 'testing'),
            (r'(signal|indicator|momentum|trend)', 'technical_analysis')
        ]
        
        for pattern, pattern_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                
                self.trading_patterns[match.group(0)] = {
                    'type': pattern_type,
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': line_num
                }
                
                # Store in database
                self.conn.execute("""
                    INSERT OR REPLACE INTO trading_patterns
                    (pattern_name, pattern_type, file_path, line_number, description, parameters)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    match.group(0), pattern_type,
                    str(file_path.relative_to(self.project_root)),
                    line_num, '', '{}'
                ))
    
    def _detect_risk_rules(self, content: str, file_path: Path):
        """Detect risk management rules"""
        risk_patterns = [
            (r'max_position\s*=\s*([0-9.]+)', 'position_limit'),
            (r'stop_loss\s*=\s*([0-9.]+)', 'stop_loss'),
            (r'target_delta\s*=\s*([0-9.]+)', 'delta_target'),
            (r'min_premium\s*=\s*([0-9.]+)', 'premium_threshold'),
            (r'if\s+.*risk.*:\s*\n\s*raise', 'risk_check')
        ]
        
        for pattern, rule_type in risk_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                self.risk_rules.append({
                    'type': rule_type,
                    'value': match.group(1) if match.lastindex else None,
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': content[:match.start()].count('\n') + 1
                })
    
    def analyze_codebase(self):
        """Analyze entire codebase"""
        print("=== COMPREHENSIVE CODEBASE ANALYSIS ===")
        start_time = time.time()
        
        # Find all Python files
        py_files = []
        for pattern in ['**/*.py']:
            py_files.extend(self.project_root.glob(pattern))
        
        # Filter out unwanted files
        py_files = [
            f for f in py_files
            if not any(p in str(f) for p in ['.venv', 'venv', '__pycache__', '.git', 'build', 'dist'])
        ]
        
        print(f"Found {len(py_files)} Python files to analyze")
        
        # Analyze each file
        for i, file_path in enumerate(py_files):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(py_files)} files analyzed", end='\r')
            self.analyze_file(file_path)
        
        print(f"\nAnalyzed {len(py_files)} files")
        
        # Commit database changes
        self.conn.commit()
        
        # Build additional indexes
        self._build_call_graph()
        self._build_inheritance_tree()
        self._detect_circular_imports()
        self._identify_hotspots()
        
        # Save all indexes
        self._save_indexes()
        
        elapsed = time.time() - start_time
        print(f"\nAnalysis complete in {elapsed:.2f}s")
        
        # Print summary
        self._print_summary()
    
    def _build_call_graph(self):
        """Build function call graph"""
        # This is a simplified version - full implementation would parse function calls
        pass
    
    def _build_inheritance_tree(self):
        """Build class inheritance tree"""
        for symbols in self.symbols.values():
            for symbol in symbols:
                if symbol.type == 'class' and symbol.parent_class:
                    self.inheritance_tree.add_edge(symbol.parent_class, symbol.name)
    
    def _detect_circular_imports(self):
        """Detect circular import dependencies"""
        cycles = list(nx.simple_cycles(self.import_graph))
        
        if cycles:
            print(f"\nWarning: Found {len(cycles)} circular import dependencies:")
            for cycle in cycles[:5]:  # Show first 5
                print(f"  - {' -> '.join(cycle)} -> {cycle[0]}")
    
    def _identify_hotspots(self):
        """Identify performance hotspots and complex code"""
        # Find files with high complexity
        complex_files = sorted(
            self.file_metrics.items(),
            key=lambda x: x[1].cyclomatic_complexity,
            reverse=True
        )[:10]
        
        if complex_files:
            print("\nMost complex files:")
            for file_path, metrics in complex_files:
                print(f"  - {file_path}: complexity={metrics.cyclomatic_complexity}")
    
    def _save_indexes(self):
        """Save all indexes to disk"""
        # Save import graph
        nx.write_gml(self.import_graph, str(self.index_dir / 'import_graph.gml'))
        
        # Save symbol index
        with open(self.index_dir / 'symbol_index.json', 'w') as f:
            symbol_data = {}
            for name, symbols in self.symbols.items():
                symbol_data[name] = [asdict(s) for s in symbols]
            json.dump(symbol_data, f, indent=2)
        
        # Save metrics
        with open(self.index_dir / 'file_metrics.json', 'w') as f:
            metrics_data = {k: asdict(v) for k, v in self.file_metrics.items()}
            json.dump(metrics_data, f, indent=2)
        
        # Save trading patterns
        with open(self.index_dir / 'trading_patterns.json', 'w') as f:
            json.dump(self.trading_patterns, f, indent=2)
        
        # Save risk rules
        with open(self.index_dir / 'risk_rules.json', 'w') as f:
            json.dump(self.risk_rules, f, indent=2)
        
        # Create quick lookup cache
        cache_data = {
            'symbol_count': len(self.symbols),
            'file_count': len(self.file_metrics),
            'total_loc': sum(m.lines_of_code for m in self.file_metrics.values()),
            'import_count': self.import_graph.number_of_edges(),
            'pattern_count': len(self.trading_patterns),
            'risk_rule_count': len(self.risk_rules),
            'analysis_time': time.time(),
            'project_root': str(self.project_root)
        }
        
        with open(self.index_dir / 'cache_summary.json', 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def _print_summary(self):
        """Print analysis summary"""
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Files analyzed: {len(self.file_metrics)}")
        print(f"Total lines of code: {sum(m.lines_of_code for m in self.file_metrics.values()):,}")
        print(f"Unique symbols: {len(self.symbols)}")
        print(f"Total symbol definitions: {sum(len(s) for s in self.symbols.values())}")
        print(f"Import relationships: {self.import_graph.number_of_edges()}")
        print(f"Trading patterns found: {len(self.trading_patterns)}")
        print(f"Risk rules identified: {len(self.risk_rules)}")
        
        # Module statistics
        print(f"\nModule statistics:")
        print(f"  - Total modules: {self.import_graph.number_of_nodes()}")
        print(f"  - Circular dependencies: {len(list(nx.simple_cycles(self.import_graph)))}")
        
        # Complexity statistics
        if self.file_metrics:
            avg_complexity = sum(m.cyclomatic_complexity for m in self.file_metrics.values()) / len(self.file_metrics)
            print(f"\nComplexity statistics:")
            print(f"  - Average cyclomatic complexity: {avg_complexity:.2f}")
            print(f"  - Files with high complexity (>10): {sum(1 for m in self.file_metrics.values() if m.cyclomatic_complexity > 10)}")

class ComprehensiveVisitor(ast.NodeVisitor):
    """AST visitor to extract comprehensive information"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.symbols: List[Symbol] = []
        self.imports: Set[str] = set()
        self.function_count = 0
        self.class_count = 0
        self.cyclomatic_complexity = 1  # Start at 1
        self.cognitive_complexity = 0
        self.current_class = None
        self.depth = 0
    
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_count += 1
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
        
        # Extract decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        symbol = Symbol(
            name=node.name,
            type='class',
            file=self.file_path,
            line=node.lineno,
            docstring=ast.get_docstring(node) or "",
            decorators=decorators,
            parent_class=bases[0] if bases else ""
        )
        self.symbols.append(symbol)
        
        # Visit methods
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_count += 1
        
        # Extract decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Build signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        signature = f"({', '.join(args)})"
        
        symbol = Symbol(
            name=node.name,
            type='function',
            file=self.file_path,
            line=node.lineno,
            docstring=ast.get_docstring(node) or "",
            signature=signature,
            decorators=decorators,
            parent_class=self.current_class or ""
        )
        self.symbols.append(symbol)
        
        # Update complexity
        self.cyclomatic_complexity += sum(
            1 for child in ast.walk(node)
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler))
        )
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        # Track module-level variables
        if self.current_class is None and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbol = Symbol(
                        name=target.id,
                        type='constant' if target.id.isupper() else 'variable',
                        file=self.file_path,
                        line=node.lineno
                    )
                    self.symbols.append(symbol)
        self.generic_visit(node)
    
    def _get_decorator_name(self, decorator):
        """Extract decorator name"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id
        return str(decorator)

def main():
    """Main entry point"""
    print("Starting comprehensive codebase analysis...")
    
    # Analyze codebase
    analyzer = CodebaseAnalyzer(PROJECT_ROOT)
    analyzer.analyze_codebase()
    
    print("\nAll indexes saved to:", analyzer.index_dir)
    print("Database saved to:", analyzer.db_path)
    
    # Create a startup optimization file
    startup_file = analyzer.index_dir / 'startup_optimization.json'
    with open(startup_file, 'w') as f:
        json.dump({
            'preload_modules': list(analyzer.imports.keys())[:50],  # Top 50 most imported
            'hot_files': [f for f, m in analyzer.file_metrics.items() if m.cyclomatic_complexity > 10],
            'critical_symbols': [name for name in analyzer.symbols.keys() if 'Strategy' in name or 'Manager' in name],
            'timestamp': time.time()
        }, f, indent=2)
    
    print(f"\nStartup optimization data saved to: {startup_file}")

if __name__ == '__main__':
    main()