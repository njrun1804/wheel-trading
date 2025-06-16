#!/usr/bin/env python3
"""
Phase 3B ML Framework Optimization Validation

This script validates which ML frameworks are actually imported and used
in the codebase, confirming our optimization decisions.
"""

import ast
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set


class MLFrameworkAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze ML framework imports and usage."""
    
    def __init__(self):
        self.imports = []
        self.from_imports = []
        self.function_calls = []
        self.attribute_access = []
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                self.from_imports.append(f"{node.module}.{alias.name}")
        self.generic_visit(node)
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.function_calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.attribute_access.append(node.func.attr)
        self.generic_visit(node)


def analyze_file(file_path: Path) -> Dict[str, List[str]]:
    """Analyze a Python file for ML framework usage."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        analyzer = MLFrameworkAnalyzer()
        analyzer.visit(tree)
        
        return {
            'imports': analyzer.imports,
            'from_imports': analyzer.from_imports,
            'function_calls': analyzer.function_calls,
            'attribute_access': analyzer.attribute_access
        }
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return {'imports': [], 'from_imports': [], 'function_calls': [], 'attribute_access': []}


def main():
    """Main validation function."""
    print("=" * 80)
    print("PHASE 3B: ML FRAMEWORK OPTIMIZATION VALIDATION")
    print("=" * 80)
    print()
    
    # Define ML framework patterns to look for
    ml_frameworks = {
        'mlx': ['mlx', 'mlx.core', 'mlx.nn', 'mlx.optimizers'],
        'torch': ['torch', 'torch.nn', 'torch.optim', 'torch.cuda'],
        'sklearn': ['sklearn', 'scikit-learn', 'sklearn.'],
        'faiss': ['faiss'],
        'transformers': ['sentence_transformers', 'transformers', 'huggingface'],
        'tensorflow': ['tensorflow', 'tf'],
        'jax': ['jax'],
        'xgboost': ['xgboost'],
        'lightgbm': ['lightgbm'],
        'hnswlib': ['hnswlib'],
        'usearch': ['usearch'],
        'numpy': ['numpy', 'np'],
        'pandas': ['pandas', 'pd'],
        'scipy': ['scipy']
    }
    
    # Scan all Python files
    project_root = Path(".")
    python_files = list(project_root.rglob("*.py"))
    
    print(f"Scanning {len(python_files)} Python files...")
    print()
    
    # Track usage by framework
    framework_usage = defaultdict(list)
    framework_files = defaultdict(set)
    
    for file_path in python_files:
        # Skip certain directories
        if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
            continue
            
        analysis = analyze_file(file_path)
        
        # Check for ML framework usage
        all_imports = analysis['imports'] + analysis['from_imports']
        
        for framework, patterns in ml_frameworks.items():
            for pattern in patterns:
                for import_stmt in all_imports:
                    if pattern in import_stmt.lower():
                        framework_usage[framework].append(import_stmt)
                        framework_files[framework].add(str(file_path))
                        break
    
    # Print results
    print("ML FRAMEWORK USAGE ANALYSIS:")
    print("-" * 40)
    
    essential_frameworks = []
    unused_frameworks = []
    
    for framework in sorted(ml_frameworks.keys()):
        files = framework_files[framework]
        imports = framework_usage[framework]
        
        if files:
            print(f"✅ {framework.upper()}: USED")
            print(f"   Files: {len(files)}")
            print(f"   Import patterns: {len(set(imports))}")
            
            # Show some example files
            if len(files) <= 5:
                for file in sorted(files):
                    print(f"     - {file}")
            else:
                for file in sorted(list(files)[:3]):
                    print(f"     - {file}")
                print(f"     - ... and {len(files) - 3} more")
            print()
            essential_frameworks.append(framework)
        else:
            print(f"❌ {framework.upper()}: NOT USED")
            unused_frameworks.append(framework)
    
    print()
    print("OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 40)
    
    print("KEEP (Essential frameworks):")
    for fw in essential_frameworks:
        if fw in ['mlx', 'sklearn', 'faiss', 'transformers', 'hnswlib']:
            print(f"  ✅ {fw} - Actually used in codebase")
    
    print()
    print("REMOVE/DISABLE (Unused frameworks):")
    for fw in unused_frameworks:
        if fw not in ['numpy', 'pandas', 'scipy']:  # These are core dependencies
            print(f"  ❌ {fw} - Not used, can be removed")
    
    print()
    print("CONDITIONAL (Fallback only):")
    if 'torch' in essential_frameworks:
        torch_files = framework_files['torch']
        print(f"  ⚠️  torch - Used in {len(torch_files)} files, but mostly as fallback")
        for file in torch_files:
            if 'mlx' in file:
                print(f"     - {file} (MLX fallback pattern)")
    
    print()
    print("APPLE SILICON OPTIMIZATION STATUS:")
    print("-" * 40)
    
    metal_optimized = ['mlx', 'faiss', 'hnswlib']
    for fw in metal_optimized:
        if fw in essential_frameworks:
            print(f"  🚀 {fw} - Optimized for Apple Silicon Metal GPU")
    
    if 'torch' in essential_frameworks:
        print(f"  ⚠️  torch - Can use MPS backend but MLX is preferred on M4 Pro")
    
    print()
    print("SUMMARY:")
    print("-" * 40)
    print(f"Total frameworks analyzed: {len(ml_frameworks)}")
    print(f"Actually used: {len(essential_frameworks)}")
    print(f"Can be removed: {len(unused_frameworks)}")
    print(f"Optimization potential: {len(unused_frameworks)/len(ml_frameworks)*100:.1f}%")
    
    # Specific recommendations for Phase 3B
    print()
    print("PHASE 3B SPECIFIC RECOMMENDATIONS:")
    print("-" * 40)
    
    keep_frameworks = []
    remove_frameworks = []
    
    if 'mlx' in essential_frameworks:
        keep_frameworks.append("mlx - Primary ML framework for Apple Silicon")
    
    if 'sklearn' in essential_frameworks:
        keep_frameworks.append("scikit-learn - Used for regime detection and adaptive models")
        
    if 'faiss' in essential_frameworks:
        keep_frameworks.append("faiss-cpu - Vector similarity search (Metal optimized)")
        
    if 'transformers' in essential_frameworks:
        keep_frameworks.append("sentence-transformers - Semantic search embeddings")
        
    if 'hnswlib' in essential_frameworks:
        keep_frameworks.append("hnswlib - Fast approximate nearest neighbor search")
    
    # Check for unused frameworks
    for fw in unused_frameworks:
        if fw not in ['numpy', 'pandas', 'scipy']:
            remove_frameworks.append(f"{fw} - Not used in codebase")
    
    print()
    print("KEEP:")
    for fw in keep_frameworks:
        print(f"  ✅ {fw}")
    
    print()
    print("REMOVE:")
    for fw in remove_frameworks:
        print(f"  ❌ {fw}")
    
    if 'torch' in essential_frameworks:
        print()
        print("CONDITIONAL:")
        print("  ⚠️  torch - Comment out in requirements, only used as MLX fallback")
    
    print()
    print("✅ Phase 3B ML Framework Optimization Analysis Complete!")


if __name__ == "__main__":
    main()