#!/usr/bin/env python3
"""Dependency graph MCP server for code analysis."""

from mcp.server import FastMCP
import ast
import os
from pathlib import Path
import json

mcp = FastMCP("dependency-graph")

@mcp.tool()
def analyze_imports(file_path: str) -> str:
    """Analyze Python imports in a file.
    
    Args:
        file_path: Path to the Python file to analyze
    """
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            return f"Error: File not found: {file_path}"
        
        with open(path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"type": "import", "module": alias.name})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name
                    })
        
        return json.dumps({
            "file": str(path),
            "imports": imports,
            "import_count": len(imports)
        }, indent=2)
        
    except SyntaxError as e:
        return f"Syntax error in {file_path}: {e}"
    except Exception as e:
        return f"Error analyzing {file_path}: {str(e)}"

@mcp.tool()
def find_dependencies(directory: str, module_name: str) -> str:
    """Find all files that import a specific module.
    
    Args:
        directory: Directory to search in
        module_name: Module name to search for
    """
    try:
        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            return f"Error: Directory not found: {directory}"
        
        dependencies = []
        
        for py_file in dir_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if module_name in alias.name:
                                dependencies.append(str(py_file))
                                break
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and module_name in node.module:
                            dependencies.append(str(py_file))
                            break
            except:
                continue
        
        return json.dumps({
            "module": module_name,
            "used_by": dependencies,
            "usage_count": len(dependencies)
        }, indent=2)
        
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def generate_dependency_graph(directory: str) -> str:
    """Generate a full dependency graph for a Python project.
    
    Args:
        directory: Root directory of the project
    """
    try:
        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            return f"Error: Directory not found: {directory}"
        
        graph = {}
        
        for py_file in dir_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                file_imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            file_imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            file_imports.append(node.module)
                
                relative_path = py_file.relative_to(dir_path)
                graph[str(relative_path)] = list(set(file_imports))
                
            except:
                continue
        
        return json.dumps(graph, indent=2)
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("Starting dependency-graph MCP server...")
    mcp.run()
