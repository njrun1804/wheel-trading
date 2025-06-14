# Examples: Migrating from MCP to Accelerated Tools

## Before (MCP way):
```python
# Slow MCP imports
from mcp_client import ripgrep, dependency_graph, python_analysis

# Usage
await ripgrep.search("pattern")
await dependency_graph.find_symbol("MyClass")
await python_analysis.analyze_file("file.py")
```

## After (Accelerated way):
```python
# Fast local imports
from unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
from unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph
from unity_wheel.accelerated_tools.python_analysis_turbo import get_python_analyzer

# Usage (10-30x faster!)
rg = get_ripgrep_turbo()
await rg.search("pattern")  # Uses all 12 CPU cores

graph = get_dependency_graph()
await graph.find_symbol("MyClass")  # GPU-accelerated

analyzer = get_python_analyzer()
await analyzer.analyze_file("file.py")  # Parallel AST parsing
```

## Drop-in Replacements:
```python
# These functions work exactly like MCP versions but 10-30x faster:

# Ripgrep
from unity_wheel.accelerated_tools.ripgrep_turbo import search, search_count

# Dependency graph  
from unity_wheel.accelerated_tools.dependency_graph_turbo import search_code_fuzzy, get_dependencies

# Python analysis
from unity_wheel.accelerated_tools.python_analysis_turbo import analyze_code

# DuckDB
from unity_wheel.accelerated_tools.duckdb_turbo import query, execute, describe_table

# Trace
from unity_wheel.accelerated_tools.trace_simple import start_trace, end_trace

# Python helpers
from unity_wheel.accelerated_tools.python_helpers_turbo import get_function_info, analyze_project
```
