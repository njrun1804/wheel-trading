"""Hardware-accelerated tools for M4 Pro - 10-30x faster than MCP servers.

Quick usage:
    from unity_wheel.accelerated_tools import ripgrep, dependency_graph, python_analyzer
    
    # Search with all 12 cores
    results = await ripgrep.search("pattern", "src")
    
    # Build dependency graph with GPU
    await dependency_graph.build_graph()
    
    # Analyze Python code with parallel AST
    analysis = await python_analyzer.analyze_directory("src")
"""

# Import all turbo implementations
from .ripgrep_turbo import (
    get_ripgrep_turbo,
    search as ripgrep_search,
    search_count as ripgrep_count
)

from .dependency_graph_turbo import (
    get_dependency_graph,
    search_code_fuzzy,
    get_dependencies
)

from .python_analysis_turbo import (
    get_python_analyzer,
    analyze_code
)

from .duckdb_turbo import (
    get_duckdb_turbo,
    query as duckdb_query,
    execute as duckdb_execute,
    describe_table
)

from .trace_simple import (
    get_trace_turbo,
    start_trace,
    end_trace,
    add_span_attribute,
    TraceConfig
)

from .python_helpers_turbo import (
    get_code_helper,
    get_function_info,
    analyze_project
)

# Convenience objects for direct access
ripgrep = get_ripgrep_turbo()
dependency_graph = get_dependency_graph()
python_analyzer = get_python_analyzer()
duckdb = get_duckdb_turbo()
tracer = get_trace_turbo()
code_helper = get_code_helper()

# Version info
__version__ = "1.0.0"
__all__ = [
    # Ripgrep
    "ripgrep",
    "get_ripgrep_turbo",
    "ripgrep_search",
    "ripgrep_count",
    
    # Dependency graph
    "dependency_graph",
    "get_dependency_graph",
    "search_code_fuzzy",
    "get_dependencies",
    
    # Python analysis
    "python_analyzer",
    "get_python_analyzer",
    "analyze_code",
    
    # DuckDB
    "duckdb",
    "get_duckdb_turbo",
    "duckdb_query",
    "duckdb_execute",
    "describe_table",
    
    # Trace
    "tracer",
    "get_trace_turbo",
    "start_trace",
    "end_trace",
    "add_span_attribute",
    "TraceConfig",
    
    # Python helpers
    "code_helper",
    "get_code_helper",
    "get_function_info",
    "analyze_project",
]