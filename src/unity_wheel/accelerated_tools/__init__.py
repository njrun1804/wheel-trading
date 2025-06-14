"""Hardware-accelerated tools for M4 Pro - 10-30x faster than MCP servers.

Quick usage:
    from ..accelerated_tools import ripgrep, dependency_graph, python_analyzer
    
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

# Lazy initialization to avoid async issues
_ripgrep = None
_dependency_graph = None
_python_analyzer = None
_duckdb = None
_tracer = None
_code_helper = None

@property
def ripgrep():
    global _ripgrep
    if _ripgrep is None:
        _ripgrep = get_ripgrep_turbo()
    return _ripgrep

@property
def dependency_graph():
    global _dependency_graph
    if _dependency_graph is None:
        _dependency_graph = get_dependency_graph()
    return _dependency_graph

@property
def python_analyzer():
    global _python_analyzer
    if _python_analyzer is None:
        _python_analyzer = get_python_analyzer()
    return _python_analyzer

@property
def duckdb():
    global _duckdb
    if _duckdb is None:
        _duckdb = get_duckdb_turbo()
    return _duckdb

@property
def tracer():
    global _tracer
    if _tracer is None:
        _tracer = get_trace_turbo()
    return _tracer

@property
def code_helper():
    global _code_helper
    if _code_helper is None:
        _code_helper = get_code_helper()
    return _code_helper

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