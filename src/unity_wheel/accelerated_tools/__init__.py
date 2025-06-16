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

# Import modules for direct access
from . import trace_simple, trace_turbo

# Import reliable initialization system for enhanced reliability
try:
    from .reliable_initialization import (
        ensure_tools_initialized,
        get_tool_fast,
        validate_all_tools,
        get_tools_manager,
    )
    HAS_RELIABLE_INIT = True
except ImportError:
    HAS_RELIABLE_INIT = False

# Import all turbo implementations
from .dependency_graph_turbo import (
    get_dependencies,
    get_dependency_graph,
    search_code_fuzzy,
)
from .duckdb_turbo import describe_table, get_duckdb_turbo
from .duckdb_turbo import execute as duckdb_execute
from .duckdb_turbo import query as duckdb_query
from .einstein_neural_integration import (
    EinsteinEmbeddingConfig,
    EinsteinNeuralBridge,
    create_ane_embedding_function,
    embed_code_files_ane,
    get_einstein_ane_pipeline,
)
from .neural_engine_turbo import (
    ANEDeviceManager,
    NeuralEngineTurbo,
    get_neural_engine_turbo,
)
from .python_analysis_turbo import analyze_code, get_python_analyzer
from .python_helpers_turbo import analyze_project, get_code_helper, get_function_info
from .ripgrep_turbo import get_ripgrep_turbo
from .ripgrep_turbo import search as ripgrep_search
from .ripgrep_turbo import search_count as ripgrep_count
from .trace_simple import TraceConfig, add_span_attribute, end_trace, start_trace
from .trace_simple import get_trace_turbo as get_trace_simple
from .trace_turbo import get_trace_turbo as get_trace_turbo_full

# Lazy initialization to avoid async issues
_ripgrep = None
_dependency_graph = None
_python_analyzer = None
_duckdb = None
_tracer = None
_code_helper = None
_neural_engine = None
_einstein_ane_pipeline = None


def get_ripgrep_instance():
    """Get ripgrep instance with reliable initialization fallback."""
    global _ripgrep
    if _ripgrep is None:
        try:
            _ripgrep = get_ripgrep_turbo()
        except Exception as e:
            # Fallback to reliable system if available
            if HAS_RELIABLE_INIT:
                _ripgrep = get_tool_fast('ripgrep_turbo')
            if _ripgrep is None:
                # Last resort - try direct initialization again
                _ripgrep = get_ripgrep_turbo()
    return _ripgrep


def get_dependency_graph_instance():
    """Get dependency graph instance with reliable initialization fallback."""
    global _dependency_graph
    if _dependency_graph is None:
        try:
            _dependency_graph = get_dependency_graph()
        except Exception as e:
            if HAS_RELIABLE_INIT:
                _dependency_graph = get_tool_fast('dependency_graph_turbo')
            if _dependency_graph is None:
                _dependency_graph = get_dependency_graph()
    return _dependency_graph


def get_python_analyzer_instance():
    """Get python analyzer instance with reliable initialization fallback."""
    global _python_analyzer
    if _python_analyzer is None:
        try:
            _python_analyzer = get_python_analyzer()
        except Exception as e:
            if HAS_RELIABLE_INIT:
                _python_analyzer = get_tool_fast('python_analysis_turbo')
            if _python_analyzer is None:
                _python_analyzer = get_python_analyzer()
    return _python_analyzer


def get_duckdb_instance():
    """Get duckdb instance with reliable initialization fallback."""
    global _duckdb
    if _duckdb is None:
        try:
            _duckdb = get_duckdb_turbo()
        except Exception as e:
            if HAS_RELIABLE_INIT:
                _duckdb = get_tool_fast('duckdb_turbo')
            if _duckdb is None:
                _duckdb = get_duckdb_turbo()
    return _duckdb


def get_tracer_instance():
    """Get tracer instance with reliable initialization fallback."""
    global _tracer
    if _tracer is None:
        try:
            # Use the simple trace implementation by default (no external dependencies)
            _tracer = get_trace_simple()
        except Exception as e:
            if HAS_RELIABLE_INIT:
                _tracer = get_tool_fast('trace_turbo')
            if _tracer is None:
                _tracer = get_trace_simple()
    return _tracer


def get_trace_turbo(use_full_implementation: bool = False):
    """Get trace implementation - simple by default, full with external deps if requested."""
    if use_full_implementation:
        try:
            return get_trace_turbo_full()
        except Exception:
            if HAS_RELIABLE_INIT:
                tool = get_tool_fast('trace_turbo')
                if tool:
                    return tool
            return get_trace_simple()
    else:
        return get_trace_simple()


def get_code_helper_instance():
    """Get code helper instance with reliable initialization fallback."""
    global _code_helper
    if _code_helper is None:
        try:
            _code_helper = get_code_helper()
        except Exception as e:
            if HAS_RELIABLE_INIT:
                _code_helper = get_tool_fast('python_helpers_turbo')
            if _code_helper is None:
                _code_helper = get_code_helper()
    return _code_helper


def get_neural_engine_instance():
    global _neural_engine
    if _neural_engine is None:
        _neural_engine = get_neural_engine_turbo()
    return _neural_engine


def get_einstein_ane_pipeline_instance():
    global _einstein_ane_pipeline
    if _einstein_ane_pipeline is None:
        _einstein_ane_pipeline = get_einstein_ane_pipeline()
    return _einstein_ane_pipeline


# Lazy initialized instances - call these functions to get instances
ripgrep = get_ripgrep_instance
dependency_graph = get_dependency_graph_instance
python_analyzer = get_python_analyzer_instance
duckdb = get_duckdb_instance
tracer = get_tracer_instance
code_helper = get_code_helper_instance
neural_engine = get_neural_engine_instance
einstein_ane_pipeline = get_einstein_ane_pipeline_instance

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
    "get_trace_simple",
    "get_trace_turbo_full",
    "start_trace",
    "end_trace",
    "add_span_attribute",
    "TraceConfig",
    "trace_simple",
    "trace_turbo",
    # Python helpers
    "code_helper",
    "get_code_helper",
    "get_function_info",
    "analyze_project",
    # Neural Engine (ANE)
    "neural_engine",
    "get_neural_engine_turbo",
    "NeuralEngineTurbo",
    "ANEDeviceManager",
    # Einstein ANE Integration
    "einstein_ane_pipeline",
    "get_einstein_ane_pipeline",
    "EinsteinEmbeddingConfig",
    "EinsteinNeuralBridge",
    "embed_code_files_ane",
    "create_ane_embedding_function",
    
    # Reliable initialization system
    "ensure_tools_initialized",
    "validate_all_tools", 
    "get_tools_manager",
    "get_tool_fast",
    "HAS_RELIABLE_INIT",
]
