"""
Fallback implementations for Bolt core functions.

These provide real functionality when the optimized accelerated tools are not available,
ensuring the system remains fully functional even without all dependencies.
"""

import logging
from typing import Any, Optional

from .code_helper_fallback import CodeHelperFallback
from .dependency_fallback import DependencyGraphFallback
from .duckdb_fallback import DuckDBFallback
from .python_analyzer_fallback import PythonAnalyzerFallback
from .ripgrep_fallback import RipgrepFallback
from .trace_fallback import TraceFallback

logger = logging.getLogger(__name__)

__all__ = [
    "RipgrepFallback",
    "DependencyGraphFallback",
    "PythonAnalyzerFallback",
    "DuckDBFallback",
    "TraceFallback",
    "CodeHelperFallback",
    "get_accelerated_tool",
]


def get_accelerated_tool(tool_name: str) -> Any | None:
    """
    Get an accelerated tool instance or fallback implementation.

    This function attempts to load accelerated tools first, falling back to
    local implementations if the accelerated versions are not available.

    Args:
        tool_name: Name of the tool to get (e.g., 'ripgrep_turbo', 'python_analysis_turbo')

    Returns:
        Tool instance or None if not available
    """
    try:
        # Ensure src is in path for accelerated tools
        import sys
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # First try to get accelerated tools
        if tool_name in ["ripgrep_turbo", "ripgrep"]:
            try:
                from src.unity_wheel.accelerated_tools.ripgrep_turbo import (
                    get_ripgrep_turbo,
                )
                return get_ripgrep_turbo()
            except ImportError as e:
                logger.debug(f"Accelerated ripgrep not available, using fallback: {e}")
                return RipgrepFallback()

        elif tool_name in ["dependency_graph_turbo", "dependency_graph"]:
            try:
                from src.unity_wheel.accelerated_tools.dependency_graph_turbo import (
                    get_dependency_graph,
                )
                return get_dependency_graph()
            except ImportError as e:
                logger.debug(f"Accelerated dependency graph not available, using fallback: {e}")
                return DependencyGraphFallback()

        elif tool_name in ["python_analysis_turbo", "python_analyzer"]:
            try:
                from src.unity_wheel.accelerated_tools.python_analysis_turbo import (
                    get_python_analyzer,
                )
                return get_python_analyzer()
            except ImportError as e:
                logger.debug(f"Accelerated python analyzer not available, using fallback: {e}")
                return PythonAnalyzerFallback()

        elif tool_name in ["duckdb_turbo", "duckdb"]:
            try:
                from src.unity_wheel.accelerated_tools.duckdb_turbo import (
                    get_duckdb_turbo,
                )
                return get_duckdb_turbo()
            except ImportError as e:
                logger.debug(f"Accelerated DuckDB not available, using fallback: {e}")
                return DuckDBFallback()

        elif tool_name in ["trace_turbo", "trace"]:
            try:
                from src.unity_wheel.accelerated_tools.trace_turbo import (
                    get_trace_turbo,
                )
                return get_trace_turbo()
            except ImportError as e:
                logger.debug(f"Accelerated trace not available, using fallback: {e}")
                return TraceFallback()

        elif tool_name in ["python_helpers_turbo", "code_helper"]:
            try:
                from src.unity_wheel.accelerated_tools.python_helpers_turbo import (
                    get_code_helper,
                )
                return get_code_helper()
            except ImportError as e:
                logger.debug(f"Accelerated code helper not available, using fallback: {e}")
                return CodeHelperFallback()

        else:
            logger.warning(f"Unknown tool requested: {tool_name}")
            return None

    except Exception as e:
        logger.error(f"Error getting accelerated tool {tool_name}: {e}")
        # Return appropriate fallback based on tool name
        if "ripgrep" in tool_name:
            return RipgrepFallback()
        elif "dependency" in tool_name:
            return DependencyGraphFallback()
        elif "python" in tool_name or "analyzer" in tool_name:
            return PythonAnalyzerFallback()
        elif "duckdb" in tool_name:
            return DuckDBFallback()
        elif "trace" in tool_name:
            return TraceFallback()
        elif "helper" in tool_name or "code" in tool_name:
            return CodeHelperFallback()
        else:
            return None
