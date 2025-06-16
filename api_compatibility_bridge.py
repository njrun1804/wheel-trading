#!/usr/bin/env python3
"""
API Compatibility Bridge

Resolves API compatibility issues between system components by providing
unified interfaces and translation layers. Ensures seamless integration
between Einstein, Bolt, accelerated tools, and the main trading system.

Key Features:
- Unified API interfaces across all components
- Automatic parameter translation and validation
- Version compatibility handling
- Error message standardization
- Performance monitoring and optimization
- Backward compatibility support
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class APICallInfo:
    """Information about an API call."""

    component: str
    function_name: str
    args: tuple
    kwargs: dict[str, Any]
    start_time: float
    end_time: float | None = None
    success: bool = False
    result: Any = None
    error: Exception | None = None
    duration: float = 0.0


@dataclass
class ComponentInterface:
    """Definition of a component's API interface."""

    name: str
    version: str
    functions: dict[str, Callable]
    compatibility_map: dict[str, str] = field(default_factory=dict)
    parameter_transforms: dict[str, Callable] = field(default_factory=dict)


class APICompatibilityBridge:
    """Provides compatibility layer between different system components."""

    def __init__(self):
        self.components: dict[str, ComponentInterface] = {}
        self.call_history: list[APICallInfo] = []
        self.performance_stats: dict[str, dict[str, Any]] = {}
        self.error_counts: dict[str, int] = {}

        # Initialize component interfaces
        self._initialize_interfaces()

    def _initialize_interfaces(self):
        """Initialize all component interfaces."""
        # Einstein interface
        self._register_einstein_interface()

        # Bolt interface
        self._register_bolt_interface()

        # Accelerated tools interface
        self._register_accelerated_tools_interface()

        # Storage interface
        self._register_storage_interface()

        # Main system interface
        self._register_main_system_interface()

    def _register_einstein_interface(self):
        """Register Einstein system interface."""
        try:
            # Import Einstein components
            from einstein import query_router, result_merger, unified_index

            functions = {
                "search": unified_index.search
                if hasattr(unified_index, "search")
                else self._not_implemented,
                "build_index": unified_index.build_index
                if hasattr(unified_index, "build_index")
                else self._not_implemented,
                "route_query": query_router.route_query
                if hasattr(query_router, "route_query")
                else self._not_implemented,
                "merge_results": result_merger.merge_results
                if hasattr(result_merger, "merge_results")
                else self._not_implemented,
            }

            # Parameter transformation for Einstein
            transforms = {
                "search": self._transform_einstein_search_params,
                "build_index": self._transform_einstein_index_params,
            }

            self.components["einstein"] = ComponentInterface(
                name="einstein",
                version="2.0",
                functions=functions,
                parameter_transforms=transforms,
            )

            logger.info("✅ Einstein interface registered")

        except ImportError as e:
            logger.warning(f"Einstein interface not available: {e}")
            self._register_mock_interface("einstein")

    def _register_bolt_interface(self):
        """Register Bolt system interface."""
        try:
            from bolt_database_fixes import ConcurrentDatabase, create_database_manager

            functions = {
                "create_manager": create_database_manager,
                "query": self._bolt_query_wrapper,
                "get_connection": self._bolt_connection_wrapper,
                "get_stats": self._bolt_stats_wrapper,
            }

            transforms = {
                "query": self._transform_bolt_query_params,
                "get_connection": self._transform_bolt_connection_params,
            }

            self.components["bolt"] = ComponentInterface(
                name="bolt",
                version="1.0",
                functions=functions,
                parameter_transforms=transforms,
            )

            logger.info("✅ Bolt interface registered")

        except ImportError as e:
            logger.warning(f"Bolt interface not available: {e}")
            self._register_mock_interface("bolt")

    def _register_accelerated_tools_interface(self):
        """Register accelerated tools interface."""
        try:
            from src.unity_wheel.accelerated_tools import (
                dependency_graph_turbo,
                duckdb_turbo,
                python_analysis_turbo,
                ripgrep_turbo,
                trace_turbo,
            )

            functions = {
                "ripgrep_search": ripgrep_turbo.search,
                "ripgrep_parallel_search": self._ripgrep_parallel_wrapper,
                "dependency_search": dependency_graph_turbo.search_code_fuzzy,
                "dependency_build": self._dependency_build_wrapper,
                "duckdb_query": self._duckdb_query_wrapper,
                "python_analyze": self._python_analyze_wrapper,
                "trace_operation": self._trace_wrapper,
            }

            transforms = {
                "ripgrep_search": self._transform_ripgrep_params,
                "dependency_search": self._transform_dependency_params,
                "duckdb_query": self._transform_duckdb_params,
            }

            self.components["accelerated_tools"] = ComponentInterface(
                name="accelerated_tools",
                version="1.0",
                functions=functions,
                parameter_transforms=transforms,
            )

            logger.info("✅ Accelerated tools interface registered")

        except ImportError as e:
            logger.warning(f"Accelerated tools interface not available: {e}")
            self._register_mock_interface("accelerated_tools")

    def _register_storage_interface(self):
        """Register storage system interface."""
        try:
            from src.unity_wheel.storage import duckdb_cache, storage

            functions = {
                "get_storage": storage.Storage,
                "create_cache": duckdb_cache.DuckDBCache,
                "get_connection": self._storage_connection_wrapper,
                "query_cache": self._cache_query_wrapper,
            }

            transforms = {"query_cache": self._transform_cache_query_params}

            self.components["storage"] = ComponentInterface(
                name="storage",
                version="1.0",
                functions=functions,
                parameter_transforms=transforms,
            )

            logger.info("✅ Storage interface registered")

        except ImportError as e:
            logger.warning(f"Storage interface not available: {e}")
            self._register_mock_interface("storage")

    def _register_main_system_interface(self):
        """Register main trading system interface."""
        try:
            # This would import main system components
            # For now, register basic interface
            functions = {
                "get_advisor": self._advisor_wrapper,
                "analyze_position": self._position_analysis_wrapper,
                "calculate_greeks": self._greeks_wrapper,
            }

            self.components["trading_system"] = ComponentInterface(
                name="trading_system", version="1.0", functions=functions
            )

            logger.info("✅ Trading system interface registered")

        except Exception as e:
            logger.warning(f"Trading system interface not available: {e}")
            self._register_mock_interface("trading_system")

    def _register_mock_interface(self, component_name: str):
        """Register a mock interface for unavailable components."""
        functions = {
            "placeholder": lambda *args, **kwargs: {
                "error": f"{component_name} not available"
            }
        }

        self.components[component_name] = ComponentInterface(
            name=component_name, version="mock", functions=functions
        )

    async def call(self, component: str, function: str, *args, **kwargs) -> Any:
        """Make a compatibility-aware API call."""
        call_info = APICallInfo(
            component=component,
            function_name=function,
            args=args,
            kwargs=kwargs,
            start_time=time.time(),
        )

        try:
            # Validate component exists
            if component not in self.components:
                raise ValueError(f"Unknown component: {component}")

            interface = self.components[component]

            # Validate function exists
            if function not in interface.functions:
                # Try compatibility mapping
                mapped_function = interface.compatibility_map.get(function)
                if mapped_function and mapped_function in interface.functions:
                    function = mapped_function
                    logger.info(
                        f"🔄 Using compatibility mapping: {component}.{function} -> {mapped_function}"
                    )
                else:
                    raise ValueError(f"Unknown function: {component}.{function}")

            # Get the actual function
            func = interface.functions[function]

            # Transform parameters if needed
            if function in interface.parameter_transforms:
                transformer = interface.parameter_transforms[function]
                args, kwargs = transformer(args, kwargs)

            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            call_info.success = True
            call_info.result = result
            call_info.end_time = time.time()
            call_info.duration = call_info.end_time - call_info.start_time

            # Update performance stats
            self._update_performance_stats(call_info)

            return result

        except Exception as e:
            call_info.error = e
            call_info.end_time = time.time()
            call_info.duration = call_info.end_time - call_info.start_time

            # Update error counts
            error_key = f"{component}.{function}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

            logger.error(f"❌ API call failed: {component}.{function}: {str(e)}")
            raise

        finally:
            self.call_history.append(call_info)

            # Keep call history manageable
            if len(self.call_history) > 1000:
                self.call_history = self.call_history[-500:]

    def _update_performance_stats(self, call_info: APICallInfo):
        """Update performance statistics."""
        key = f"{call_info.component}.{call_info.function_name}"

        if key not in self.performance_stats:
            self.performance_stats[key] = {
                "call_count": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "success_count": 0,
                "error_count": 0,
            }

        stats = self.performance_stats[key]
        stats["call_count"] += 1
        stats["total_time"] += call_info.duration
        stats["min_time"] = min(stats["min_time"], call_info.duration)
        stats["max_time"] = max(stats["max_time"], call_info.duration)

        if call_info.success:
            stats["success_count"] += 1
        else:
            stats["error_count"] += 1

    # Parameter transformation functions

    def _transform_einstein_search_params(
        self, args: tuple, kwargs: dict
    ) -> tuple[tuple, dict]:
        """Transform parameters for Einstein search calls."""
        # Handle different parameter patterns
        if args and isinstance(args[0], str):
            # Simple search pattern
            query = args[0]
            kwargs.setdefault("max_results", kwargs.get("limit", 10))
            kwargs.setdefault("context_lines", 3)
            return (query,), kwargs

        return args, kwargs

    def _transform_einstein_index_params(
        self, args: tuple, kwargs: dict
    ) -> tuple[tuple, dict]:
        """Transform parameters for Einstein index building."""
        kwargs.setdefault("force_rebuild", False)
        kwargs.setdefault("parallel", True)
        return args, kwargs

    def _transform_bolt_query_params(
        self, args: tuple, kwargs: dict
    ) -> tuple[tuple, dict]:
        """Transform parameters for Bolt database queries."""
        # Ensure proper connection handling
        kwargs.setdefault("timeout", 30.0)
        kwargs.setdefault("retry_attempts", 3)
        return args, kwargs

    def _transform_bolt_connection_params(
        self, args: tuple, kwargs: dict
    ) -> tuple[tuple, dict]:
        """Transform parameters for Bolt connections."""
        kwargs.setdefault("read_only", False)
        kwargs.setdefault("pool_size", 4)
        return args, kwargs

    def _transform_ripgrep_params(
        self, args: tuple, kwargs: dict
    ) -> tuple[tuple, dict]:
        """Transform parameters for ripgrep calls."""
        # Handle legacy parameter names
        if "limit" in kwargs:
            kwargs["max_results"] = kwargs.pop("limit")
        if "type" in kwargs:
            kwargs["file_type"] = kwargs.pop("type")

        kwargs.setdefault("max_results", 100)
        return args, kwargs

    def _transform_dependency_params(
        self, args: tuple, kwargs: dict
    ) -> tuple[tuple, dict]:
        """Transform parameters for dependency graph calls."""
        kwargs.setdefault("fuzzy", True)
        kwargs.setdefault("max_results", 50)
        return args, kwargs

    def _transform_duckdb_params(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Transform parameters for DuckDB calls."""
        kwargs.setdefault("return_type", "pandas")
        kwargs.setdefault("parallel", True)
        return args, kwargs

    def _transform_cache_query_params(
        self, args: tuple, kwargs: dict
    ) -> tuple[tuple, dict]:
        """Transform parameters for cache queries."""
        kwargs.setdefault("max_age_minutes", 15)
        return args, kwargs

    # Wrapper functions for complex integrations

    async def _ripgrep_parallel_wrapper(
        self, patterns: list[str], *args, **kwargs
    ) -> dict[str, Any]:
        """Wrapper for parallel ripgrep searches."""
        from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo

        rg = get_ripgrep_turbo()
        path = kwargs.get("path", ".")
        return await rg.parallel_search(patterns, path)

    async def _dependency_build_wrapper(self, *args, **kwargs) -> dict[str, Any]:
        """Wrapper for dependency graph building."""
        from src.unity_wheel.accelerated_tools.dependency_graph_turbo import (
            get_dependency_graph,
        )

        graph = get_dependency_graph()
        return await graph.build_graph()

    async def _duckdb_query_wrapper(self, query: str, *args, **kwargs) -> Any:
        """Wrapper for DuckDB queries."""
        from src.unity_wheel.accelerated_tools.duckdb_turbo import get_duckdb_turbo

        db_path = kwargs.get("db_path", "data/wheel_trading_master.duckdb")
        db = get_duckdb_turbo(db_path)

        return_type = kwargs.get("return_type", "pandas")
        if return_type == "pandas":
            return await db.query_to_pandas(query)
        else:
            return await db.query(query)

    async def _python_analyze_wrapper(self, *args, **kwargs) -> dict[str, Any]:
        """Wrapper for Python analysis."""
        from src.unity_wheel.accelerated_tools.python_analysis_turbo import (
            get_python_analyzer,
        )

        analyzer = get_python_analyzer()
        path = kwargs.get("path", ".")
        return await analyzer.analyze_directory(path)

    async def _trace_wrapper(self, operation_name: str, *args, **kwargs) -> Any:
        """Wrapper for tracing operations."""
        from src.unity_wheel.accelerated_tools.trace_turbo import get_trace_turbo

        tracer = get_trace_turbo()
        async with tracer.trace_span(operation_name):
            # Execute the wrapped operation
            func = kwargs.get("func")
            if func:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args)
                else:
                    return func(*args)
            return {"traced": True, "operation": operation_name}

    def _bolt_query_wrapper(self, db_path: str, query: str, *args, **kwargs) -> Any:
        """Wrapper for Bolt database queries."""
        from bolt_database_fixes import create_database_manager

        db_manager = create_database_manager(db_path)
        try:
            return db_manager.query(query, kwargs.get("params"))
        finally:
            db_manager.close()

    def _bolt_connection_wrapper(self, db_path: str, *args, **kwargs) -> Any:
        """Wrapper for Bolt database connections."""
        from bolt_database_fixes import create_database_manager

        return create_database_manager(db_path, **kwargs)

    def _bolt_stats_wrapper(self, db_path: str, *args, **kwargs) -> dict[str, Any]:
        """Wrapper for Bolt performance stats."""
        from bolt_database_fixes import create_database_manager

        db_manager = create_database_manager(db_path)
        try:
            return db_manager.get_performance_stats()
        finally:
            db_manager.close()

    def _storage_connection_wrapper(self, *args, **kwargs) -> Any:
        """Wrapper for storage connections."""
        from src.unity_wheel.storage.storage import Storage

        storage = Storage()
        return storage

    def _cache_query_wrapper(self, cache_type: str, *args, **kwargs) -> Any:
        """Wrapper for cache queries."""
        from src.unity_wheel.storage.duckdb_cache import DuckDBCache

        cache = DuckDBCache()
        return cache

    def _advisor_wrapper(self, *args, **kwargs) -> dict[str, Any]:
        """Wrapper for trading advisor."""
        # This would integrate with the main trading system
        return {"advisor": "not_implemented"}

    def _position_analysis_wrapper(self, *args, **kwargs) -> dict[str, Any]:
        """Wrapper for position analysis."""
        # This would integrate with position analysis
        return {"analysis": "not_implemented"}

    def _greeks_wrapper(self, *args, **kwargs) -> dict[str, Any]:
        """Wrapper for Greeks calculation."""
        # This would integrate with Greeks calculation
        return {"greeks": "not_implemented"}

    def _not_implemented(self, *args, **kwargs) -> dict[str, Any]:
        """Placeholder for not implemented functions."""
        return {"error": "Function not implemented"}

    # Utility and monitoring functions

    def get_component_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered components."""
        status = {}

        for name, interface in self.components.items():
            # Count available functions
            available_functions = len(
                [f for f in interface.functions.values() if f != self._not_implemented]
            )
            total_functions = len(interface.functions)

            status[name] = {
                "version": interface.version,
                "available_functions": available_functions,
                "total_functions": total_functions,
                "availability": available_functions / max(1, total_functions),
                "has_transforms": len(interface.parameter_transforms) > 0,
                "compatibility_mappings": len(interface.compatibility_map),
            }

        return status

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        total_calls = sum(
            stats["call_count"] for stats in self.performance_stats.values()
        )
        total_errors = sum(self.error_counts.values())

        # Calculate averages
        avg_times = {}
        for key, stats in self.performance_stats.items():
            if stats["call_count"] > 0:
                avg_times[key] = stats["total_time"] / stats["call_count"]

        # Find slowest and fastest calls
        slowest = (
            max(avg_times.items(), key=lambda x: x[1]) if avg_times else ("none", 0)
        )
        fastest = (
            min(avg_times.items(), key=lambda x: x[1]) if avg_times else ("none", 0)
        )

        return {
            "summary": {
                "total_calls": total_calls,
                "total_errors": total_errors,
                "error_rate": total_errors / max(1, total_calls),
                "components_registered": len(self.components),
                "active_interfaces": len(
                    [c for c in self.components.values() if c.version != "mock"]
                ),
            },
            "performance": {
                "slowest_call": {"function": slowest[0], "avg_time": slowest[1]},
                "fastest_call": {"function": fastest[0], "avg_time": fastest[1]},
                "avg_times": avg_times,
            },
            "detailed_stats": self.performance_stats,
            "error_counts": self.error_counts,
        }

    def validate_compatibility(self) -> dict[str, Any]:
        """Validate compatibility between all components."""
        validation_results = {}

        for component_name, interface in self.components.items():
            results = {
                "version_compatible": interface.version != "mock",
                "functions_available": [],
                "functions_missing": [],
                "parameter_transforms": len(interface.parameter_transforms),
                "compatibility_mappings": len(interface.compatibility_map),
            }

            # Check each function
            for func_name, func in interface.functions.items():
                if func == self._not_implemented:
                    results["functions_missing"].append(func_name)
                else:
                    results["functions_available"].append(func_name)

            # Calculate compatibility score
            total_functions = len(interface.functions)
            available_functions = len(results["functions_available"])
            results["compatibility_score"] = available_functions / max(
                1, total_functions
            )

            validation_results[component_name] = results

        # Overall compatibility
        overall_score = sum(
            r["compatibility_score"] for r in validation_results.values()
        ) / len(validation_results)

        return {
            "overall_compatibility_score": overall_score,
            "component_results": validation_results,
            "critical_issues": [
                name
                for name, result in validation_results.items()
                if result["compatibility_score"] < 0.5
            ],
        }


# Global instance
_api_bridge: APICompatibilityBridge | None = None


def get_api_bridge() -> APICompatibilityBridge:
    """Get or create the global API compatibility bridge."""
    global _api_bridge
    if _api_bridge is None:
        _api_bridge = APICompatibilityBridge()
    return _api_bridge


# Convenience functions for common operations
async def call_einstein(function: str, *args, **kwargs) -> Any:
    """Make a call to Einstein system."""
    bridge = get_api_bridge()
    return await bridge.call("einstein", function, *args, **kwargs)


async def call_bolt(function: str, *args, **kwargs) -> Any:
    """Make a call to Bolt system."""
    bridge = get_api_bridge()
    return await bridge.call("bolt", function, *args, **kwargs)


async def call_accelerated_tools(function: str, *args, **kwargs) -> Any:
    """Make a call to accelerated tools."""
    bridge = get_api_bridge()
    return await bridge.call("accelerated_tools", function, *args, **kwargs)


async def call_storage(function: str, *args, **kwargs) -> Any:
    """Make a call to storage system."""
    bridge = get_api_bridge()
    return await bridge.call("storage", function, *args, **kwargs)


async def call_trading_system(function: str, *args, **kwargs) -> Any:
    """Make a call to trading system."""
    bridge = get_api_bridge()
    return await bridge.call("trading_system", function, *args, **kwargs)


# Decorator for automatic compatibility handling
def compatible_api(component: str, function: str, **transform_kwargs):
    """Decorator to make functions compatible with the API bridge."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            bridge = get_api_bridge()

            # Merge transform kwargs
            for key, value in transform_kwargs.items():
                kwargs.setdefault(key, value)

            return await bridge.call(component, function, *args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":

    async def main():
        # Test the API bridge
        bridge = APICompatibilityBridge()

        # Test component status
        status = bridge.get_component_status()
        print("Component Status:")
        for name, info in status.items():
            print(
                f"  {name}: {info['availability']:.1%} available ({info['available_functions']}/{info['total_functions']} functions)"
            )

        # Test compatibility validation
        compatibility = bridge.validate_compatibility()
        print(
            f"\nOverall Compatibility: {compatibility['overall_compatibility_score']:.1%}"
        )

        if compatibility["critical_issues"]:
            print(f"Critical Issues: {', '.join(compatibility['critical_issues'])}")
        else:
            print("✅ No critical compatibility issues found")

    asyncio.run(main())
