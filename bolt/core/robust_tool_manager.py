#!/usr/bin/env python3
"""
Robust Tool Access Manager for Bolt Agents

Provides reliable access to accelerated tools with:
- Automatic retry logic with exponential backoff
- Graceful fallbacks when tools are unavailable
- Health monitoring and recovery
- Circuit breaker pattern for failing tools
- Agent-specific tool isolation
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolState(Enum):
    """State of tool availability."""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RECOVERING = "recovering"


@dataclass
class ToolHealthMetrics:
    """Health metrics for a tool."""
    success_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    error_count: int = 0
    last_success: Optional[float] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    total_calls: int = 0


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    initial_delay_ms: float = 100.0
    max_delay_ms: float = 5000.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


class CircuitBreaker:
    """Circuit breaker for tool access."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = ToolState.AVAILABLE
        self.lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self.lock:
            if self.state == ToolState.AVAILABLE:
                return True
            elif self.state == ToolState.UNAVAILABLE:
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = ToolState.RECOVERING
                    return True
                return False
            elif self.state == ToolState.RECOVERING:
                return True
            return False
    
    def record_success(self):
        """Record successful execution."""
        with self.lock:
            self.failure_count = 0
            self.state = ToolState.AVAILABLE
    
    def record_failure(self):
        """Record failed execution."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = ToolState.UNAVAILABLE
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class ToolWrapper:
    """Wrapper for a tool with health monitoring and retry logic."""
    
    def __init__(self, tool_name: str, tool_instance: Any, retry_config: RetryConfig):
        self.tool_name = tool_name
        self.tool_instance = tool_instance
        self.retry_config = retry_config
        self.health_metrics = ToolHealthMetrics()
        self.circuit_breaker = CircuitBreaker()
        self.lock = asyncio.Lock()
    
    async def execute_with_retry(self, method_name: str, *args, **kwargs) -> Any:
        """Execute tool method with retry logic and circuit breaker."""
        if not self.circuit_breaker.can_execute():
            raise RuntimeError(f"Circuit breaker open for {self.tool_name}")
        
        start_time = time.perf_counter()
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                # Get the method from the tool instance
                if hasattr(self.tool_instance, method_name):
                    method = getattr(self.tool_instance, method_name)
                else:
                    # Try calling the tool instance directly if it's callable
                    if callable(self.tool_instance):
                        method = self.tool_instance
                    else:
                        raise AttributeError(f"{self.tool_name} has no method {method_name}")
                
                # Execute the method
                if asyncio.iscoroutinefunction(method):
                    result = await method(*args, **kwargs)
                else:
                    # Run in thread pool for sync methods
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, method, *args, **kwargs)
                
                # Record success
                duration_ms = (time.perf_counter() - start_time) * 1000
                await self._record_success(duration_ms)
                self.circuit_breaker.record_success()
                
                return result
                
            except Exception as e:
                last_exception = e
                await self._record_failure(str(e))
                
                if attempt < self.retry_config.max_attempts - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay_ms = min(
                        self.retry_config.initial_delay_ms * (self.retry_config.backoff_multiplier ** attempt),
                        self.retry_config.max_delay_ms
                    )
                    
                    if self.retry_config.jitter:
                        import random
                        delay_ms *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
                    
                    logger.debug(f"Retry {attempt + 1} for {self.tool_name}.{method_name} after {delay_ms:.1f}ms")
                    await asyncio.sleep(delay_ms / 1000.0)
        
        # All attempts failed
        self.circuit_breaker.record_failure()
        raise RuntimeError(f"All {self.retry_config.max_attempts} attempts failed for {self.tool_name}.{method_name}: {last_exception}")
    
    async def _record_success(self, duration_ms: float):
        """Record successful operation."""
        async with self.lock:
            self.health_metrics.total_calls += 1
            self.health_metrics.consecutive_failures = 0
            self.health_metrics.last_success = time.time()
            
            # Update moving average response time
            if self.health_metrics.avg_response_time_ms == 0:
                self.health_metrics.avg_response_time_ms = duration_ms
            else:
                self.health_metrics.avg_response_time_ms = (
                    self.health_metrics.avg_response_time_ms * 0.9 + duration_ms * 0.1
                )
            
            # Update success rate
            success_count = self.health_metrics.total_calls - self.health_metrics.error_count
            self.health_metrics.success_rate = success_count / self.health_metrics.total_calls
    
    async def _record_failure(self, error_msg: str):
        """Record failed operation."""
        async with self.lock:
            self.health_metrics.total_calls += 1
            self.health_metrics.error_count += 1
            self.health_metrics.consecutive_failures += 1
            self.health_metrics.last_error = error_msg
            
            # Update success rate
            success_count = self.health_metrics.total_calls - self.health_metrics.error_count
            self.health_metrics.success_rate = success_count / self.health_metrics.total_calls
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "tool_name": self.tool_name,
            "state": self.circuit_breaker.state.value,
            "metrics": {
                "success_rate": self.health_metrics.success_rate,
                "avg_response_time_ms": self.health_metrics.avg_response_time_ms,
                "error_count": self.health_metrics.error_count,
                "consecutive_failures": self.health_metrics.consecutive_failures,
                "total_calls": self.health_metrics.total_calls,
                "last_success": self.health_metrics.last_success,
                "last_error": self.health_metrics.last_error
            }
        }


class RobustToolManager:
    """Manages robust access to accelerated tools for Bolt agents."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.tools: Dict[str, ToolWrapper] = {}
        self.fallback_tools: Dict[str, Any] = {}
        self.retry_config = RetryConfig()
        self.initialization_lock = asyncio.Lock()
        self.initialized = False
        self.health_check_interval = 30.0  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Load hardware optimization config
        self.hw_config = self._load_hardware_config()
        
        logger.info(f"RobustToolManager created for agent {agent_id}")
    
    def _load_hardware_config(self) -> Dict[str, Any]:
        """Load hardware optimization configuration."""
        try:
            config_path = Path("optimization_config.json")
            if config_path.exists():
                with open(config_path) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load hardware config: {e}")
        
        # Default M4 Pro configuration
        return {
            "cpu": {"max_workers": 8, "batch_size": 16},
            "memory": {"cache_size_mb": 1024},
            "gpu": {"batch_size": 32}
        }
    
    async def initialize_tools(self, force_reinit: bool = False) -> Dict[str, ToolState]:
        """Initialize all accelerated tools with robust error handling."""
        async with self.initialization_lock:
            if self.initialized and not force_reinit:
                return {name: wrapper.circuit_breaker.state for name, wrapper in self.tools.items()}
            
            logger.info(f"Initializing accelerated tools for agent {self.agent_id}")
            initialization_results = {}
            
            # Tool initialization with fallbacks
            tool_configs = [
                ("ripgrep", self._init_ripgrep_tool),
                ("dependency_graph", self._init_dependency_graph_tool),
                ("python_analyzer", self._init_python_analyzer_tool),
                ("duckdb", self._init_duckdb_tool),
                ("trace", self._init_trace_tool),
                ("code_helper", self._init_code_helper_tool),
            ]
            
            # Initialize tools concurrently
            init_tasks = []
            for tool_name, init_func in tool_configs:
                init_tasks.append(self._safe_tool_init(tool_name, init_func))
            
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            for i, (tool_name, _) in enumerate(tool_configs):
                result = results[i]
                if isinstance(result, Exception):
                    logger.warning(f"Failed to initialize {tool_name}: {result}")
                    initialization_results[tool_name] = ToolState.UNAVAILABLE
                    self._setup_fallback_tool(tool_name)
                else:
                    initialization_results[tool_name] = result
            
            self.initialized = True
            
            # Start health monitoring
            if not self.health_check_task or self.health_check_task.done():
                self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info(f"Tool initialization complete for agent {self.agent_id}: {initialization_results}")
            return initialization_results
    
    async def _safe_tool_init(self, tool_name: str, init_func) -> ToolState:
        """Safely initialize a tool with timeout and error handling."""
        try:
            # Use timeout to prevent hanging initialization
            tool_instance = await asyncio.wait_for(init_func(), timeout=10.0)
            if tool_instance:
                wrapper = ToolWrapper(tool_name, tool_instance, self.retry_config)
                self.tools[tool_name] = wrapper
                return ToolState.AVAILABLE
            else:
                return ToolState.UNAVAILABLE
        except asyncio.TimeoutError:
            logger.warning(f"Tool {tool_name} initialization timed out")
            return ToolState.UNAVAILABLE
        except Exception as e:
            logger.warning(f"Tool {tool_name} initialization failed: {e}")
            return ToolState.UNAVAILABLE
    
    async def _init_ripgrep_tool(self) -> Any:
        """Initialize ripgrep tool with error handling."""
        try:
            from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
            return get_ripgrep_turbo()
        except ImportError as e:
            logger.warning(f"Failed to import ripgrep_turbo: {e}")
            return None
    
    async def _init_dependency_graph_tool(self) -> Any:
        """Initialize dependency graph tool with error handling."""
        try:
            from src.unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph
            return get_dependency_graph()
        except ImportError as e:
            logger.warning(f"Failed to import dependency_graph_turbo: {e}")
            return None
    
    async def _init_python_analyzer_tool(self) -> Any:
        """Initialize Python analyzer tool with error handling."""
        try:
            from src.unity_wheel.accelerated_tools.python_analysis_turbo import get_python_analyzer
            return get_python_analyzer()
        except ImportError as e:
            logger.warning(f"Failed to import python_analysis_turbo: {e}")
            return None
    
    async def _init_duckdb_tool(self) -> Any:
        """Initialize DuckDB tool with error handling."""
        try:
            from src.unity_wheel.accelerated_tools.duckdb_turbo import get_duckdb_turbo
            # Use memory database for agent isolation
            return get_duckdb_turbo(":memory:")
        except ImportError as e:
            logger.warning(f"Failed to import duckdb_turbo: {e}")
            return None
    
    async def _init_trace_tool(self) -> Any:
        """Initialize trace tool with error handling."""
        try:
            from src.unity_wheel.accelerated_tools.trace_simple import get_trace_turbo
            return get_trace_turbo()
        except ImportError as e:
            logger.warning(f"Failed to import trace_simple: {e}")
            return None
    
    async def _init_code_helper_tool(self) -> Any:
        """Initialize code helper tool with error handling."""
        try:
            from src.unity_wheel.accelerated_tools.python_helpers_turbo import get_code_helper
            return get_code_helper()
        except ImportError as e:
            logger.warning(f"Failed to import python_helpers_turbo: {e}")
            return None
    
    def _setup_fallback_tool(self, tool_name: str):
        """Setup fallback implementation for unavailable tools."""
        fallback_implementations = {
            "ripgrep": self._fallback_ripgrep,
            "dependency_graph": self._fallback_dependency_graph,
            "python_analyzer": self._fallback_python_analyzer,
            "duckdb": self._fallback_duckdb,
            "trace": self._fallback_trace,
            "code_helper": self._fallback_code_helper,
        }
        
        if tool_name in fallback_implementations:
            self.fallback_tools[tool_name] = fallback_implementations[tool_name]
            logger.info(f"Setup fallback implementation for {tool_name}")
    
    async def get_tool(self, tool_name: str) -> Union[ToolWrapper, Any]:
        """Get tool with automatic fallback."""
        if not self.initialized:
            await self.initialize_tools()
        
        if tool_name in self.tools:
            wrapper = self.tools[tool_name]
            if wrapper.circuit_breaker.can_execute():
                return wrapper
        
        # Use fallback if available
        if tool_name in self.fallback_tools:
            logger.debug(f"Using fallback implementation for {tool_name}")
            return self.fallback_tools[tool_name]
        
        raise RuntimeError(f"Tool {tool_name} is unavailable and no fallback exists")
    
    @asynccontextmanager
    async def use_tool(self, tool_name: str, method_name: str = None):
        """Context manager for safe tool usage."""
        try:
            tool = await self.get_tool(tool_name)
            if isinstance(tool, ToolWrapper):
                if method_name:
                    yield lambda *args, **kwargs: tool.execute_with_retry(method_name, *args, **kwargs)
                else:
                    yield tool
            else:
                # Fallback tool
                yield tool
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all tools."""
        if not self.initialized:
            return {"status": "not_initialized", "tools": {}}
        
        tool_health = {}
        for tool_name, wrapper in self.tools.items():
            tool_health[tool_name] = wrapper.get_health_status()
        
        # Add fallback status
        fallback_status = {name: "available" for name in self.fallback_tools.keys()}
        
        overall_health = "healthy"
        if any(status["state"] == "unavailable" for status in tool_health.values()):
            overall_health = "degraded"
        if all(status["state"] == "unavailable" for status in tool_health.values()):
            overall_health = "unhealthy"
        
        return {
            "agent_id": self.agent_id,
            "status": overall_health,
            "initialized": self.initialized,
            "tools": tool_health,
            "fallbacks": fallback_status,
            "hardware_config": self.hw_config
        }
    
    async def _health_check_loop(self):
        """Background health monitoring."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Perform lightweight health checks
                for tool_name, wrapper in self.tools.items():
                    # Check if tool is responding
                    if wrapper.health_metrics.consecutive_failures > 3:
                        logger.warning(f"Tool {tool_name} has {wrapper.health_metrics.consecutive_failures} consecutive failures")
                        
                        # Try to recover
                        try:
                            await self._attempt_tool_recovery(tool_name)
                        except Exception as e:
                            logger.error(f"Failed to recover tool {tool_name}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _attempt_tool_recovery(self, tool_name: str):
        """Attempt to recover a failing tool."""
        logger.info(f"Attempting recovery for tool {tool_name}")
        
        # Remove current tool instance
        if tool_name in self.tools:
            del self.tools[tool_name]
        
        # Re-initialize
        tool_configs = {
            "ripgrep": self._init_ripgrep_tool,
            "dependency_graph": self._init_dependency_graph_tool,
            "python_analyzer": self._init_python_analyzer_tool,
            "duckdb": self._init_duckdb_tool,
            "trace": self._init_trace_tool,
            "code_helper": self._init_code_helper_tool,
        }
        
        if tool_name in tool_configs:
            result = await self._safe_tool_init(tool_name, tool_configs[tool_name])
            if result == ToolState.AVAILABLE:
                logger.info(f"Successfully recovered tool {tool_name}")
            else:
                logger.warning(f"Failed to recover tool {tool_name}")
    
    # Fallback implementations
    def _fallback_ripgrep(self):
        """Basic fallback for ripgrep."""
        class FallbackRipgrep:
            async def search(self, pattern: str, path: str = ".", **kwargs):
                import subprocess
                try:
                    result = subprocess.run(
                        ["rg", pattern, path, "--json"], 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    return [] if result.returncode != 0 else [{"content": line} for line in result.stdout.splitlines()[:10]]
                except:
                    return []
        return FallbackRipgrep()
    
    def _fallback_dependency_graph(self):
        """Basic fallback for dependency graph."""
        class FallbackDependencyGraph:
            async def build_graph(self, **kwargs):
                return {"message": "Fallback dependency graph"}
            async def find_symbol(self, symbol: str):
                return []
        return FallbackDependencyGraph()
    
    def _fallback_python_analyzer(self):
        """Basic fallback for Python analyzer."""
        class FallbackPythonAnalyzer:
            async def analyze_file(self, file_path: str):
                return {"file_path": file_path, "analysis": "fallback"}
            async def analyze_directory(self, directory: str):
                return {"directory": directory, "analysis": "fallback"}
        return FallbackPythonAnalyzer()
    
    def _fallback_duckdb(self):
        """Basic fallback for DuckDB."""
        class FallbackDuckDB:
            async def query(self, sql: str):
                return []
            async def execute(self, sql: str):
                return True
        return FallbackDuckDB()
    
    def _fallback_trace(self):
        """Basic fallback for tracing."""
        class FallbackTrace:
            @asynccontextmanager
            async def trace_span(self, name: str, **kwargs):
                yield {"name": name, "fallback": True}
        return FallbackTrace()
    
    def _fallback_code_helper(self):
        """Basic fallback for code helper."""
        class FallbackCodeHelper:
            async def get_function_info(self, file_path: str, function_name: str):
                return {"function": function_name, "fallback": True}
        return FallbackCodeHelper()
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup tool instances
        for wrapper in self.tools.values():
            if hasattr(wrapper.tool_instance, 'cleanup'):
                try:
                    if asyncio.iscoroutinefunction(wrapper.tool_instance.cleanup):
                        await wrapper.tool_instance.cleanup()
                    else:
                        wrapper.tool_instance.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up {wrapper.tool_name}: {e}")


# Global manager registry for agents
_agent_tool_managers: Dict[str, RobustToolManager] = {}
_manager_lock = threading.Lock()


def get_tool_manager(agent_id: str) -> RobustToolManager:
    """Get or create tool manager for agent."""
    with _manager_lock:
        if agent_id not in _agent_tool_managers:
            _agent_tool_managers[agent_id] = RobustToolManager(agent_id)
        return _agent_tool_managers[agent_id]


async def cleanup_all_managers():
    """Cleanup all agent tool managers."""
    with _manager_lock:
        managers = list(_agent_tool_managers.values())
        _agent_tool_managers.clear()
    
    # Cleanup all managers concurrently
    if managers:
        await asyncio.gather(*[manager.cleanup() for manager in managers], return_exceptions=True)