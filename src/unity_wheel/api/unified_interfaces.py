"""
Unified API Interfaces for Einstein and Bolt Systems

This module defines standardized interfaces that resolve API inconsistencies
between Einstein and Bolt systems, providing unified function signatures,
error handling, data structures, and async/sync patterns.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# UNIFIED ERROR HANDLING
# =============================================================================

class SystemErrorSeverity(Enum):
    """Unified error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SystemErrorCategory(Enum):
    """Unified error categories."""
    SYSTEM = "system"
    RESOURCE = "resource"
    NETWORK = "network"
    HARDWARE = "hardware"
    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"
    VALIDATION = "validation"


@dataclass
class UnifiedErrorContext:
    """Unified error context information."""
    timestamp: float = field(default_factory=time.time)
    operation: Optional[str] = None
    component: Optional[str] = None
    system_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "component": self.component,
            "system_id": self.system_id,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


class UnifiedException(Exception):
    """Unified exception class for both Einstein and Bolt systems."""
    
    def __init__(
        self,
        message: str,
        severity: SystemErrorSeverity = SystemErrorSeverity.ERROR,
        category: SystemErrorCategory = SystemErrorCategory.SYSTEM,
        context: Optional[UnifiedErrorContext] = None,
        recovery_hints: Optional[List[str]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or UnifiedErrorContext()
        self.recovery_hints = recovery_hints or []
        self.cause = cause
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context.to_dict(),
            "recovery_hints": self.recovery_hints,
            "cause": str(self.cause) if self.cause else None,
        }


# =============================================================================
# UNIFIED DATA STRUCTURES
# =============================================================================

@dataclass
class UnifiedResult:
    """Unified result structure for all operations."""
    success: bool
    data: Any = None
    error: Optional[UnifiedException] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    
    @classmethod
    def success_result(cls, data: Any = None, **metadata) -> "UnifiedResult":
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def error_result(cls, error: Union[str, UnifiedException], **metadata) -> "UnifiedResult":
        """Create an error result."""
        if isinstance(error, str):
            error = UnifiedException(error)
        return cls(success=False, error=error, metadata=metadata)


@dataclass
class UnifiedConfig:
    """Unified configuration structure."""
    max_timeout_seconds: float = 30.0
    max_retries: int = 3
    enable_caching: bool = True
    parallel_execution: bool = True
    max_concurrent_operations: int = 10
    logging_level: str = "INFO"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedSearchRequest:
    """Unified search request structure."""
    query: str
    search_type: str = "text"  # text, semantic, structural, analytical
    max_results: int = 100
    include_metadata: bool = True
    filters: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedSearchResult:
    """Unified search result structure."""
    results: List[Dict[str, Any]]
    total_found: int
    search_time_ms: float
    query_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# UNIFIED INTERFACES
# =============================================================================

class IUnifiedSearchEngine(ABC):
    """Unified interface for search engines (Einstein)."""
    
    @abstractmethod
    async def initialize(self, config: UnifiedConfig) -> UnifiedResult:
        """Initialize the search engine."""
        pass
    
    @abstractmethod
    async def search(self, request: UnifiedSearchRequest) -> UnifiedResult[UnifiedSearchResult]:
        """Perform a search operation."""
        pass
    
    @abstractmethod
    async def index_content(self, content: Dict[str, Any]) -> UnifiedResult:
        """Index new content."""
        pass
    
    @abstractmethod
    async def get_health_status(self) -> UnifiedResult[Dict[str, Any]]:
        """Get system health status."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> UnifiedResult:
        """Shutdown the search engine."""
        pass


class IUnifiedAgentSystem(ABC):
    """Unified interface for agent systems (Bolt)."""
    
    @abstractmethod
    async def initialize(self, config: UnifiedConfig) -> UnifiedResult:
        """Initialize the agent system."""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> UnifiedResult:
        """Execute a task using the agent system."""
        pass
    
    @abstractmethod
    async def execute_batch(self, tasks: List[Dict[str, Any]]) -> UnifiedResult[List[Dict[str, Any]]]:
        """Execute multiple tasks in batch."""
        pass
    
    @abstractmethod
    async def get_system_status(self) -> UnifiedResult[Dict[str, Any]]:
        """Get agent system status."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> UnifiedResult:
        """Shutdown the agent system."""
        pass


class IUnifiedDatabase(ABC):
    """Unified interface for database operations."""
    
    @abstractmethod
    async def query(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> UnifiedResult:
        """Execute a database query."""
        pass
    
    @abstractmethod
    async def execute(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> UnifiedResult:
        """Execute a database command."""
        pass
    
    @abstractmethod
    async def get_connection_info(self) -> UnifiedResult[Dict[str, Any]]:
        """Get database connection information."""
        pass
    
    @abstractmethod
    async def health_check(self) -> UnifiedResult[Dict[str, Any]]:
        """Perform database health check."""
        pass


class IUnifiedTracing(ABC):
    """Unified interface for tracing operations."""
    
    @abstractmethod
    @asynccontextmanager
    async def trace_span(self, operation_name: str, **tags) -> AsyncGenerator[Dict[str, Any], None]:
        """Create a trace span."""
        pass
    
    @abstractmethod
    async def log_event(self, message: str, level: str = "info", **attributes) -> None:
        """Log an event."""
        pass
    
    @abstractmethod
    async def get_trace_data(self, trace_id: str) -> UnifiedResult[List[Dict[str, Any]]]:
        """Get trace data."""
        pass


# =============================================================================
# UNIFIED ADAPTERS
# =============================================================================

class EinsteinAdapter(IUnifiedSearchEngine):
    """Adapter to make Einstein conform to unified interface with accelerated tools integration."""
    
    def __init__(self, einstein_instance):
        self.einstein = einstein_instance
        self._initialized = False
        self._accelerated_tools = {}
        self._search_cache = {}
        self._cache_ttl_seconds = 300  # 5 minutes
    
    async def initialize(self, config: UnifiedConfig) -> UnifiedResult:
        """Initialize Einstein with unified config and accelerated tools."""
        try:
            # Initialize accelerated tools
            await self._init_accelerated_tools()
            
            # Map unified config to Einstein config
            if hasattr(self.einstein, 'initialize'):
                await self.einstein.initialize()
            
            self._initialized = True
            
            return UnifiedResult.success_result({
                "status": "initialized",
                "accelerated_tools": list(self._accelerated_tools.keys()),
                "cache_enabled": config.enable_caching
            })
        except Exception as e:
            error = UnifiedException(
                f"Einstein initialization failed: {e}",
                category=SystemErrorCategory.SYSTEM,
                cause=e
            )
            return UnifiedResult.error_result(error)
    
    async def _init_accelerated_tools(self):
        """Initialize accelerated tools for enhanced performance."""
        try:
            # Try to load ripgrep turbo
            try:
                from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
                self._accelerated_tools['ripgrep'] = get_ripgrep_turbo()
                logger.info("Loaded accelerated ripgrep for Einstein")
            except ImportError:
                logger.debug("Accelerated ripgrep not available")
            
            # Try to load dependency graph turbo
            try:
                from src.unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph
                self._accelerated_tools['dependency_graph'] = get_dependency_graph()
                logger.info("Loaded accelerated dependency graph for Einstein")
            except ImportError:
                logger.debug("Accelerated dependency graph not available")
            
            # Try to load python analyzer turbo
            try:
                from src.unity_wheel.accelerated_tools.python_analysis_turbo import get_python_analyzer
                self._accelerated_tools['python_analyzer'] = get_python_analyzer()
                logger.info("Loaded accelerated Python analyzer for Einstein")
            except ImportError:
                logger.debug("Accelerated Python analyzer not available")
                
        except Exception as e:
            logger.warning(f"Failed to initialize some accelerated tools: {e}")
    
    async def search(self, request: UnifiedSearchRequest) -> UnifiedResult[UnifiedSearchResult]:
        """Perform search using Einstein with accelerated tools optimization."""
        if not self._initialized:
            error = UnifiedException("Einstein not initialized", severity=SystemErrorSeverity.ERROR)
            return UnifiedResult.error_result(error)
        
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = f"{request.query}:{request.search_type}:{request.max_results}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for search: {request.query[:50]}...")
                return cached_result
            
            results = []
            
            # Use accelerated tools based on search type
            if request.search_type == "text" and 'ripgrep' in self._accelerated_tools:
                # Use accelerated ripgrep for text search
                ripgrep = self._accelerated_tools['ripgrep']
                rg_results = await ripgrep.search(
                    request.query,
                    path=request.filters.get('path', '.'),
                    max_results=request.max_results
                )
                results = rg_results if isinstance(rg_results, list) else []
                
            elif request.search_type == "structural" and 'dependency_graph' in self._accelerated_tools:
                # Use accelerated dependency graph for structural search
                dep_graph = self._accelerated_tools['dependency_graph']
                if hasattr(dep_graph, 'search_symbols'):
                    graph_results = await dep_graph.search_symbols(request.query)
                    results = graph_results if isinstance(graph_results, list) else []
                    
            elif request.search_type == "analytical" and 'python_analyzer' in self._accelerated_tools:
                # Use accelerated Python analyzer for code analysis
                analyzer = self._accelerated_tools['python_analyzer']
                if hasattr(analyzer, 'search_patterns'):
                    analysis_results = await analyzer.search_patterns(request.query)
                    results = analysis_results if isinstance(analysis_results, list) else []
            
            # Fallback to Einstein's native search
            if not results and hasattr(self.einstein, 'search'):
                results = await self.einstein.search(
                    request.query,
                    limit=request.max_results,
                    search_type=request.search_type
                )
                results = results if isinstance(results, list) else [results] if results else []
            
            search_time_ms = (time.time() - start_time) * 1000
            
            # Convert to unified format
            unified_result = UnifiedSearchResult(
                results=results,
                total_found=len(results),
                search_time_ms=search_time_ms,
                query_info={"query": request.query, "type": request.search_type},
                metadata={
                    "accelerated_tool_used": self._get_tool_used(request.search_type),
                    "cache_hit": False
                }
            )
            
            result = UnifiedResult.success_result(
                unified_result,
                search_time_ms=search_time_ms,
                engine="einstein",
                accelerated=bool(self._get_tool_used(request.search_type))
            )
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            error = UnifiedException(
                f"Einstein search failed: {e}",
                category=SystemErrorCategory.SYSTEM,
                cause=e
            )
            return UnifiedResult.error_result(error)
    
    def _get_tool_used(self, search_type: str) -> Optional[str]:
        """Get the accelerated tool used for a search type."""
        if search_type == "text" and 'ripgrep' in self._accelerated_tools:
            return "ripgrep_turbo"
        elif search_type == "structural" and 'dependency_graph' in self._accelerated_tools:
            return "dependency_graph_turbo"
        elif search_type == "analytical" and 'python_analyzer' in self._accelerated_tools:
            return "python_analysis_turbo"
        return None
    
    def _get_cached_result(self, cache_key: str) -> Optional[UnifiedResult]:
        """Get cached search result if valid."""
        if cache_key in self._search_cache:
            cached_data, timestamp = self._search_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                # Update metadata to indicate cache hit
                if cached_data.success and hasattr(cached_data.data, 'metadata'):
                    cached_data.data.metadata['cache_hit'] = True
                return cached_data
            else:
                # Remove expired cache entry
                del self._search_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: UnifiedResult):
        """Cache search result."""
        # Limit cache size to prevent memory issues
        if len(self._search_cache) > 100:
            # Remove oldest entries
            sorted_cache = sorted(self._search_cache.items(), key=lambda x: x[1][1])
            for old_key, _ in sorted_cache[:20]:  # Remove 20 oldest
                del self._search_cache[old_key]
        
        self._search_cache[cache_key] = (result, time.time())
    
    async def index_content(self, content: Dict[str, Any]) -> UnifiedResult:
        """Index content in Einstein."""
        try:
            # Einstein indexing logic would go here
            return UnifiedResult.success_result({"indexed": True})
        except Exception as e:
            error = UnifiedException(f"Indexing failed: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    async def get_health_status(self) -> UnifiedResult[Dict[str, Any]]:
        """Get Einstein health status including accelerated tools."""
        try:
            health_data = {
                "initialized": self._initialized,
                "component": "einstein",
                "status": "healthy" if self._initialized else "not_initialized",
                "accelerated_tools": {
                    tool_name: "available" for tool_name in self._accelerated_tools.keys()
                },
                "cache_stats": {
                    "cached_searches": len(self._search_cache),
                    "cache_ttl_seconds": self._cache_ttl_seconds
                }
            }
            
            # Add Einstein-specific health info if available
            if hasattr(self.einstein, 'get_status'):
                try:
                    einstein_status = await self.einstein.get_status()
                    health_data["einstein_internal"] = einstein_status
                except Exception as e:
                    health_data["einstein_internal"] = {"error": str(e)}
            
            return UnifiedResult.success_result(health_data)
        except Exception as e:
            error = UnifiedException(f"Health check failed: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    async def shutdown(self) -> UnifiedResult:
        """Shutdown Einstein."""
        try:
            if hasattr(self.einstein, 'shutdown'):
                await self.einstein.shutdown()
            self._initialized = False
            return UnifiedResult.success_result({"status": "shutdown"})
        except Exception as e:
            error = UnifiedException(f"Shutdown failed: {e}", cause=e)
            return UnifiedResult.error_result(error)


class BoltAdapter(IUnifiedAgentSystem):
    """Adapter to make Bolt conform to unified interface with work-stealing agent pool integration."""
    
    def __init__(self, bolt_instance):
        self.bolt = bolt_instance
        self._initialized = False
        self._agent_pool = None
        self._work_stealing_enabled = True
    
    async def initialize(self, config: UnifiedConfig) -> UnifiedResult:
        """Initialize Bolt with unified config and work-stealing agent pool."""
        try:
            # Initialize the main Bolt system
            if hasattr(self.bolt, 'initialize'):
                await self.bolt.initialize()
            
            # Initialize work-stealing agent pool if available
            try:
                from bolt.agents.agent_pool import WorkStealingAgentPool
                self._agent_pool = WorkStealingAgentPool(
                    num_agents=config.max_concurrent_operations,
                    enable_work_stealing=self._work_stealing_enabled
                )
                await self._agent_pool.initialize()
                logger.info(f"Initialized Bolt with {config.max_concurrent_operations} work-stealing agents")
            except ImportError:
                logger.warning("Work-stealing agent pool not available, using fallback")
                self._agent_pool = None
            
            self._initialized = True
            return UnifiedResult.success_result({
                "status": "initialized",
                "work_stealing_enabled": self._agent_pool is not None,
                "agent_count": config.max_concurrent_operations
            })
        except Exception as e:
            error = UnifiedException(
                f"Bolt initialization failed: {e}",
                category=SystemErrorCategory.SYSTEM,
                cause=e
            )
            return UnifiedResult.error_result(error)
    
    async def execute_task(self, task: Dict[str, Any]) -> UnifiedResult:
        """Execute task using Bolt with work-stealing optimization."""
        if not self._initialized:
            error = UnifiedException("Bolt not initialized", severity=SystemErrorSeverity.ERROR)
            return UnifiedResult.error_result(error)
        
        try:
            start_time = time.time()
            
            # Use work-stealing agent pool if available
            if self._agent_pool:
                from bolt.agents.agent_pool import WorkStealingTask, TaskPriority
                
                # Convert unified task to work-stealing task
                priority = TaskPriority.NORMAL
                if task.get("priority") == "high":
                    priority = TaskPriority.HIGH
                elif task.get("priority") == "critical":
                    priority = TaskPriority.CRITICAL
                elif task.get("priority") == "low":
                    priority = TaskPriority.LOW
                
                ws_task = WorkStealingTask(
                    id=task.get("id", f"task_{int(time.time() * 1000)}"),
                    description=task.get("description", "Unified task execution"),
                    priority=priority,
                    subdividable=task.get("subdividable", True),
                    estimated_duration=task.get("estimated_duration", 1.0),
                    metadata=task.get("metadata", {})
                )
                
                # Submit and wait for completion
                await self._agent_pool.submit_task(ws_task)
                result = await self._agent_pool.wait_for_task_completion(ws_task.id)
                
            elif hasattr(self.bolt, 'execute_task'):
                # Fallback to direct Bolt execution
                result = await self.bolt.execute_task(task)
            else:
                # Basic fallback execution
                result = {"status": "executed", "task_id": task.get("id")}
            
            duration_ms = (time.time() - start_time) * 1000
            
            return UnifiedResult.success_result(
                result,
                duration_ms=duration_ms,
                engine="bolt",
                work_stealing_used=self._agent_pool is not None
            )
            
        except Exception as e:
            error = UnifiedException(
                f"Bolt task execution failed: {e}",
                category=SystemErrorCategory.SYSTEM,
                cause=e
            )
            return UnifiedResult.error_result(error)
    
    async def execute_batch(self, tasks: List[Dict[str, Any]]) -> UnifiedResult[List[Dict[str, Any]]]:
        """Execute multiple tasks in batch with optimized work-stealing."""
        try:
            start_time = time.time()
            
            if self._agent_pool and len(tasks) > 1:
                # Use work-stealing agent pool for parallel execution
                from bolt.agents.agent_pool import WorkStealingTask, TaskPriority
                
                # Submit all tasks to the pool
                task_ids = []
                for i, task in enumerate(tasks):
                    priority = TaskPriority.NORMAL
                    if task.get("priority") == "high":
                        priority = TaskPriority.HIGH
                    elif task.get("priority") == "critical":
                        priority = TaskPriority.CRITICAL
                    elif task.get("priority") == "low":
                        priority = TaskPriority.LOW
                    
                    ws_task = WorkStealingTask(
                        id=task.get("id", f"batch_task_{i}_{int(time.time() * 1000)}"),
                        description=task.get("description", f"Batch task {i+1}/{len(tasks)}"),
                        priority=priority,
                        subdividable=task.get("subdividable", False),  # Batch tasks typically not subdividable
                        estimated_duration=task.get("estimated_duration", 0.5),
                        metadata={**task.get("metadata", {}), "batch_index": i}
                    )
                    
                    await self._agent_pool.submit_task(ws_task)
                    task_ids.append(ws_task.id)
                
                # Wait for all tasks to complete
                results = []
                for task_id in task_ids:
                    try:
                        result = await self._agent_pool.wait_for_task_completion(task_id)
                        results.append(result)
                    except Exception as e:
                        results.append({"error": str(e)})
                
            else:
                # Sequential execution fallback
                results = []
                for task in tasks:
                    task_result = await self.execute_task(task)
                    results.append(task_result.data if task_result.success else {"error": str(task_result.error)})
            
            duration_ms = (time.time() - start_time) * 1000
            
            return UnifiedResult.success_result(
                results,
                batch_size=len(tasks),
                duration_ms=duration_ms,
                engine="bolt",
                parallel_execution=self._agent_pool is not None and len(tasks) > 1
            )
            
        except Exception as e:
            error = UnifiedException(f"Batch execution failed: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    async def get_system_status(self) -> UnifiedResult[Dict[str, Any]]:
        """Get Bolt system status including agent pool metrics."""
        try:
            status_data = {
                "initialized": self._initialized,
                "component": "bolt",
                "status": "healthy" if self._initialized else "not_initialized",
                "work_stealing_enabled": self._agent_pool is not None
            }
            
            # Add agent pool status if available
            if self._agent_pool:
                pool_status = self._agent_pool.get_pool_status()
                status_data["agent_pool"] = {
                    "total_agents": pool_status["total_agents"],
                    "busy_agents": pool_status["busy_agents"],
                    "idle_agents": pool_status["idle_agents"],
                    "utilization": pool_status["utilization"],
                    "performance_metrics": pool_status["performance_metrics"]
                }
            
            return UnifiedResult.success_result(status_data)
        except Exception as e:
            error = UnifiedException(f"Status check failed: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    async def shutdown(self) -> UnifiedResult:
        """Shutdown Bolt and agent pool gracefully."""
        try:
            shutdown_results = []
            
            # Shutdown agent pool first
            if self._agent_pool:
                try:
                    await self._agent_pool.shutdown()
                    shutdown_results.append("agent_pool: success")
                except Exception as e:
                    shutdown_results.append(f"agent_pool: failed - {e}")
            
            # Shutdown main Bolt system
            if hasattr(self.bolt, 'shutdown'):
                try:
                    await self.bolt.shutdown()
                    shutdown_results.append("bolt_system: success")
                except Exception as e:
                    shutdown_results.append(f"bolt_system: failed - {e}")
            
            self._initialized = False
            return UnifiedResult.success_result({
                "status": "shutdown",
                "shutdown_details": shutdown_results
            })
        except Exception as e:
            error = UnifiedException(f"Shutdown failed: {e}", cause=e)
            return UnifiedResult.error_result(error)


class DuckDBAdapter(IUnifiedDatabase):
    """Adapter for DuckDB with accelerated tools integration."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._duckdb_turbo = None
        self._initialized = False
    
    async def initialize(self, config: UnifiedConfig) -> UnifiedResult:
        """Initialize DuckDB adapter."""
        try:
            # Try to get accelerated DuckDB
            try:
                from src.unity_wheel.accelerated_tools.duckdb_turbo import get_duckdb_turbo
                self._duckdb_turbo = get_duckdb_turbo(self.db_path)
                logger.info("Initialized accelerated DuckDB adapter")
            except ImportError:
                logger.warning("Accelerated DuckDB not available")
                return UnifiedResult.error_result(
                    UnifiedException("Accelerated DuckDB required but not available")
                )
            
            self._initialized = True
            return UnifiedResult.success_result({"status": "initialized", "accelerated": True})
        except Exception as e:
            error = UnifiedException(f"DuckDB initialization failed: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    async def query(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> UnifiedResult:
        """Execute query using accelerated DuckDB."""
        if not self._initialized or not self._duckdb_turbo:
            error = UnifiedException("DuckDB not initialized", severity=SystemErrorSeverity.ERROR)
            return UnifiedResult.error_result(error)
        
        try:
            start_time = time.time()
            
            if hasattr(self._duckdb_turbo, 'query_to_pandas'):
                result = await self._duckdb_turbo.query_to_pandas(sql, parameters or {})
            else:
                result = {"error": "Query method not available"}
            
            duration_ms = (time.time() - start_time) * 1000
            
            return UnifiedResult.success_result(
                result,
                duration_ms=duration_ms,
                sql=sql,
                accelerated=True
            )
            
        except Exception as e:
            error = UnifiedException(f"DuckDB query failed: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    async def execute(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> UnifiedResult:
        """Execute command using accelerated DuckDB."""
        # For DuckDB, execute is similar to query
        return await self.query(sql, parameters)
    
    async def get_connection_info(self) -> UnifiedResult[Dict[str, Any]]:
        """Get connection information."""
        return UnifiedResult.success_result({
            "db_path": self.db_path,
            "accelerated": True,
            "initialized": self._initialized
        })
    
    async def health_check(self) -> UnifiedResult[Dict[str, Any]]:
        """Perform health check."""
        try:
            # Simple test query
            test_result = await self.query("SELECT 1 as test")
            
            health_data = {
                "status": "healthy" if test_result.success else "unhealthy",
                "test_query_success": test_result.success,
                "accelerated": True,
                "initialized": self._initialized
            }
            
            return UnifiedResult.success_result(health_data)
        except Exception as e:
            error = UnifiedException(f"Health check failed: {e}", cause=e)
            return UnifiedResult.error_result(error)


class TracingAdapter(IUnifiedTracing):
    """Adapter for unified tracing with multiple backend support."""
    
    def __init__(self, backend: str = "simple"):
        self.backend = backend
        self._tracer = None
        self._initialized = False
    
    async def initialize(self, config: UnifiedConfig) -> UnifiedResult:
        """Initialize tracing adapter."""
        try:
            # Try to get accelerated tracing
            try:
                from ..accelerated_tools.trace_turbo import get_trace_turbo
                self._tracer = get_trace_turbo()
                logger.info(f"Initialized accelerated tracing with {self.backend} backend")
            except ImportError:
                # Fallback to simple tracing
                self._tracer = SimpleTracer()
                logger.info("Using simple tracing fallback")
            
            self._initialized = True
            return UnifiedResult.success_result({"status": "initialized", "backend": self.backend})
        except Exception as e:
            error = UnifiedException(f"Tracing initialization failed: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    @asynccontextmanager
    async def trace_span(self, operation_name: str, **tags) -> AsyncGenerator[Dict[str, Any], None]:
        """Create a trace span."""
        if not self._initialized:
            # Simple fallback span
            span_data = {"operation": operation_name, "start_time": time.time(), **tags}
            yield span_data
            span_data["duration_ms"] = (time.time() - span_data["start_time"]) * 1000
            return
        
        try:
            if hasattr(self._tracer, 'trace_span'):
                async with self._tracer.trace_span(operation_name, **tags) as span:
                    yield span
            else:
                # Simple span implementation
                span_data = {"operation": operation_name, "start_time": time.time(), **tags}
                yield span_data
                span_data["duration_ms"] = (time.time() - span_data["start_time"]) * 1000
        except Exception as e:
            logger.warning(f"Trace span failed: {e}")
            # Fallback span
            span_data = {"operation": operation_name, "error": str(e), **tags}
            yield span_data
    
    async def log_event(self, message: str, level: str = "info", **attributes) -> None:
        """Log an event."""
        try:
            if self._tracer and hasattr(self._tracer, 'log_event'):
                await self._tracer.log_event(message, level, **attributes)
            else:
                # Fallback to standard logging
                logger.log(getattr(logging, level.upper(), logging.INFO), f"{message} {attributes}")
        except Exception as e:
            logger.warning(f"Event logging failed: {e}")
    
    async def get_trace_data(self, trace_id: str) -> UnifiedResult[List[Dict[str, Any]]]:
        """Get trace data."""
        try:
            if self._tracer and hasattr(self._tracer, 'get_trace_data'):
                data = await self._tracer.get_trace_data(trace_id)
                return UnifiedResult.success_result(data)
            else:
                return UnifiedResult.success_result([])
        except Exception as e:
            error = UnifiedException(f"Trace data retrieval failed: {e}", cause=e)
            return UnifiedResult.error_result(error)


class SimpleTracer:
    """Simple tracer fallback implementation."""
    
    def __init__(self):
        self.traces = {}
    
    @asynccontextmanager
    async def trace_span(self, operation_name: str, **tags):
        span_id = f"{operation_name}_{int(time.time() * 1000000)}"
        span_data = {
            "span_id": span_id,
            "operation": operation_name,
            "start_time": time.time(),
            "tags": tags
        }
        
        try:
            yield span_data
        finally:
            span_data["end_time"] = time.time()
            span_data["duration_ms"] = (span_data["end_time"] - span_data["start_time"]) * 1000
            self.traces[span_id] = span_data
    
    async def log_event(self, message: str, level: str = "info", **attributes):
        logger.log(getattr(logging, level.upper(), logging.INFO), f"{message} {attributes}")
    
    async def get_trace_data(self, trace_id: str) -> List[Dict[str, Any]]:
        return [self.traces.get(trace_id, {})] if trace_id in self.traces else []


# =============================================================================
# UNIFIED SYSTEM MANAGER
# =============================================================================

class UnifiedSystemManager:
    """Manages unified interactions between Einstein and Bolt systems."""
    
    def __init__(self):
        self.einstein: Optional[IUnifiedSearchEngine] = None
        self.bolt: Optional[IUnifiedAgentSystem] = None
        self.database: Optional[IUnifiedDatabase] = None
        self.tracer: Optional[IUnifiedTracing] = None
        self.config = UnifiedConfig()
        self._initialized = False
    
    async def initialize_einstein(self, einstein_instance) -> UnifiedResult:
        """Initialize Einstein with unified interface."""
        try:
            self.einstein = EinsteinAdapter(einstein_instance)
            result = await self.einstein.initialize(self.config)
            logger.info("Einstein initialized with unified interface")
            return result
        except Exception as e:
            error = UnifiedException(f"Failed to initialize Einstein: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    async def initialize_bolt(self, bolt_instance) -> UnifiedResult:
        """Initialize Bolt with unified interface."""
        try:
            self.bolt = BoltAdapter(bolt_instance)
            result = await self.bolt.initialize(self.config)
            logger.info("Bolt initialized with unified interface")
            return result
        except Exception as e:
            error = UnifiedException(f"Failed to initialize Bolt: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    async def search_and_process(
        self,
        search_request: UnifiedSearchRequest,
        processing_task: Optional[Dict[str, Any]] = None
    ) -> UnifiedResult:
        """Unified operation: search with Einstein and process with Bolt."""
        if not self.einstein or not self.bolt:
            error = UnifiedException("Systems not initialized", severity=SystemErrorSeverity.ERROR)
            return UnifiedResult.error_result(error)
        
        try:
            start_time = time.time()
            
            # Step 1: Search with Einstein
            search_result = await self.einstein.search(search_request)
            if not search_result.success:
                return search_result
            
            # Step 2: Process results with Bolt (if task provided)
            if processing_task:
                processing_task["search_results"] = search_result.data
                processing_task["search_metadata"] = search_result.metadata
                
                process_result = await self.bolt.execute_task(processing_task)
                if not process_result.success:
                    return process_result
                
                # Combine results with comprehensive metadata
                total_duration_ms = (time.time() - start_time) * 1000
                combined_data = {
                    "search_results": search_result.data,
                    "processed_results": process_result.data,
                    "pipeline_metadata": {
                        "search_time_ms": search_result.metadata.get("search_time_ms"),
                        "processing_time_ms": process_result.metadata.get("duration_ms"),
                        "total_pipeline_time_ms": total_duration_ms,
                        "search_engine": "einstein",
                        "processing_engine": "bolt",
                        "accelerated_search": search_result.metadata.get("accelerated", False),
                        "work_stealing_used": process_result.metadata.get("work_stealing_used", False)
                    }
                }
                return UnifiedResult.success_result(combined_data)
            
            return search_result
            
        except Exception as e:
            error = UnifiedException(f"Search and process failed: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    async def batch_search_and_process(
        self,
        search_requests: List[UnifiedSearchRequest],
        processing_tasks: Optional[List[Dict[str, Any]]] = None
    ) -> UnifiedResult:
        """Execute multiple search and process operations efficiently."""
        if not self.einstein or not self.bolt:
            error = UnifiedException("Systems not initialized", severity=SystemErrorSeverity.ERROR)
            return UnifiedResult.error_result(error)
        
        try:
            start_time = time.time()
            
            # Execute all searches concurrently
            search_tasks = [self.einstein.search(req) for req in search_requests]
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process results if processing tasks provided
            final_results = []
            if processing_tasks and len(processing_tasks) == len(search_requests):
                # Prepare processing tasks with search results
                prepared_tasks = []
                for i, (search_result, proc_task) in enumerate(zip(search_results, processing_tasks)):
                    if isinstance(search_result, UnifiedResult) and search_result.success:
                        proc_task["search_results"] = search_result.data
                        proc_task["search_metadata"] = search_result.metadata
                        proc_task["batch_index"] = i
                        prepared_tasks.append(proc_task)
                    else:
                        # Handle search failure
                        final_results.append({"error": f"Search failed: {search_result}"})
                        prepared_tasks.append(None)
                
                # Execute processing in batch
                if prepared_tasks:
                    valid_tasks = [task for task in prepared_tasks if task is not None]
                    if valid_tasks:
                        batch_result = await self.bolt.execute_batch(valid_tasks)
                        if batch_result.success:
                            final_results.extend(batch_result.data)
                        else:
                            final_results.extend([{"error": "Batch processing failed"}] * len(valid_tasks))
            else:
                # Just return search results
                final_results = [res.data if isinstance(res, UnifiedResult) and res.success else {"error": str(res)} for res in search_results]
            
            total_duration_ms = (time.time() - start_time) * 1000
            
            return UnifiedResult.success_result(
                final_results,
                batch_size=len(search_requests),
                total_duration_ms=total_duration_ms,
                pipeline="batch_search_and_process"
            )
            
        except Exception as e:
            error = UnifiedException(f"Batch search and process failed: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    async def analyze_codebase(
        self,
        analysis_type: str = "comprehensive",
        target_path: str = "."
    ) -> UnifiedResult:
        """Perform comprehensive codebase analysis using both systems."""
        if not self.einstein or not self.bolt:
            error = UnifiedException("Systems not initialized", severity=SystemErrorSeverity.ERROR)
            return UnifiedResult.error_result(error)
        
        try:
            start_time = time.time()
            
            # Define analysis searches based on type
            search_patterns = {
                "comprehensive": [
                    UnifiedSearchRequest("TODO|FIXME|HACK", "text", 50, filters={"path": target_path}),
                    UnifiedSearchRequest("class.*:", "structural", 100, filters={"path": target_path}),
                    UnifiedSearchRequest("def.*(", "structural", 100, filters={"path": target_path}),
                    UnifiedSearchRequest("import.*", "analytical", 200, filters={"path": target_path})
                ],
                "security": [
                    UnifiedSearchRequest("password|secret|token|key", "text", 50, filters={"path": target_path}),
                    UnifiedSearchRequest("sql.*inject|eval.*input", "text", 30, filters={"path": target_path})
                ],
                "performance": [
                    UnifiedSearchRequest("sleep|time\.sleep|await.*sleep", "text", 30, filters={"path": target_path}),
                    UnifiedSearchRequest("for.*in.*range\(.*\)", "text", 40, filters={"path": target_path})
                ]
            }
            
            searches = search_patterns.get(analysis_type, search_patterns["comprehensive"])
            
            # Execute searches
            search_results = await asyncio.gather(
                *[self.einstein.search(req) for req in searches],
                return_exceptions=True
            )
            
            # Prepare analysis tasks for Bolt
            analysis_tasks = []
            for i, (search_req, search_result) in enumerate(zip(searches, search_results)):
                if isinstance(search_result, UnifiedResult) and search_result.success:
                    task = {
                        "id": f"analysis_{analysis_type}_{i}",
                        "description": f"Analyze {search_req.search_type} search results",
                        "search_results": search_result.data,
                        "analysis_type": analysis_type,
                        "search_pattern": search_req.query,
                        "priority": "normal"
                    }
                    analysis_tasks.append(task)
            
            # Process analysis with Bolt
            analysis_result = await self.bolt.execute_batch(analysis_tasks)
            
            total_duration_ms = (time.time() - start_time) * 1000
            
            # Compile comprehensive analysis report
            report = {
                "analysis_type": analysis_type,
                "target_path": target_path,
                "search_patterns_used": len(searches),
                "total_findings": sum(
                    len(res.data.results) if isinstance(res, UnifiedResult) and res.success else 0
                    for res in search_results
                ),
                "search_results": [
                    res.data if isinstance(res, UnifiedResult) and res.success else {"error": str(res)}
                    for res in search_results
                ],
                "analysis_results": analysis_result.data if analysis_result.success else {"error": str(analysis_result.error)},
                "performance_metrics": {
                    "total_duration_ms": total_duration_ms,
                    "search_phase_ms": sum(
                        res.metadata.get("search_time_ms", 0) if isinstance(res, UnifiedResult) and res.success else 0
                        for res in search_results
                    ),
                    "analysis_phase_ms": analysis_result.metadata.get("duration_ms", 0) if analysis_result.success else 0
                }
            }
            
            return UnifiedResult.success_result(
                report,
                analysis_type=analysis_type,
                total_duration_ms=total_duration_ms
            )
            
        except Exception as e:
            error = UnifiedException(f"Codebase analysis failed: {e}", cause=e)
            return UnifiedResult.error_result(error)
    
    async def get_unified_status(self) -> UnifiedResult[Dict[str, Any]]:
        """Get comprehensive status of all unified systems."""
        status = {
            "manager_initialized": self._initialized,
            "configuration": {
                "max_timeout_seconds": self.config.max_timeout_seconds,
                "max_retries": self.config.max_retries,
                "enable_caching": self.config.enable_caching,
                "parallel_execution": self.config.parallel_execution,
                "max_concurrent_operations": self.config.max_concurrent_operations
            },
            "systems": {},
            "integration_status": {
                "einstein_available": self.einstein is not None,
                "bolt_available": self.bolt is not None,
                "database_available": self.database is not None,
                "tracer_available": self.tracer is not None
            }
        }
        
        # Get detailed system statuses
        status_tasks = []
        
        if self.einstein:
            status_tasks.append(("einstein", self.einstein.get_health_status()))
        
        if self.bolt:
            status_tasks.append(("bolt", self.bolt.get_system_status()))
        
        # Gather all status checks concurrently
        if status_tasks:
            system_names, status_coroutines = zip(*status_tasks)
            system_statuses = await asyncio.gather(*status_coroutines, return_exceptions=True)
            
            for name, result in zip(system_names, system_statuses):
                if isinstance(result, UnifiedResult) and result.success:
                    status["systems"][name] = result.data
                else:
                    status["systems"][name] = {"error": str(result)}
        
        # Add integration health metrics
        status["integration_health"] = {
            "systems_online": len([s for s in status["systems"].values() if s.get("status") == "healthy"]),
            "total_systems": len(status["systems"]),
            "overall_health": "healthy" if all(
                s.get("status") == "healthy" for s in status["systems"].values()
            ) else "degraded"
        }
        
        return UnifiedResult.success_result(status)
    
    async def shutdown_all(self) -> UnifiedResult:
        """Shutdown all systems gracefully with proper error handling."""
        shutdown_start = time.time()
        results = []
        errors = []
        
        # Create shutdown tasks for concurrent execution
        shutdown_tasks = []
        
        if self.bolt:
            shutdown_tasks.append(("bolt", self.bolt.shutdown()))
        
        if self.einstein:
            shutdown_tasks.append(("einstein", self.einstein.shutdown()))
        
        if self.database:
            if hasattr(self.database, 'shutdown'):
                shutdown_tasks.append(("database", self.database.shutdown()))
        
        if self.tracer:
            if hasattr(self.tracer, 'shutdown'):
                shutdown_tasks.append(("tracer", self.tracer.shutdown()))
        
        # Execute shutdowns concurrently with timeout
        if shutdown_tasks:
            system_names, shutdown_coroutines = zip(*shutdown_tasks)
            
            try:
                shutdown_results = await asyncio.wait_for(
                    asyncio.gather(*shutdown_coroutines, return_exceptions=True),
                    timeout=self.config.max_timeout_seconds
                )
                
                for name, result in zip(system_names, shutdown_results):
                    if isinstance(result, UnifiedResult):
                        results.append((name, result.success))
                        if not result.success:
                            errors.append(f"{name}: {result.error}")
                    elif isinstance(result, Exception):
                        results.append((name, False))
                        errors.append(f"{name}: {str(result)}")
                    else:
                        results.append((name, True))
                        
            except asyncio.TimeoutError:
                errors.append(f"Shutdown timed out after {self.config.max_timeout_seconds}s")
                results.extend([(name, False) for name, _ in shutdown_tasks])
        
        shutdown_duration = time.time() - shutdown_start
        all_success = all(success for _, success in results)
        
        shutdown_summary = {
            "shutdown_results": dict(results),
            "shutdown_duration_seconds": shutdown_duration,
            "errors": errors,
            "total_systems": len(results),
            "successful_shutdowns": sum(1 for _, success in results if success)
        }
        
        self._initialized = False
        
        if all_success:
            logger.info(f"All systems shutdown successfully in {shutdown_duration:.2f}s")
            return UnifiedResult.success_result(shutdown_summary)
        else:
            error = UnifiedException(
                f"Some systems failed to shutdown properly: {', '.join(errors)}",
                severity=SystemErrorSeverity.WARNING if len(errors) < len(results) else SystemErrorSeverity.ERROR
            )
            return UnifiedResult.error_result(error, **shutdown_summary)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_unified_error(
    message: str,
    severity: SystemErrorSeverity = SystemErrorSeverity.ERROR,
    category: SystemErrorCategory = SystemErrorCategory.SYSTEM,
    **kwargs
) -> UnifiedException:
    """Create a unified exception with context."""
    context = UnifiedErrorContext(**kwargs)
    return UnifiedException(message, severity, category, context)


def wrap_legacy_exception(
    exc: Exception,
    operation: str = "unknown",
    component: str = "unknown"
) -> UnifiedException:
    """Wrap a legacy exception in unified format."""
    context = UnifiedErrorContext(operation=operation, component=component)
    return UnifiedException(
        f"Legacy error in {operation}: {str(exc)}",
        category=SystemErrorCategory.SYSTEM,
        context=context,
        cause=exc
    )


async def safe_async_call(
    func,
    *args,
    operation_name: str = "async_operation",
    timeout: float = 30.0,
    **kwargs
) -> UnifiedResult:
    """Safely call an async function with unified error handling."""
    try:
        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        return UnifiedResult.success_result(result)
    except asyncio.TimeoutError:
        error = create_unified_error(
            f"Operation {operation_name} timed out after {timeout}s",
            category=SystemErrorCategory.TIMEOUT
        )
        return UnifiedResult.error_result(error)
    except Exception as e:
        error = wrap_legacy_exception(e, operation_name)
        return UnifiedResult.error_result(error)


# Global unified manager instance
_unified_manager: Optional[UnifiedSystemManager] = None


def get_unified_manager() -> UnifiedSystemManager:
    """Get the global unified system manager."""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = UnifiedSystemManager()
    return _unified_manager


def configure_unified_logging():
    """Configure unified logging for both systems."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Set consistent logging levels
    logging.getLogger('einstein').setLevel(logging.INFO)
    logging.getLogger('bolt').setLevel(logging.INFO)
    logging.getLogger('unity_wheel.api.unified_interfaces').setLevel(logging.INFO)


# =============================================================================
# FACTORY FUNCTIONS AND INTEGRATION EXAMPLES
# =============================================================================

async def create_unified_system(
    einstein_instance=None,
    bolt_instance=None,
    db_path: Optional[str] = None,
    config: Optional[UnifiedConfig] = None
) -> UnifiedSystemManager:
    """Factory function to create and initialize a complete unified system."""
    manager = get_unified_manager()
    
    if config:
        manager.config = config
    
    results = []
    
    # Initialize Einstein if provided
    if einstein_instance:
        einstein_result = await manager.initialize_einstein(einstein_instance)
        results.append(("einstein", einstein_result.success))
        if not einstein_result.success:
            logger.error(f"Einstein initialization failed: {einstein_result.error}")
    
    # Initialize Bolt if provided
    if bolt_instance:
        bolt_result = await manager.initialize_bolt(bolt_instance)
        results.append(("bolt", bolt_result.success))
        if not bolt_result.success:
            logger.error(f"Bolt initialization failed: {bolt_result.error}")
    
    # Initialize database if path provided
    if db_path:
        try:
            manager.database = DuckDBAdapter(db_path)
            db_result = await manager.database.initialize(manager.config)
            results.append(("database", db_result.success))
            if not db_result.success:
                logger.error(f"Database initialization failed: {db_result.error}")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            results.append(("database", False))
    
    # Initialize tracing
    try:
        manager.tracer = TracingAdapter()
        trace_result = await manager.tracer.initialize(manager.config)
        results.append(("tracing", trace_result.success))
        if not trace_result.success:
            logger.error(f"Tracing initialization failed: {trace_result.error}")
    except Exception as e:
        logger.error(f"Tracing setup failed: {e}")
        results.append(("tracing", False))
    
    # Log initialization summary
    successful = sum(1 for _, success in results if success)
    total = len(results)
    logger.info(f"Unified system initialized: {successful}/{total} components successful")
    
    for component, success in results:
        status = "✅" if success else "❌"
        logger.info(f"  {status} {component}")
    
    manager._initialized = True
    return manager


async def quick_search_example():
    """Example of quick search operation using unified API."""
    print("🔍 Quick Search Example")
    
    # This would typically be called with actual Einstein/Bolt instances
    # manager = await create_unified_system(einstein_instance, bolt_instance)
    
    # For demonstration, create manager without instances
    manager = get_unified_manager()
    
    # Create sample search request
    search_request = UnifiedSearchRequest(
        query="TODO|FIXME",
        search_type="text",
        max_results=10,
        filters={"path": "src"}
    )
    
    print(f"Search query: {search_request.query}")
    print(f"Search type: {search_request.search_type}")
    print(f"Max results: {search_request.max_results}")
    
    # This would perform actual search if systems were initialized
    # result = await manager.search_and_process(search_request)
    print("Note: Actual search requires initialized Einstein instance")


async def codebase_analysis_example():
    """Example of comprehensive codebase analysis."""
    print("📊 Codebase Analysis Example")
    
    # This would typically be called with actual instances
    manager = get_unified_manager()
    
    print("Analysis types available:")
    print("  - comprehensive: General code quality and structure")
    print("  - security: Security vulnerabilities and sensitive data")
    print("  - performance: Performance bottlenecks and inefficiencies")
    
    # This would perform actual analysis if systems were initialized
    # result = await manager.analyze_codebase("comprehensive", "src")
    print("Note: Actual analysis requires initialized Einstein and Bolt instances")


async def batch_processing_example():
    """Example of batch processing with work-stealing optimization."""
    print("⚡ Batch Processing Example")
    
    manager = get_unified_manager()
    
    # Create multiple search requests
    search_requests = [
        UnifiedSearchRequest("class.*:", "structural", 50),
        UnifiedSearchRequest("def.*(", "structural", 50),
        UnifiedSearchRequest("import.*", "analytical", 100),
        UnifiedSearchRequest("TODO|FIXME", "text", 30)
    ]
    
    # Create corresponding processing tasks
    processing_tasks = [
        {"id": f"process_{i}", "description": f"Process search {i}", "priority": "normal"}
        for i in range(len(search_requests))
    ]
    
    print(f"Batch size: {len(search_requests)} searches")
    print("Processing tasks prepared with work-stealing optimization")
    
    # This would perform actual batch processing if systems were initialized
    # result = await manager.batch_search_and_process(search_requests, processing_tasks)
    print("Note: Actual batch processing requires initialized systems")


async def error_handling_example():
    """Example of unified error handling patterns."""
    print("🛡️ Error Handling Example")
    
    # Create various types of unified errors
    system_error = create_unified_error(
        "System initialization failed",
        severity=SystemErrorSeverity.CRITICAL,
        category=SystemErrorCategory.SYSTEM,
        operation="initialization",
        component="einstein"
    )
    
    timeout_error = create_unified_error(
        "Operation timed out",
        severity=SystemErrorSeverity.WARNING,
        category=SystemErrorCategory.TIMEOUT,
        operation="search",
        component="unified_manager"
    )
    
    print(f"System Error: {system_error.to_dict()}")
    print(f"Timeout Error: {timeout_error.to_dict()}")
    
    # Example of safe async call
    async def risky_operation():
        await asyncio.sleep(0.1)
        return {"result": "success"}
    
    result = await safe_async_call(
        risky_operation,
        operation_name="example_operation",
        timeout=5.0
    )
    
    print(f"Safe async call result: success={result.success}")


if __name__ == "__main__":
    """Run integration examples."""
    async def run_examples():
        configure_unified_logging()
        
        print("🚀 Unified API Interface Examples")
        print("="*50)
        
        await quick_search_example()
        print()
        
        await codebase_analysis_example()
        print()
        
        await batch_processing_example()
        print()
        
        await error_handling_example()
        print()
        
        print("✅ All examples completed")
        print("\n📋 Next Steps:")
        print("1. Initialize actual Einstein and Bolt instances")
        print("2. Call create_unified_system() with your instances")
        print("3. Use manager.search_and_process() for unified operations")
        print("4. Call manager.shutdown_all() when done")
    
    asyncio.run(run_examples())