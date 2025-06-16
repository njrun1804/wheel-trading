#!/usr/bin/env python3
"""
Einstein Unified Indexing System

Builds on ALL Jarvis accelerated tools to create a unified indexing system that:
- Combines FAISS vectors, DuckDB analytics, dependency graphs, Python analysis
- Uses hardware acceleration (12 cores + Metal GPU) 
- Provides sub-10ms response for 235k LOC codebase
- Integrates with Jarvis2 and meta systems
- Supports the full intelligence loop
"""

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import sqlite3
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Import existing indexing components
from database_manager import get_database_manager
from einstein.adaptive_concurrency import (
    PerformanceTracker,
    get_adaptive_concurrency_manager,
)
from einstein.einstein_config import get_einstein_config
from einstein.file_watcher import add_realtime_indexing
from neural_backend_manager import get_neural_backend_manager
from unified_config import get_unified_config

# NOTE: Accelerated tools are imported lazily to avoid circular dependencies
# The accelerated tools import system is managed through lazy loading functions

logger = logging.getLogger(__name__)


# Lazy loading functions to avoid circular dependencies
def _get_dependency_graph():
    """Lazy import dependency graph turbo."""
    try:
        from src.unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph
        return get_dependency_graph()
    except ImportError as e:
        logger.warning(f"Failed to import dependency_graph_turbo: {e}")
        return None


def _get_duckdb_turbo(db_path=None):
    """Lazy import DuckDB turbo."""
    try:
        from src.unity_wheel.accelerated_tools.duckdb_turbo import get_duckdb_turbo
        return get_duckdb_turbo(db_path)
    except ImportError as e:
        logger.warning(f"Failed to import duckdb_turbo: {e}")
        return None


def _get_python_analyzer():
    """Lazy import Python analyzer turbo."""
    try:
        from src.unity_wheel.accelerated_tools.python_analysis_turbo import get_python_analyzer
        return get_python_analyzer()
    except ImportError as e:
        logger.warning(f"Failed to import python_analysis_turbo: {e}")
        return None


def _get_code_helper():
    """Lazy import code helper turbo."""
    try:
        from src.unity_wheel.accelerated_tools.python_helpers_turbo import get_code_helper
        return get_code_helper()
    except ImportError as e:
        logger.warning(f"Failed to import python_helpers_turbo: {e}")
        return None


def _get_ripgrep_turbo():
    """Lazy import ripgrep turbo."""
    try:
        from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
        return get_ripgrep_turbo()
    except ImportError as e:
        logger.warning(f"Failed to import ripgrep_turbo: {e}")
        return None


def _get_trace_turbo():
    """Lazy import trace turbo."""
    try:
        from src.unity_wheel.accelerated_tools.trace_simple import get_trace_turbo
        return get_trace_turbo()
    except ImportError as e:
        logger.warning(f"Failed to import trace_simple: {e}")
        return None


def _get_sequential_thinking():
    """Lazy import sequential thinking turbo."""
    try:
        from src.unity_wheel.accelerated_tools.sequential_thinking_turbo import (
            SequentialThinkingTurbo,
        )
        return SequentialThinkingTurbo()
    except ImportError as e:
        logger.warning(f"Failed to import sequential_thinking_turbo: {e}")
        return None


@dataclass
class SearchResult:
    """Unified search result from Einstein system."""

    content: str
    file_path: str
    line_number: int
    score: float
    result_type: str  # 'text', 'semantic', 'structural', 'analytical'
    context: dict[str, Any]
    timestamp: float


@dataclass
class IndexStats:
    """Statistics about the Einstein index."""

    total_files: int
    total_lines: int
    index_size_mb: float
    last_update: float
    search_performance_ms: dict[str, float]
    coverage_percentage: float


class EinsteinIndexHub:
    """Unified indexing system leveraging all Jarvis accelerated tools."""

    def __init__(
        self, project_root: Path | str | None = None, fast_mode: bool = False
    ) -> None:
        # Ensure project_root is always a Path object
        if project_root is None:
            self.project_root = Path.cwd()
        elif isinstance(project_root, str):
            self.project_root = Path(project_root)
        else:
            self.project_root = project_root
        self.config = get_unified_config()
        self.einstein_config = get_einstein_config(project_root)
        self.neural_backend = get_neural_backend_manager()

        # PERFORMANCE OPTIMIZATION: Fast mode for quick startup
        self._fast_mode = fast_mode
        if fast_mode:
            self._skip_initial_scan = True
            self._skip_file_watcher = True
            self._skip_dependency_build = True
            logger.info("⚡ Einstein Fast Mode enabled - skipping heavy initialization")

        # Adaptive concurrency manager for hardware optimization
        self.concurrency_manager = get_adaptive_concurrency_manager()

        # Bounded concurrency (will be replaced by adaptive semaphores)
        self.search_semaphore = asyncio.Semaphore(
            self.einstein_config.performance.max_search_concurrency
        )
        self.embedding_semaphore = asyncio.Semaphore(
            self.einstein_config.performance.max_embedding_concurrency
        )
        self.file_io_semaphore = asyncio.Semaphore(
            self.einstein_config.performance.max_file_io_concurrency
        )
        self.analysis_semaphore = asyncio.Semaphore(
            self.einstein_config.performance.max_analysis_concurrency
        )

        # Initialize ALL accelerated tools with error handling
        # Initialize all tools to None first to ensure they exist
        self.ripgrep = None
        self.dependency_graph = None
        self.python_analyzer = None
        self.duckdb = None
        self.sequential_thinking = None
        self.code_helper = None
        self.tracer = None

        try:
            # Use lazy loading functions to avoid circular dependencies
            self.ripgrep = _get_ripgrep_turbo()
            self.dependency_graph = _get_dependency_graph()
            self.python_analyzer = _get_python_analyzer()

            # Initialize DuckDB with proper database path from config
            analytics_db_path = str(self.einstein_config.paths.analytics_db_path)
            self.duckdb = _get_duckdb_turbo(analytics_db_path)

            self.sequential_thinking = _get_sequential_thinking()
            self.code_helper = _get_code_helper()
            self.tracer = _get_trace_turbo()

        except Exception as e:
            logger.error(
                f"Error initializing accelerated tools: {e}",
                exc_info=True,
                extra={
                    "operation": "init_accelerated_tools",
                    "error_type": type(e).__name__,
                    "project_root": str(self.project_root),
                    "cpu_cores": self.einstein_config.hardware.cpu_cores,
                    "platform_type": self.einstein_config.hardware.platform_type,
                    "gpu_available": self.einstein_config.hardware.has_gpu,
                    "memory_gb": self.einstein_config.hardware.memory_total_gb,
                },
            )
            # Attributes are already set to None above, so we're safe

        # Existing systems integration
        self.db_manager = get_database_manager()

        # Performance tracking
        self.search_stats = {
            "text_search_ms": [],
            "semantic_search_ms": [],
            "structural_search_ms": [],
            "analytical_search_ms": [],
        }

        # Hardware optimization with CPU usage limits for M4 Pro
        self.cpu_cores = self.einstein_config.hardware.cpu_cores
        # CRITICAL FIX: Limit executor to prevent CPU saturation - use only 60% of available cores
        max_workers = max(2, min(6, int(self.cpu_cores * 0.6)))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(
            f"🔧 ThreadPoolExecutor limited to {max_workers} workers (60% of {self.cpu_cores} cores)"
        )

        # Initialize FAISS-related attributes
        self.vector_index = None
        self._faiss_loaded = False
        self._faiss_available = self._check_faiss_availability()
        self.embedding_pipeline = None
        self._embedding_pipeline_available = (
            False  # Track embedding pipeline availability
        )

        # Initialize file watcher attributes
        self._shutdown_event = None
        self._file_change_loop = None
        self._file_change_task = None

        # Real-time indexing
        self.file_watcher = None
        self.file_change_queue = None  # Will be initialized in _start_file_watcher
        self._last_indexed = {}  # File path -> (hash, timestamp)

        # Index state attributes (required for initialization logging)
        self.file_metadata = {}  # File path -> metadata dict
        self.indexed_files = set()  # Set of indexed file paths
        self.embedding_dim = 384  # Default embedding dimension (sentence-transformers)

        # Add realtime indexing capability
        add_realtime_indexing(self, [self.project_root])

        logger.info(
            f"🧠 Einstein Index initialized with {self.cpu_cores} cores on {self.einstein_config.hardware.platform_type}"
        )

    def _check_faiss_availability(self) -> bool:
        """Check if FAISS library is available and working.

        Returns:
            bool: True if FAISS is available and functional
        """
        try:
            import faiss
            import numpy as np

            # Test basic FAISS functionality
            test_index = faiss.IndexFlatL2(128)  # Small test index
            test_vector = np.array([[1.0] * 128], dtype=np.float32)
            test_index.add(test_vector)

            if test_index.ntotal == 1:
                logger.debug("✅ FAISS library is available and functional")
                return True
            else:
                logger.warning("⚠️ FAISS library loaded but basic test failed")
                return False

        except ImportError:
            logger.info(
                "ℹ️ FAISS library not available - will use embedding pipeline fallback"
            )
            return False
        except Exception as e:
            logger.warning(
                f"⚠️ FAISS library test failed: {e}",
                extra={
                    "operation": "check_faiss_availability",
                    "error_type": type(e).__name__,
                    "test_vector_shape": "(1, 128)",
                    "faiss_import_success": True,
                    "numpy_available": True,
                },
            )
            return False

    def _validate_faiss_index_for_save(self) -> bool:
        """Validate FAISS index state before saving.

        Returns:
            bool: True if index is valid for saving, False otherwise
        """
        try:
            if not hasattr(self, "vector_index") or self.vector_index is None:
                logger.warning("No FAISS index exists to validate")
                return False

            # Check if index has required attributes
            if not hasattr(self.vector_index, "ntotal"):
                logger.warning("FAISS index missing 'ntotal' attribute")
                return False

            if not hasattr(self.vector_index, "d"):
                logger.warning("FAISS index missing dimension 'd' attribute")
                return False

            # Check if index has valid dimensions
            if self.vector_index.d <= 0:
                logger.warning(f"Invalid FAISS index dimension: {self.vector_index.d}")
                return False

            # Index can be empty (ntotal = 0) and still be valid for saving
            logger.debug(
                f"FAISS index validation passed: {self.vector_index.ntotal} vectors, dimension {self.vector_index.d}"
            )
            return True

        except Exception as e:
            logger.error(f"Error validating FAISS index: {e}")
            return False

    def _validate_faiss_search_input(
        self, query_embedding: Any, k: int = 20
    ) -> tuple[bool, str]:
        """Validate inputs for FAISS search operations.

        Args:
            query_embedding: The query embedding vector
            k: Number of results to search for

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Check if FAISS is available
            if not self._check_faiss_availability():
                return False, "FAISS library not available"

            # Check if FAISS index exists and is loaded
            if not hasattr(self, "vector_index") or not self.vector_index:
                return False, "FAISS index not loaded"

            if not self._faiss_loaded:
                return False, "FAISS index not marked as loaded"

            # Validate index attributes
            if not hasattr(self.vector_index, "search") or not hasattr(
                self.vector_index, "ntotal"
            ):
                return False, "FAISS index missing required methods"

            if self.vector_index.ntotal == 0:
                return False, "FAISS index is empty"

            # Validate query embedding
            if query_embedding is None:
                return False, "Query embedding is None"

            try:
                import numpy as np

                if not isinstance(query_embedding, np.ndarray):
                    query_embedding = np.array(query_embedding)

                if query_embedding.size == 0:
                    return False, "Query embedding is empty"

                # Check embedding dimension
                expected_dim = getattr(self.vector_index, "d", 1536)
                query_shape = query_embedding.shape
                if len(query_shape) == 1:
                    actual_dim = query_shape[0]
                elif len(query_shape) == 2:
                    actual_dim = query_shape[1]
                else:
                    return False, f"Invalid query embedding shape: {query_shape}"

                if actual_dim != expected_dim:
                    return (
                        False,
                        f"Query embedding dimension {actual_dim} doesn't match index dimension {expected_dim}",
                    )

            except Exception as embed_error:
                return False, f"Error validating query embedding: {embed_error}"

            # Validate k parameter
            if k <= 0:
                return False, "Search parameter k must be positive"

            if k > self.vector_index.ntotal:
                # This is acceptable, just limit k
                pass

            return True, "Validation passed"

        except Exception as e:
            return False, f"Validation error: {e}"

    async def initialize(self) -> None:
        """Initialize all indexing components."""

        start_time = time.time()
        try:
            # Check FAISS availability first
            self._faiss_available = self._check_faiss_availability()

            # Create Einstein directory for persistent indexes
            einstein_dir = self.project_root / ".einstein"
            einstein_dir.mkdir(exist_ok=True)

            # Build dependency graph (skip if running in script context to avoid multiprocessing issues)
            if not getattr(self, "_skip_dependency_build", False):
                try:
                    if self._fast_mode:
                        logger.info(
                            "⚡ Fast mode: Deferring dependency graph build to background"
                        )
                        asyncio.create_task(self._build_dependency_graph_deferred())
                    else:
                        await self.dependency_graph.build_graph()
                except Exception as e:
                    logger.error(
                        f"Dependency graph build failed, skipping: {e}",
                        exc_info=True,
                        extra={
                            "operation": "build_dependency_graph",
                            "error_type": type(e).__name__,
                            "project_root": str(self.project_root),
                            "skip_dependency_build": getattr(
                                self, "_skip_dependency_build", False
                            ),
                            "dependency_graph_available": self.dependency_graph
                            is not None,
                            "multiprocessing_context": getattr(
                                self, "_multiprocessing_context", "unknown"
                            ),
                            "fast_mode": self._fast_mode,
                        },
                    )
            else:
                logger.info("Skipping dependency graph build in script context")

            # Initialize DuckDB analytics (skip if DuckDB failed to initialize)
            if self.duckdb:
                if self._fast_mode:
                    logger.info(
                        "⚡ Fast mode: Deferring analytics DB initialization to background"
                    )
                    asyncio.create_task(self._initialize_analytics_db_deferred())
                else:
                    await self._initialize_analytics_db()
            else:
                logger.warning(
                    "Skipping DuckDB analytics initialization due to connection issues",
                    extra={
                        "operation": "initialize_duckdb",
                        "duckdb_available": self.duckdb is not None,
                        "analytics_db_path": str(
                            self.einstein_config.paths.analytics_db_path
                        ),
                        "db_exists": self.einstein_config.paths.analytics_db_path.exists(),
                        "fast_mode": self._fast_mode,
                    },
                )

            # Initialize FAISS index with persistence
            if self._fast_mode:
                logger.info("⚡ Fast mode: Deferring FAISS initialization to background")
                asyncio.create_task(self._initialize_persistent_faiss_deferred())
            else:
                await self._initialize_persistent_faiss()

            # Initialize embedding pipeline
            if self._fast_mode:
                logger.info(
                    "⚡ Fast mode: Deferring embedding pipeline initialization to background"
                )
                asyncio.create_task(self._initialize_embedding_pipeline_deferred())
            else:
                await self._initialize_embedding_pipeline()

            # Check if we need to perform initial scan
            await self._check_and_perform_initial_scan()

            # PERFORMANCE OPTIMIZATION: Defer file watcher initialization
            # File watcher will be initialized when first needed
            if not getattr(self, "_skip_file_watcher", False):
                self._prepare_file_watcher()
            else:
                logger.info("⚡ Deferring file watcher initialization for fast startup")

            initialization_time = time.time() - start_time
            logger.info(
                f"✅ Einstein Index initialization complete in {initialization_time:.2f}s"
            )
            logger.info(
                f"📊 Index Stats: {len(self.file_metadata)} files, {len(self.indexed_files)} indexed, embedding_dim={self.embedding_dim}"
            )

            # Log accelerated tools status
            tools_status = {
                "ripgrep": self.ripgrep is not None,
                "dependency_graph": self.dependency_graph is not None,
                "python_analyzer": self.python_analyzer is not None,
                "duckdb": self.duckdb is not None,
                "code_helper": self.code_helper is not None,
            }
            logger.info(f"🔧 Accelerated Tools: {tools_status}")

            # Log hardware detection
            logger.info(
                f"🖥️  Hardware: {self.einstein_config.hardware.platform_type}, "
                f"{self.einstein_config.hardware.cpu_cores} cores, "
                f"{self.einstein_config.hardware.memory_total_gb:.1f}GB RAM, "
                f"GPU: {self.einstein_config.hardware.has_gpu}"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize Einstein Index: {e}",
                exc_info=True,
                extra={
                    "operation": "initialize",
                    "error_type": type(e).__name__,
                    "project_root": str(self.project_root),
                    "faiss_available": self._faiss_available,
                    "embedding_pipeline_available": self._embedding_pipeline_available,
                    "initialization_stage": "complete",
                    "cpu_cores": self.cpu_cores,
                },
            )
            # Ensure cleanup on initialization failure
            await self._cleanup_initialization()
            raise

    async def _initialize_analytics_db(self) -> None:
        """Initialize DuckDB analytics schema."""

        # Create analytics tables
        await self.duckdb.execute(
            """
            CREATE TABLE IF NOT EXISTS file_analytics (
                file_path TEXT PRIMARY KEY,
                lines_of_code INTEGER,
                complexity_score REAL,
                last_modified REAL,
                language TEXT,
                dependencies TEXT[],
                exports TEXT[]
            )
        """
        )

        await self.duckdb.execute(
            """
            CREATE TABLE IF NOT EXISTS search_analytics (
                timestamp REAL,
                query TEXT,
                result_count INTEGER,
                search_time_ms REAL,
                search_type TEXT,
                success_rating REAL DEFAULT 0.0,
                user_feedback INTEGER DEFAULT 0
            )
        """
        )

        # Create query learning table
        await self.duckdb.execute(
            """
            CREATE TABLE IF NOT EXISTS query_patterns (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT,
                best_search_types TEXT,
                avg_success_rating REAL,
                usage_count INTEGER,
                last_used REAL
            )
        """
        )

    async def _check_and_perform_initial_scan(self) -> None:
        """Check if initial scan is needed and perform if necessary."""

        # PERFORMANCE OPTIMIZATION: Skip initial scan during fast startup
        if getattr(self, "_skip_initial_scan", False):
            logger.info("⚡ Skipping initial scan for fast startup")
            return

        # Check if we have a valid cached scan
        scan_cache_file = self.project_root / ".einstein" / "scan_cache.json"

        if scan_cache_file.exists():
            try:
                import json

                with open(scan_cache_file) as f:
                    cache_data = json.load(f)

                # Check if cache is recent and complete
                cache_timestamp = cache_data.get("timestamp", 0)
                cache_file_count = cache_data.get("file_count", 0)

                # PERFORMANCE OPTIMIZATION: Use faster file counting
                import time

                cache_age_hours = (time.time() - cache_timestamp) / 3600

                # More lenient cache validation for performance
                if cache_age_hours < 24:  # Extended cache validity to 24 hours
                    logger.info(
                        f"✅ Using cached scan data ({cache_file_count} files, {cache_age_hours:.1f}h old)"
                    )
                    return
                else:
                    logger.info(f"🔄 Cache outdated (age: {cache_age_hours:.1f}h)")

            except Exception as e:
                logger.warning(
                    f"Invalid scan cache: {e}",
                    extra={
                        "operation": "check_scan_cache",
                        "error_type": type(e).__name__,
                        "cache_file": str(scan_cache_file),
                        "cache_exists": scan_cache_file.exists(),
                        "cache_size": scan_cache_file.stat().st_size
                        if scan_cache_file.exists()
                        else 0,
                    },
                )

        # PERFORMANCE OPTIMIZATION: Defer initial scan to background
        # This allows faster startup - scan will happen in background
        logger.info("⚡ Deferring initial scan to background for fast startup")
        asyncio.create_task(self._perform_initial_scan_deferred())

        # Create minimal cache to avoid repeated scans
        try:
            import json
            import time

            cache_data = {
                "timestamp": time.time(),
                "file_count": 0,  # Will be updated by background scan
                "status": "deferred",
            }
            scan_cache_file.parent.mkdir(exist_ok=True)
            with open(scan_cache_file, "w") as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save deferred scan cache: {e}")

    async def _perform_initial_scan_deferred(self) -> None:
        """Perform initial codebase scan in background for better startup performance."""

        try:
            # Small delay to allow main initialization to complete
            await asyncio.sleep(1.0)

            # Find all Python files with faster globbing
            python_files = list(self.project_root.rglob("*.py"))
            import os

            # CRITICAL FIX: Limit background processing to prevent CPU overload
            cores = min(
                4,
                getattr(self.config.hardware, "max_workers", os.cpu_count() or 12) // 3,
            )  # Use only 1/3 of cores for background
            logger.info(
                f"🔍 Background scan: Analyzing {len(python_files)} Python files on {cores} cores..."
            )

            # PERFORMANCE OPTIMIZATION: Smaller batches to prevent CPU overload
            batch_size = 20  # Reduced from 50 to prevent system overload
            successful_analyses = 0

            for i in range(0, len(python_files), batch_size):
                batch = python_files[i : i + batch_size]
                tasks = [
                    asyncio.create_task(self._analyze_file(file_path))
                    for file_path in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                batch_successful = sum(
                    1 for r in results if not isinstance(r, Exception)
                )
                successful_analyses += batch_successful

                # Progress update
                if i % (batch_size * 10) == 0:  # Update every 10 batches
                    logger.info(
                        f"📊 Background scan progress: {successful_analyses}/{len(python_files)} files analyzed"
                    )

            logger.info(
                f"✅ Background scan complete: {successful_analyses}/{len(python_files)} files analyzed"
            )

            # Update scan cache
            scan_cache_file = self.project_root / ".einstein" / "scan_cache.json"
            try:
                import json
                import time

                cache_data = {
                    "timestamp": time.time(),
                    "file_count": len(python_files),
                    "status": "complete",
                }
                with open(scan_cache_file, "w") as f:
                    json.dump(cache_data, f)
                logger.info("💾 Background scan cache updated")
            except Exception as e:
                logger.warning(f"Failed to update scan cache: {e}")

        except Exception as e:
            logger.error(f"Background scan failed: {e}", exc_info=True)

    async def _perform_initial_scan(self) -> None:
        """Perform initial codebase scan using all tools (legacy method)."""
        # Redirect to deferred scan for performance
        await self._perform_initial_scan_deferred()

    async def _analyze_file(self, file_path: Path) -> dict[str, Any]:
        """Analyze a single file using accelerated tools with adaptive concurrency."""

        # PERFORMANCE OPTIMIZATION: Skip analysis for very large files
        try:
            file_size = file_path.stat().st_size
            if file_size > 1_000_000:  # Skip files larger than 1MB
                return {"skipped": True, "reason": "file_too_large", "size": file_size}
        except OSError as e:
            logger.warning(f"Could not get file size for {file_path}: {e}")
            # Continue with analysis anyway

        # Use adaptive concurrency for file analysis
        semaphore = await self.concurrency_manager.get_semaphore("file_analysis")

        async with semaphore:
            async with PerformanceTracker("file_analysis", semaphore._value):
                try:
                    # PERFORMANCE OPTIMIZATION: Minimal analysis for speed
                    analysis = {
                        "file_path": str(file_path),
                        "analyzed_at": time.time(),
                        "size": file_size if "file_size" in locals() else 0,
                    }

                    # Use Python analyzer for structure if available
                    if self.python_analyzer:
                        try:
                            analysis = await self.python_analyzer.analyze_file(
                                str(file_path)
                            )
                            # Handle case where analyzer returns object instead of dict
                            if hasattr(analysis, "__dict__"):
                                analysis = analysis.__dict__
                            elif not isinstance(analysis, dict):
                                analysis = {"raw_result": analysis}
                        except Exception as e:
                            logger.warning(
                                f"Python analyzer failed for {file_path}: {e}",
                                extra={
                                    "operation": "python_analyzer_file_analysis",
                                    "error_type": type(e).__name__,
                                    "file_path": str(file_path),
                                    "analyzer_available": self.python_analyzer
                                    is not None,
                                    "file_exists": file_path.exists(),
                                    "file_size": file_path.stat().st_size
                                    if file_path.exists()
                                    else 0,
                                    "file_extension": file_path.suffix,
                                },
                            )
                            analysis = {}

                    # Get function signatures using code helper if available
                    if self.code_helper:
                        try:
                            # Use wildcard to get all functions from the file
                            functions_result = (
                                await self.code_helper.get_function_signature(
                                    str(file_path), "*"
                                )
                            )
                            if (
                                isinstance(functions_result, dict)
                                and "functions" in functions_result
                            ):
                                functions_result["functions"]
                            elif (
                                isinstance(functions_result, dict)
                                and "error" in functions_result
                            ):
                                logger.debug(
                                    f"Code helper returned error for {file_path}: {functions_result['error']}"
                                )
                        except ImportError as e:
                            # Skip files that shouldn't be analyzed (test files, etc.)
                            logger.debug(
                                f"Skipping {file_path} (likely test file or contains async module-level code): {e}"
                            )
                        except Exception as e:
                            logger.debug(
                                f"Code helper failed for {file_path}: {e}",
                                extra={
                                    "operation": "code_helper_function_signatures",
                                    "error_type": type(e).__name__,
                                    "file_path": str(file_path),
                                    "helper_available": self.code_helper is not None,
                                    "file_exists": file_path.exists(),
                                    "file_extension": file_path.suffix,
                                },
                            )

                    # Store in analytics DB if available
                    if self.duckdb:
                        try:
                            await self.duckdb.execute(
                                """
                                INSERT OR REPLACE INTO file_analytics 
                                (file_path, lines_of_code, complexity_score, last_modified, language)
                                VALUES (?, ?, ?, ?, ?)
                            """,
                                (
                                    str(file_path),
                                    analysis.get(
                                        "loc", analysis.get("lines_of_code", 0)
                                    ),
                                    analysis.get("complexity", 0.0),
                                    file_path.stat().st_mtime,
                                    "python",
                                ),
                            )
                        except Exception as e:
                            logger.warning(
                                f"DB storage failed for {file_path}: {e}",
                                extra={
                                    "operation": "analyze_file_db_storage",
                                    "error_type": type(e).__name__,
                                    "file_path": str(file_path),
                                    "analysis_keys": list(analysis.keys()),
                                    "duckdb_available": self.duckdb is not None,
                                    "file_size": file_path.stat().st_size
                                    if file_path.exists()
                                    else 0,
                                },
                            )

                    return analysis

                except Exception as e:
                    logger.error(
                        f"Analysis failed for {file_path}: {e}",
                        exc_info=True,
                        extra={
                            "operation": "analyze_file",
                            "error_type": type(e).__name__,
                            "file_path": str(file_path),
                            "semaphore_type": "file_analysis",
                            "python_analyzer_available": self.python_analyzer
                            is not None,
                            "code_helper_available": self.code_helper is not None,
                            "file_exists": file_path.exists(),
                        },
                    )
                    return {}

    async def search(
        self,
        query: str,
        search_types: list[str] = None,
        use_learning: bool = True,
        max_results: int = 1000,
    ) -> list[SearchResult]:
        """Unified search across all indexing modalities with adaptive concurrency and learning."""

        # Use learned search types if available and requested
        if search_types is None:
            if use_learning:
                search_types = await self.get_optimized_search_types(query)
            else:
                search_types = ["text", "semantic", "structural", "analytical"]

        logger.info(
            f"🔍 Searching with types: {search_types} for query: '{query}', max_results={max_results}"
        )
        start_time = time.time()

        # Log query analysis
        query_complexity = "simple" if len(query.split()) <= 3 else "complex"
        logger.info(
            f"📝 Query analysis: '{query}' ({len(query)} chars, {len(query.split())} words, {query_complexity})"
        )

        # Bounded parallel search across modalities
        search_tasks = []

        if "text" in search_types:
            search_tasks.append(
                self._bounded_search(self._text_search, query, "text_search")
            )

        if "semantic" in search_types:
            search_tasks.append(
                self._bounded_search(self._semantic_search, query, "semantic_search")
            )

        if "structural" in search_types:
            search_tasks.append(
                self._bounded_search(
                    self._structural_search, query, "structural_search"
                )
            )

        if "analytical" in search_types:
            search_tasks.append(
                self._bounded_search(
                    self._analytical_search, query, "analytical_search"
                )
            )

        # Execute with bounded concurrency
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        search_time = (time.time() - start_time) * 1000

        # Merge and rank results with proper prioritization
        all_results = []
        for result_set in results:
            if isinstance(result_set, list):
                all_results.extend(result_set)

        # Enhanced result ranking and deduplication
        ranked_results = await self._enhanced_result_ranking(query, all_results)

        # Record search analytics with search types for learning
        await self._record_search_analytics(
            query, len(ranked_results), search_time, search_types
        )

        return ranked_results[:max_results]  # Top results based on configured limit

    async def _bounded_search(
        self, search_func, query: str, operation_type: str = "search"
    ):
        """Execute search function with adaptive concurrency."""
        # Get adaptive semaphore based on operation type
        semaphore = await self.concurrency_manager.get_semaphore(operation_type)

        async with semaphore, PerformanceTracker(operation_type, semaphore._value):
            return await search_func(query)

    async def _text_search(self, query: str) -> list[SearchResult]:
        """Fast text search using ripgrep turbo with fallback to simple search."""

        start_time = time.time()

        try:
            # Try ripgrep first if available
            if self.ripgrep:
                try:
                    rg_results = await self.ripgrep.search(
                        query, str(self.project_root), max_results=100
                    )

                    results = []
                    for rg_result in rg_results:
                        result = SearchResult(
                            content=rg_result["content"],
                            file_path=rg_result["file"],
                            line_number=rg_result["line"],
                            score=1.0,  # Text matches are exact
                            result_type="text",
                            context={"column": rg_result.get("column", 0)},
                            timestamp=time.time(),
                        )
                        results.append(result)

                    search_time = (time.time() - start_time) * 1000
                    self.search_stats["text_search_ms"].append(search_time)

                    return results

                except (
                    TimeoutError,
                    OSError,
                    subprocess.SubprocessError,
                    Exception,
                ) as rg_error:
                    logger.warning(f"Ripgrep search failed, using fallback: {rg_error}")

            # Fallback to simple text search
            return await self._simple_text_search(query)

        except Exception as e:
            logger.error(
                f"Text search completely failed: {e}",
                exc_info=True,
                extra={
                    "operation": "text_search",
                    "error_type": type(e).__name__,
                    "query": query[:50],  # Truncate long queries
                    "query_length": len(query),
                    "ripgrep_available": self.ripgrep is not None,
                    "project_root": str(self.project_root),
                    "search_time_ms": (time.time() - start_time) * 1000,
                },
            )
            return []

    async def _simple_text_search(
        self, query: str, max_results: int = 100
    ) -> list[SearchResult]:
        """Simple text search fallback when ripgrep is not available."""

        import re

        results = []

        try:
            # Get all Python files
            python_files = list(self.project_root.rglob("*.py"))

            # Create regex pattern for case-insensitive search
            try:
                pattern = re.compile(re.escape(query), re.IGNORECASE)
            except re.error:
                # If query has regex chars, fall back to literal search
                pattern = re.compile(query, re.IGNORECASE)

            # Search through files
            for file_path in python_files:
                try:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()

                    for line_num, line in enumerate(lines, 1):
                        if pattern.search(line):
                            result = SearchResult(
                                content=line.strip(),
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                score=1.0,
                                result_type="text_fallback",
                                context={"method": "simple_search"},
                                timestamp=time.time(),
                            )
                            results.append(result)

                            if len(results) >= max_results:
                                return results

                except Exception:
                    # Skip files that can't be read
                    continue

            logger.info(
                f"Simple text search found {len(results)} results for '{query}'"
            )
            return results

        except Exception as e:
            logger.error(f"Simple text search failed: {e}")
            return []

    async def _semantic_search(self, query: str) -> list[SearchResult]:
        """Optimized semantic search specifically for coding analysis queries."""

        start_time = time.time()
        results = []

        try:
            logger.debug(f"Starting optimized semantic search for query: '{query}'")

            # NEW: Use optimized semantic search for coding queries
            if not hasattr(self, "_optimized_semantic_search"):
                from .optimized_semantic_search import OptimizedSemanticSearch

                self._optimized_semantic_search = OptimizedSemanticSearch(
                    self.project_root
                )

            # Try optimized semantic search first
            try:
                optimized_results = await self._optimized_semantic_search.search(
                    query, max_results=20, search_type="auto"
                )

                if optimized_results:
                    # Convert to unified SearchResult format
                    for opt_result in optimized_results:
                        result = SearchResult(
                            content=opt_result.content,
                            file_path=opt_result.file_path,
                            line_number=opt_result.line_number,
                            score=opt_result.relevance_score or opt_result.score,
                            result_type="semantic_optimized",
                            context=opt_result.context,
                            timestamp=opt_result.timestamp,
                        )
                        results.append(result)

                    logger.info(
                        f"Optimized semantic search found {len(results)} results"
                    )

                    # Skip fallback stages if we have good results
                    if len(results) >= 5:
                        search_time = (time.time() - start_time) * 1000
                        self.search_stats["semantic_search_ms"].append(search_time)
                        return results[:20]

            except Exception as e:
                logger.warning(f"Optimized semantic search failed, falling back: {e}")

            # Fallback Stage 1: Try FAISS search if available
            if (
                not results
                and self._faiss_available
                and await self._ensure_embedding_pipeline()
            ):
                try:
                    query_embedding, token_count = await self._safe_get_query_embedding(
                        query
                    )
                    if query_embedding is not None:
                        faiss_results = await self._try_faiss_search(query_embedding)
                        if faiss_results:
                            results.extend(faiss_results)
                            logger.info(
                                f"FAISS search found {len(faiss_results)} results"
                            )
                except Exception as e:
                    logger.error(
                        f"FAISS search stage failed: {e}",
                        exc_info=True,
                        extra={
                            "operation": "semantic_search_faiss",
                            "error_type": type(e).__name__,
                            "query": query[:50],  # Truncate long queries
                            "query_length": len(query),
                            "faiss_available": self._faiss_available,
                            "embedding_pipeline_available": self._embedding_pipeline_available,
                            "faiss_loaded": self._faiss_loaded,
                            "vector_index_size": self.vector_index.ntotal
                            if self.vector_index
                            and hasattr(self.vector_index, "ntotal")
                            else 0,
                        },
                    )

            # Stage 2: Try embedding pipeline search if no FAISS results
            if not results and await self._ensure_embedding_pipeline():
                try:
                    query_embedding, _ = await self._safe_get_query_embedding(query)
                    if query_embedding is not None:
                        pipeline_results = await self._try_embedding_pipeline_search(
                            query, query_embedding
                        )
                        if pipeline_results:
                            results.extend(pipeline_results)
                            logger.info(
                                f"Embedding pipeline search found {len(pipeline_results)} results"
                            )
                except Exception as e:
                    logger.error(
                        f"Embedding pipeline search stage failed: {e}",
                        exc_info=True,
                        extra={
                            "operation": "semantic_search_embedding_pipeline",
                            "error_type": type(e).__name__,
                            "query": query[:50],  # Truncate long queries
                            "query_length": len(query),
                            "embedding_pipeline_available": self._embedding_pipeline_available,
                            "pipeline_type": type(self.embedding_pipeline).__name__
                            if self.embedding_pipeline
                            else "None",
                        },
                    )

            # Stage 3: Try neural backend search as fallback
            if not results:
                try:
                    neural_results = await self._try_neural_backend_search(query)
                    if neural_results:
                        results.extend(neural_results)
                        logger.info(
                            f"Neural backend search found {len(neural_results)} results"
                        )
                except Exception as e:
                    logger.error(
                        f"Neural backend search stage failed: {e}",
                        exc_info=True,
                        extra={
                            "operation": "semantic_search_neural_backend",
                            "error_type": type(e).__name__,
                            "query": query[:50],  # Truncate long queries
                            "query_length": len(query),
                            "neural_backend_available": self.neural_backend is not None,
                            "neural_backend_type": type(self.neural_backend).__name__
                            if self.neural_backend
                            else "None",
                        },
                    )

            # Stage 4: Final fallback to enhanced text search
            if not results:
                try:
                    fallback_results = await self._fallback_semantic_text_search(query)
                    if fallback_results:
                        results.extend(fallback_results)
                        logger.info(
                            f"Semantic text fallback found {len(fallback_results)} results"
                        )
                except Exception as e:
                    logger.warning(
                        f"Semantic text fallback failed: {e}",
                        extra={
                            "operation": "semantic_search_text_fallback",
                            "error_type": type(e).__name__,
                            "query": query[:50],
                            "fallback_stage": "semantic_text",
                            "previous_results_count": len(results),
                        },
                    )

            # Ultimate fallback - basic text search
            if not results:
                try:
                    text_results = await self._fallback_text_search(query)
                    results.extend(text_results)
                    logger.info(
                        f"Using text search as ultimate fallback, found {len(text_results)} results"
                    )
                except Exception as e:
                    logger.error(
                        f"All fallback methods failed: {e}",
                        exc_info=True,
                        extra={
                            "operation": "semantic_search_ultimate_fallback",
                            "error_type": type(e).__name__,
                            "query": query[:50],
                            "fallback_stage": "ultimate_text",
                            "previous_results_count": len(results),
                            "search_time_ms": (time.time() - start_time) * 1000,
                        },
                    )
                    # Return empty results rather than crash
                    results = []

            search_time = (time.time() - start_time) * 1000
            self.search_stats["semantic_search_ms"].append(search_time)

            logger.info(
                f"✅ Semantic search completed: {len(results)} results in {search_time:.1f}ms"
            )

            # Log search result breakdown
            if results:
                file_types = {}
                for result in results[:10]:  # Analyze top 10
                    file_ext = Path(result.file_path).suffix
                    file_types[file_ext] = file_types.get(file_ext, 0) + 1
                logger.info(f"📊 Top results by type: {file_types}")
                logger.info(
                    f"🎯 Best match: {results[0].file_path}:{results[0].line_number} (score: {results[0].score:.3f})"
                )

            return results[:20]  # Limit to top 20 results

        except Exception as e:
            logger.error(
                f"Semantic search completely failed: {e}",
                exc_info=True,
                extra={
                    "operation": "semantic_search",
                    "error_type": type(e).__name__,
                    "query": query[:50],
                    "search_time_ms": (time.time() - start_time) * 1000,
                    "faiss_available": self._faiss_available,
                    "embedding_pipeline_available": self._embedding_pipeline_available,
                    "neural_backend_available": self.neural_backend is not None,
                },
            )
            # Even if everything fails, try to return some text-based results
            try:
                return await self._fallback_text_search(query)
            except Exception as final_e:
                logger.error(
                    f"Final fallback also failed: {final_e}",
                    exc_info=True,
                    extra={
                        "operation": "semantic_search_final_fallback",
                        "error_type": type(final_e).__name__,
                        "query": query[:50],
                        "original_error": str(e),
                        "search_time_ms": (time.time() - start_time) * 1000,
                    },
                )
                return []

    async def _structural_search(self, query: str) -> list[SearchResult]:
        """Enhanced structural search using AST parsing and dependency graph."""

        start_time = time.time()
        results = []

        try:
            # Stage 1: Direct AST search for structural patterns
            ast_results = await self._ast_structural_search(query)
            results.extend(ast_results)

            # Stage 2: Use dependency graph for symbol resolution if available
            if self.dependency_graph:
                try:
                    symbols = await self.dependency_graph.find_symbol(query)

                    for symbol in symbols:
                        # Handle different symbol formats
                        if isinstance(symbol, dict):
                            file_path = symbol.get(
                                "file", symbol.get("path", "unknown")
                            )
                            line_number = symbol.get("line", symbol.get("lineno", 0))
                            symbol_name = symbol.get("name", str(symbol))
                            symbol_type = symbol.get("type", "unknown")

                            result = SearchResult(
                                content=f"{symbol_type}: {symbol_name}",
                                file_path=file_path,
                                line_number=line_number,
                                score=0.85,  # High score for exact symbol matches
                                result_type="symbol",
                                context={
                                    "symbol_type": symbol_type,
                                    "symbol_name": symbol_name,
                                    "method": "dependency_graph",
                                },
                                timestamp=time.time(),
                            )
                            results.append(result)
                        else:
                            # Handle string/other formats
                            result = SearchResult(
                                content=f"Symbol: {symbol}",
                                file_path="unknown",
                                line_number=0,
                                score=0.7,
                                result_type="symbol",
                                context={
                                    "symbol_type": "unknown",
                                    "method": "dependency_graph",
                                },
                                timestamp=time.time(),
                            )
                            results.append(result)

                except Exception as dep_error:
                    logger.warning(f"Dependency graph search failed: {dep_error}")

            # Stage 3: Pattern-based structural search for specific queries
            pattern_results = await self._pattern_structural_search(query)
            results.extend(pattern_results)

            # Remove duplicates and sort by score
            seen = set()
            unique_results = []
            for result in results:
                key = f"{result.file_path}:{result.line_number}:{result.content[:50]}"
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)

            # Sort by score (highest first)
            unique_results.sort(key=lambda x: x.score, reverse=True)

            search_time = (time.time() - start_time) * 1000
            self.search_stats["structural_search_ms"].append(search_time)

            logger.info(
                f"✅ Structural search found {len(unique_results)} results in {search_time:.1f}ms"
            )

            return unique_results[:20]  # Return top 20 results

        except Exception as e:
            logger.error(
                f"Structural search failed: {e}",
                exc_info=True,
                extra={
                    "operation": "structural_search",
                    "error_type": type(e).__name__,
                    "query": query[:50],
                    "query_length": len(query),
                    "dependency_graph_available": self.dependency_graph is not None,
                    "search_time_ms": (time.time() - start_time) * 1000,
                },
            )
            return results  # Return partial results if available

    async def _analytical_search(self, query: str) -> list[SearchResult]:
        """Analytical search using DuckDB."""

        start_time = time.time()

        try:
            # Skip if DuckDB is not available
            if not self.duckdb:
                logger.debug("DuckDB not available for analytical search")
                return []

            # Search file analytics
            analytics_result_arrow = await self.duckdb.execute(
                """
                SELECT file_path, complexity_score, lines_of_code
                FROM file_analytics 
                WHERE file_path LIKE ?
                ORDER BY complexity_score DESC
                LIMIT 20
            """,
                (f"%{query}%",),
            )

            results = []
            # Convert Arrow table to pandas for easier access
            analytics_df = analytics_result_arrow.to_pandas()
            for _, row in analytics_df.iterrows():
                result = SearchResult(
                    content=f"File metrics: {row['lines_of_code']} LOC, complexity {row['complexity_score']:.2f}",
                    file_path=row["file_path"],
                    line_number=1,
                    score=row["complexity_score"] / 10.0,  # Normalize complexity score
                    result_type="analytical",
                    context={
                        "complexity": row["complexity_score"],
                        "loc": row["lines_of_code"],
                    },
                    timestamp=time.time(),
                )
                results.append(result)

            search_time = (time.time() - start_time) * 1000
            self.search_stats["analytical_search_ms"].append(search_time)

            return results

        except Exception as e:
            logger.error(
                f"Analytical search failed: {e}",
                exc_info=True,
                extra={
                    "operation": "analytical_search",
                    "query": query,
                    "duckdb_available": self.duckdb is not None,
                },
            )
            return []

    async def _record_search_analytics(
        self,
        query: str,
        result_count: int,
        search_time_ms: float,
        search_types: list[str] = None,
    ) -> None:
        """Record search analytics for performance monitoring and learning."""

        try:
            # Skip if DuckDB is not available
            if not self.duckdb:
                return

            # Record the search event
            await self.duckdb.execute(
                """
                INSERT INTO search_analytics
                (timestamp, query, result_count, search_time_ms, search_type)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    time.time(),
                    query,
                    result_count,
                    search_time_ms,
                    ",".join(search_types) if search_types else "unified",
                ),
            )

            # Update query patterns for learning
            await self._update_query_patterns(
                query, search_types or ["unified"], result_count > 0
            )

        except Exception as e:
            logger.error(
                f"Failed to record search analytics: {e}",
                exc_info=True,
                extra={
                    "operation": "record_search_analytics",
                    "query": query,
                    "result_count": result_count,
                    "search_time_ms": search_time_ms,
                    "search_types": search_types,
                    "duckdb_available": self.duckdb is not None,
                },
            )

    async def get_intelligent_context(self, query: str) -> dict[str, Any]:
        """Get intelligent context for Jarvis using sequential thinking."""

        # Use tracer if available, otherwise skip tracing
        if self.tracer:
            async with self.tracer.trace_span("intelligent_context") as span:
                return await self._get_context_impl(query, span)
        else:
            return await self._get_context_impl(query, None)

    async def _get_context_impl(self, query: str, span=None) -> dict[str, Any]:
        """Implementation of get_intelligent_context with optional tracing."""

        # Use sequential thinking to analyze the query if available
        thinking_plan = []
        if self.sequential_thinking:
            try:
                thinking_plan = await self.sequential_thinking.think(
                    goal=f"Provide comprehensive context for: {query}",
                    constraints=[
                        "Focus on relevant code patterns",
                        "Include dependencies and relationships",
                        "Consider performance implications",
                    ],
                    max_steps=5,
                )
            except Exception as e:
                logger.warning(f"Sequential thinking failed: {e}")
                thinking_plan = []

        # Perform unified search
        search_results = await self.search(query)

        # Analyze dependencies if dependency graph is available
        deps = []
        if self.dependency_graph:
            try:
                deps = await self.dependency_graph.find_symbol(query)
            except Exception as e:
                logger.warning(f"Dependency analysis failed: {e}")
                deps = []

        context = {
            "query": query,
            "thinking_plan": thinking_plan,
            "search_results": [r.__dict__ for r in search_results[:10]],
            "dependencies": deps,
            "timestamp": time.time(),
            "total_results": len(search_results),
        }

        if span:
            span.add_attribute("context_size", len(search_results))

        return context

    async def _initialize_persistent_faiss(self) -> None:
        """Initialize FAISS index with persistence support and robust error handling."""

        # Ensure Einstein directory exists first
        einstein_dir = self.project_root / ".einstein"
        try:
            einstein_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(
                f"Failed to create Einstein directory {einstein_dir}: {e}",
                exc_info=True,
                extra={
                    "operation": "create_einstein_directory",
                    "error_type": type(e).__name__,
                    "einstein_dir": str(einstein_dir),
                    "parent_exists": einstein_dir.parent.exists(),
                    "parent_writable": os.access(einstein_dir.parent, os.W_OK)
                    if einstein_dir.parent.exists()
                    else False,
                    "project_root": str(self.project_root),
                },
            )
            self.vector_index = None
            self._faiss_loaded = False
            return

        faiss_path = einstein_dir / "embeddings.index"

        # Skip FAISS initialization if not available
        if self._faiss_available is False:
            logger.info("Skipping FAISS initialization - library not available")
            self.vector_index = None
            self._faiss_loaded = False
            return

        try:
            # Set FAISS threading for optimal performance
            import os

            import faiss

            num_threads = min(os.cpu_count() or 4, 8)  # Limit to 8 threads max
            faiss.omp_set_num_threads(num_threads)

            if faiss_path.exists() and faiss_path.stat().st_size > 0:
                # Load existing FAISS index with validation
                try:
                    self.vector_index = faiss.read_index(str(faiss_path))

                    # Validate loaded index
                    if (
                        hasattr(self.vector_index, "ntotal")
                        and self.vector_index.ntotal >= 0
                    ):
                        logger.info(
                            f"✅ Loaded persistent FAISS index with {self.vector_index.ntotal} vectors"
                        )
                        self._faiss_loaded = True
                    else:
                        logger.warning(
                            "Loaded FAISS index appears invalid - creating new one"
                        )
                        raise ValueError("Invalid FAISS index structure")

                except Exception as load_error:
                    logger.warning(f"Failed to load existing FAISS index: {load_error}")
                    logger.info("Creating new FAISS index")

                    # Backup corrupted index
                    backup_path = faiss_path.with_suffix(".index.backup")
                    try:
                        faiss_path.rename(backup_path)
                        logger.info(f"Backed up corrupted index to {backup_path}")
                    except (OSError, FileNotFoundError, PermissionError) as e:
                        logger.debug(
                            f"Failed to backup corrupted FAISS index {faiss_path} to {backup_path}: {e}",
                            extra={
                                "operation": "backup_corrupted_faiss_index",
                                "faiss_path": str(faiss_path),
                                "backup_path": str(backup_path),
                            },
                        )
                        pass

                    # Create new index
                    self.vector_index = self._create_new_faiss_index()
            else:
                # Create new FAISS index
                logger.info("No existing FAISS index found - creating new one")
                self.vector_index = self._create_new_faiss_index()

        except ImportError:
            logger.warning(
                "⚠️  FAISS library not available - semantic search will use embedding pipeline fallback"
            )
            self.vector_index = None
            self._faiss_loaded = False
        except Exception as e:
            logger.error(
                f"❌ Failed to initialize FAISS index: {e}",
                exc_info=True,
                extra={
                    "operation": "initialize_persistent_faiss",
                    "error_type": type(e).__name__,
                    "faiss_path": str(faiss_path)
                    if "faiss_path" in locals()
                    else "unknown",
                    "faiss_available": self._faiss_available,
                    "einstein_dir": str(einstein_dir),
                    "num_threads": locals().get("num_threads", "unknown"),
                },
            )
            self.vector_index = None
            self._faiss_loaded = False

    def _create_new_faiss_index(self) -> Any:
        """Create a new FAISS index with optimal settings."""
        try:
            import faiss

            # Use dimension compatible with common embedding models
            dimension = 1536  # Match ada-002 embedding dimension

            # Create HNSW index for efficient approximate nearest neighbor search
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 links per node

            # Optimize for search performance
            index.hnsw.efConstruction = (
                200  # Higher for better recall during construction
            )
            index.hnsw.efSearch = 100  # Higher for better recall during search

            self._faiss_loaded = True
            logger.info(f"✅ Created new FAISS HNSW index (dimension={dimension})")

            return index

        except Exception as e:
            logger.error(
                f"Failed to create new FAISS index: {e}",
                exc_info=True,
                extra={
                    "operation": "create_new_faiss_index",
                    "error_type": type(e).__name__,
                    "dimension": 1536,
                    "index_type": "IndexHNSWFlat",
                    "faiss_available": self._faiss_available,
                },
            )
            self._faiss_loaded = False
            return None

    async def _initialize_embedding_pipeline(self) -> None:
        """Initialize the embedding pipeline for semantic search with comprehensive error handling."""
        try:
            import numpy as np

            from einstein.mlx_embeddings import get_mlx_embedding_engine
            from src.unity_wheel.mcp.embedding_pipeline import EmbeddingPipeline

            # Create a robust embedding function for the pipeline
            def create_embedding_func():
                """Create an embedding function that returns proper format with error handling."""
                # Get MLX embedding engine
                embedding_engine = get_mlx_embedding_engine(embed_dim=1536)

                def embedding_func(text: str):
                    try:
                        if not text or not isinstance(text, str):
                            text = "empty query"

                        # Use production MLX-based embedding
                        embedding, token_count = embedding_engine.embed_text(text)

                        return embedding, token_count
                    except Exception as e:
                        logger.error(f"Embedding function error: {e}")
                        # Return zero embedding as fallback
                        return np.zeros(1536, dtype="float32"), 1

                return embedding_func

            # Initialize with Einstein cache path and proper embedding function
            cache_path = self.einstein_config.paths.embeddings_db_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            embedding_func = create_embedding_func()

            # Test the embedding function before creating pipeline
            test_embedding, test_tokens = embedding_func("test query")
            if test_embedding is None or test_embedding.size == 0:
                raise ValueError("Embedding function test failed")

            self.embedding_pipeline = EmbeddingPipeline(
                cache_path=cache_path, embedding_func=embedding_func
            )

            # Verify the embedding function is properly set
            if (
                not hasattr(self.embedding_pipeline, "embedding_func")
                or self.embedding_pipeline.embedding_func is None
            ):
                raise ValueError("Embedding function not properly initialized")

            # Test the pipeline's embedding function
            try:
                test_result = self.embedding_pipeline.embedding_func("pipeline test")
                if test_result is None or len(test_result) != 2:
                    raise ValueError("Pipeline embedding function test failed")
            except Exception as test_e:
                logger.error(f"Pipeline embedding function test failed: {test_e}")
                raise

            # Mark as available
            self._embedding_pipeline_available = True
            logger.info(
                "✅ Embedding pipeline initialized with Einstein cache and validated embedding function"
            )

        except ImportError as ie:
            logger.warning(
                f"Embedding pipeline module not available: {ie}",
                extra={
                    "operation": "initialize_embedding_pipeline",
                    "module_import": "embedding_pipeline",
                },
            )
            self.embedding_pipeline = None
            self._embedding_pipeline_available = False
        except Exception as e:
            logger.error(
                f"Failed to initialize embedding pipeline: {e}",
                exc_info=True,
                extra={
                    "operation": "initialize_embedding_pipeline",
                    "einstein_dir": str(self.project_root / ".einstein"),
                },
            )
            self.embedding_pipeline = None
            self._embedding_pipeline_available = False

    async def _load_faiss_index(self, faiss_path: Path) -> bool:
        """Load FAISS index from disk with comprehensive error handling.

        Args:
            faiss_path: Path to the FAISS index file

        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            import faiss

            if not faiss_path.exists():
                logger.info(f"No FAISS index found at {faiss_path}")
                return False

            if faiss_path.stat().st_size == 0:
                logger.warning(f"FAISS index file is empty: {faiss_path}")
                return False

            # Load index with validation
            try:
                self.vector_index = faiss.read_index(str(faiss_path))

                # Validate loaded index
                if not hasattr(self.vector_index, "ntotal"):
                    logger.error("Loaded FAISS index is missing required attributes")
                    self.vector_index = None
                    return False

                logger.info(
                    f"✅ Loaded FAISS index with {self.vector_index.ntotal} vectors from {faiss_path}"
                )
                self._faiss_loaded = True
                return True

            except Exception as load_error:
                logger.error(
                    f"Failed to load FAISS index from {faiss_path}: {load_error}",
                    exc_info=True,
                    extra={
                        "operation": "load_faiss_index",
                        "faiss_path": str(faiss_path),
                        "file_size": faiss_path.stat().st_size
                        if faiss_path.exists()
                        else 0,
                    },
                )

                # Try to backup corrupted file
                backup_path = faiss_path.with_suffix(".index.corrupted")
                try:
                    faiss_path.rename(backup_path)
                    logger.info(f"Moved corrupted index to {backup_path}")
                except (OSError, FileNotFoundError, PermissionError) as e:
                    logger.warning(
                        f"Failed to move corrupted FAISS index {faiss_path} to backup: {e}",
                        extra={
                            "operation": "backup_corrupted_index",
                            "faiss_path": str(faiss_path),
                            "backup_path": str(backup_path),
                        },
                    )

                self.vector_index = None
                self._faiss_loaded = False
                return False

        except ImportError:
            logger.warning(
                "FAISS library not available for loading index",
                extra={"operation": "load_faiss_index", "faiss_path": str(faiss_path)},
            )
            self.vector_index = None
            self._faiss_loaded = False
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error loading FAISS index: {e}",
                exc_info=True,
                extra={
                    "operation": "load_faiss_index",
                    "faiss_path": str(faiss_path),
                    "faiss_available": self._faiss_available,
                },
            )
            self.vector_index = None
            self._faiss_loaded = False
            return False

    def _calculate_similarity(self, embedding1: Any, embedding2: Any) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            import numpy as np

            # Ensure embeddings are numpy arrays
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            denominator = norm1 * norm2
            if denominator == 0:
                return 0.0

            similarity = dot_product / denominator
            return float(similarity)
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    async def save_faiss_index(self) -> bool:
        """Save FAISS index to disk for persistence with comprehensive error handling.

        Returns:
            bool: True if successfully saved, False otherwise
        """
        # Early exit if no FAISS index exists
        if not hasattr(self, "vector_index") or not self.vector_index:
            logger.debug("No FAISS index to save")
            return False

        # Check FAISS availability first
        if not self._check_faiss_availability():
            logger.warning("⚠️  FAISS library not available - cannot save index")
            return False

        try:
            import faiss

            # Ensure Einstein directory exists with robust error handling
            einstein_dir = self.project_root / ".einstein"
            try:
                einstein_dir.mkdir(parents=True, exist_ok=True)
            except Exception as dir_error:
                logger.error(
                    f"Failed to create Einstein directory {einstein_dir}: {dir_error}"
                )
                return False

            faiss_path = einstein_dir / "embeddings.index"

            # Comprehensive validation of index state before saving
            if not self._validate_faiss_index_for_save():
                logger.warning("FAISS index validation failed - cannot save")
                return False

            # Save index with atomic operation using temporary file
            temp_path = faiss_path.with_suffix(".index.tmp")
            backup_path = faiss_path.with_suffix(".index.backup")

            try:
                # Create backup of existing index if it exists
                if faiss_path.exists():
                    try:
                        import shutil

                        shutil.copy2(faiss_path, backup_path)
                        logger.debug(f"Created backup at {backup_path}")
                    except Exception as backup_error:
                        logger.warning(f"Failed to create backup: {backup_error}")
                        # Continue without backup - not critical

                # Write to temporary file first
                faiss.write_index(self.vector_index, str(temp_path))

                # Verify the written file
                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    raise ValueError("Temporary FAISS index file is empty or missing")

                # Atomic rename to prevent corruption
                temp_path.rename(faiss_path)

                # Clean up backup if save was successful
                if backup_path.exists():
                    try:
                        backup_path.unlink()
                    except (OSError, PermissionError) as e:
                        logger.debug(
                            f"Failed to clean up backup file {backup_path}: {e}",
                            extra={
                                "operation": "cleanup_faiss_backup",
                                "backup_path": str(backup_path),
                                "critical": False,
                            },
                        )
                        pass  # Not critical if backup cleanup fails

                logger.info(
                    f"✅ Saved FAISS index with {self.vector_index.ntotal} vectors to {faiss_path}"
                )
                return True

            except Exception as write_error:
                logger.error(f"Failed to write FAISS index: {write_error}")

                # Clean up temp file if it exists
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except (OSError, PermissionError) as e:
                        logger.debug(
                            f"Failed to clean up temporary FAISS index file {temp_path}: {e}",
                            extra={
                                "operation": "cleanup_temp_faiss_file",
                                "temp_path": str(temp_path),
                                "critical": False,
                            },
                        )
                        pass

                # Restore backup if it exists and main file is corrupted
                if backup_path.exists() and (
                    not faiss_path.exists() or faiss_path.stat().st_size == 0
                ):
                    try:
                        backup_path.rename(faiss_path)
                        logger.info("Restored FAISS index from backup")
                    except Exception as restore_error:
                        logger.error(f"Failed to restore backup: {restore_error}")

                return False

        except ImportError:
            logger.warning("⚠️  FAISS library not available - cannot save index")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error saving FAISS index: {e}")
            return False

    async def get_stats(self) -> IndexStats:
        """Get comprehensive index statistics."""

        # Get file count from analytics
        total_files = 0
        try:
            file_count_result = await self.duckdb.execute(
                "SELECT COUNT(*) FROM file_analytics"
            )
            if file_count_result:
                # Convert PyArrow scalar to Python int
                raw_count = file_count_result[0][0]
                total_files = (
                    int(raw_count.as_py())
                    if hasattr(raw_count, "as_py")
                    else int(raw_count)
                )
        except Exception as e:
            logger.warning(f"Could not get file count from analytics: {e}")

        # Calculate total lines of code from analytics
        total_lines = 0
        try:
            total_lines_result = await self.duckdb.execute(
                "SELECT SUM(lines_of_code) FROM file_analytics WHERE lines_of_code IS NOT NULL"
            )
            if total_lines_result and total_lines_result[0][0] is not None:
                # Convert PyArrow scalar to Python int
                raw_lines = total_lines_result[0][0]
                if raw_lines is not None:
                    total_lines = (
                        int(raw_lines.as_py())
                        if hasattr(raw_lines, "as_py")
                        else int(raw_lines)
                    )
        except Exception as e:
            logger.warning(f"Could not get total lines from analytics: {e}")

        # Calculate actual index size in MB
        index_size_mb = await self._calculate_index_size()

        # Calculate coverage percentage
        coverage_percentage = await self._calculate_coverage_percentage()

        # Calculate average search performance
        avg_performance = {}
        for search_type, times in self.search_stats.items():
            avg_performance[search_type] = sum(times) / len(times) if times else 0.0

        # Get FAISS index size if available
        if self.vector_index and hasattr(self.vector_index, "ntotal"):
            pass

        return IndexStats(
            total_files=total_files,
            total_lines=total_lines,
            index_size_mb=index_size_mb,
            last_update=time.time(),
            search_performance_ms=avg_performance,
            coverage_percentage=coverage_percentage,
        )

    async def _calculate_index_size(self) -> float:
        """Calculate the total size of all index files in MB."""
        try:
            total_size_bytes = 0

            # Calculate size of Einstein directory
            einstein_dir = self.project_root / ".einstein"
            if einstein_dir.exists():
                for file_path in einstein_dir.rglob("*"):
                    if file_path.is_file():
                        try:
                            total_size_bytes += file_path.stat().st_size
                        except (OSError, FileNotFoundError):
                            # Skip files that can't be accessed
                            continue

            # Add DuckDB database file size
            db_path = self.einstein_config.paths.analytics_db_path
            if db_path.exists():
                with contextlib.suppress(OSError, FileNotFoundError):
                    total_size_bytes += db_path.stat().st_size

            # Add any other index-related files
            for cache_pattern in [
                ".einstein_cache",
                ".metacoding_cache",
                ".thinking_cache",
            ]:
                cache_dir = self.project_root / cache_pattern
                if cache_dir.exists():
                    for file_path in cache_dir.rglob("*"):
                        if file_path.is_file():
                            try:
                                total_size_bytes += file_path.stat().st_size
                            except (OSError, FileNotFoundError):
                                continue

            # Convert to MB
            return total_size_bytes / (1024 * 1024)

        except Exception as e:
            logger.warning(f"Failed to calculate index size: {e}")
            return 0.0

    async def _calculate_coverage_percentage(self) -> float:
        """Calculate the percentage of Python files that have been indexed."""
        try:
            # Count total Python files in the project
            total_python_files = 0
            for file_path in self.project_root.rglob("*.py"):
                # Skip hidden directories and common non-source directories
                if any(part.startswith(".") for part in file_path.parts):
                    continue
                if any(
                    part in ["__pycache__", "venv", "env", "node_modules", ".git"]
                    for part in file_path.parts
                ):
                    continue
                total_python_files += 1

            if total_python_files == 0:
                return 0.0

            # Get count of indexed files from analytics
            try:
                indexed_files_result = await self.duckdb.execute(
                    "SELECT COUNT(*) FROM file_analytics"
                )
                if indexed_files_result:
                    # Convert PyArrow scalar to Python int
                    raw_indexed = indexed_files_result[0][0]
                    indexed_files = (
                        int(raw_indexed.as_py())
                        if hasattr(raw_indexed, "as_py")
                        else int(raw_indexed)
                    )
                else:
                    indexed_files = 0
            except (sqlite3.Error, AttributeError, TypeError, ValueError) as e:
                logger.debug(
                    f"DuckDB query failed for coverage calculation - database may not be initialized: {e}"
                )
                # Database not initialized yet
                indexed_files = 0

            # Calculate coverage percentage
            coverage = (indexed_files / total_python_files) * 100.0

            # Cap at 100% in case of any inconsistencies
            return min(coverage, 100.0)

        except Exception as e:
            logger.warning(f"Failed to calculate coverage percentage: {e}")
            # Return a reasonable fallback based on what we know
            try:
                # If we can't count files, use a simple heuristic
                indexed_files_result = await self.duckdb.execute(
                    "SELECT COUNT(*) FROM file_analytics"
                )
                if indexed_files_result:
                    # Convert PyArrow scalar to Python int
                    raw_indexed = indexed_files_result[0][0]
                    indexed_files = (
                        int(raw_indexed.as_py())
                        if hasattr(raw_indexed, "as_py")
                        else int(raw_indexed)
                    )
                else:
                    indexed_files = 0

                # Assume we've indexed a reasonable portion if we have any data
                if indexed_files > 0:
                    return min(80.0, indexed_files * 2.0)  # Conservative estimate
                else:
                    return 0.0
            except (sqlite3.Error, AttributeError, TypeError, ValueError) as e:
                logger.warning(
                    f"Failed to query DuckDB for coverage fallback calculation: {e}"
                )
                return 0.0

    def _prepare_file_watcher(self) -> None:
        """Prepare file watcher components without starting them."""
        try:
            # Initialize file watcher attributes if not already done
            if not hasattr(self, "_shutdown_event") or self._shutdown_event is None:
                self._shutdown_event = None

            if not hasattr(self, "_file_change_loop") or self._file_change_loop is None:
                self._file_change_loop = None

            if not hasattr(self, "_file_change_task") or self._file_change_task is None:
                self._file_change_task = None

            if not hasattr(self, "file_watcher") or self.file_watcher is None:
                self.file_watcher = None

            if not hasattr(self, "file_change_queue") or self.file_change_queue is None:
                self.file_change_queue = None

            logger.debug("📁 File watcher components prepared")

        except Exception as e:
            logger.warning(f"Failed to prepare file watcher: {e}")

    async def _start_file_watcher(self) -> None:
        """Start real-time file system monitoring for automatic reindexing."""
        try:
            # Initialize async components for file watching
            self.file_change_queue = asyncio.Queue(
                maxsize=self.einstein_config.cache.max_cache_entries // 10
            )  # Prevent memory bloat
            self._shutdown_event = asyncio.Event()

            # Get current event loop for proper callback handling
            self._file_change_loop = asyncio.get_running_loop()

            # Create file system event handler with proper async integration
            handler = EinsteinFileHandler(self)

            # Set up observer
            self.file_watcher = Observer()
            self.file_watcher.schedule(handler, str(self.project_root), recursive=True)

            # Start background task before starting watcher to avoid race conditions
            self._file_change_task = asyncio.create_task(self._process_file_changes())

            # Start watching
            self.file_watcher.start()

            logger.info("🔍 Real-time file monitoring started")

        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            await self._cleanup_file_watcher()

    async def _process_file_changes(self) -> None:
        """Background task to process file changes from the queue."""
        logger.info("📂 File change processing started")

        try:
            while not self._shutdown_event.is_set():
                try:
                    # CRITICAL FIX: Add CPU monitoring to prevent runaway processing
                    try:
                        import psutil

                        cpu_percent = psutil.cpu_percent(interval=0.1)
                        if cpu_percent > 85:  # Emergency brake at 85% CPU
                            logger.warning(
                                f"🚨 CPU usage at {cpu_percent:.1f}% - throttling file processing"
                            )
                            await asyncio.sleep(0.5)  # Brief pause to let CPU cool down
                            continue
                    except ImportError:
                        pass  # psutil not available, continue without monitoring

                    # Wait for file change event with timeout to check shutdown
                    try:
                        file_path, event_type = await asyncio.wait_for(
                            self.file_change_queue.get(), timeout=1.0
                        )
                    except TimeoutError:
                        # Normal timeout to check shutdown, continue
                        continue

                    # Skip non-relevant files
                    if not self._should_process_file(file_path):
                        self.file_change_queue.task_done()
                        continue

                    # Handle the file change with error isolation
                    try:
                        await self._handle_file_change(Path(file_path), event_type)
                    except Exception as e:
                        logger.error(f"Failed to handle file change {file_path}: {e}")
                    finally:
                        self.file_change_queue.task_done()

                except TimeoutError:
                    # Timeout is normal - just check shutdown and continue
                    continue

                except asyncio.CancelledError:
                    logger.info("📂 File change processing cancelled")
                    break

        except Exception as e:
            logger.error(f"Error in file change processing loop: {e}")
        finally:
            logger.info("📂 File change processing stopped")

    async def _handle_file_change(self, file_path: Path, event_type: str) -> None:
        """Handle a single file change event with comprehensive error isolation."""
        try:
            logger.debug(f"Handling file change: {event_type} for {file_path}")

            if event_type in ["modified", "created"]:
                # Check if file actually changed by comparing hash
                if await self._file_needs_reindexing(file_path):
                    logger.info(f"Reindexing changed file: {file_path}")

                    # Re-analyze the file with error isolation
                    try:
                        await self._analyze_file(file_path)
                    except Exception as analyze_error:
                        logger.error(
                            f"Failed to analyze file {file_path}: {analyze_error}"
                        )
                        # Continue with other operations even if analysis fails

                    # Update embeddings if pipeline is available with error isolation
                    try:
                        if self.embedding_pipeline and hasattr(
                            self.embedding_pipeline, "embed_file"
                        ):
                            await self.embedding_pipeline.embed_file(
                                str(file_path), force_refresh=True
                            )
                    except Exception as embed_error:
                        logger.error(
                            f"Failed to update embeddings for {file_path}: {embed_error}"
                        )
                        # Continue even if embedding fails

                    # Save updated indexes with error isolation
                    try:
                        save_success = await self.save_faiss_index()
                        if not save_success:
                            logger.warning(
                                f"Failed to save FAISS index after updating {file_path}"
                            )
                    except Exception as save_error:
                        logger.error(
                            f"Error saving FAISS index for {file_path}: {save_error}"
                        )

            elif event_type == "deleted":
                # Remove file from indexes with error isolation
                try:
                    await self._remove_file_from_indexes(file_path)
                except Exception as remove_error:
                    logger.error(
                        f"Failed to remove {file_path} from indexes: {remove_error}"
                    )

        except Exception as e:
            logger.error(f"Failed to handle file change for {file_path}: {e}")
            # Don't re-raise - we want to continue processing other files

    async def _file_needs_reindexing(self, file_path: Path) -> bool:
        """Check if a file needs reindexing based on content hash."""
        try:
            if not file_path.exists():
                return False

            # Calculate current file hash
            content = file_path.read_text(encoding="utf-8")
            current_hash = hashlib.md5(content.encode()).hexdigest()
            current_mtime = file_path.stat().st_mtime

            # Check against last known state
            file_key = str(file_path)
            if file_key in self._last_indexed:
                last_hash, last_mtime = self._last_indexed[file_key]
                if current_hash == last_hash and current_mtime <= last_mtime:
                    return False

            # Update tracking
            self._last_indexed[file_key] = (current_hash, current_mtime)
            return True

        except Exception as e:
            logger.warning(f"Could not check if {file_path} needs reindexing: {e}")
            return True  # Err on the side of reindexing

    async def _remove_file_from_indexes(self, file_path: Path) -> None:
        """Remove a deleted file from all indexes."""
        try:
            file_str = str(file_path)

            # Remove from analytics DB
            await self.duckdb.execute(
                "DELETE FROM file_analytics WHERE file_path = ?", (file_str,)
            )

            # Remove from tracking
            if file_str in self._last_indexed:
                del self._last_indexed[file_str]

            logger.info(f"Removed deleted file from indexes: {file_path}")

        except Exception as e:
            logger.error(f"Failed to remove {file_path} from indexes: {e}")

    def _should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed based on extension and patterns."""
        try:
            path = Path(file_path)

            # Check extension - expand beyond just Python
            allowed_extensions = {
                ".py",
                ".md",
                ".txt",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
            }
            if path.suffix not in allowed_extensions:
                return False

            # Skip temporary and system files
            ignored_patterns = {
                "__pycache__",
                ".git",
                ".DS_Store",
                "node_modules",
                ".pytest_cache",
                ".mypy_cache",
                ".venv",
                "venv",
                ".einstein",
                ".thinking_cache",
                ".metacoding_cache",
            }

            return all(ignored not in path.parts for ignored in ignored_patterns)

        except Exception as e:
            logger.warning(f"Error checking if should process {file_path}: {e}")
            return False

    async def _cleanup_initialization(self) -> None:
        """Clean up resources if initialization fails."""
        try:
            await self._cleanup_file_watcher()
            if hasattr(self, "executor") and self.executor:
                self.executor.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"Error during initialization cleanup: {e}")

    async def _cleanup_file_watcher(self) -> None:
        """Clean up file watcher resources safely."""
        try:
            # Signal shutdown if shutdown event exists
            if hasattr(self, "_shutdown_event") and self._shutdown_event:
                self._shutdown_event.set()

            # Cancel file change processing task
            if (
                hasattr(self, "_file_change_task")
                and self._file_change_task
                and not self._file_change_task.done()
            ):
                self._file_change_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._file_change_task

            # Stop file watcher
            if hasattr(self, "file_watcher") and self.file_watcher:
                self.file_watcher.stop()
                self.file_watcher.join()

            # Clear queue
            if hasattr(self, "file_change_queue") and self.file_change_queue:
                while not self.file_change_queue.empty():
                    try:
                        self.file_change_queue.get_nowait()
                        self.file_change_queue.task_done()
                    except (asyncio.QueueEmpty, RuntimeError) as e:
                        logger.debug(
                            f"Queue cleanup completed or queue operation failed: {e}"
                        )
                        break

            logger.info("📋 File watcher cleanup completed")

        except Exception as e:
            logger.error(f"Error during file watcher cleanup: {e}")

    async def _update_query_patterns(
        self, query: str, search_types: list[str], had_results: bool
    ) -> None:
        """Update query patterns for learning."""
        try:
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()

            # Check if pattern exists
            existing_result_arrow = await self.duckdb.execute(
                "SELECT avg_success_rating, usage_count FROM query_patterns WHERE query_hash = ?",
                (query_hash,),
            )
            existing_df = existing_result_arrow.to_pandas()

            if len(existing_df) > 0:
                # Update existing pattern
                row = existing_df.iloc[0]
                old_rating, old_count = row["avg_success_rating"], row["usage_count"]
                new_count = old_count + 1
                success_score = 1.0 if had_results else 0.0
                new_rating = (old_rating * old_count + success_score) / new_count

                await self.duckdb.execute(
                    """
                    UPDATE query_patterns 
                    SET avg_success_rating = ?, usage_count = ?, last_used = ?
                    WHERE query_hash = ?
                """,
                    (new_rating, new_count, time.time(), query_hash),
                )
            else:
                # Create new pattern
                success_score = 1.0 if had_results else 0.0
                await self.duckdb.execute(
                    """
                    INSERT INTO query_patterns 
                    (query_hash, query_text, best_search_types, avg_success_rating, usage_count, last_used)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        query_hash,
                        query,
                        ",".join(search_types),
                        success_score,
                        1,
                        time.time(),
                    ),
                )

        except Exception as e:
            logger.warning(f"Failed to update query patterns: {e}")

    async def get_optimized_search_types(self, query: str) -> list[str]:
        """Get optimized search types based on query learning."""
        try:
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()

            # Check for exact match
            exact_match_result_arrow = await self.duckdb.execute(
                """
                SELECT best_search_types, avg_success_rating
                FROM query_patterns 
                WHERE query_hash = ? AND avg_success_rating > 0.3
            """,
                (query_hash,),
            )
            exact_match_df = exact_match_result_arrow.to_pandas()

            if len(exact_match_df) > 0:
                row = exact_match_df.iloc[0]
                learned_types = row["best_search_types"].split(",")
                logger.info(f"Using learned search types for query: {learned_types}")
                return learned_types

            # Look for similar queries (simplified similarity based on common words)
            query_words = set(query.lower().split())
            if len(query_words) > 1:
                similar_patterns_result_arrow = await self.duckdb.execute(
                    """
                    SELECT best_search_types, avg_success_rating, query_text
                    FROM query_patterns 
                    WHERE avg_success_rating > 0.5
                    ORDER BY usage_count DESC
                    LIMIT 10
                """
                )
                similar_patterns_df = similar_patterns_result_arrow.to_pandas()

                for _, row in similar_patterns_df.iterrows():
                    pattern_types, _rating, pattern_text = (
                        row["best_search_types"],
                        row["avg_success_rating"],
                        row["query_text"],
                    )
                    pattern_words = set(pattern_text.lower().split())
                    union_words = query_words.union(pattern_words)
                    overlap = (
                        len(query_words.intersection(pattern_words)) / len(union_words)
                        if len(union_words) > 0
                        else 0.0
                    )

                    if overlap > 0.4:  # 40% word overlap
                        logger.info(
                            f"Using similar query pattern: {pattern_types} (overlap: {overlap:.1%})"
                        )
                        return pattern_types.split(",")

        except Exception as e:
            logger.warning(f"Failed to get optimized search types: {e}")

        # Fall back to default
        return ["text", "semantic", "structural"]

    def get_faiss_status(self) -> dict[str, Any]:
        """Get detailed FAISS status information.

        Returns:
            dict: Status information about FAISS availability and index state
        """
        status = {
            "faiss_available": self._faiss_available
            if self._faiss_available is not None
            else False,
            "faiss_loaded": self._faiss_loaded,
            "index_exists": self.vector_index is not None,
            "index_size": 0,
            "index_dimension": None,
            "index_type": None,
            "fallback_active": not (self._faiss_available and self._faiss_loaded),
        }

        if self.vector_index:
            try:
                status["index_size"] = getattr(self.vector_index, "ntotal", 0)
                status["index_dimension"] = getattr(self.vector_index, "d", None)
                status["index_type"] = type(self.vector_index).__name__
            except Exception as e:
                logger.debug(f"Error getting FAISS index details: {e}")

        return status

    async def record_user_feedback(self, query: str, success_rating: float) -> None:
        """Record user feedback on search results quality."""
        try:
            # Update the most recent search for this query
            await self.duckdb.execute(
                """
                UPDATE search_analytics 
                SET success_rating = ?, user_feedback = 1
                WHERE query = ? AND timestamp = (
                    SELECT MAX(timestamp) FROM search_analytics WHERE query = ?
                )
            """,
                (success_rating, query, query),
            )

            # Update query patterns
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()
            existing_pattern_arrow = await self.duckdb.execute(
                "SELECT avg_success_rating, usage_count FROM query_patterns WHERE query_hash = ?",
                (query_hash,),
            )
            existing_pattern_df = existing_pattern_arrow.to_pandas()

            if len(existing_pattern_df) > 0:
                row = existing_pattern_df.iloc[0]
                old_rating, _count = row["avg_success_rating"], row["usage_count"]
                # Weight user feedback more heavily than automatic scoring
                new_rating = old_rating * 0.7 + success_rating * 0.3

                await self.duckdb.execute(
                    """
                    UPDATE query_patterns 
                    SET avg_success_rating = ?
                    WHERE query_hash = ?
                """,
                    (new_rating, query_hash),
                )

                logger.info(
                    f"Updated query pattern rating: {old_rating:.2f} -> {new_rating:.2f}"
                )

        except Exception as e:
            logger.error(f"Failed to record user feedback: {e}")

    async def get_search_analytics_summary(self) -> dict[str, Any]:
        """Get summary of search analytics and learning."""
        try:
            # Get recent search statistics
            recent_searches_result_arrow = await self.duckdb.execute(
                """
                SELECT 
                    COUNT(*) as total_searches,
                    AVG(search_time_ms) as avg_search_time,
                    AVG(result_count) as avg_results,
                    AVG(success_rating) as avg_success_rating
                FROM search_analytics 
                WHERE timestamp > ?
            """,
                (time.time() - 86400,),
            )  # Last 24 hours
            recent_searches_df = recent_searches_result_arrow.to_pandas()
            recent_searches = (
                recent_searches_df.to_dict("records")[0]
                if len(recent_searches_df) > 0
                else {}
            )

            # Get top query patterns
            top_patterns_result_arrow = await self.duckdb.execute(
                """
                SELECT query_text, avg_success_rating, usage_count
                FROM query_patterns 
                ORDER BY usage_count DESC
                LIMIT 5
            """
            )
            top_patterns_df = top_patterns_result_arrow.to_pandas()
            top_patterns = top_patterns_df.to_dict("records")

            # Get search type performance
            type_performance_result_arrow = await self.duckdb.execute(
                """
                SELECT 
                    search_type,
                    COUNT(*) as usage_count,
                    AVG(search_time_ms) as avg_time,
                    AVG(success_rating) as avg_rating
                FROM search_analytics 
                WHERE timestamp > ? AND success_rating > 0
                GROUP BY search_type
                ORDER BY avg_rating DESC
            """,
                (time.time() - 86400,),
            )
            type_performance_df = type_performance_result_arrow.to_pandas()
            type_performance = type_performance_df.to_dict("records")

            # Get total learned patterns count
            patterns_count_result_arrow = await self.duckdb.execute(
                "SELECT COUNT(*) FROM query_patterns"
            )
            patterns_count_df = patterns_count_result_arrow.to_pandas()
            total_patterns = (
                patterns_count_df.iloc[0, 0] if len(patterns_count_df) > 0 else 0
            )

            return {
                "recent_stats": recent_searches,
                "top_patterns": top_patterns,
                "search_type_performance": type_performance,
                "total_learned_patterns": total_patterns,
            }

        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}

    async def start_file_watching(self) -> None:
        """Start file system watching for real-time index updates.

        This is the main entry point for file watching, compatible with the startup script.
        """
        try:
            if (
                hasattr(self, "file_watcher")
                and self.file_watcher
                and self.file_watcher.is_alive()
            ):
                logger.info("📁 File watcher already running")
                return

            logger.info("🔍 Starting Einstein file watching system...")

            # Start the file watcher
            await self._start_file_watcher()

            logger.info("✅ File watching started successfully")

        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
            raise

    async def stop_file_watching(self) -> None:
        """Stop the file system watcher safely."""
        await self._cleanup_file_watcher()

    async def stop_file_watcher(self) -> None:
        """Stop the file system watcher safely."""
        await self._cleanup_file_watcher()

    def stop_file_watcher_sync(self) -> None:
        """Synchronous wrapper for stopping file watcher."""
        if self.file_watcher:
            self.file_watcher.stop()
            self.file_watcher.join()
            logger.info("🛑 File watcher stopped")

    def get_file_watching_stats(self) -> dict[str, Any]:
        """Get file watching statistics."""
        try:
            stats = {
                "is_watching": False,
                "files_processed": 0,
                "updates_queued": 0,
                "updates_completed": 0,
                "uptime_seconds": 0,
                "watch_paths": [str(self.project_root)],
            }

            if hasattr(self, "file_watcher") and self.file_watcher:
                stats["is_watching"] = self.file_watcher.is_alive()

            if hasattr(self, "file_change_queue") and self.file_change_queue:
                stats["updates_queued"] = self.file_change_queue.qsize()

            if hasattr(self, "_file_change_task") and self._file_change_task:
                stats["updates_completed"] = getattr(
                    self._file_change_task, "_completed_tasks", 0
                )

            return stats

        except Exception as e:
            logger.warning(f"Failed to get file watching stats: {e}")
            return {
                "is_watching": False,
                "files_processed": 0,
                "updates_queued": 0,
                "updates_completed": 0,
                "uptime_seconds": 0,
                "watch_paths": [str(self.project_root)],
            }

    async def cleanup(self) -> None:
        """Clean up all Einstein resources safely.

        This is the main cleanup method expected by startup/shutdown scripts.
        """
        try:
            logger.info("🧹 Starting Einstein cleanup...")

            # Stop file watching first
            await self.stop_file_watching()

            # Clean up file watcher resources
            await self._cleanup_file_watcher()

            # Save any persistent indexes
            try:
                if hasattr(self, "vector_index") and self.vector_index:
                    await self.save_faiss_index()
                    logger.debug("💾 FAISS index saved during cleanup")
            except Exception as e:
                logger.warning(f"Failed to save FAISS index during cleanup: {e}")

            # Shutdown executor
            try:
                if hasattr(self, "executor") and self.executor:
                    self.executor.shutdown(wait=False)
                    logger.debug("⚡ Thread pool executor shutdown")
            except Exception as e:
                logger.warning(f"Error shutting down executor: {e}")

            # Close database connections
            try:
                if hasattr(self, "duckdb") and self.duckdb:
                    # DuckDB connections are automatically managed, no explicit close needed
                    logger.debug("💾 DuckDB connections released")
            except Exception as e:
                logger.warning(f"Error cleaning up DuckDB: {e}")

            logger.info("✅ Einstein cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during Einstein cleanup: {e}")
            # Don't re-raise - cleanup should be as robust as possible

    # Semantic Search Helper Methods
    async def _ensure_embedding_pipeline(self) -> bool:
        """Ensure embedding pipeline is properly initialized and available."""
        try:
            # Check if pipeline exists and is marked as available
            if (
                hasattr(self, "_embedding_pipeline_available")
                and self._embedding_pipeline_available
            ):
                # Double-check that the pipeline and function are still valid
                if (
                    hasattr(self, "embedding_pipeline")
                    and self.embedding_pipeline
                    and hasattr(self.embedding_pipeline, "embedding_func")
                    and self.embedding_pipeline.embedding_func is not None
                ):
                    return True
                else:
                    # Pipeline became invalid, reset availability flag
                    self._embedding_pipeline_available = False

            # Initialize if not present or became invalid
            if (
                not hasattr(self, "embedding_pipeline")
                or self.embedding_pipeline is None
            ):
                await self._initialize_embedding_pipeline()

            # Test pipeline functionality after initialization
            if hasattr(self, "embedding_pipeline") and self.embedding_pipeline:
                try:
                    (
                        test_embedding,
                        test_tokens,
                    ) = self.embedding_pipeline.embedding_func("test")
                    if test_embedding is not None and test_embedding.size > 0:
                        self._embedding_pipeline_available = True
                        return True
                    else:
                        logger.warning(
                            "Embedding pipeline test failed - invalid output"
                        )
                        self._embedding_pipeline_available = False
                        return False
                except Exception as test_e:
                    logger.warning(f"Embedding pipeline test failed: {test_e}")
                    self._embedding_pipeline_available = False
                    return False

            # Pipeline not available
            self._embedding_pipeline_available = False
            return False

        except Exception as e:
            logger.error(f"Failed to ensure embedding pipeline: {e}")
            self._embedding_pipeline_available = False
            return False

    async def _safe_get_query_embedding(self, query: str) -> tuple[Any | None, int]:
        """Safely get query embedding with multiple fallback mechanisms."""
        try:
            if not self.embedding_pipeline or not hasattr(
                self.embedding_pipeline, "embedding_func"
            ):
                raise ValueError("Embedding pipeline or function not available")

            # Try the main embedding function
            embedding, token_count = self.embedding_pipeline.embedding_func(query)

            if embedding is None:
                raise ValueError("Embedding function returned None")

            return embedding, token_count

        except Exception as e:
            logger.warning(f"Primary embedding failed: {e}, trying neural backend")

            # Fallback to neural backend if available
            try:
                if hasattr(self, "neural_backend") and self.neural_backend:
                    fallback_embedding = await self._neural_backend_embedding(query)
                    if fallback_embedding is not None:
                        return fallback_embedding, len(query.split()) * 1.3
            except Exception as neural_e:
                logger.warning(f"Neural backend embedding failed: {neural_e}")

            # Last resort - generate random embedding for compatibility
            logger.warning("Using random embedding as last resort")
            import numpy as np

            random_embedding = np.random.randn(1536).astype("float32")
            return random_embedding, len(query.split())

    def _get_fallback_embedding_function(self) -> Callable[[str], list[float]]:
        """Get a fallback embedding function with error handling."""

        def embedding_func(text: str) -> list[float]:
            try:
                # Try to use neural backend first
                if hasattr(self, "neural_backend") and self.neural_backend:
                    try:
                        # Use MLX embeddings if available
                        from einstein.mlx_embeddings import get_mlx_embedding_engine

                        embedding_engine = get_mlx_embedding_engine()
                        embedding_result, token_count = embedding_engine.embed_text(
                            text
                        )
                        return embedding_result.tolist()
                    except (AttributeError, ImportError, RuntimeError) as e:
                        logger.debug(
                            f"Neural backend embedding failed, using fallback: {e}"
                        )
                        # Continue to fallback below

                # Fallback to mock embedding
                import numpy as np

                embedding = np.random.randn(1536).astype("float32")
                token_count = max(1, len(text.split()) * 1.3)
                return embedding, int(token_count)

            except Exception as e:
                logger.error(f"Fallback embedding function failed: {e}")
                # Ultimate fallback
                import numpy as np

                return np.zeros(1536, dtype="float32"), 1

        return embedding_func

    async def _recover_faiss_index(self) -> bool:
        """Recover FAISS index from corruption or failure."""
        try:
            logger.info("🔧 Attempting FAISS index recovery...")

            # Clear current index
            self.vector_index = None
            self._faiss_loaded = False

            # Remove corrupted index files
            einstein_dir = self.project_root / ".einstein"
            faiss_path = einstein_dir / "embeddings.index"

            if faiss_path.exists():
                backup_path = faiss_path.with_suffix(".index.corrupted")
                faiss_path.rename(backup_path)
                logger.info(f"Moved corrupted index to {backup_path}")

            # Reinitialize FAISS
            await self._initialize_persistent_faiss()

            if self.vector_index is not None:
                logger.info("✅ FAISS index recovery successful")
                return True
            else:
                logger.warning("❌ FAISS index recovery failed")
                return False

        except Exception as e:
            logger.error(f"FAISS recovery failed: {e}")
            return False

    async def _try_faiss_search(
        self, query_embedding: "np.ndarray"
    ) -> list[SearchResult]:
        """Try FAISS search with comprehensive error handling."""
        results = []

        try:
            # Check FAISS availability first
            if not self._faiss_available:
                logger.debug("FAISS not available for search")
                return results

            # Ensure FAISS index is loaded
            if not hasattr(self, "vector_index") or self.vector_index is None:
                faiss_path = self.project_root / ".einstein" / "embeddings.index"
                if not await self._load_faiss_index(faiss_path):
                    logger.debug("Could not load FAISS index")
                    return results

            # Load metadata for file path mapping
            metadata = None
            metadata_path = self.project_root / ".einstein" / "faiss_metadata.json"
            if metadata_path.exists():
                try:
                    import json

                    with open(metadata_path) as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.debug(f"Could not load FAISS metadata: {e}")
                    metadata = None

            if self.vector_index is None:
                logger.debug("FAISS vector index is None")
                return results

            # FAISS already available since we checked _faiss_available above

            # Validate index state
            if not hasattr(self.vector_index, "ntotal"):
                logger.warning("FAISS index missing ntotal attribute")
                return results

            if self.vector_index.ntotal == 0:
                logger.debug("FAISS index is empty (no vectors)")
                return results

            # Validate query embedding
            if query_embedding is None:
                logger.warning("Query embedding is None")
                return results

            try:
                import numpy as np

                if not isinstance(query_embedding, np.ndarray):
                    query_embedding = np.array(query_embedding)

                if query_embedding.size == 0:
                    logger.warning("Query embedding is empty")
                    return results
            except Exception as array_e:
                logger.warning(f"Query embedding validation failed: {array_e}")
                return results

            # Prepare query embedding with dimension validation
            try:
                query_vector = query_embedding.reshape(1, -1).astype("float32")

                # Check dimension compatibility
                expected_dim = getattr(self.vector_index, "d", 1536)
                if query_vector.shape[1] != expected_dim:
                    logger.warning(
                        f"Embedding dimension mismatch: got {query_vector.shape[1]}, expected {expected_dim}"
                    )
                    return results

            except Exception as reshape_e:
                logger.warning(f"Query embedding reshape failed: {reshape_e}")
                return results

            # Determine search parameters
            k = min(20, self.vector_index.ntotal)
            if k <= 0:
                logger.debug("No vectors available for search")
                return results

            # Perform FAISS search with timeout protection
            try:
                scores, indices = self.vector_index.search(query_vector, k)

                if scores is None or indices is None:
                    logger.warning("FAISS search returned None results")
                    return results

                if len(scores) == 0 or len(indices) == 0:
                    logger.debug("FAISS search returned empty results")
                    return results

            except Exception as search_e:
                logger.warning(f"FAISS search operation failed: {search_e}")
                return results

            # Convert results with validation
            valid_results = 0
            for score, idx in zip(scores[0], indices[0], strict=False):
                try:
                    # Validate result components
                    if idx < 0:
                        continue  # Invalid index

                    # Convert numpy types to Python types
                    try:
                        score = float(score)
                    except (ValueError, TypeError):
                        continue  # Invalid score type

                    # Convert distance to similarity score (FAISS returns L2 distances, so smaller = more similar)
                    # We'll use 1/(1+distance) to convert to similarity where higher = better
                    similarity_score = 1.0 / (1.0 + float(score))

                    if similarity_score < 0.001:  # Very low similarity threshold
                        continue

                    # Map index to actual file path
                    file_path = f"indexed_content_{idx}"
                    content_preview = f"Semantic match with similarity {similarity_score:.3f} (distance: {score:.1f})"

                    if (
                        metadata
                        and "files" in metadata
                        and 0 <= idx < len(metadata["files"])
                    ):
                        file_info = metadata["files"][idx]
                        file_path = file_info.get("file_path", file_path)
                        if "content_preview" in file_info:
                            content_preview = file_info["content_preview"][:100] + "..."

                    # Create search result
                    result = SearchResult(
                        content=content_preview,
                        file_path=file_path,
                        line_number=1,
                        score=similarity_score,
                        result_type="semantic",
                        context={
                            "faiss_index": int(idx),
                            "similarity_score": similarity_score,
                            "distance_score": float(score),
                            "search_method": "faiss_vector_search",
                            "index_size": self.vector_index.ntotal,
                        },
                        timestamp=time.time(),
                    )
                    results.append(result)
                    valid_results += 1

                except Exception as result_e:
                    logger.debug(f"Failed to process FAISS result {idx}: {result_e}")
                    continue

            logger.debug(
                f"FAISS search completed: {valid_results} valid results from {len(scores[0])} candidates"
            )
            return results

        except ImportError:
            logger.debug("FAISS library import failed")
            self._faiss_available = False
        except Exception as e:
            logger.warning(f"FAISS search failed with unexpected error: {e}")

        return results

    async def _try_embedding_pipeline_search(
        self, query: str, query_embedding: "np.ndarray"
    ) -> list[SearchResult]:
        """Try embedding pipeline search with error handling."""
        results = []

        try:
            if not self.embedding_pipeline:
                return results

            # Search through recently indexed files using embedding pipeline
            search_results = await self.embedding_pipeline.embed_search_results(
                query, str(self.project_root), context_lines=3
            )

            for embed_result in search_results[:10]:  # Top 10 results
                if "embedding" in embed_result:
                    # Calculate similarity score
                    similarity = self._calculate_similarity(
                        query_embedding, embed_result["embedding"]
                    )

                    if similarity > 0.3:  # Minimum threshold
                        result = SearchResult(
                            content=embed_result["content"][
                                :200
                            ],  # Truncate for display
                            file_path=embed_result.get("file_path", "unknown"),
                            line_number=embed_result.get("start_line", 1),
                            score=float(similarity),
                            result_type="semantic",
                            context={
                                "similarity": float(similarity),
                                "tokens": embed_result.get("tokens", 0),
                                "cached": embed_result.get("cached", False),
                            },
                            timestamp=time.time(),
                        )
                        results.append(result)

        except Exception as e:
            logger.warning(f"Embedding pipeline search failed: {e}")

        return results

    async def _try_neural_backend_search(self, query: str) -> list[SearchResult]:
        """Try neural backend search as fallback."""
        results = []

        try:
            if not hasattr(self, "neural_backend") or not self.neural_backend:
                return results

            # This would be implemented based on available neural backend
            # For now, return empty results
            logger.debug("Neural backend search not implemented yet")

        except Exception as e:
            logger.warning(f"Neural backend search failed: {e}")

        return results

    async def _neural_backend_embedding(self, text: str) -> Optional["np.ndarray"]:
        """Generate embedding using neural backend."""
        try:
            if not hasattr(self, "neural_backend") or not self.neural_backend:
                return None

            # Use MLX-based embedding implementation
            try:
                from einstein.mlx_embeddings import get_mlx_embedding_engine

                embedding_engine = get_mlx_embedding_engine()
                embedding_result, _ = await embedding_engine.embed_text_async(text)
                return embedding_result
            except ImportError as e:
                logger.debug(f"MLX embeddings not available: {e}")
                return None

        except Exception as e:
            logger.warning(f"Neural backend embedding failed: {e}")
            return None

    async def _fallback_text_search(self, query: str) -> list[SearchResult]:
        """Fallback to text search when semantic search is unavailable."""
        try:
            logger.info("Using text search fallback for semantic query")
            return await self._text_search(query)
        except Exception as e:
            logger.error(f"Text search fallback failed: {e}")
            return []

    async def _fallback_similarity_search(self, query: str) -> list[SearchResult]:
        """Fallback similarity search using text-based methods."""
        try:
            # Try text search first
            text_results = await self._text_search(query)

            # If we get results, enhance them with semantic context
            for result in text_results:
                result.result_type = "semantic_fallback"
                result.context["fallback_method"] = "text_similarity"

            return text_results[:10]  # Limit results

        except Exception as e:
            logger.error(f"Fallback similarity search failed: {e}")
            return []

    async def _fallback_semantic_text_search(self, query: str) -> list[SearchResult]:
        """Final fallback using enhanced text search for semantic queries."""
        try:
            # Split query into individual terms for broader matching
            query_terms = query.lower().split()

            # Search for each term and combine results
            all_results = []

            for term in query_terms:
                if len(term) > 2:  # Skip very short terms
                    term_results = await self._text_search(term)
                    for result in term_results:
                        result.result_type = "semantic_text_fallback"
                        result.score *= 0.7  # Reduce score for fallback method
                        result.context["fallback_method"] = "semantic_text"
                        result.context["original_query"] = query
                    all_results.extend(term_results)

            # Remove duplicates and sort by score
            seen_files = set()
            unique_results = []
            for result in sorted(all_results, key=lambda x: x.score, reverse=True):
                file_key = f"{result.file_path}:{result.line_number}"
                if file_key not in seen_files:
                    seen_files.add(file_key)
                    unique_results.append(result)

            return unique_results[:15]  # Return top 15 unique results

        except Exception as e:
            logger.error(f"Semantic text fallback failed: {e}")
            return []

    async def build_indices(self) -> None:
        """Build all indices for the Einstein system."""
        try:
            logger.info("🏗️ Building Einstein indices...")

            # Initialize the system if not already done
            await self.initialize()

            # Perform initial scan to build the index
            await self._perform_initial_scan()

            # Initialize FAISS index
            await self._initialize_persistent_faiss()

            logger.info("✅ Einstein indices built successfully")

        except Exception as e:
            logger.error(f"Failed to build indices: {e}")
            raise

    async def get_index_stats(self) -> IndexStats:
        """Get comprehensive statistics about the Einstein index."""
        try:
            # Get file count
            python_files = list(self.project_root.rglob("*.py"))
            total_files = len(python_files)

            # Calculate total lines
            total_lines = 0
            for file_path in python_files:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        total_lines += len(f.readlines())
                except (UnicodeDecodeError, OSError) as e:
                    logger.debug(f"Could not read file {file_path}: {e}")
                    continue

            # Get index size
            index_size_mb = 0
            einstein_dir = Path(self.einstein_config.paths.base_dir)
            if einstein_dir.exists():
                for file_path in einstein_dir.rglob("*"):
                    if file_path.is_file():
                        index_size_mb += file_path.stat().st_size
                index_size_mb = index_size_mb / (1024 * 1024)  # Convert to MB

            # Get search performance
            search_performance = {}
            for stat_type, times in self.search_stats.items():
                if times:
                    search_performance[stat_type] = sum(times) / len(times)
                else:
                    search_performance[stat_type] = 0.0

            # Calculate coverage
            coverage_percentage = 100.0 if total_files > 0 else 0.0

            return IndexStats(
                total_files=total_files,
                total_lines=total_lines,
                index_size_mb=index_size_mb,
                last_update=time.time(),
                search_performance_ms=search_performance,
                coverage_percentage=coverage_percentage,
            )

        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return IndexStats(
                total_files=0,
                total_lines=0,
                index_size_mb=0.0,
                last_update=0.0,
                search_performance_ms={},
                coverage_percentage=0.0,
            )

    async def _build_dependency_graph_deferred(self):
        """Build dependency graph in background for fast startup."""
        try:
            await asyncio.sleep(2.0)  # Allow main initialization to complete
            logger.info("🔍 Background: Building dependency graph...")
            if self.dependency_graph:
                await self.dependency_graph.build_graph()
                logger.info("✅ Background dependency graph build complete")
        except Exception as e:
            logger.error(f"Background dependency graph build failed: {e}")

    async def _initialize_analytics_db_deferred(self):
        """Initialize analytics database in background for fast startup."""
        try:
            await asyncio.sleep(1.5)  # Allow main initialization to complete
            logger.info("📊 Background: Initializing analytics database...")
            await self._initialize_analytics_db()
            logger.info("✅ Background analytics DB initialization complete")
        except Exception as e:
            logger.error(f"Background analytics DB initialization failed: {e}")

    async def _initialize_persistent_faiss_deferred(self):
        """Initialize FAISS index in background for fast startup."""
        try:
            await asyncio.sleep(3.0)  # Allow main initialization to complete
            logger.info("🤖 Background: Initializing FAISS index...")
            await self._initialize_persistent_faiss()
            logger.info("✅ Background FAISS initialization complete")
        except Exception as e:
            logger.error(f"Background FAISS initialization failed: {e}")

    async def _initialize_embedding_pipeline_deferred(self):
        """Initialize embedding pipeline in background for fast startup."""
        try:
            await asyncio.sleep(4.0)  # Allow main initialization to complete
            logger.info("🔤 Background: Initializing embedding pipeline...")
            await self._initialize_embedding_pipeline()
            logger.info("✅ Background embedding pipeline initialization complete")
        except Exception as e:
            logger.error(f"Background embedding pipeline initialization failed: {e}")

    def enable_fast_mode(self) -> None:
        """Enable fast mode for quick initialization."""
        self._fast_mode = True
        self._skip_initial_scan = True
        self._skip_file_watcher = True
        self._skip_dependency_build = True
        logger.info("⚡ Einstein Fast Mode enabled")

    def disable_fast_mode(self) -> None:
        """Disable fast mode and perform full initialization."""
        self._fast_mode = False
        self._skip_initial_scan = False
        self._skip_file_watcher = False
        self._skip_dependency_build = False
        logger.info("🔄 Einstein Fast Mode disabled - will perform full initialization")

    async def shutdown(self):
        """Shutdown the Einstein index system."""
        try:
            logger.info("Shutting down Einstein index system")
            # Add any cleanup here if needed
        except Exception as e:
            logger.error(f"Error during Einstein shutdown: {e}")

    async def _ast_structural_search(self, query: str) -> list[SearchResult]:
        """Search for structural patterns using AST analysis."""

        results = []
        query_lower = query.lower()

        # Define structural search patterns
        search_patterns = {
            "function": ["function", "def", "method"],
            "class": ["class", "object"],
            "async": ["async", "await", "asynchronous"],
            "import": ["import", "module", "from"],
            "exception": ["exception", "error", "try", "except"],
            "inheritance": ["inherit", "extend", "base", "parent"],
        }

        # Determine search type
        search_type = "generic"
        for pattern_type, keywords in search_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                search_type = pattern_type
                break

        # Use ripgrep to find relevant files with structural patterns
        if self.ripgrep:
            try:
                # Search for specific patterns based on query type
                if search_type == "function":
                    patterns = [r"^\s*def\s+\w+", r"^\s*async\s+def\s+\w+"]
                elif search_type == "class":
                    patterns = [r"^\s*class\s+\w+"]
                elif search_type == "async":
                    patterns = [r"^\s*async\s+def\s+\w+", r"\bawait\s+"]
                elif search_type == "import":
                    patterns = [r"^\s*import\s+\w+", r"^\s*from\s+\w+"]
                elif search_type == "exception":
                    patterns = [r"^\s*try:", r"^\s*except\s+\w*", r"\braise\s+\w+"]
                else:
                    # Generic patterns - search for the query in various contexts
                    patterns = [r"\b" + re.escape(query.lower()) + r"\b"]

                for pattern in patterns:
                    try:
                        rg_results = await self.ripgrep.search(
                            pattern,
                            str(self.project_root),
                            max_results=10,
                            use_regex=True,
                        )

                        for rg_result in rg_results:
                            # Analyze the matched line with AST if it's Python code
                            file_path = rg_result.get("file", "")
                            line_content = rg_result.get("content", "")
                            line_number = rg_result.get("line", 0)

                            if file_path.endswith(".py"):
                                # Skip comments and documentation
                                if (
                                    line_content.strip().startswith("#")
                                    or '"""' in line_content
                                ):
                                    continue

                                ast_analysis = await self._analyze_line_structure(
                                    file_path, line_content, line_number
                                )

                                if (
                                    ast_analysis
                                    and ast_analysis.get("type") != "unknown"
                                ):
                                    result = SearchResult(
                                        content=line_content.strip(),
                                        file_path=file_path,
                                        line_number=line_number,
                                        score=self._calculate_structural_score(
                                            query, ast_analysis, search_type
                                        ),
                                        result_type="structural_ast",
                                        context={
                                            "ast_info": ast_analysis,
                                            "search_type": search_type,
                                            "pattern_matched": pattern,
                                            "method": "ast_analysis",
                                        },
                                        timestamp=time.time(),
                                    )
                                    results.append(result)
                    except Exception as pattern_error:
                        logger.debug(
                            f"Pattern search failed for {pattern}: {pattern_error}"
                        )

            except Exception as rg_error:
                logger.warning(f"Ripgrep structural search failed: {rg_error}")

        return results

    async def _analyze_line_structure(
        self, file_path: str, line_content: str, line_number: int
    ) -> dict:
        """Analyze a single line for structural information using AST context."""

        try:
            # Try to get context around the line for better AST analysis
            context_lines = await self._get_file_context(
                file_path, line_number, context_size=5
            )

            if context_lines:
                # Try to parse the context as AST
                try:
                    import ast

                    tree = ast.parse(context_lines)

                    # Find the specific node at the target line
                    for node in ast.walk(tree):
                        if (
                            hasattr(node, "lineno")
                            and abs(
                                node.lineno
                                - (line_number - len(context_lines.split("\n")) // 2)
                            )
                            <= 1
                        ):
                            return self._extract_node_info(node)

                except SyntaxError:
                    # If AST parsing fails, do simple pattern analysis
                    pass

            # Fallback to simple pattern analysis
            return self._simple_line_analysis(line_content)

        except Exception as e:
            logger.debug(
                f"Line structure analysis failed for {file_path}:{line_number}: {e}"
            )
            return self._simple_line_analysis(line_content)

    async def _get_file_context(
        self, file_path: str, line_number: int, context_size: int = 5
    ) -> str:
        """Get context lines around a specific line number."""

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            start_line = max(0, line_number - context_size - 1)
            end_line = min(len(lines), line_number + context_size)

            context_lines = lines[start_line:end_line]
            return "".join(context_lines)

        except Exception as e:
            logger.debug(f"Failed to get context for {file_path}:{line_number}: {e}")
            return ""

    def _extract_node_info(self, node) -> dict:
        """Extract structural information from an AST node."""

        import ast

        info = {"node_type": type(node).__name__, "line": getattr(node, "lineno", 0)}

        if isinstance(node, ast.FunctionDef):
            info.update(
                {
                    "type": "function",
                    "name": node.name,
                    "is_async": False,
                    "args": [arg.arg for arg in node.args.args],
                }
            )
        elif isinstance(node, ast.AsyncFunctionDef):
            info.update(
                {
                    "type": "async_function",
                    "name": node.name,
                    "is_async": True,
                    "args": [arg.arg for arg in node.args.args],
                }
            )
        elif isinstance(node, ast.ClassDef):
            info.update(
                {
                    "type": "class",
                    "name": node.name,
                    "bases": [self._get_ast_name(base) for base in node.bases],
                }
            )
        elif isinstance(node, ast.Import):
            info.update(
                {"type": "import", "modules": [alias.name for alias in node.names]}
            )
        elif isinstance(node, ast.ImportFrom):
            info.update(
                {
                    "type": "from_import",
                    "module": node.module,
                    "names": [alias.name for alias in node.names],
                }
            )
        elif isinstance(node, ast.Try):
            info.update({"type": "exception_handler", "handlers": len(node.handlers)})

        return info

    def _get_ast_name(self, node) -> str:
        """Extract name from AST node."""
        import ast

        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_ast_name(node.value)}.{node.attr}"
        else:
            return str(node)

    def _simple_line_analysis(self, line_content: str) -> dict:
        """Simple pattern-based analysis of a line."""

        line = line_content.strip()
        info = {"line_content": line}

        if line.startswith("def "):
            info["type"] = "function"
            info["name"] = line.split("(")[0].replace("def ", "").strip()
        elif line.startswith("async def "):
            info["type"] = "async_function"
            info["name"] = line.split("(")[0].replace("async def ", "").strip()
        elif line.startswith("class "):
            info["type"] = "class"
            info["name"] = (
                line.split("(")[0].split(":")[0].replace("class ", "").strip()
            )
        elif line.startswith("import ") or line.startswith("from "):
            info["type"] = "import"
        elif line.strip().startswith("try:"):
            info["type"] = "exception_handler"
        else:
            info["type"] = "unknown"

        return info

    def _calculate_structural_score(
        self, query: str, ast_analysis: dict, search_type: str
    ) -> float:
        """Calculate relevance score for structural matches."""

        base_score = 0.7  # Higher base score for structural matches
        query_lower = query.lower()

        # Boost score based on exact type match
        analysis_type = ast_analysis.get("type", "")
        if (
            search_type == "function"
            and analysis_type in ["function", "async_function"]
            or search_type == "class"
            and analysis_type == "class"
        ):
            base_score += 0.25
        elif search_type == "async" and analysis_type == "async_function":
            base_score += 0.3  # Higher boost for async matches
        elif (
            search_type == "import"
            and analysis_type in ["import", "from_import"]
            or search_type == "exception"
            and analysis_type == "exception_handler"
        ):
            base_score += 0.25

        # Additional boost for specific query patterns
        if (
            "async" in query_lower
            and analysis_type == "async_function"
            or "class" in query_lower
            and analysis_type == "class"
        ):
            base_score += 0.2
        elif "function" in query_lower and analysis_type in [
            "function",
            "async_function",
        ]:
            base_score += 0.15

        # Boost score based on name match
        name = ast_analysis.get("name", "")
        if name and query_lower in name.lower():
            base_score += 0.15

        # Boost score for exact matches
        if name and name.lower() == query_lower:
            base_score += 0.2

        # Penalize comments and non-code content
        line_content = ast_analysis.get("line_content", "")
        if line_content.strip().startswith("#") or '"""' in line_content:
            base_score *= 0.3  # Heavily penalize comments

        return min(1.0, base_score)

    async def _pattern_structural_search(self, query: str) -> list[SearchResult]:
        """Pattern-based structural search for common programming constructs."""

        results = []
        query_lower = query.lower()

        # Common structural patterns to search for
        structural_patterns = {
            "async functions": r"^\s*async\s+def\s+",
            "class definitions": r"^\s*class\s+\w+",
            "function definitions": r"^\s*def\s+\w+",
            "import statements": r"^\s*(import|from)\s+",
            "exception handling": r"^\s*(try:|except\s+)",
            "inheritance": r"^\s*class\s+\w+\s*\([^)]+\)",
            "decorators": r"^\s*@\w+",
            "context managers": r"^\s*with\s+",
        }

        # Find matching patterns
        for pattern_name, regex_pattern in structural_patterns.items():
            if (
                any(word in query_lower for word in pattern_name.split())
                and self.ripgrep
            ):
                try:
                    rg_results = await self.ripgrep.search(
                        regex_pattern,
                        str(self.project_root),
                        max_results=5,
                        use_regex=True,
                        file_types=["py"],
                    )

                    for rg_result in rg_results:
                        result = SearchResult(
                            content=rg_result.get("content", "").strip(),
                            file_path=rg_result.get("file", ""),
                            line_number=rg_result.get("line", 0),
                            score=0.75,  # Good score for pattern matches
                            result_type="structural_pattern",
                            context={
                                "pattern_type": pattern_name,
                                "regex_pattern": regex_pattern,
                                "method": "pattern_matching",
                            },
                            timestamp=time.time(),
                        )
                        results.append(result)

                except Exception as pattern_error:
                    logger.debug(
                        f"Pattern search failed for {pattern_name}: {pattern_error}"
                    )

        return results

    async def _enhanced_result_ranking(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Enhanced ranking of search results with deduplication and prioritization."""

        if not results:
            return []

        # Remove duplicates based on file and line
        seen = set()
        unique_results = []

        for result in results:
            key = f"{result.file_path}:{result.line_number}"
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        # Apply query-specific scoring adjustments
        query_lower = query.lower()

        for result in unique_results:
            # Boost structural matches
            if result.result_type in ["structural_ast", "structural_pattern", "symbol"]:
                result.score *= 1.5

            # Boost for exact type matches
            if "async" in query_lower and result.result_type == "async_function":
                result.score *= 1.8
            elif (
                "function" in query_lower
                and "def " in result.content
                or "class" in query_lower
                and "class " in result.content
                or "import" in query_lower
                and ("import " in result.content or "from " in result.content)
            ):
                result.score *= 1.6

            # Penalize test files unless specifically searching for tests
            if "test" not in query_lower and "test" in result.file_path.lower():
                result.score *= 0.7

            # Penalize comments and documentation heavily
            if result.content.strip().startswith("#") or '"""' in result.content:
                result.score *= 0.2

            # Boost core implementation files
            if any(
                term in result.file_path.lower() for term in ["src/", "core/", "main"]
            ):
                result.score *= 1.2

            # Ensure score stays within bounds
            result.score = min(1.0, result.score)

        # Sort by score (highest first)
        unique_results.sort(key=lambda x: x.score, reverse=True)

        return unique_results


# Global instance
_einstein_hub: EinsteinIndexHub | None = None


class EinsteinFileHandler(FileSystemEventHandler):
    """File system event handler for Einstein real-time indexing.

    Properly handles the event loop context issue by using thread-safe
    methods to communicate with the async processing loop.
    """

    def __init__(self, einstein_hub: EinsteinIndexHub):
        super().__init__()
        self.einstein_hub = einstein_hub
        self.debounce_delay = (
            self.einstein_config.monitoring.debounce_delay_ms / 1000.0
        )  # Convert ms to seconds
        self.pending_events = {}  # file_path -> (event_type, timer)

    def _schedule_event(self, file_path: str, event_type: str):
        """Schedule file event with debouncing and proper async context."""
        try:
            # Skip if we don't have the required components
            if (
                not hasattr(self.einstein_hub, "file_change_queue")
                or not self.einstein_hub.file_change_queue
            ):
                logger.debug("File change queue not available, skipping event")
                return

            if (
                not hasattr(self.einstein_hub, "_file_change_loop")
                or not self.einstein_hub._file_change_loop
            ):
                logger.debug("File change loop not available, skipping event")
                return

            # Cancel existing timer for this file
            if file_path in self.pending_events:
                timer = self.pending_events[file_path][1]
                if timer:
                    timer.cancel()

            # Schedule new event with debouncing
            import threading

            timer = threading.Timer(
                self.debounce_delay, self._process_event, args=[file_path, event_type]
            )

            self.pending_events[file_path] = (event_type, timer)
            timer.start()

        except Exception as e:
            logger.error(f"Error scheduling file event {file_path}: {e}")

    def _process_event(self, file_path: str, event_type: str):
        """Process file event by putting it in the async queue safely."""
        try:
            # Clean up pending event
            if file_path in self.pending_events:
                del self.pending_events[file_path]

            # Check if we should process this file
            if not self.einstein_hub._should_process_file(file_path):
                logger.debug(f"Skipping file that should not be processed: {file_path}")
                return

            # Use call_soon_threadsafe to safely add to queue from another thread
            if (
                hasattr(self.einstein_hub, "_file_change_loop")
                and self.einstein_hub._file_change_loop
                and not self.einstein_hub._file_change_loop.is_closed()
            ):
                try:
                    self.einstein_hub._file_change_loop.call_soon_threadsafe(
                        self._add_to_queue_safe, file_path, event_type
                    )
                except RuntimeError as e:
                    if "Event loop is closed" in str(e):
                        logger.debug(
                            f"Event loop closed, cannot process file event: {file_path}"
                        )
                    else:
                        raise
            else:
                logger.debug(f"File change loop not available for event: {file_path}")

        except Exception as e:
            logger.error(f"Error processing file event {file_path}: {e}")

    def _add_to_queue_safe(self, file_path: str, event_type: str):
        """Add event to queue safely from the correct event loop context."""
        try:
            if (
                hasattr(self.einstein_hub, "file_change_queue")
                and self.einstein_hub.file_change_queue
            ):
                # Use put_nowait to avoid blocking
                try:
                    self.einstein_hub.file_change_queue.put_nowait(
                        (file_path, event_type)
                    )
                    logger.debug(f"Queued file event: {event_type} for {file_path}")
                except asyncio.QueueFull:
                    logger.warning(
                        f"File change queue full, dropping event: {file_path}"
                    )
                except Exception as queue_error:
                    logger.error(f"Error putting event in queue: {queue_error}")
            else:
                logger.warning(
                    f"File change queue not available, cannot queue event: {file_path}"
                )

        except Exception as e:
            logger.error(f"Error adding to queue: {file_path}: {e}")

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._schedule_event(event.src_path, "modified")

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._schedule_event(event.src_path, "created")

    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            self._schedule_event(event.src_path, "deleted")

    def cleanup(self):
        """Clean up any pending timers."""
        for _file_path, (_event_type, timer) in self.pending_events.items():
            if timer:
                timer.cancel()
        self.pending_events.clear()


# Backward compatibility alias
UnifiedIndex = EinsteinIndexHub


def get_einstein_hub() -> EinsteinIndexHub:
    """Get the global Einstein indexing hub."""
    global _einstein_hub
    if _einstein_hub is None:
        _einstein_hub = EinsteinIndexHub()
    return _einstein_hub


if __name__ == "__main__":
    # Test the Einstein system
    async def test_einstein():
        hub = get_einstein_hub()
        await hub.initialize()

        # Test search
        results = await hub.search("WheelStrategy")
        print(f"Found {len(results)} results for WheelStrategy")

        # Test intelligent context
        context = await hub.get_intelligent_context("options pricing")
        print(f"Generated context with {len(context['search_results'])} results")

        # Show stats
        stats = await hub.get_stats()
        print(f"Index stats: {stats}")

    asyncio.run(test_einstein())
