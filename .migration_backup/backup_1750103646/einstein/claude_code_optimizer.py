#!/usr/bin/env python3
"""
Einstein System Optimizer for Claude Code CLI

Optimizes the Einstein system specifically for Claude Code CLI's rapid code gathering needs:
- Fast search responses (configured via performance targets)
- Minimal startup time (configured via startup targets)  
- Efficient memory usage (configured via memory limits)
- Comprehensive coverage (100% of codebase)

Key optimizations:
1. Lazy initialization with pre-warmed caches
2. Streaming search with early termination
3. Memory-mapped indexes with compression
4. Hardware-specific optimizations
5. Predictive prefetching based on query patterns
"""

import asyncio
import hashlib
import json
import logging
import mmap
import os
import pickle
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

try:
    import lru
except ImportError:
    # Fallback to functools.lru_cache or simple dict
    class LRU:
        def __init__(self, maxsize: int) -> None:
            self.maxsize = maxsize
            self.data = {}

        def __getitem__(self, key: Any) -> Any:
            return self.data[key]

        def __setitem__(self, key: Any, value: Any) -> None:
            if len(self.data) >= self.maxsize:
                # Remove oldest item (simple approach)
                oldest_key = next(iter(self.data))
                del self.data[oldest_key]
            self.data[key] = value

        def get(self, key: Any, default: Any = None) -> Any:
            return self.data.get(key, default)

        def __contains__(self, key: Any) -> bool:
            return key in self.data

    class lru:
        @staticmethod
        def LRU(maxsize: int) -> "LRU":
            return LRU(maxsize)


import psutil

from einstein.einstein_config import get_einstein_config
from src.unity_wheel.accelerated_tools.dependency_graph_turbo import (
    DependencyGraphTurbo,
)

# Import accelerated components
from src.unity_wheel.accelerated_tools.ripgrep_turbo import RipgrepTurbo

logger = logging.getLogger(__name__)


class ClaudeCodeConfig:
    """Configuration optimized for Claude Code CLI usage patterns."""

    def __init__(self) -> None:
        self.config = get_einstein_config()

        # Performance targets from Einstein config
        self.performance = self.config.performance  # Expose performance config
        self.max_search_time_ms = self.config.performance.max_search_time_ms
        self.max_startup_time_ms = self.config.performance.max_startup_time_ms
        self.max_memory_usage_gb = self.config.performance.max_memory_usage_gb

        # Optimization flags
        self.use_lazy_initialization = True
        self.use_streaming_search = True
        self.use_predictive_prefetch = self.config.enable_predictive_prefetch
        self.use_compressed_indexes = True
        self.use_memory_mapping = True

        # Hardware configuration from Einstein config
        self.cpu_cores = self.config.hardware.cpu_cores
        self.memory_gb = self.config.hardware.memory_total_gb
        self.use_gpu_acceleration = self.config.enable_gpu_acceleration

        # Cache configuration from Einstein config
        self.hot_cache_size = self.config.cache.hot_cache_size
        self.warm_cache_size = self.config.cache.warm_cache_size
        self.index_cache_size_mb = self.config.cache.index_cache_size_mb


class OptimizedSearchIndex:
    """Memory-mapped, compressed search index optimized for configured search performance targets."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        if cache_dir is None:
            config = get_einstein_config()
            cache_dir = config.paths.optimized_cache_dir

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Memory-mapped index files
        self.text_index_path = cache_dir / "text.idx"
        self.symbol_index_path = cache_dir / "symbols.idx"
        self.content_index_path = cache_dir / "content.idx"

        # In-memory hot caches using LRU
        config = get_einstein_config()
        self.hot_text_cache = lru.LRU(config.cache.hot_cache_size)
        self.hot_symbol_cache = lru.LRU(config.cache.hot_cache_size)

        # Memory-mapped files
        self._text_mmap = None
        self._symbol_mmap = None
        self._content_mmap = None

        # Index metadata
        self._metadata = {}
        self._index_loaded = False

    async def initialize(self, force_rebuild: bool = False):
        """Initialize indexes with lazy loading."""
        start_time = time.time()

        # Check if indexes exist and are fresh
        if not force_rebuild and self._indexes_exist() and self._indexes_fresh():
            self._load_existing_indexes()
        else:
            await self._build_optimized_indexes()

        # Memory-map the index files
        self._memory_map_indexes()

        init_time = (time.time() - start_time) * 1000
        logger.info(f"Optimized search index initialized in {init_time:.1f}ms")

    def _indexes_exist(self) -> bool:
        """Check if all required index files exist."""
        return (
            self.text_index_path.exists()
            and self.symbol_index_path.exists()
            and self.content_index_path.exists()
        )

    def _indexes_fresh(self) -> bool:
        """Check if indexes are fresh (less than 1 hour old)."""
        if not self.text_index_path.exists():
            return False

        index_age = time.time() - self.text_index_path.stat().st_mtime
        return index_age < 3600  # 1 hour

    async def _build_optimized_indexes(self):
        """Build compressed, optimized indexes."""
        logger.info("Building optimized search indexes...")

        # Use ripgrep to build text index
        rg = RipgrepTurbo()

        # Build comprehensive text index for common patterns
        text_patterns = [
            r"class\s+\w+",
            r"def\s+\w+",
            r"import\s+\w+",
            r"from\s+\w+",
            r"TODO|FIXME|NOTE",
            r"async\s+def",
            r"@\w+",
        ]

        text_index = {}
        for pattern in text_patterns:
            results = await rg.search(pattern, ".", max_results=10000)
            for result in results:
                key = f"{result['file']}:{result['line']}"
                text_index[key] = {
                    "content": result["content"],
                    "pattern": pattern,
                    "score": 1.0,
                }

        # Build symbol index using dependency graph
        dep_graph = DependencyGraphTurbo(".")
        await dep_graph.build_graph()

        symbol_index = {}
        for symbol, locations in dep_graph.symbol_locations.items():
            for file_path, line_num in locations:
                key = f"{file_path}:{line_num}"
                symbol_index[key] = {
                    "symbol": symbol,
                    "type": "definition",
                    "score": 0.9,
                }

        # Save compressed indexes
        self._save_compressed_index(self.text_index_path, text_index)
        self._save_compressed_index(self.symbol_index_path, symbol_index)

        # Save metadata
        metadata = {
            "build_time": time.time(),
            "text_entries": len(text_index),
            "symbol_entries": len(symbol_index),
            "version": "1.0",
        }

        with open(self.cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

    def _save_compressed_index(self, path: Path, index: dict):
        """Save index with compression."""
        import gzip

        data = pickle.dumps(index)

        with gzip.open(path, "wb") as f:
            f.write(data)

    def _load_compressed_index(self, path: Path) -> dict:
        """Load compressed index."""
        import gzip

        with gzip.open(path, "rb") as f:
            data = f.read()

        return pickle.loads(data)

    def _load_existing_indexes(self):
        """Load existing indexes."""
        logger.info("Loading existing optimized indexes...")

        # Load metadata
        metadata_path = self.cache_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)

    def _memory_map_indexes(self):
        """Memory-map index files for ultra-fast access."""
        try:
            # Memory map text index
            if self.text_index_path.exists():
                with open(self.text_index_path, "rb") as f:
                    self._text_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Memory map symbol index
            if self.symbol_index_path.exists():
                with open(self.symbol_index_path, "rb") as f:
                    self._symbol_mmap = mmap.mmap(
                        f.fileno(), 0, access=mmap.ACCESS_READ
                    )

            self._index_loaded = True
            logger.info("Indexes memory-mapped successfully")

        except Exception as e:
            logger.error(
                f"Failed to memory-map indexes: {e}",
                exc_info=True,
                extra={
                    "operation": "memory_map_indexes",
                    "error_type": type(e).__name__,
                    "text_index_exists": self.text_index_path.exists(),
                    "symbol_index_exists": self.symbol_index_path.exists(),
                    "cache_dir": str(self.cache_dir),
                    "index_sizes": {
                        "text": self.text_index_path.stat().st_size
                        if self.text_index_path.exists()
                        else 0,
                        "symbol": self.symbol_index_path.stat().st_size
                        if self.symbol_index_path.exists()
                        else 0,
                    },
                    "cache_dir_writable": os.access(self.cache_dir, os.W_OK)
                    if self.cache_dir.exists()
                    else False,
                },
            )

    async def search_streaming(
        self, query: str, max_results: int = 20
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streaming search with early termination for configured response time targets."""
        start_time = time.time()

        # Check hot cache first (should be very fast)
        cache_key = hashlib.md5(query.encode()).hexdigest()[:16]

        if cache_key in self.hot_text_cache:
            cached_results = self.hot_text_cache[cache_key]
            for result in cached_results[:max_results]:
                yield result
            return

        # Search across indexes in parallel
        search_tasks = [
            self._search_text_index(query, max_results // 2),
            self._search_symbol_index(query, max_results // 2),
        ]

        results = []

        # Use asyncio.as_completed for streaming results
        for coro in asyncio.as_completed(search_tasks):
            try:
                batch_results = await coro
                for result in batch_results:
                    if len(results) >= max_results:
                        break
                    results.append(result)
                    yield result

                    # Early termination if we have enough good results
                    elapsed_ms = (time.time() - start_time) * 1000
                    if (
                        elapsed_ms > (self.config.performance.max_search_time_ms * 0.8)
                        and len(results) >= 5
                    ):  # Stop at 80% of target if we have 5+ results
                        break

            except Exception as e:
                logger.error(
                    f"Search task failed: {e}",
                    exc_info=True,
                    extra={
                        "operation": "search_streaming",
                        "error_type": type(e).__name__,
                        "query": query[:50],  # Truncate long queries
                        "query_length": len(query),
                        "max_results": max_results,
                        "elapsed_ms": (time.time() - start_time) * 1000,
                        "results_count": len(results),
                        "index_loaded": self._index_loaded,
                        "cache_hit": cache_key in self.hot_text_cache,
                        "cache_key": cache_key,
                    },
                )
                continue

        # Cache successful results
        if results:
            self.hot_text_cache[cache_key] = results

    async def _search_text_index(
        self, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search text index with optimized string matching."""
        if not self._index_loaded:
            return []

        # For demo purposes, use ripgrep for actual text search
        # In production, this would use the memory-mapped index
        rg = RipgrepTurbo()
        results = await rg.search(query, ".", max_results=max_results)

        return [
            {
                "content": r["content"],
                "file_path": r["file"],
                "line_number": r["line"],
                "score": 1.0,
                "result_type": "text",
                "source": "optimized_text_index",
            }
            for r in results
        ]

    async def _search_symbol_index(
        self, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search symbol index."""
        if not self._index_loaded:
            return []

        # For demo purposes, use dependency graph
        # In production, this would use the memory-mapped index
        dep_graph = DependencyGraphTurbo(".")
        symbol_results = await dep_graph.find_symbol(query)

        return [
            {
                "content": f"Symbol: {query}",
                "file_path": r.get("file", "unknown"),
                "line_number": r.get("line", 1),
                "score": 0.9,
                "result_type": "structural",
                "source": "optimized_symbol_index",
            }
            for r in symbol_results[:max_results]
        ]


class PredictivePrefetcher:
    """Predictive prefetching based on Claude Code CLI usage patterns."""

    def __init__(self, config: ClaudeCodeConfig):
        self.config = config
        self.query_patterns = {}
        config = get_einstein_config()
        self.prefetch_cache = lru.LRU(config.cache.prefetch_cache_size)
        self.usage_stats = {}

        # Common Claude Code CLI query patterns
        self.common_patterns = [
            "class",
            "def ",
            "import",
            "from",
            "async def",
            "TODO",
            "FIXME",
            "@",
            "Exception",
            "test_",
        ]

    async def initialize(self):
        """Initialize predictive prefetching."""
        # Pre-warm cache with common patterns
        for pattern in self.common_patterns:
            asyncio.create_task(self._prefetch_pattern(pattern))

    async def _prefetch_pattern(self, pattern: str):
        """Prefetch results for a common pattern."""
        try:
            # Use background thread to avoid blocking
            rg = RipgrepTurbo()
            results = await rg.search(pattern, ".", max_results=50)

            cache_key = hashlib.md5(pattern.encode()).hexdigest()[:16]
            self.prefetch_cache[cache_key] = results

        except Exception as e:
            logger.debug(f"Prefetch failed for pattern {pattern}: {e}")

    def should_prefetch(self, query: str) -> bool:
        """Determine if query should trigger prefetching."""
        # Prefetch for partial queries that are likely to be extended
        return len(query) >= 3 and any(
            pattern in query for pattern in self.common_patterns
        )

    async def get_prefetched_results(self, query: str) -> list[dict[str, Any]] | None:
        """Get prefetched results if available."""
        cache_key = hashlib.md5(query.encode()).hexdigest()[:16]
        return self.prefetch_cache.get(cache_key)


class ClaudeCodeOptimizer:
    """Main optimizer for Claude Code CLI integration."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.config = ClaudeCodeConfig()

        # Optimized components
        self.search_index = OptimizedSearchIndex()  # Uses config for path
        self.prefetcher = PredictivePrefetcher(self.config)

        # Performance tracking
        self.search_times = []
        self.cache_hit_rate = 0.0
        self.initialization_complete = False

        # Background initialization
        self._init_task = None

        # Claude Code CLI specific optimizations
        self.claude_patterns = self._initialize_claude_patterns()
        config = get_einstein_config()
        self.rapid_cache = lru.LRU(
            config.cache.warm_cache_size
        )  # Larger cache for rapid access
        self.query_sequence_predictor = {}
        self.last_queries = []

    async def initialize_lazy(self):
        """Lazy initialization for minimal startup time with proper async coordination."""
        if self.initialization_complete:
            return

        start_time = time.time()
        initialization_failed = False
        critical_components_ready = False

        try:
            logger.info("🚀 Starting Claude Code optimizer initialization...")

            # Start background initialization with proper error handling
            if not self._init_task:
                self._init_task = asyncio.create_task(self._background_initialization())
                logger.debug("Background initialization task started")

            # Wait for critical components with timeout
            try:
                await asyncio.wait_for(
                    self._wait_for_critical_components(),
                    timeout=self.config.max_startup_time_ms
                    / 1000
                    * 0.8,  # Use 80% of max time
                )
                critical_components_ready = True
                logger.debug("Critical components are ready")

            except TimeoutError:
                logger.error(
                    f"Critical components initialization timed out after {self.config.max_startup_time_ms * 0.8:.0f}ms"
                )
                initialization_failed = True
            except Exception as e:
                logger.error(f"Critical components initialization failed: {e}")
                initialization_failed = True

            # Calculate initialization time
            init_time = (time.time() - start_time) * 1000

            # Assess initialization results
            if not initialization_failed and critical_components_ready:
                if init_time > self.config.max_startup_time_ms:
                    logger.warning(
                        f"⚠️ Startup time {init_time:.1f}ms exceeds target {self.config.max_startup_time_ms}ms"
                    )
                else:
                    logger.info(
                        f"✅ Claude Code optimizer initialized in {init_time:.1f}ms"
                    )

                self.initialization_complete = True

                # Start monitoring background initialization completion
                asyncio.create_task(self._monitor_background_initialization())

            else:
                logger.error("❌ Initialization failed - critical components not ready")
                # Attempt partial recovery
                await self._attempt_recovery_initialization()

        except Exception as e:
            logger.error(
                f"❌ Unexpected error during initialization: {e}", exc_info=True
            )
            initialization_failed = True

            # Attempt minimal fallback initialization
            try:
                await self._fallback_initialization()
                logger.warning(
                    "⚠️ Fallback initialization completed with limited functionality"
                )
                self.initialization_complete = True
            except Exception as fallback_error:
                logger.error(f"❌ Fallback initialization also failed: {fallback_error}")
                raise RuntimeError(
                    f"Claude Code optimizer initialization completely failed: {e}"
                ) from e

    async def _background_initialization(self):
        """Background initialization of non-critical components with comprehensive error handling."""
        background_start_time = time.time()
        components_initialized = []
        components_failed = []

        try:
            logger.info("🔧 Starting background component initialization...")

            # Initialize search index with error handling
            try:
                logger.debug("Initializing search index...")
                await self.search_index.initialize()
                components_initialized.append("search_index")
                logger.debug("✅ Search index initialized successfully")
            except Exception as e:
                logger.error(f"❌ Search index initialization failed: {e}")
                components_failed.append(("search_index", str(e)))

            # Initialize prefetcher with error handling
            try:
                logger.debug("Initializing prefetcher...")
                await self.prefetcher.initialize()
                components_initialized.append("prefetcher")
                logger.debug("✅ Prefetcher initialized successfully")
            except Exception as e:
                logger.error(f"❌ Prefetcher initialization failed: {e}")
                components_failed.append(("prefetcher", str(e)))

            # Initialize Claude Code specific optimizations
            try:
                logger.debug("Applying Claude Code optimizations...")
                await self.optimize_for_claude_code()
                components_initialized.append("claude_code_optimizations")
                logger.debug("✅ Claude Code optimizations applied successfully")
            except Exception as e:
                logger.error(f"❌ Claude Code optimizations failed: {e}")
                components_failed.append(("claude_code_optimizations", str(e)))

            # Calculate background initialization time
            background_time = (time.time() - background_start_time) * 1000

            # Report results
            if components_failed:
                logger.warning(
                    f"⚠️ Background initialization completed with issues in {background_time:.1f}ms:\n"
                    f"   ✅ Successful: {components_initialized}\n"
                    f"   ❌ Failed: {[name for name, _ in components_failed]}"
                )
            else:
                logger.info(
                    f"✅ Background initialization completed successfully in {background_time:.1f}ms\n"
                    f"   Components: {components_initialized}"
                )

        except Exception as e:
            logger.error(
                f"❌ Critical error in background initialization: {e}", exc_info=True
            )
            components_failed.append(("background_initialization", str(e)))

    async def _wait_for_critical_components(self):
        """Wait for critical components needed for immediate operation with comprehensive validation."""
        critical_components = {
            "ripgrep": False,
            "search_index_basic": False,
            "prefetcher_basic": False,
        }

        logger.debug("Validating critical components...")

        # Test basic ripgrep functionality
        try:
            logger.debug("Testing RipgrepTurbo...")
            rg = RipgrepTurbo()

            # Perform a minimal test search with timeout
            config = get_einstein_config()
            test_results = await asyncio.wait_for(
                rg.search("import", ".", max_results=1),
                timeout=config.performance.max_search_time_ms
                / 200.0,  # Proportional timeout based on search target
            )

            if test_results is not None:
                critical_components["ripgrep"] = True
                logger.debug("✅ RipgrepTurbo is functional")
            else:
                logger.warning("⚠️ RipgrepTurbo returned None results")

        except TimeoutError:
            logger.error("❌ RipgrepTurbo test search timed out")
        except Exception as e:
            logger.error(f"❌ RipgrepTurbo validation failed: {e}")

        # Test basic search index availability
        try:
            logger.debug("Testing search index basic functionality...")
            if hasattr(self.search_index, "hot_text_cache"):
                # Search index structure is available
                critical_components["search_index_basic"] = True
                logger.debug("✅ Search index basic structure is available")
            else:
                logger.warning("⚠️ Search index structure not properly initialized")
        except Exception as e:
            logger.error(f"❌ Search index basic validation failed: {e}")

        # Test basic prefetcher availability
        try:
            logger.debug("Testing prefetcher basic functionality...")
            if hasattr(self.prefetcher, "prefetch_cache"):
                # Prefetcher structure is available
                critical_components["prefetcher_basic"] = True
                logger.debug("✅ Prefetcher basic structure is available")
            else:
                logger.warning("⚠️ Prefetcher structure not properly initialized")
        except Exception as e:
            logger.error(f"❌ Prefetcher basic validation failed: {e}")

        # Evaluate critical component status
        ready_count = sum(critical_components.values())
        total_count = len(critical_components)

        logger.info(f"Critical components status: {ready_count}/{total_count} ready")

        if ready_count == 0:
            raise RuntimeError("No critical components are functional")
        elif ready_count < total_count:
            logger.warning(
                f"⚠️ Only {ready_count}/{total_count} critical components are ready - proceeding with limited functionality"
            )
        else:
            logger.info("✅ All critical components are ready")

        return ready_count >= 1  # At least one critical component must be ready

    async def _monitor_background_initialization(self):
        """Monitor the completion of background initialization."""
        try:
            if self._init_task and not self._init_task.done():
                # Wait for background initialization with timeout
                config = get_einstein_config()
                await asyncio.wait_for(
                    self._init_task,
                    timeout=config.performance.max_background_init_ms / 1000.0 * 15,
                )  # 15x background init time
                logger.info("✅ Background initialization monitoring completed")
            else:
                logger.debug(
                    "Background initialization already completed or not started"
                )
        except TimeoutError:
            logger.warning("⚠️ Background initialization monitoring timed out")
        except Exception as e:
            logger.error(f"❌ Background initialization monitoring failed: {e}")

    async def _attempt_recovery_initialization(self):
        """Attempt recovery initialization with minimal components."""
        logger.info("🔧 Attempting recovery initialization...")

        try:
            # Try to initialize just the most basic components
            logger.debug("Initializing minimal RipgrepTurbo...")
            test_rg = RipgrepTurbo()

            # Try a very simple search to verify it works
            test_result = await test_rg.search(".", ".", max_results=1)

            if test_result is not None:
                logger.info(
                    "✅ Recovery initialization successful - basic search functionality available"
                )
                self.initialization_complete = True
                return True
            else:
                logger.error(
                    "❌ Recovery initialization failed - basic search not working"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Recovery initialization failed: {e}")
            return False

    async def _fallback_initialization(self):
        """Fallback initialization with absolute minimal functionality."""
        logger.warning(
            "🆘 Attempting fallback initialization with minimal functionality..."
        )

        try:
            # Create minimal search functionality
            class MinimalSearchIndex:
                def __init__(self):
                    self.hot_text_cache = {}

                async def search_streaming(self, query: str, max_results: int = 20):
                    """Stream search results using RipgrepTurbo fallback.

                    Args:
                        query: Search query string
                        max_results: Maximum number of results to return

                    Yields:
                        dict: Search result with content, file_path, line_number,
                              score, result_type, and source fields.
                    """
                    # Minimal search using just RipgrepTurbo
                    try:
                        rg = RipgrepTurbo()
                        results = await rg.search(query, ".", max_results=max_results)
                        for result in results:
                            yield {
                                "content": result.get("content", ""),
                                "file_path": result.get("file", ""),
                                "line_number": result.get("line", 0),
                                "score": 1.0,
                                "result_type": "text",
                                "source": "fallback_search",
                            }
                    except Exception as e:
                        logger.error(f"Fallback search failed: {e}")
                        return

            # Replace search index with minimal version
            self.search_index = MinimalSearchIndex()

            # Create minimal prefetcher
            class MinimalPrefetcher:
                def __init__(self):
                    self.config = self.config if hasattr(self, "config") else None

                async def get_prefetched_results(self, query: str):
                    """Get prefetched results (fallback mode).

                    Args:
                        query: Search query string

                    Returns:
                        None: No prefetching available in fallback mode.
                    """
                    return None  # No prefetching in fallback mode

                def should_prefetch(self, query: str):
                    """Check if query should trigger prefetching (fallback mode).

                    Args:
                        query: Search query string

                    Returns:
                        bool: Always False in fallback mode.
                    """
                    return False  # No prefetching in fallback mode

            self.prefetcher = MinimalPrefetcher()

            logger.warning(
                "⚠️ Fallback initialization completed - limited functionality available"
            )

        except Exception as e:
            logger.error(f"❌ Fallback initialization failed: {e}")
            raise RuntimeError(f"All initialization methods failed: {e}") from e

    def _initialize_claude_patterns(self):
        """Initialize Claude-specific patterns for code analysis."""
        return {
            "optimization": [
                "TODO",
                "FIXME",
                "OPTIMIZE",
                "SLOW",
                "bottleneck",
                "performance",
            ],
            "error_handling": ["error", "exception", "try", "catch", "raise", "Error"],
            "complexity": ["complex", "refactor", "simplify", "nested", "duplicate"],
            "structure": ["class", "def ", "async def", "import", "from"],
            "testing": ["test_", "assert", "mock", "pytest", "unittest"],
            "documentation": ["docstring", '"""', "'''", "README", "docs"],
        }

    async def rapid_search(
        self, query: str, max_results: int = 20
    ) -> list[dict[str, Any]]:
        """Perform rapid search optimized for Claude Code CLI."""
        start_time = time.time()

        # Ensure lazy initialization
        await self.initialize_lazy()

        results = []

        # Check prefetch cache first
        prefetched = await self.prefetcher.get_prefetched_results(query)
        if prefetched:
            results.extend(prefetched[:max_results])
            logger.debug(f"Used prefetched results for query: {query}")

        # If we don't have enough results, search normally
        if len(results) < max_results:
            remaining_results = max_results - len(results)

            # Use streaming search for fastest response
            async for result in self.search_index.search_streaming(
                query, remaining_results
            ):
                results.append(result)

                # Early termination if we have enough results
                elapsed_ms = (time.time() - start_time) * 1000
                if (
                    elapsed_ms > self.config.performance.max_search_time_ms * 0.8
                ):  # Target 80% of max
                    break

        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)

        # Log performance warning if we exceed target
        if search_time > self.config.max_search_time_ms:
            logger.warning(
                f"Search time {search_time:.1f}ms exceeds target {self.config.max_search_time_ms}ms"
            )

        # Trigger prefetching for related queries
        if self.prefetcher.should_prefetch(query):
            asyncio.create_task(self._trigger_related_prefetch(query))

        return results

    async def _trigger_related_prefetch(self, query: str):
        """Trigger prefetching for queries related to current query."""
        # Generate related query patterns
        related_patterns = [
            f"{query}*",
            f"*{query}*",
            query.split()[0] if " " in query else query[:3],
        ]

        for pattern in related_patterns:
            if pattern != query:
                asyncio.create_task(self.prefetcher._prefetch_pattern(pattern))

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        if not self.search_times:
            return {"status": "No searches performed yet"}

        avg_search_time = sum(self.search_times) / len(self.search_times)
        max_search_time = max(self.search_times)
        min_search_time = min(self.search_times)

        # Calculate cache hit rate
        total_searches = len(self.search_times)
        config = get_einstein_config()
        estimated_cache_hits = len(
            [
                t
                for t in self.search_times
                if t < config.performance.target_text_search_ms * 2
            ]
        )  # Fast searches likely cache hits
        cache_hit_rate = (
            estimated_cache_hits / total_searches if total_searches > 0 else 0
        )

        return {
            "avg_search_time_ms": round(avg_search_time, 1),
            "max_search_time_ms": round(max_search_time, 1),
            "min_search_time_ms": round(min_search_time, 1),
            "total_searches": total_searches,
            "cache_hit_rate": round(cache_hit_rate * 100, 1),
            "target_met_percentage": round(
                len(
                    [
                        t
                        for t in self.search_times
                        if t <= self.config.max_search_time_ms
                    ]
                )
                / total_searches
                * 100,
                1,
            ),
            "memory_usage_mb": round(
                psutil.Process().memory_info().rss / 1024 / 1024, 1
            ),
        }

    async def optimize_for_claude_code(self):
        """Apply Claude Code CLI specific optimizations."""
        logger.info("Applying Claude Code CLI optimizations...")

        # Optimize for common Claude Code patterns
        claude_patterns = [
            "class ",
            "def ",
            "async def",
            "import ",
            "from ",
            "TODO",
            "FIXME",
            "NOTE",
            "Exception",
            "Error",
            "@property",
            "@staticmethod",
            "@classmethod",
            "if __name__",
            "test_",
            "pytest",
            "assert",
            "raise",
            "try:",
            "except:",
            "finally:",
            "with ",
            "async with",
            "await ",
            "yield ",
            "return ",
            "print(",
            "logging",
            "logger",
            "config",
            "settings",
        ]

        # Pre-warm caches with Claude Code patterns
        for pattern in claude_patterns:
            asyncio.create_task(self.prefetcher._prefetch_pattern(pattern))

        logger.info(
            f"Pre-warming cache with {len(claude_patterns)} Claude Code patterns"
        )


# Global optimizer instance
_optimizer: ClaudeCodeOptimizer | None = None


def get_claude_code_optimizer() -> ClaudeCodeOptimizer:
    """Get the global Claude Code optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ClaudeCodeOptimizer()
    return _optimizer


async def benchmark_claude_code_performance():
    """Benchmark Claude Code optimizer performance."""
    print("🚀 Benchmarking Claude Code Optimizer Performance...")

    optimizer = get_claude_code_optimizer()

    # Test startup time
    startup_start = time.time()
    await optimizer.initialize_lazy()
    startup_time = (time.time() - startup_start) * 1000

    print(
        f"📊 Startup time: {startup_time:.1f}ms (target: {optimizer.config.max_startup_time_ms}ms)"
    )

    # Test search performance with Claude Code patterns
    test_queries = [
        "class WheelStrategy",
        "def calculate_delta",
        "import pandas",
        "async def process",
        "Exception",
        "TODO",
        "test_",
        "@property",
        "if __name__",
        "logging",
    ]

    search_times = []

    for query in test_queries:
        start_time = time.time()
        results = await optimizer.rapid_search(query, max_results=10)
        search_time = (time.time() - start_time) * 1000
        search_times.append(search_time)

        print(f"📊 '{query}': {search_time:.1f}ms, {len(results)} results")

    # Performance summary
    avg_search_time = sum(search_times) / len(search_times)
    max_search_time = max(search_times)
    target_met = len(
        [t for t in search_times if t <= optimizer.config.max_search_time_ms]
    )

    print("\n📈 Performance Summary:")
    print(f"   Average search time: {avg_search_time:.1f}ms")
    print(f"   Maximum search time: {max_search_time:.1f}ms")
    print(
        f"   Target met: {target_met}/{len(search_times)} queries ({target_met/len(search_times)*100:.1f}%)"
    )
    print(f"   Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")

    # Performance rating
    if avg_search_time <= 25 and startup_time <= 500:
        print("🏆 EXCELLENT: Exceeds Claude Code CLI performance targets")
    elif avg_search_time <= 40 and startup_time <= 750:
        print("✅ GOOD: Meets Claude Code CLI performance targets")
    elif avg_search_time <= 60 and startup_time <= 1000:
        print("⚠️ ACCEPTABLE: Close to Claude Code CLI performance targets")
    else:
        print("❌ NEEDS IMPROVEMENT: Below Claude Code CLI performance targets")


if __name__ == "__main__":
    asyncio.run(benchmark_claude_code_performance())
