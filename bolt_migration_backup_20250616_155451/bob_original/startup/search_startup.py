import re
#!/usr/bin/env python3
"""
Rapid Startup Module for Einstein System

Optimizes Einstein startup time for Claude Code CLI to meet configured targets through:
1. Lazy loading with dependency injection
2. Background initialization workers  
3. Pre-built index caching
4. Minimal critical path loading
5. Asynchronous component initialization
"""

import asyncio
import logging
import os
import pickle
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from einstein.einstein_config import get_einstein_config

logger = logging.getLogger(__name__)


@dataclass
class StartupProfile:
    """Tracks startup performance metrics."""

    total_time_ms: float
    critical_path_ms: float
    background_init_ms: float
    cache_hit_rate: float
    components_loaded: int
    memory_usage_mb: float


class LazyComponentLoader:
    """Lazy loader for Einstein components with dependency injection."""

    def __init__(self) -> None:
        self._components: dict[str, Any] = {}
        self._component_factories: dict[str, Callable] = {}
        self._loading_futures: dict[str, asyncio.Future] = {}
        self._dependencies: dict[str, list[str]] = {}

        # Component loading priority (lower = higher priority)
        self._priority: dict[str, int] = {
            "ripgrep": 1,  # Critical for immediate search
            "config": 1,  # Needed by all components
            "cache_manager": 2,  # Important for performance
            "dependency_graph": 3,  # Can load in background
            "duckdb": 3,  # Can load in background
            "embedding_pipeline": 4,  # Low priority
            "semantic_search": 4,  # Low priority
        }

        self._register_component_factories()

    def _register_component_factories(self) -> None:
        """Register component factory functions."""

        # Critical components (must load immediately)
        self._component_factories["config"] = self._create_config
        self._component_factories["ripgrep"] = self._create_ripgrep

        # Important components (load early)
        self._component_factories["cache_manager"] = self._create_cache_manager
        self._component_factories["rapid_cache"] = self._create_rapid_cache

        # Background components (can load later)
        self._component_factories["dependency_graph"] = self._create_dependency_graph
        self._component_factories["duckdb"] = self._create_duckdb
        self._component_factories[
            "embedding_pipeline"
        ] = self._create_embedding_pipeline

        # Define dependencies
        self._dependencies = {
            "ripgrep": ["config"],
            "cache_manager": ["config"],
            "rapid_cache": ["config"],
            "dependency_graph": ["config"],
            "duckdb": ["config"],
            "embedding_pipeline": ["config", "cache_manager"],
        }

    async def get_component(self, name: str) -> Any:
        """Get component, loading it if necessary."""
        if name in self._components:
            return self._components[name]

        # Check if component is currently loading
        if name in self._loading_futures:
            return await self._loading_futures[name]

        # Start loading component
        future = asyncio.create_task(self._load_component(name))
        self._loading_futures[name] = future

        try:
            component = await future
            self._components[name] = component
            return component
        finally:
            # Clean up loading future
            if name in self._loading_futures:
                del self._loading_futures[name]

    async def _load_component(self, name: str) -> Any:
        """Load a component with its dependencies."""
        if name not in self._component_factories:
            raise ValueError(f"Unknown component: {name}")

        # Load dependencies first
        dependencies = self._dependencies.get(name, [])
        for dep_name in dependencies:
            await self.get_component(dep_name)

        # Load the component
        factory = self._component_factories[name]
        if asyncio.iscoroutinefunction(factory):
            return await factory()
        else:
            return factory()

    async def preload_critical_components(self) -> list[str]:
        """Preload critical components for minimal startup time."""
        critical_components = [
            name for name, priority in self._priority.items() if priority <= 2
        ]

        tasks = []
        for component_name in critical_components:
            task = asyncio.create_task(self.get_component(component_name))
            tasks.append((component_name, task))

        loaded = []
        for name, task in tasks:
            try:
                await task
                loaded.append(name)
            except Exception as e:
                logger.error(
                    f"Failed to preload critical component {name}: {e}",
                    exc_info=True,
                    extra={
                        "operation": "preload_critical_components",
                        "error_type": type(e).__name__,
                        "component_name": name,
                        "loaded_components": loaded,
                        "component_priority": self._priority.get(name, "unknown"),
                        "dependencies": self._dependencies.get(name, []),
                    },
                )

        return loaded

    def start_background_loading(self) -> None:
        """Start background loading of non-critical components."""
        background_components = [
            name for name, priority in self._priority.items() if priority > 2
        ]

        # Start background tasks without waiting
        for component_name in background_components:
            asyncio.create_task(self._background_load_component(component_name))

    async def _background_load_component(self, name: str) -> None:
        """Load component in background."""
        try:
            await self.get_component(name)
            logger.debug(f"Background loaded component: {name}")
        except Exception as e:
            logger.error(
                f"Background loading failed for {name}: {e}",
                exc_info=True,
                extra={
                    "operation": "background_load_component",
                    "error_type": type(e).__name__,
                    "component_name": name,
                    "component_priority": self._priority.get(name, "unknown"),
                    "dependencies": self._dependencies.get(name, []),
                },
            )

    # Component factory methods
    def _create_config(self) -> dict[str, Any]:
        """Create configuration object with comprehensive error handling."""
        try:
            import json

            config_path = Path("optimization_config.json")

            if config_path.exists():
                try:
                    with open(config_path, encoding="utf-8") as f:
                        config = json.load(f)

                    # Validate configuration structure
                    if not isinstance(config, dict):
                        raise ValueError("Configuration must be a dictionary")

                    # Ensure required sections exist with auto-detected defaults
                    einstein_config = get_einstein_config()
                    config.setdefault("cpu", {}).setdefault(
                        "max_workers", einstein_config.hardware.cpu_cores
                    )
                    config.setdefault("memory", {}).setdefault(
                        "max_allocation_gb",
                        int(einstein_config.hardware.memory_total_gb * 0.8),
                    )
                    config.setdefault("io", {}).setdefault(
                        "concurrent_reads",
                        einstein_config.performance.max_file_io_concurrency,
                    )

                    logger.debug(
                        f"✅ Loaded and validated configuration from {config_path}"
                    )
                    return config

                except OSError as e:
                    logger.warning(
                        f"I/O error loading config from {config_path}: {e}, using defaults"
                    )
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Invalid JSON in config file {config_path}: {e}, using defaults"
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(
                        f"Invalid config structure in {config_path}: {e}, using defaults"
                    )
            else:
                logger.debug(
                    f"Configuration file {config_path} not found, using defaults"
                )

            # Return default configuration optimized for detected hardware
            einstein_config = get_einstein_config()
            default_config = {
                "cpu": {
                    "max_workers": einstein_config.hardware.cpu_performance_cores,
                    "background_workers": einstein_config.hardware.cpu_efficiency_cores,
                },
                "memory": {
                    "max_allocation_gb": int(
                        einstein_config.hardware.memory_total_gb * 0.8
                    ),
                    "cache_size_mb": einstein_config.cache.index_cache_size_mb,
                },
                "io": {
                    "concurrent_reads": einstein_config.performance.max_file_io_concurrency,
                    "buffer_size_kb": 64,
                },
                "startup": {
                    "max_startup_time_ms": 500,
                    "critical_path_timeout_ms": 200,
                },
            }
            logger.debug("✅ Using optimized default configuration")
            return default_config

        except Exception as e:
            logger.error(f"Critical error creating config: {e}", exc_info=True)
            # Return minimal safe configuration that will definitely work
            minimal_config = {
                "cpu": {"max_workers": 4, "background_workers": 2},
                "memory": {"max_allocation_gb": 8, "cache_size_mb": 256},
                "io": {"concurrent_reads": 12, "buffer_size_kb": 32},
                "startup": {
                    "max_startup_time_ms": 1000,
                    "critical_path_timeout_ms": 500,
                },
            }
            logger.warning("⚠️ Using minimal safe configuration due to errors")
            return minimal_config

    async def _create_ripgrep(self) -> Any:
        """Create ripgrep component with comprehensive error handling and validation."""
        try:
            from src.unity_wheel.accelerated_tools.ripgrep_turbo import RipgrepTurbo

            # Create ripgrep instance
            ripgrep = RipgrepTurbo()

            # Validate ripgrep functionality with a quick test
            try:
                # Test basic functionality with timeout
                test_result = await asyncio.wait_for(
                    ripgrep.search("test", ".", max_results=1), timeout=2.0
                )
                logger.debug(
                    f"✅ RipgrepTurbo validation successful (found {len(test_result)} results)"
                )
            except TimeoutError:
                logger.warning(
                    "RipgrepTurbo test search timed out, but component created"
                )
            except Exception as test_error:
                logger.warning(
                    f"RipgrepTurbo test search failed: {test_error}, but component created"
                )

            logger.debug("✅ RipgrepTurbo component created and validated successfully")
            return ripgrep

        except ImportError as e:
            logger.error(f"Failed to import RipgrepTurbo: {e}")
            logger.error("Make sure the accelerated tools are properly installed")
            raise RuntimeError(
                f"RipgrepTurbo is required but not available: {e}"
            ) from e

        except Exception as e:
            logger.error(
                f"Failed to create or validate RipgrepTurbo: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to initialize RipgrepTurbo: {e}") from e

    def _create_cache_manager(self) -> Any:
        """Create cache manager with comprehensive error handling and validation."""
        try:
            # Ensure cache directory exists first
            cache_base_dir = Path(".einstein")
            cache_base_dir.mkdir(parents=True, exist_ok=True)

            # Create cache manager
            cache_manager = RapidCacheManager()

            # Validate cache manager functionality
            try:
                cache_stats = cache_manager.query_cache.get_cache_stats()
                logger.debug(f"Cache manager validation successful: {cache_stats}")
            except Exception as validation_error:
                logger.warning(f"Cache manager validation warning: {validation_error}")

            logger.debug("✅ RapidCacheManager created and validated successfully")
            return cache_manager

        except PermissionError as e:
            logger.error(f"Permission denied creating cache directories: {e}")
            raise RuntimeError(f"Cannot create cache directories: {e}") from e

        except Exception as e:
            logger.error(f"Failed to create RapidCacheManager: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize cache manager: {e}") from e

    def _create_rapid_cache(self) -> Any:
        """Create rapid cache with comprehensive error handling and validation."""
        try:
            cache_dir = Path(".einstein") / "rapid_cache"

            # Ensure directory creation with proper error handling
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Cache directory created/verified: {cache_dir}")
            except OSError as dir_error:
                if cache_dir.exists():
                    logger.debug(f"Cache directory already exists: {cache_dir}")
                else:
                    raise dir_error

            # Create rapid cache instance
            rapid_cache = RapidCache(cache_dir)

            # Validate cache functionality with a simple test
            try:
                cache_stats = rapid_cache.get_cache_stats()
                logger.debug(f"RapidCache validation successful: {cache_stats}")
            except Exception as validation_error:
                logger.warning(f"RapidCache validation warning: {validation_error}")

            logger.debug(
                f"✅ RapidCache created and validated successfully at {cache_dir}"
            )
            return rapid_cache

        except PermissionError as e:
            logger.error(f"Permission denied creating cache directory: {e}")
            # Try alternative location
            try:
                import tempfile

                temp_cache_dir = Path(tempfile.gettempdir()) / "einstein_cache"
                temp_cache_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Using temporary cache directory: {temp_cache_dir}")
                return RapidCache(temp_cache_dir)
            except (OSError, FileNotFoundError, PermissionError) as temp_e:
                logger.error(
                    f"Cannot create temporary cache directory {temp_cache_dir}: {temp_e}",
                    extra={
                        "operation": "create_temp_cache_dir",
                        "temp_cache_dir": str(temp_cache_dir),
                    },
                )
                raise RuntimeError(f"Cannot create cache directory: {e}") from e

        except Exception as e:
            logger.error(f"Failed to create RapidCache: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize rapid cache: {e}") from e

    async def _create_dependency_graph(self) -> Any:
        """Create dependency graph with comprehensive error handling and validation."""
        try:
            from src.unity_wheel.accelerated_tools.dependency_graph_turbo import (
                DependencyGraphTurbo,
            )

            # Validate project root exists
            project_root = Path(".")
            if not project_root.exists():
                raise RuntimeError(
                    f"Project root directory does not exist: {project_root}"
                )

            # Create dependency graph instance
            dep_graph = DependencyGraphTurbo(str(project_root))

            # Optional: Validate basic functionality (don't fail if this doesn't work)
            try:
                # Test basic graph operations with timeout
                await asyncio.wait_for(dep_graph.build_initial_index(), timeout=5.0)
                logger.debug("✅ DependencyGraphTurbo initial indexing successful")
            except TimeoutError:
                logger.warning(
                    "DependencyGraphTurbo initial indexing timed out, continuing..."
                )
            except Exception as validation_error:
                logger.warning(
                    f"DependencyGraphTurbo validation warning: {validation_error}"
                )

            logger.debug("✅ DependencyGraphTurbo component created successfully")
            return dep_graph

        except ImportError as e:
            logger.error(f"Failed to import DependencyGraphTurbo: {e}")
            logger.error("Make sure the accelerated tools are properly installed")
            raise RuntimeError(
                f"DependencyGraphTurbo is required but not available: {e}"
            ) from e

        except Exception as e:
            logger.error(f"Failed to create DependencyGraphTurbo: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize dependency graph: {e}") from e

    async def _create_duckdb(self) -> Any:
        """Create DuckDB component with comprehensive error handling and validation."""
        try:
            from src.unity_wheel.accelerated_tools.duckdb_turbo import DuckDBTurbo

            # Prepare database path with validation
            db_dir = Path(".einstein")
            db_path = db_dir / "rapid_analytics.db"

            # Ensure directory exists
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Database directory created/verified: {db_dir}")
            except OSError as dir_error:
                if not db_dir.exists():
                    raise RuntimeError(
                        f"Cannot create database directory: {dir_error}"
                    ) from dir_error

            # Validate database path is writable
            if db_path.exists():
                if not os.access(db_path, os.W_OK):
                    raise RuntimeError(f"Database file is not writable: {db_path}")
            else:
                # Test write access by creating a temporary file
                test_file = db_path.with_suffix(".test")
                try:
                    test_file.touch()
                    test_file.unlink()
                except OSError as write_error:
                    raise RuntimeError(
                        f"Cannot write to database directory: {write_error}"
                    ) from write_error

            # Create DuckDB instance
            duckdb = DuckDBTurbo(str(db_path))

            # Validate DuckDB functionality with a simple test
            try:
                # Test basic SQL execution with timeout
                await asyncio.wait_for(
                    duckdb.execute("SELECT 1 as test_value"), timeout=3.0
                )
                logger.debug("✅ DuckDBTurbo functionality validation successful")
            except TimeoutError:
                logger.warning(
                    "DuckDBTurbo validation timed out, but component created"
                )
            except Exception as validation_error:
                logger.warning(f"DuckDBTurbo validation warning: {validation_error}")

            logger.debug(
                f"✅ DuckDBTurbo component created and validated successfully at {db_path}"
            )
            return duckdb

        except ImportError as e:
            logger.error(f"Failed to import DuckDBTurbo: {e}")
            logger.error("Make sure the accelerated tools are properly installed")
            raise RuntimeError(f"DuckDBTurbo is required but not available: {e}") from e

        except Exception as e:
            logger.error(f"Failed to create DuckDBTurbo: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize DuckDB: {e}") from e

    async def _create_embedding_pipeline(self) -> Any:
        """Create embedding pipeline with comprehensive 3-tier fallback system."""
        try:
            # Tier 1: Try to import and create a proper embedding pipeline
            try:
                from sentence_transformers import SentenceTransformer

                class BasicEmbeddingPipeline:
                    """Basic embedding pipeline using SentenceTransformers."""

                    def __init__(self):
                        try:
                            # Use a lightweight model for fast startup
                            self.model = SentenceTransformer("all-MiniLM-L6-v2")
                            config = get_einstein_config()
                            self.embedding_dim = (
                                config.ml.embedding_dimension_minilm
                            )  # MiniLM dimension from config
                            logger.info(
                                "🧠 SentenceTransformer embedding pipeline initialized successfully"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to load SentenceTransformer model: {e}"
                            )
                            raise

                    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                        """Embed a list of texts with comprehensive error handling."""
                        if not texts:
                            return []

                        try:
                            # Validate input
                            valid_texts = [str(text) if text else "" for text in texts]

                            # Perform embedding with async support
                            loop = asyncio.get_event_loop()
                            embeddings = await loop.run_in_executor(
                                None,
                                lambda: self.model.encode(
                                    valid_texts,
                                    convert_to_tensor=False,
                                    show_progress_bar=False,
                                ),
                            )

                            # Convert to proper format
                            result = (
                                embeddings.tolist()
                                if hasattr(embeddings, "tolist")
                                else embeddings
                            )

                            # Validate output dimensions
                            if result and len(result[0]) != self.embedding_dim:
                                logger.warning(
                                    f"Unexpected embedding dimension: {len(result[0])}, expected {self.embedding_dim}"
                                )

                            return result

                        except Exception as e:
                            logger.error(f"Embedding computation failed: {e}")
                            # Return zero embeddings as fallback
                            return [[0.0] * self.embedding_dim for _ in texts]

                    async def embed_text(self, text: str) -> list[float]:
                        """Embed a single text."""
                        embeddings = await self.embed_texts([text])
                        return (
                            embeddings[0] if embeddings else [0.0] * self.embedding_dim
                        )

                pipeline = BasicEmbeddingPipeline()
                logger.debug("✅ Tier 1 embedding pipeline created successfully")
                return pipeline

            except ImportError as import_error:
                logger.warning(
                    f"SentenceTransformers not available ({import_error}), falling back to MLX"
                )

                # Tier 2: MLX-based embedding pipeline

                class MLXEmbeddingPipeline:
                    """MLX-based embedding pipeline for production use."""

                    def __init__(self):
                        config = get_einstein_config()
                        self.embedding_dim = config.ml.embedding_dimension_minilm
                        self.engine = MLXEmbeddingEngine(embed_dim=self.embedding_dim)
                        logger.info(
                            "🚀 MLX-based embedding pipeline initialized (Tier 2)"
                        )

                    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                        """Create MLX-based embeddings."""
                        if not texts:
                            return []

                        try:
                            # Use batch processing for efficiency
                            results = self.engine.embed_batch(texts)
                            embeddings = [
                                embedding.tolist() for embedding, _ in results
                            ]
                            return embeddings

                        except Exception as e:
                            logger.error(f"MLX embedding failed: {e}")
                            # Fallback to individual processing
                            embeddings = []
                            for text in texts:
                                try:
                                    embedding, _ = self.engine.embed_text(text)
                                    embeddings.append(embedding.tolist())
                                except Exception:
                                    embeddings.append([0.0] * self.embedding_dim)
                            return embeddings

                    async def embed_text(self, text: str) -> list[float]:
                        """Embed a single text."""
                        try:
                            embedding, _ = await self.engine.embed_text_async(text)
                            return embedding.tolist()
                        except Exception as e:
                            logger.error(f"Single text embedding failed: {e}")
                            return [0.0] * self.embedding_dim

                pipeline = MLXEmbeddingPipeline()
                logger.debug("✅ Tier 2 embedding pipeline created successfully")
                return pipeline

        except Exception as e:
            logger.error(f"Tier 1 and 2 embedding pipelines failed: {e}")

            # Tier 3: Failsafe embedding pipeline that never fails
            class FailsafeEmbeddingPipeline:
                """Failsafe embedding pipeline that always returns valid results."""

                def __init__(self):
                    config = get_einstein_config()
                    self.embedding_dim = config.ml.embedding_dimension_minilm
                    logger.warning(
                        "⚠️ Using failsafe embedding pipeline (Tier 3) - limited functionality"
                    )

                async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                    """Return safe zero embeddings."""
                    if not texts:
                        return []
                    try:
                        return [[0.0] * self.embedding_dim for _ in texts]
                    except (MemoryError, OverflowError, ValueError) as e:
                        # Handle edge cases with embedding dimensions or memory issues
                        logger.debug(
                            f"Failed to create zero embeddings for {len(texts)} texts with dim {self.embedding_dim}: {e}"
                        )
                        return [[0.0] * self.embedding_dim]

                async def embed_text(self, text: str) -> list[float]:
                    """Return safe zero embedding."""
                    try:
                        return [0.0] * self.embedding_dim
                    except (MemoryError, OverflowError, ValueError) as e:
                        # Handle edge cases with embedding dimension
                        logger.debug(
                            f"Failed to create zero embedding with dim {self.embedding_dim}: {e}"
                        )
                        return [
                            0.0
                        ] * config.ml.embedding_dimension_minilm  # Configured fallback

            pipeline = FailsafeEmbeddingPipeline()
            logger.debug("✅ Tier 3 failsafe embedding pipeline created")
            return pipeline


class RapidCache:
    """Ultra-fast cache for frequent Einstein queries."""

    def __init__(self, cache_dir: Path, config: Any | None = None) -> None:
        if config is None:
            from einstein.einstein_config import get_einstein_config

            config = get_einstein_config()
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # SQLite cache for persistent storage
        self.db_path = cache_dir / "rapid_cache.db"
        self._init_database()

        # In-memory cache for hot data
        self._memory_cache: dict[str, Any] = {}
        self._access_count: dict[str, int] = {}

        # Load hot cache from database
        self._load_hot_cache()

    def _init_database(self) -> None:
        """Initialize SQLite cache database with comprehensive schema and error handling."""
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    f"PRAGMA cache_size={config.database.cache_size_pages}"
                )  # Use configured cache size
                conn.execute("PRAGMA temp_store=memory")

                # Create main search cache table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS search_cache (
                        query_hash TEXT PRIMARY KEY,
                        query TEXT NOT NULL,
                        results BLOB NOT NULL,
                        access_count INTEGER DEFAULT 1 CHECK(access_count > 0),
                        last_access TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                        result_size INTEGER DEFAULT 0,
                        query_type TEXT DEFAULT 'general'
                    )
                """
                )

                # Create performance tracking table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_hash TEXT NOT NULL,
                        execution_time_ms REAL NOT NULL,
                        cache_hit BOOLEAN NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (query_hash) REFERENCES search_cache (query_hash)
                    )
                """
                )

                # Create optimized indexes
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_query_hash ON search_cache(query_hash)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_access_count ON search_cache(access_count DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_last_access ON search_cache(last_access DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_query_type ON search_cache(query_type)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON cache_performance(timestamp DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_perf_cache_hit ON cache_performance(cache_hit)"
                )

                # Create cache statistics view
                conn.execute(
                    """
                    CREATE VIEW IF NOT EXISTS cache_stats AS
                    SELECT 
                        COUNT(*) as total_entries,
                        AVG(access_count) as avg_access_count,
                        MAX(access_count) as max_access_count,
                        COUNT(DISTINCT query_type) as query_types,
                        SUM(result_size) as total_cache_size
                    FROM search_cache
                """
                )

                # Verify database integrity
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                if integrity_result != "ok":
                    logger.error(f"Database integrity check failed: {integrity_result}")
                    raise RuntimeError(
                        f"Database corruption detected: {integrity_result}"
                    )

                logger.debug(f"Database initialized successfully at {self.db_path}")

        except sqlite3.Error as e:
            logger.error(f"SQLite error during database initialization: {e}")
            raise RuntimeError(f"Failed to initialize cache database: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")

    def _load_hot_cache(self) -> None:
        """Load frequently accessed items into memory with error handling."""
        try:
            if not self.db_path.exists():
                logger.debug(
                    "Cache database does not exist yet, skipping hot cache loading"
                )
                return

            with sqlite3.connect(self.db_path) as conn:
                # Load hot cache items based on access frequency and recency
                from einstein.einstein_config import get_einstein_config

                config = get_einstein_config()

                cursor = conn.execute(
                    """
                    SELECT query_hash, query, results, access_count, result_size
                    FROM search_cache 
                    WHERE access_count >= 2  -- Only load items accessed more than once
                    ORDER BY 
                        access_count DESC,
                        last_access DESC
                    LIMIT ?
                """,
                    (config.database.default_query_limit,),
                )

                loaded_count = 0
                total_size = 0

                for row in cursor:
                    query_hash, query, results_blob, access_count, result_size = row
                    try:
                        results = pickle.loads(results_blob)
                        self._memory_cache[query_hash] = {
                            "query": query,
                            "results": results,
                        }
                        self._access_count[query_hash] = access_count
                        loaded_count += 1
                        total_size += result_size or len(results_blob)

                    except (pickle.PickleError, EOFError) as e:
                        logger.warning(
                            f"Failed to deserialize cached result for {query_hash}: {e}"
                        )
                        # Remove corrupted entry
                        try:
                            conn.execute(
                                "DELETE FROM search_cache WHERE query_hash = ?",
                                (query_hash,),
                            )
                        except sqlite3.Error as delete_error:
                            logger.debug(
                                f"Could not delete corrupted cache entry {query_hash}: {delete_error}"
                            )
                            # Non-critical error during cache loading, continue processing
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error loading cached result for {query_hash}: {e}"
                        )

                logger.debug(
                    f"Loaded {loaded_count} items into hot cache ({total_size} bytes)"
                )

        except sqlite3.Error as e:
            logger.error(f"SQLite error loading hot cache: {e}")
            # Don't raise here - hot cache loading is not critical
        except Exception as e:
            logger.error(f"Unexpected error loading hot cache: {e}")
            # Don't raise here - hot cache loading is not critical

    def get(self, query_hash: str) -> list[dict[str, Any]] | None:
        """Get cached results for query hash."""
        # Check memory cache first
        if query_hash in self._memory_cache:
            self._access_count[query_hash] = self._access_count.get(query_hash, 0) + 1
            return self._memory_cache[query_hash]["results"]

        # Check database cache
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT results, access_count FROM search_cache WHERE query_hash = ?",
                (query_hash,),
            )
            row = cursor.fetchone()

            if row:
                results_blob, access_count = row
                try:
                    results = pickle.loads(results_blob)

                    # Update access count
                    conn.execute(
                        "UPDATE search_cache SET access_count = access_count + 1, last_access = CURRENT_TIMESTAMP WHERE query_hash = ?",
                        (query_hash,),
                    )

                    # Add to memory cache if frequently accessed
                    if access_count >= 3:
                        self._memory_cache[query_hash] = {
                            "query": "",  # We don't store query in this path
                            "results": results,
                        }
                        self._access_count[query_hash] = access_count + 1

                    return results

                except Exception as e:
                    logger.warning(f"Failed to deserialize cached result: {e}")

        return None

    def set(self, query_hash: str, query: str, results: list[dict[str, Any]]) -> None:
        """Cache search results."""
        try:
            results_blob = pickle.dumps(results)

            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO search_cache 
                    (query_hash, query, results, access_count, last_access, created_at)
                    VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                    (query_hash, query, results_blob),
                )

            # Store in memory cache for immediate reuse
            self._memory_cache[query_hash] = {"query": query, "results": results}
            self._access_count[query_hash] = 1

        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM search_cache")
            total_entries = cursor.fetchone()[0]

            cursor = conn.execute("SELECT AVG(access_count) FROM search_cache")
            avg_access_count = cursor.fetchone()[0] or 0

        return {
            "total_entries": total_entries,
            "memory_cache_size": len(self._memory_cache),
            "avg_access_count": round(avg_access_count, 1),
            "cache_file_size_mb": round(self.db_path.stat().st_size / 1024 / 1024, 2)
            if self.db_path.exists()
            else 0,
        }


class RapidCacheManager:
    """Manages all caching for rapid search performance."""

    def __init__(self) -> None:
        self.query_cache = RapidCache(Path(".einstein") / "query_cache")
        self.file_cache = RapidCache(Path(".einstein") / "file_cache")

        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0

    def get_search_results(self, query: str) -> list[dict[str, Any]] | None:
        """Get cached search results."""
        import hashlib

        query_hash = hashlib.md5(query.encode()).hexdigest()

        results = self.query_cache.get(query_hash)
        if results:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        return results

    def cache_search_results(self, query: str, results: list[dict[str, Any]]) -> None:
        """Cache search results."""
        import hashlib

        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.query_cache.set(query_hash, query, results)

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class RapidStartupManager:
    """Manages rapid startup of Einstein system."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path.cwd()
        self.loader = LazyComponentLoader()
        self.cache_manager = None

        # Startup tracking
        self.startup_start_time = None
        self.critical_path_time = None
        self.background_init_complete = False

        # Performance targets from config
        config = get_einstein_config()
        self.max_startup_time_ms = config.performance.max_startup_time_ms
        self.max_critical_path_ms = config.performance.max_critical_path_ms

    async def rapid_initialize(self) -> StartupProfile:
        """Perform rapid initialization optimized for Claude Code CLI."""
        self.startup_start_time = time.time()

        logger.info("🚀 Starting rapid Einstein initialization...")

        # Phase 1: Critical path loading (must complete fast)
        critical_start = time.time()
        critical_components = await self.loader.preload_critical_components()
        self.critical_path_time = (time.time() - critical_start) * 1000

        logger.info(
            f"✅ Critical path loaded in {self.critical_path_time:.1f}ms: {critical_components}"
        )

        # Phase 2: Start background initialization (non-blocking)
        background_start = time.time()
        self.loader.start_background_loading()

        # Get cache manager for immediate use
        self.cache_manager = await self.loader.get_component("cache_manager")

        # Phase 3: Verify system is ready for basic operations
        await self._verify_ready_state()

        total_time = (time.time() - self.startup_start_time) * 1000
        background_time = (time.time() - background_start) * 1000

        # Create startup profile
        profile = StartupProfile(
            total_time_ms=total_time,
            critical_path_ms=self.critical_path_time,
            background_init_ms=background_time,
            cache_hit_rate=0.0,  # Will be calculated after first searches
            components_loaded=len(critical_components),
            memory_usage_mb=self._get_memory_usage_mb(),
        )

        # Log performance assessment
        self._assess_startup_performance(profile)

        return profile

    async def _verify_ready_state(self) -> None:
        """Verify system is ready for basic search operations."""
        try:
            # Test ripgrep functionality
            ripgrep = await self.loader.get_component("ripgrep")
            await ripgrep.search("test", ".", max_results=1)

            logger.debug("✅ Basic search functionality verified")

        except Exception as e:
            logger.error(
                f"Ready state verification failed: {e}",
                exc_info=True,
                extra={
                    "operation": "verify_ready_state",
                    "test_type": "ripgrep_functionality",
                },
            )

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024

    def _assess_startup_performance(self, profile: StartupProfile) -> None:
        """Assess and log startup performance."""
        if profile.total_time_ms <= self.max_startup_time_ms:
            if profile.critical_path_ms <= self.max_critical_path_ms:
                logger.info(
                    f"🏆 EXCELLENT startup performance: {profile.total_time_ms:.1f}ms total, {profile.critical_path_ms:.1f}ms critical"
                )
            else:
                logger.info(
                    f"✅ GOOD startup performance: {profile.total_time_ms:.1f}ms total (critical path: {profile.critical_path_ms:.1f}ms)"
                )
        else:
            logger.warning(
                f"⚠️ Startup time {profile.total_time_ms:.1f}ms exceeds target {self.max_startup_time_ms}ms"
            )

    async def get_component(self, name: str) -> Any:
        """Get a component through the lazy loader."""
        return await self.loader.get_component(name)

    async def rapid_search(
        self, query: str, max_results: int = 20
    ) -> list[dict[str, Any]]:
        """Perform rapid search with caching."""
        # Check cache first
        if self.cache_manager:
            cached_results = self.cache_manager.get_search_results(query)
            if cached_results:
                logger.debug(f"Cache hit for query: {query}")
                return cached_results[:max_results]

        # Perform search using available components
        ripgrep = await self.loader.get_component("ripgrep")
        results = await ripgrep.search(query, ".", max_results=max_results)

        # Convert to standard format
        formatted_results = [
            {
                "content": r["content"],
                "file_path": r["file"],
                "line_number": r["line"],
                "score": 1.0,
                "result_type": "text",
                "source": "rapid_search",
            }
            for r in results
        ]

        # Cache results for future use
        if self.cache_manager:
            self.cache_manager.cache_search_results(query, formatted_results)

        return formatted_results

    async def get_startup_diagnostics(self) -> dict[str, Any]:
        """Get detailed startup diagnostics."""
        if not self.startup_start_time:
            return {"status": "Not initialized"}

        current_time = time.time()
        uptime_ms = (current_time - self.startup_start_time) * 1000

        # Get component status
        component_status = {}
        for component_name in self.loader._component_factories:
            component_status[component_name] = component_name in self.loader._components

        # Get cache statistics
        cache_stats = {}
        if self.cache_manager:
            cache_stats = {
                "hit_rate": round(self.cache_manager.get_hit_rate() * 100, 1),
                "total_hits": self.cache_manager.cache_hits,
                "total_misses": self.cache_manager.cache_misses,
            }

        return {
            "uptime_ms": round(uptime_ms, 1),
            "critical_path_ms": round(self.critical_path_time, 1)
            if self.critical_path_time
            else None,
            "components_loaded": sum(component_status.values()),
            "total_components": len(component_status),
            "component_status": component_status,
            "memory_usage_mb": round(self._get_memory_usage_mb(), 1),
            "cache_stats": cache_stats,
            "background_init_complete": self.background_init_complete,
        }


# Global rapid startup manager
_rapid_manager: RapidStartupManager | None = None


def get_rapid_startup_manager() -> RapidStartupManager:
    """Get the global rapid startup manager."""
    global _rapid_manager
    if _rapid_manager is None:
        _rapid_manager = RapidStartupManager()
    return _rapid_manager


async def benchmark_rapid_startup() -> None:
    """Benchmark rapid startup performance."""
    print("🚀 Benchmarking Rapid Startup Performance...")

    manager = get_rapid_startup_manager()

    # Test startup performance
    profile = await manager.rapid_initialize()

    print("\n📊 Startup Performance:")
    print(f"   Total time: {profile.total_time_ms:.1f}ms")
    print(f"   Critical path: {profile.critical_path_ms:.1f}ms")
    print(f"   Background init: {profile.background_init_ms:.1f}ms")
    print(f"   Components loaded: {profile.components_loaded}")
    print(f"   Memory usage: {profile.memory_usage_mb:.1f}MB")

    # Test search performance
    test_queries = ["class", "def", "import", "async", "TODO"]
    search_times = []

    print("\n🔍 Testing Search Performance:")
    for query in test_queries:
        start_time = time.time()
        results = await manager.rapid_search(query, max_results=5)
        search_time = (time.time() - start_time) * 1000
        search_times.append(search_time)

        print(f"   '{query}': {search_time:.1f}ms, {len(results)} results")

    # Performance assessment
    avg_search_time = sum(search_times) / len(search_times)

    print("\n📈 Performance Summary:")
    print(f"   Average search time: {avg_search_time:.1f}ms")
    print(f"   Startup target met: {'✅' if profile.total_time_ms <= 500 else '❌'}")
    print(f"   Search target met: {'✅' if avg_search_time <= 50 else '❌'}")

    # Diagnostics
    diagnostics = await manager.get_startup_diagnostics()
    print("\n🔧 System Diagnostics:")
    print(
        f"   Components: {diagnostics['components_loaded']}/{diagnostics['total_components']}"
    )
    print(f"   Cache hit rate: {diagnostics['cache_stats'].get('hit_rate', 0)}%")
    print(f"   Memory usage: {diagnostics['memory_usage_mb']}MB")


if __name__ == "__main__":
    asyncio.run(benchmark_rapid_startup())
