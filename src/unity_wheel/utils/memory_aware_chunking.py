"""Memory-aware chunking strategies for different data types.

Provides intelligent chunking algorithms that adapt to available system resources
and data characteristics to optimize memory usage and processing performance.
"""

from __future__ import annotations

import asyncio
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

import psutil

from .logging import get_logger
from .recovery import with_recovery
from .stream_processors import DataType, StreamConfig

logger = get_logger(__name__)

T = TypeVar("T")


class ChunkingStrategy(Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"  # Fixed chunk size
    ADAPTIVE = "adaptive"  # Adapt to memory conditions
    CONTENT_AWARE = "content_aware"  # Based on content structure
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Optimize for processing speed
    MEMORY_CONSERVATIVE = "memory_conservative"  # Minimize memory usage


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""

    # Strategy selection
    strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE
    
    # Size configuration
    min_chunk_size: int = 1024  # 1KB minimum
    max_chunk_size: int = 50 * 1024 * 1024  # 50MB maximum
    target_chunk_size: int = 1024 * 1024  # 1MB target
    
    # Memory management
    memory_limit_mb: int = 100  # Memory limit per chunking operation
    system_memory_threshold: float = 0.8  # Use file-based chunking if system memory > 80%
    gc_interval: int = 100  # Run garbage collection every N chunks
    
    # Performance tuning
    parallel_processing: bool = True
    max_concurrent_chunks: int = 4
    prefetch_chunks: int = 2
    
    # Content-aware settings
    respect_boundaries: bool = True  # Respect JSON object/array boundaries
    boundary_chars: set[str] = None  # Characters that indicate boundaries
    
    # Optimization settings
    profile_memory: bool = False
    adaptive_threshold: float = 0.1  # Adjust chunk size if memory usage changes by 10%


@dataclass
class ChunkMetrics:
    """Metrics for chunk processing."""

    chunk_id: int
    size_bytes: int
    processing_time_ms: float
    memory_usage_mb: float
    boundary_respected: bool = True
    optimization_applied: bool = False


class BaseChunkingStrategy(ABC, Generic[T]):
    """Abstract base class for chunking strategies."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.metrics: list[ChunkMetrics] = []
        self._chunk_counter = 0

    @abstractmethod
    async def chunk_data(self, data: T) -> AsyncGenerator[tuple[int, T], None]:
        """Chunk data according to the strategy."""
        pass

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def _get_system_memory_usage(self) -> float:
        """Get system memory usage as percentage."""
        return psutil.virtual_memory().percent / 100

    def _should_use_conservative_chunking(self) -> bool:
        """Determine if conservative chunking should be used."""
        return (
            self._get_system_memory_usage() > self.config.system_memory_threshold
            or self._get_memory_usage_mb() > self.config.memory_limit_mb
        )

    def _calculate_adaptive_chunk_size(self, data_size: int) -> int:
        """Calculate adaptive chunk size based on system conditions."""
        base_size = self.config.target_chunk_size
        
        # Adjust based on data size
        if data_size < base_size:
            return data_size
        
        # Adjust based on memory pressure
        memory_factor = 1.0
        if self._should_use_conservative_chunking():
            memory_factor = 0.5  # Use smaller chunks under memory pressure
        
        # Adjust based on available memory
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        if available_memory_mb < 500:  # Less than 500MB available
            memory_factor *= 0.5
        elif available_memory_mb > 2000:  # More than 2GB available
            memory_factor *= 1.5
        
        adjusted_size = int(base_size * memory_factor)
        
        # Ensure within bounds
        return max(
            self.config.min_chunk_size,
            min(adjusted_size, self.config.max_chunk_size)
        )

    async def _record_metrics(
        self,
        chunk_id: int,
        chunk_size: int,
        start_time: float,
        memory_before: float,
        **kwargs
    ) -> None:
        """Record metrics for a chunk."""
        import time
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        memory_after = self._get_memory_usage_mb()
        
        metrics = ChunkMetrics(
            chunk_id=chunk_id,
            size_bytes=chunk_size,
            processing_time_ms=processing_time,
            memory_usage_mb=memory_after - memory_before,
            **kwargs
        )
        
        self.metrics.append(metrics)
        
        # Log performance issues
        if processing_time > 1000:  # More than 1 second
            logger.warning(f"Slow chunk processing: {processing_time:.1f}ms for chunk {chunk_id}")
        
        if memory_after - memory_before > 50:  # More than 50MB increase
            logger.warning(f"High memory usage: {memory_after - memory_before:.1f}MB for chunk {chunk_id}")

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary of chunking operation."""
        if not self.metrics:
            return {}
        
        total_bytes = sum(m.size_bytes for m in self.metrics)
        total_time = sum(m.processing_time_ms for m in self.metrics)
        total_memory = sum(m.memory_usage_mb for m in self.metrics)
        
        return {
            "total_chunks": len(self.metrics),
            "total_bytes": total_bytes,
            "total_time_ms": total_time,
            "total_memory_mb": total_memory,
            "avg_chunk_size": total_bytes / len(self.metrics),
            "avg_processing_time_ms": total_time / len(self.metrics),
            "avg_memory_per_chunk_mb": total_memory / len(self.metrics),
            "throughput_mbps": (total_bytes / (1024 * 1024)) / (total_time / 1000) if total_time > 0 else 0,
            "boundaries_respected": sum(1 for m in self.metrics if m.boundary_respected),
            "optimizations_applied": sum(1 for m in self.metrics if m.optimization_applied),
        }


class BytesChunkingStrategy(BaseChunkingStrategy[bytes]):
    """Chunking strategy for bytes data."""

    async def chunk_data(self, data: bytes) -> AsyncGenerator[tuple[int, bytes], None]:
        """Chunk bytes data with adaptive sizing."""
        data_size = len(data)
        chunk_size = self._calculate_adaptive_chunk_size(data_size)
        
        offset = 0
        while offset < data_size:
            import time
            start_time = time.time()
            memory_before = self._get_memory_usage_mb()
            
            # Calculate actual chunk size (may be smaller at end)
            actual_chunk_size = min(chunk_size, data_size - offset)
            chunk = data[offset:offset + actual_chunk_size]
            
            chunk_id = self._chunk_counter
            self._chunk_counter += 1
            
            # Record metrics
            await self._record_metrics(
                chunk_id, actual_chunk_size, start_time, memory_before
            )
            
            yield chunk_id, chunk
            offset += actual_chunk_size
            
            # Garbage collection and yielding control
            if chunk_id % self.config.gc_interval == 0:
                import gc
                gc.collect()
                await asyncio.sleep(0)  # Yield control


class TextChunkingStrategy(BaseChunkingStrategy[str]):
    """Chunking strategy for text data with boundary awareness."""

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.boundary_chars = config.boundary_chars or {'\n', '.', '!', '?', ';'}

    def _find_boundary(self, text: str, start: int, end: int) -> int:
        """Find a good boundary for text chunking."""
        if not self.config.respect_boundaries:
            return end
        
        # Look for boundary characters near the end
        search_start = max(start, end - 500)  # Search within 500 chars of end
        
        for i in range(end - 1, search_start - 1, -1):
            if text[i] in self.boundary_chars:
                return i + 1
        
        # No boundary found, use original end
        return end

    async def chunk_data(self, data: str) -> AsyncGenerator[tuple[int, str], None]:
        """Chunk text data with boundary awareness."""
        data_size = len(data.encode('utf-8'))
        base_chunk_size = self._calculate_adaptive_chunk_size(data_size)
        
        # Convert byte size to approximate character count
        # Assume average 2 bytes per character for text
        char_chunk_size = base_chunk_size // 2
        
        offset = 0
        text_length = len(data)
        
        while offset < text_length:
            import time
            start_time = time.time()
            memory_before = self._get_memory_usage_mb()
            
            # Calculate end position
            end_pos = min(offset + char_chunk_size, text_length)
            
            # Find good boundary
            boundary_pos = self._find_boundary(data, offset, end_pos)
            boundary_respected = boundary_pos != end_pos
            
            chunk = data[offset:boundary_pos]
            chunk_size = len(chunk.encode('utf-8'))
            
            chunk_id = self._chunk_counter
            self._chunk_counter += 1
            
            # Record metrics
            await self._record_metrics(
                chunk_id, chunk_size, start_time, memory_before,
                boundary_respected=boundary_respected
            )
            
            yield chunk_id, chunk
            offset = boundary_pos
            
            # Garbage collection and yielding control
            if chunk_id % self.config.gc_interval == 0:
                import gc
                gc.collect()
                await asyncio.sleep(0)


class JSONChunkingStrategy(BaseChunkingStrategy[list]):
    """Chunking strategy for JSON data with object/array awareness."""

    async def chunk_data(self, data: list) -> AsyncGenerator[tuple[int, list], None]:
        """Chunk JSON array data with object boundary awareness."""
        if not isinstance(data, list):
            # Single object, return as-is
            yield 0, data
            return
        
        total_items = len(data)
        if total_items == 0:
            return
        
        # Estimate size of individual items
        sample_size = min(10, total_items)
        sample_items = data[:sample_size]
        avg_item_size = sum(len(str(item)) for item in sample_items) / sample_size
        
        # Calculate items per chunk
        target_chunk_size = self._calculate_adaptive_chunk_size(int(avg_item_size * total_items))
        items_per_chunk = max(1, int(target_chunk_size / avg_item_size))
        
        offset = 0
        while offset < total_items:
            import time
            start_time = time.time()
            memory_before = self._get_memory_usage_mb()
            
            # Get chunk items
            end_pos = min(offset + items_per_chunk, total_items)
            chunk = data[offset:end_pos]
            chunk_size = sum(len(str(item)) for item in chunk)
            
            chunk_id = self._chunk_counter
            self._chunk_counter += 1
            
            # Record metrics
            await self._record_metrics(
                chunk_id, chunk_size, start_time, memory_before,
                boundary_respected=True  # Always respect object boundaries
            )
            
            yield chunk_id, chunk
            offset = end_pos
            
            # Garbage collection and yielding control
            if chunk_id % self.config.gc_interval == 0:
                import gc
                gc.collect()
                await asyncio.sleep(0)


class DataFrameChunkingStrategy(BaseChunkingStrategy):
    """Chunking strategy for DataFrame-like data."""

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)

    async def chunk_data(self, data) -> AsyncGenerator[tuple[int, Any], None]:
        """Chunk DataFrame data by rows."""
        try:
            import pandas as pd
            
            if not isinstance(data, pd.DataFrame):
                yield 0, data
                return
            
            total_rows = len(data)
            if total_rows == 0:
                return
            
            # Estimate memory usage per row
            memory_usage = data.memory_usage(deep=True).sum()
            avg_bytes_per_row = memory_usage / total_rows
            
            # Calculate rows per chunk
            target_chunk_size = self._calculate_adaptive_chunk_size(memory_usage)
            rows_per_chunk = max(1, int(target_chunk_size / avg_bytes_per_row))
            
            offset = 0
            while offset < total_rows:
                import time
                start_time = time.time()
                memory_before = self._get_memory_usage_mb()
                
                # Get chunk rows
                end_pos = min(offset + rows_per_chunk, total_rows)
                chunk = data.iloc[offset:end_pos]
                chunk_size = chunk.memory_usage(deep=True).sum()
                
                chunk_id = self._chunk_counter
                self._chunk_counter += 1
                
                # Record metrics
                await self._record_metrics(
                    chunk_id, chunk_size, start_time, memory_before,
                    boundary_respected=True  # Always respect row boundaries
                )
                
                yield chunk_id, chunk
                offset = end_pos
                
                # Garbage collection and yielding control
                if chunk_id % self.config.gc_interval == 0:
                    import gc
                    gc.collect()
                    await asyncio.sleep(0)
                    
        except ImportError:
            # pandas not available, treat as regular data
            yield 0, data


class AdaptiveChunker:
    """Adaptive chunker that selects the best strategy based on data type."""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self._strategies: dict[type, BaseChunkingStrategy] = {}

    def _get_strategy(self, data: Any) -> BaseChunkingStrategy:
        """Get appropriate chunking strategy for data type."""
        data_type = type(data)
        
        if data_type not in self._strategies:
            if isinstance(data, bytes):
                self._strategies[data_type] = BytesChunkingStrategy(self.config)
            elif isinstance(data, str):
                self._strategies[data_type] = TextChunkingStrategy(self.config)
            elif isinstance(data, list):
                self._strategies[data_type] = JSONChunkingStrategy(self.config)
            else:
                try:
                    import pandas as pd
                    if isinstance(data, pd.DataFrame):
                        self._strategies[data_type] = DataFrameChunkingStrategy(self.config)
                    else:
                        # Default to bytes strategy after converting to string
                        self._strategies[data_type] = BytesChunkingStrategy(self.config)
                except ImportError:
                    self._strategies[data_type] = BytesChunkingStrategy(self.config)
        
        return self._strategies[data_type]

    async def chunk_data(self, data: Any) -> AsyncGenerator[tuple[int, Any], None]:
        """Chunk data using the appropriate strategy."""
        strategy = self._get_strategy(data)
        
        # Convert data if needed for bytes strategy
        if isinstance(strategy, BytesChunkingStrategy) and not isinstance(data, bytes):
            if isinstance(data, str):
                data = data.encode('utf-8')
            else:
                data = str(data).encode('utf-8')
        
        async for chunk_id, chunk in strategy.chunk_data(data):
            yield chunk_id, chunk

    def get_strategy_metrics(self, data_type: type) -> dict[str, Any]:
        """Get metrics for a specific data type strategy."""
        if data_type in self._strategies:
            return self._strategies[data_type].get_performance_summary()
        return {}

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all strategies."""
        return {
            str(data_type): strategy.get_performance_summary()
            for data_type, strategy in self._strategies.items()
        }


# Convenience functions
@with_recovery(max_attempts=3, backoff_factor=1.5)
async def chunk_large_data(
    data: Any,
    config: ChunkingConfig | None = None,
    chunk_processor: Callable[[int, Any], Any] | None = None,
) -> AsyncGenerator[tuple[int, Any], None]:
    """Chunk large data with automatic strategy selection."""
    chunker = AdaptiveChunker(config)
    
    async for chunk_id, chunk in chunker.chunk_data(data):
        if chunk_processor:
            processed_chunk = chunk_processor(chunk_id, chunk)
            yield chunk_id, processed_chunk
        else:
            yield chunk_id, chunk


async def chunk_and_process(
    data: Any,
    processor: Callable[[Any], Any],
    config: ChunkingConfig | None = None,
    parallel: bool = True,
) -> list[Any]:
    """Chunk data and process chunks with optional parallelization."""
    results = []
    
    if parallel and (config is None or config.parallel_processing):
        # Process chunks in parallel
        semaphore = asyncio.Semaphore(
            config.max_concurrent_chunks if config else 4
        )
        
        async def process_chunk(chunk_id: int, chunk: Any) -> tuple[int, Any]:
            async with semaphore:
                return chunk_id, processor(chunk)
        
        # Collect all chunks first
        chunks = []
        async for chunk_id, chunk in chunk_large_data(data, config):
            chunks.append((chunk_id, chunk))
        
        # Process in parallel
        tasks = [process_chunk(chunk_id, chunk) for chunk_id, chunk in chunks]
        processed_chunks = await asyncio.gather(*tasks)
        
        # Sort by chunk_id to maintain order
        processed_chunks.sort(key=lambda x: x[0])
        results = [result for _, result in processed_chunks]
    
    else:
        # Process chunks sequentially
        async for chunk_id, chunk in chunk_large_data(data, config):
            result = processor(chunk)
            results.append(result)
    
    return results


def get_optimal_chunk_config(
    data_size_mb: float,
    data_type: DataType,
    available_memory_mb: float,
) -> ChunkingConfig:
    """Get optimal chunking configuration for given constraints."""
    config = ChunkingConfig()
    
    # Adjust strategy based on data size and available memory
    if available_memory_mb < 100:  # Low memory
        config.strategy = ChunkingStrategy.MEMORY_CONSERVATIVE
        config.target_chunk_size = 512 * 1024  # 512KB
        config.max_concurrent_chunks = 2
    elif data_size_mb > 1000:  # Large data
        config.strategy = ChunkingStrategy.PERFORMANCE_OPTIMIZED
        config.target_chunk_size = 10 * 1024 * 1024  # 10MB
        config.max_concurrent_chunks = 8
    else:
        config.strategy = ChunkingStrategy.ADAPTIVE
    
    # Adjust based on data type
    if data_type in (DataType.JSON, DataType.TEXT):
        config.respect_boundaries = True
    elif data_type in (DataType.PARQUET, DataType.ARROW):
        config.target_chunk_size *= 4  # Larger chunks for columnar data
    
    # Set memory limit
    config.memory_limit_mb = min(available_memory_mb * 0.3, 200)  # Use max 30% of available memory
    
    return config