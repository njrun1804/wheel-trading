"""Streaming data processors to prevent string overflow errors in Claude Code.

Provides robust streaming classes that handle large datasets without memory overflow,
with intelligent chunking for different data types and integration with existing systems.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import tempfile
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import uuid4

import psutil

from .logging import get_logger
from .recovery import with_recovery

logger = get_logger(__name__)

T = TypeVar("T")


class DataType(Enum):
    """Data type enumeration for optimal chunking strategies."""

    JSON = "json"
    TEXT = "text"
    BINARY = "binary"
    CSV = "csv"
    PARQUET = "parquet"
    ARROW = "arrow"


class StreamingMode(Enum):
    """Streaming operation modes."""

    MEMORY = "memory"  # Stream through memory (default)
    FILE = "file"  # Stream through temporary files
    HYBRID = "hybrid"  # Adaptive based on data size


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for streaming operations."""

    # Memory limits
    max_memory_mb: int = 100  # Max memory per stream
    max_total_memory_mb: int = 500  # Max total memory for all streams
    memory_check_interval: int = 100  # Check memory every N chunks

    # Chunking configuration
    default_chunk_size: int = 64 * 1024  # 64KB default
    max_chunk_size: int = 10 * 1024 * 1024  # 10MB max
    adaptive_chunking: bool = True

    # File configuration
    temp_dir: Path | None = None
    compress_temp_files: bool = True
    auto_cleanup: bool = True

    # Performance
    max_concurrent_streams: int = 10
    prefetch_size: int = 3  # Number of chunks to prefetch

    # Recovery
    max_retries: int = 3
    retry_delay: float = 0.1


@dataclass
class StreamMetrics:
    """Metrics for stream processing operations."""

    stream_id: str
    data_type: DataType
    total_bytes: int = 0
    chunks_processed: int = 0
    memory_peak_mb: float = 0.0
    processing_time_ms: float = 0.0
    error_count: int = 0
    temp_files_created: int = 0
    compression_ratio: float = 1.0


class MemoryMonitor:
    """Monitor system memory usage and trigger adaptive behavior."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self._current_streams: dict[str, float] = {}  # stream_id -> memory_mb

    def get_system_memory_mb(self) -> float:
        """Get current system memory usage in MB."""
        return psutil.virtual_memory().used / (1024 * 1024)

    def get_available_memory_mb(self) -> float:
        """Get available memory in MB."""
        return psutil.virtual_memory().available / (1024 * 1024)

    def register_stream(self, stream_id: str) -> None:
        """Register a new stream for monitoring."""
        self._current_streams[stream_id] = 0.0

    def update_stream_memory(self, stream_id: str, memory_mb: float) -> None:
        """Update memory usage for a stream."""
        if stream_id in self._current_streams:
            self._current_streams[stream_id] = memory_mb

    def unregister_stream(self, stream_id: str) -> None:
        """Unregister a stream from monitoring."""
        self._current_streams.pop(stream_id, None)

    def get_total_stream_memory_mb(self) -> float:
        """Get total memory used by all registered streams."""
        return sum(self._current_streams.values())

    def should_use_file_mode(self, estimated_size_mb: float) -> bool:
        """Determine if file mode should be used based on memory constraints."""
        total_stream_memory = self.get_total_stream_memory_mb()
        available_memory = self.get_available_memory_mb()

        # Use file mode if:
        # 1. Estimated size exceeds per-stream limit
        # 2. Total stream memory would exceed limit
        # 3. Available system memory is low
        return (
            estimated_size_mb > self.config.max_memory_mb
            or (total_stream_memory + estimated_size_mb) > self.config.max_total_memory_mb
            or available_memory < (estimated_size_mb * 2)  # Need 2x for safety
        )

    def get_adaptive_chunk_size(
        self, data_type: DataType, available_memory_mb: float
    ) -> int:
        """Calculate adaptive chunk size based on available memory and data type."""
        base_size = self.config.default_chunk_size

        # Adjust based on data type
        type_multipliers = {
            DataType.JSON: 1.5,  # Larger chunks for JSON
            DataType.TEXT: 1.0,  # Standard chunks for text
            DataType.BINARY: 0.8,  # Smaller chunks for binary
            DataType.CSV: 2.0,  # Larger chunks for structured data
            DataType.PARQUET: 4.0,  # Much larger chunks for columnar data
            DataType.ARROW: 4.0,  # Much larger chunks for columnar data
        }

        adjusted_size = int(base_size * type_multipliers.get(data_type, 1.0))

        # Adjust based on available memory
        if available_memory_mb < 100:  # Low memory
            adjusted_size = min(adjusted_size, 16 * 1024)  # 16KB max
        elif available_memory_mb > 1000:  # High memory
            adjusted_size = min(adjusted_size * 2, self.config.max_chunk_size)

        return max(4096, min(adjusted_size, self.config.max_chunk_size))


class StreamProcessor(Generic[T]):
    """Base streaming processor with memory-aware chunking and error recovery."""

    def __init__(
        self,
        data_type: DataType,
        config: StreamConfig | None = None,
        memory_monitor: MemoryMonitor | None = None,
    ):
        self.data_type = data_type
        self.config = config or StreamConfig()
        self.memory_monitor = memory_monitor or MemoryMonitor(self.config)
        self.stream_id = str(uuid4())
        self.metrics = StreamMetrics(stream_id=self.stream_id, data_type=data_type)
        self._temp_files: list[Path] = []
        self._start_time = datetime.now()

        # Register with memory monitor
        self.memory_monitor.register_stream(self.stream_id)

    async def __aenter__(self) -> StreamProcessor[T]:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.cleanup()

    async def cleanup(self) -> None:
        """Clean up resources and temporary files."""
        try:
            self.memory_monitor.unregister_stream(self.stream_id)

            if self.config.auto_cleanup:
                for temp_file in self._temp_files:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")

            # Update final metrics
            self.metrics.processing_time_ms = (
                datetime.now() - self._start_time
            ).total_seconds() * 1000

            logger.info(
                "Stream processing completed",
                extra={
                    "stream_id": self.stream_id,
                    "metrics": {
                        "total_bytes": self.metrics.total_bytes,
                        "chunks_processed": self.metrics.chunks_processed,
                        "memory_peak_mb": self.metrics.memory_peak_mb,
                        "processing_time_ms": self.metrics.processing_time_ms,
                        "error_count": self.metrics.error_count,
                        "temp_files_created": self.metrics.temp_files_created,
                        "compression_ratio": self.metrics.compression_ratio,
                    },
                },
            )

        except Exception as e:
            logger.error(f"Error during stream cleanup: {e}")

    def _estimate_memory_usage(self, data: Any) -> float:
        """Estimate memory usage of data in MB."""
        try:
            if isinstance(data, (str, bytes)):
                return len(data) / (1024 * 1024)
            elif isinstance(data, dict):
                return len(json.dumps(data)) / (1024 * 1024)
            elif hasattr(data, "__sizeof__"):
                return data.__sizeof__() / (1024 * 1024)
            else:
                # Rough estimate
                return len(str(data)) / (1024 * 1024)
        except Exception:
            return 0.1  # Default estimate

    def _get_chunk_size(self) -> int:
        """Get adaptive chunk size based on current conditions."""
        if not self.config.adaptive_chunking:
            return self.config.default_chunk_size

        available_memory = self.memory_monitor.get_available_memory_mb()
        return self.memory_monitor.get_adaptive_chunk_size(
            self.data_type, available_memory
        )

    @asynccontextmanager
    async def _temp_file(self, suffix: str = ".tmp") -> AsyncIterator[Path]:
        """Create and manage a temporary file."""
        temp_dir = self.config.temp_dir or Path(tempfile.gettempdir())
        temp_file = temp_dir / f"stream_{self.stream_id}_{len(self._temp_files)}{suffix}"

        try:
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            self._temp_files.append(temp_file)
            self.metrics.temp_files_created += 1
            yield temp_file
        finally:
            pass  # Cleanup handled in __aexit__

    async def _write_compressed(self, data: bytes, file_path: Path) -> None:
        """Write data to file with optional compression."""
        if self.config.compress_temp_files:
            with gzip.open(file_path, "wb") as f:
                f.write(data)
            # Update compression ratio
            original_size = len(data)
            compressed_size = file_path.stat().st_size
            if original_size > 0:
                self.metrics.compression_ratio = compressed_size / original_size
        else:
            file_path.write_bytes(data)

    async def _read_compressed(self, file_path: Path) -> bytes:
        """Read data from file with optional decompression."""
        if self.config.compress_temp_files:
            with gzip.open(file_path, "rb") as f:
                return f.read()
        else:
            return file_path.read_bytes()


class DataStreamProcessor(StreamProcessor[bytes]):
    """Processor for streaming raw data with chunking and overflow protection."""

    async def process_data_stream(
        self, data_source: AsyncIterator[bytes] | bytes | str
    ) -> AsyncGenerator[bytes, None]:
        """Process data stream with adaptive chunking."""
        if isinstance(data_source, (bytes, str)):
            # Handle static data
            if isinstance(data_source, str):
                data_source = data_source.encode("utf-8")

            data_size_mb = self._estimate_memory_usage(data_source)
            
            if self.memory_monitor.should_use_file_mode(data_size_mb):
                # Use file-based streaming for large data
                async for chunk in self._process_large_data_via_file(data_source):
                    yield chunk
            else:
                # Use memory-based streaming
                async for chunk in self._process_data_in_memory(data_source):
                    yield chunk
        else:
            # Handle async iterator
            async for chunk in self._process_async_stream(data_source):
                yield chunk

    async def _process_data_in_memory(self, data: bytes) -> AsyncGenerator[bytes, None]:
        """Process data in memory with chunking."""
        chunk_size = self._get_chunk_size()
        offset = 0
        
        while offset < len(data):
            # Memory check
            if self.metrics.chunks_processed % self.config.memory_check_interval == 0:
                current_memory = self._estimate_memory_usage(data[offset:offset + chunk_size])
                self.memory_monitor.update_stream_memory(self.stream_id, current_memory)
                self.metrics.memory_peak_mb = max(
                    self.metrics.memory_peak_mb, current_memory
                )

            chunk = data[offset : offset + chunk_size]
            self.metrics.total_bytes += len(chunk)
            self.metrics.chunks_processed += 1
            
            yield chunk
            offset += chunk_size

            # Allow other tasks to run
            await asyncio.sleep(0)

    async def _process_large_data_via_file(
        self, data: bytes
    ) -> AsyncGenerator[bytes, None]:
        """Process large data using temporary files."""
        async with self._temp_file(".dat") as temp_file:
            # Write data to temp file
            await self._write_compressed(data, temp_file)
            
            # Read back in chunks
            chunk_size = self._get_chunk_size()
            
            if self.config.compress_temp_files:
                with gzip.open(temp_file, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        
                        self.metrics.total_bytes += len(chunk)
                        self.metrics.chunks_processed += 1
                        yield chunk
                        await asyncio.sleep(0)
            else:
                with open(temp_file, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        
                        self.metrics.total_bytes += len(chunk)
                        self.metrics.chunks_processed += 1
                        yield chunk
                        await asyncio.sleep(0)

    async def _process_async_stream(
        self, data_source: AsyncIterator[bytes]
    ) -> AsyncGenerator[bytes, None]:
        """Process async data stream with buffering."""
        buffer = b""
        chunk_size = self._get_chunk_size()
        
        async for data_chunk in data_source:
            buffer += data_chunk
            
            # Yield complete chunks
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                
                self.metrics.total_bytes += len(chunk)
                self.metrics.chunks_processed += 1
                yield chunk
                await asyncio.sleep(0)
            
            # Memory pressure check
            if len(buffer) > self.config.max_memory_mb * 1024 * 1024:
                # Write buffer to temp file and continue
                async with self._temp_file(".buffer") as temp_file:
                    await self._write_compressed(buffer, temp_file)
                    buffer = b""
                    
                    # Yield from temp file
                    async for chunk in self._yield_from_temp_file(temp_file, chunk_size):
                        yield chunk
        
        # Yield remaining buffer
        if buffer:
            self.metrics.total_bytes += len(buffer)
            self.metrics.chunks_processed += 1
            yield buffer

    async def _yield_from_temp_file(
        self, temp_file: Path, chunk_size: int
    ) -> AsyncGenerator[bytes, None]:
        """Yield chunks from temporary file."""
        data = await self._read_compressed(temp_file)
        offset = 0
        
        while offset < len(data):
            chunk = data[offset : offset + chunk_size]
            self.metrics.total_bytes += len(chunk)
            self.metrics.chunks_processed += 1
            yield chunk
            offset += chunk_size
            await asyncio.sleep(0)


class JSONStreamProcessor(StreamProcessor[dict]):
    """Processor for streaming JSON data with intelligent parsing."""

    async def process_json_stream(
        self, data_source: AsyncIterator[str] | str | list[dict] | dict
    ) -> AsyncGenerator[dict, None]:
        """Process JSON stream with parsing and validation."""
        if isinstance(data_source, str):
            # Single JSON string
            try:
                data = json.loads(data_source)
                if isinstance(data, list):
                    for item in data:
                        yield item
                        await asyncio.sleep(0)
                else:
                    yield data
            except json.JSONDecodeError as e:
                self.metrics.error_count += 1
                logger.error(f"JSON parsing error: {e}")
                raise
        
        elif isinstance(data_source, dict):
            yield data_source
        
        elif isinstance(data_source, list):
            for item in data_source:
                yield item
                await asyncio.sleep(0)
        
        else:
            # Async iterator of JSON strings
            async for json_str in data_source:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        for item in data:
                            yield item
                            await asyncio.sleep(0)
                    else:
                        yield data
                    
                    self.metrics.chunks_processed += 1
                except json.JSONDecodeError as e:
                    self.metrics.error_count += 1
                    logger.warning(f"Skipping invalid JSON chunk: {e}")
                    continue

    async def process_large_json_file(self, file_path: Path) -> AsyncGenerator[dict, None]:
        """Process large JSON file line by line."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        self.metrics.chunks_processed += 1
                        self.metrics.total_bytes += len(line.encode('utf-8'))
                        yield data
                        
                        # Periodic memory check
                        if line_no % self.config.memory_check_interval == 0:
                            await asyncio.sleep(0)
                            
                    except json.JSONDecodeError as e:
                        self.metrics.error_count += 1
                        logger.warning(f"Invalid JSON at line {line_no}: {e}")
                        continue
                        
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error reading JSON file {file_path}: {e}")
            raise


class TextStreamProcessor(StreamProcessor[str]):
    """Processor for streaming text data with encoding handling."""

    def __init__(
        self,
        config: StreamConfig | None = None,
        memory_monitor: MemoryMonitor | None = None,
        encoding: str = "utf-8",
        line_based: bool = False,
    ):
        super().__init__(DataType.TEXT, config, memory_monitor)
        self.encoding = encoding
        self.line_based = line_based

    async def process_text_stream(
        self, data_source: AsyncIterator[str] | str | bytes
    ) -> AsyncGenerator[str, None]:
        """Process text stream with encoding and chunking."""
        if isinstance(data_source, bytes):
            data_source = data_source.decode(self.encoding)
        
        if isinstance(data_source, str):
            if self.line_based:
                async for line in self._process_text_lines(data_source):
                    yield line
            else:
                async for chunk in self._process_text_chunks(data_source):
                    yield chunk
        else:
            # Async iterator
            async for text_chunk in data_source:
                if isinstance(text_chunk, bytes):
                    text_chunk = text_chunk.decode(self.encoding)
                
                if self.line_based:
                    async for line in self._process_text_lines(text_chunk):
                        yield line
                else:
                    async for chunk in self._process_text_chunks(text_chunk):
                        yield chunk

    async def _process_text_lines(self, text: str) -> AsyncGenerator[str, None]:
        """Process text line by line."""
        lines = text.split('\n')
        for line_no, line in enumerate(lines):
            self.metrics.chunks_processed += 1
            self.metrics.total_bytes += len(line.encode(self.encoding))
            yield line
            
            if line_no % self.config.memory_check_interval == 0:
                await asyncio.sleep(0)

    async def _process_text_chunks(self, text: str) -> AsyncGenerator[str, None]:
        """Process text in fixed-size chunks."""
        chunk_size = self._get_chunk_size() // 4  # Assuming average 4 bytes per char
        offset = 0
        
        while offset < len(text):
            chunk = text[offset : offset + chunk_size]
            self.metrics.chunks_processed += 1
            self.metrics.total_bytes += len(chunk.encode(self.encoding))
            yield chunk
            offset += chunk_size
            await asyncio.sleep(0)


# Global memory monitor instance
_global_memory_monitor: MemoryMonitor | None = None


def get_memory_monitor(config: StreamConfig | None = None) -> MemoryMonitor:
    """Get global memory monitor instance."""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor(config or StreamConfig())
    return _global_memory_monitor


@with_recovery(max_attempts=3, backoff_factor=1.5)
async def create_data_stream_processor(
    data_type: DataType = DataType.BINARY,
    config: StreamConfig | None = None,
    memory_monitor: MemoryMonitor | None = None,
) -> DataStreamProcessor:
    """Factory function to create data stream processor with recovery."""
    if memory_monitor is None:
        memory_monitor = get_memory_monitor(config)
    
    return DataStreamProcessor(data_type, config, memory_monitor)


@with_recovery(max_attempts=3, backoff_factor=1.5)
async def create_json_stream_processor(
    config: StreamConfig | None = None,
    memory_monitor: MemoryMonitor | None = None,
) -> JSONStreamProcessor:
    """Factory function to create JSON stream processor with recovery."""
    if memory_monitor is None:
        memory_monitor = get_memory_monitor(config)
    
    return JSONStreamProcessor(DataType.JSON, config, memory_monitor)


@with_recovery(max_attempts=3, backoff_factor=1.5)
async def create_text_stream_processor(
    config: StreamConfig | None = None,
    memory_monitor: MemoryMonitor | None = None,
    encoding: str = "utf-8",
    line_based: bool = False,
) -> TextStreamProcessor:
    """Factory function to create text stream processor with recovery."""
    if memory_monitor is None:
        memory_monitor = get_memory_monitor(config)
    
    return TextStreamProcessor(config, memory_monitor, encoding, line_based)


# Convenience functions for common use cases
async def stream_large_json(
    data: str | dict | list | Path,
    chunk_callback: Callable[[dict], None] | None = None,
    config: StreamConfig | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream large JSON data with automatic processing mode selection."""
    async with await create_json_stream_processor(config) as processor:
        if isinstance(data, Path):
            async for item in processor.process_large_json_file(data):
                if chunk_callback:
                    chunk_callback(item)
                yield item
        else:
            async for item in processor.process_json_stream(data):
                if chunk_callback:
                    chunk_callback(item)
                yield item


async def stream_large_text(
    data: str | bytes | Path,
    line_based: bool = False,
    encoding: str = "utf-8",
    chunk_callback: Callable[[str], None] | None = None,
    config: StreamConfig | None = None,
) -> AsyncGenerator[str, None]:
    """Stream large text data with automatic processing mode selection."""
    async with await create_text_stream_processor(
        config, encoding=encoding, line_based=line_based
    ) as processor:
        if isinstance(data, Path):
            with open(data, 'r', encoding=encoding) as f:
                content = f.read()
            data_source = content
        else:
            data_source = data
        
        async for chunk in processor.process_text_stream(data_source):
            if chunk_callback:
                chunk_callback(chunk)
            yield chunk


async def stream_large_data(
    data: bytes | str | AsyncIterator[bytes],
    data_type: DataType = DataType.BINARY,
    chunk_callback: Callable[[bytes], None] | None = None,
    config: StreamConfig | None = None,
) -> AsyncGenerator[bytes, None]:
    """Stream large binary data with automatic processing mode selection."""
    async with await create_data_stream_processor(data_type, config) as processor:
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        async for chunk in processor.process_data_stream(data):
            if chunk_callback:
                chunk_callback(chunk)
            yield chunk