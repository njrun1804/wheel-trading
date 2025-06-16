"""Safe output handling to prevent string overflow errors in Claude Code.

Provides intelligent output management with automatic fallback to file-based
output when data exceeds memory or string limits.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Union
from uuid import uuid4

from .logging import get_logger
from .recovery import with_recovery
from .stream_processors import DataType, StreamConfig, get_memory_monitor

logger = get_logger(__name__)

OutputData = Union[str, bytes, dict, list, Any]


@dataclass
class OutputConfig:
    """Configuration for safe output handling."""

    # Size limits (in bytes)
    max_string_length: int = 500_000  # 500KB max string output
    max_memory_mb: int = 50  # 50MB max memory usage
    max_json_size: int = 1_000_000  # 1MB max JSON output

    # File output configuration
    use_temp_files: bool = True
    temp_dir: Path | None = None
    file_prefix: str = "safe_output"
    auto_cleanup: bool = True
    compress_files: bool = True

    # Truncation settings
    enable_truncation: bool = True
    truncation_suffix: str = "\n... [OUTPUT TRUNCATED - See full results in file]"
    preview_lines: int = 50  # Number of lines to show in preview

    # Formatting
    pretty_json: bool = True
    include_metadata: bool = True


@dataclass
class OutputResult:
    """Result of safe output processing."""

    content: str  # Safe content for display
    is_truncated: bool = False
    file_path: Path | None = None
    original_size: int = 0
    compressed_size: int = 0
    content_hash: str = ""
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SafeOutputHandler:
    """Handler for safe output with automatic overflow protection."""

    def __init__(self, config: OutputConfig | None = None):
        self.config = config or OutputConfig()
        self.memory_monitor = get_memory_monitor()
        self._temp_files: list[Path] = []

    def __enter__(self) -> SafeOutputHandler:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        if self.config.auto_cleanup:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        self._temp_files.clear()

    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        try:
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, bytes):
                return len(data)
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data).encode('utf-8'))
            else:
                return len(str(data).encode('utf-8'))
        except Exception:
            return len(str(data).encode('utf-8'))

    def _calculate_hash(self, content: str | bytes) -> str:
        """Calculate SHA-256 hash of content."""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()[:16]

    def _create_temp_file(self, suffix: str = ".txt") -> Path:
        """Create a temporary file and track it for cleanup."""
        temp_dir = self.config.temp_dir or Path(tempfile.gettempdir())
        temp_file = temp_dir / f"{self.config.file_prefix}_{uuid4().hex}{suffix}"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        self._temp_files.append(temp_file)
        return temp_file

    def _format_json(self, data: dict | list) -> str:
        """Format JSON data with optional pretty printing."""
        if self.config.pretty_json:
            return json.dumps(data, indent=2, default=str, ensure_ascii=False)
        else:
            return json.dumps(data, default=str, ensure_ascii=False)

    def _truncate_content(self, content: str, max_length: int) -> tuple[str, bool]:
        """Truncate content to max length with preview."""
        if len(content) <= max_length:
            return content, False

        # Try to truncate at line boundaries
        lines = content.split('\n')
        if len(lines) > self.config.preview_lines:
            truncated_lines = lines[:self.config.preview_lines]
            truncated_content = '\n'.join(truncated_lines)
            if len(truncated_content) <= max_length - len(self.config.truncation_suffix):
                return truncated_content + self.config.truncation_suffix, True

        # Fallback to character-based truncation
        safe_length = max_length - len(self.config.truncation_suffix)
        truncated = content[:safe_length] + self.config.truncation_suffix
        return truncated, True

    def _write_to_file(self, content: str | bytes, file_path: Path) -> int:
        """Write content to file with optional compression."""
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content

        if self.config.compress_files:
            import gzip
            with gzip.open(file_path.with_suffix(file_path.suffix + '.gz'), 'wb') as f:
                f.write(content_bytes)
            # Update file path to compressed version
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            file_path.unlink()  # Remove uncompressed file
            self._temp_files.remove(file_path)
            self._temp_files.append(compressed_path)
            return compressed_path.stat().st_size
        else:
            file_path.write_bytes(content_bytes)
            return file_path.stat().st_size

    @with_recovery(max_attempts=3, backoff_factor=1.5)
    def handle_output(self, data: OutputData, output_type: str = "general") -> OutputResult:
        """Handle output data with safe overflow protection."""
        try:
            # Convert data to string representation
            if isinstance(data, (dict, list)):
                content = self._format_json(data)
                data_type = "json"
            elif isinstance(data, bytes):
                try:
                    content = data.decode('utf-8')
                    data_type = "text"
                except UnicodeDecodeError:
                    content = repr(data)
                    data_type = "binary"
            else:
                content = str(data)
                data_type = "text"

            original_size = len(content.encode('utf-8'))
            content_hash = self._calculate_hash(content)

            # Check if content exceeds limits
            memory_mb = original_size / (1024 * 1024)
            needs_file_output = (
                original_size > self.config.max_string_length
                or memory_mb > self.config.max_memory_mb
                or (isinstance(data, (dict, list)) and original_size > self.config.max_json_size)
                or self.memory_monitor.should_use_file_mode(memory_mb)
            )

            # Create metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "output_type": output_type,
                "data_type": data_type,
                "original_size": original_size,
                "content_hash": content_hash,
            } if self.config.include_metadata else {}

            if needs_file_output and self.config.use_temp_files:
                # Write full content to file
                file_ext = ".json" if data_type == "json" else ".txt"
                temp_file = self._create_temp_file(file_ext)
                compressed_size = self._write_to_file(content, temp_file)

                # Create truncated preview
                preview_content, is_truncated = self._truncate_content(
                    content, self.config.max_string_length
                )

                # Add file info to metadata
                metadata.update({
                    "file_output": True,
                    "file_path": str(temp_file),
                    "compressed_size": compressed_size,
                    "compression_ratio": compressed_size / original_size if original_size > 0 else 1.0,
                })

                logger.info(
                    f"Large output written to file: {temp_file}",
                    extra={
                        "output_handler": {
                            "original_size": original_size,
                            "compressed_size": compressed_size,
                            "file_path": str(temp_file),
                            "truncated": is_truncated,
                        }
                    },
                )

                return OutputResult(
                    content=preview_content,
                    is_truncated=is_truncated,
                    file_path=temp_file,
                    original_size=original_size,
                    compressed_size=compressed_size,
                    content_hash=content_hash,
                    metadata=metadata,
                )

            elif self.config.enable_truncation and original_size > self.config.max_string_length:
                # Truncate content without file output
                truncated_content, is_truncated = self._truncate_content(
                    content, self.config.max_string_length
                )

                metadata.update({
                    "file_output": False,
                    "truncated": is_truncated,
                })

                return OutputResult(
                    content=truncated_content,
                    is_truncated=is_truncated,
                    original_size=original_size,
                    content_hash=content_hash,
                    metadata=metadata,
                )

            else:
                # Content is safe to return as-is
                metadata.update({
                    "file_output": False,
                    "truncated": False,
                })

                return OutputResult(
                    content=content,
                    is_truncated=False,
                    original_size=original_size,
                    content_hash=content_hash,
                    metadata=metadata,
                )

        except Exception as e:
            logger.error(f"Error in safe output handling: {e}")
            # Fallback to simple string representation
            fallback_content = f"Error processing output: {e}\nFallback representation: {repr(data)[:1000]}"
            return OutputResult(
                content=fallback_content,
                is_truncated=True,
                original_size=len(fallback_content),
                content_hash=self._calculate_hash(fallback_content),
                metadata={"error": str(e), "fallback": True},
            )

    def handle_dataframe_output(self, df: Any, max_rows: int = 1000) -> OutputResult:
        """Handle pandas DataFrame output with smart truncation."""
        try:
            import pandas as pd
            
            if not isinstance(df, pd.DataFrame):
                return self.handle_output(df, "dataframe")

            # Get basic info
            rows, cols = df.shape
            memory_usage = df.memory_usage(deep=True).sum()

            # Decide on output format
            if rows > max_rows or memory_usage > self.config.max_memory_mb * 1024 * 1024:
                # Create summary with head/tail
                head_rows = min(25, max_rows // 2)
                tail_rows = min(25, max_rows // 2)
                
                summary_parts = [
                    f"DataFrame Shape: {rows:,} rows × {cols} columns",
                    f"Memory Usage: {memory_usage / 1024 / 1024:.2f} MB",
                    "",
                    "First {} rows:".format(head_rows),
                    df.head(head_rows).to_string(),
                ]

                if rows > head_rows * 2:
                    summary_parts.extend([
                        "",
                        f"... ({rows - head_rows - tail_rows:,} rows omitted) ...",
                        "",
                        f"Last {tail_rows} rows:",
                        df.tail(tail_rows).to_string(),
                    ])

                summary_parts.extend([
                    "",
                    "Column Info:",
                    str(df.dtypes),
                    "",
                    "Summary Statistics:",
                    df.describe().to_string(),
                ])

                content = "\n".join(summary_parts)
                
                # Also save full DataFrame to file if enabled
                if self.config.use_temp_files:
                    temp_file = self._create_temp_file(".csv")
                    df.to_csv(temp_file, index=False)
                    file_size = temp_file.stat().st_size
                    
                    return OutputResult(
                        content=content,
                        is_truncated=True,
                        file_path=temp_file,
                        original_size=memory_usage,
                        compressed_size=file_size,
                        content_hash=self._calculate_hash(content),
                        metadata={
                            "dataframe_shape": [rows, cols],
                            "memory_usage_mb": memory_usage / 1024 / 1024,
                            "full_data_in_file": True,
                        },
                    )
                else:
                    return OutputResult(
                        content=content,
                        is_truncated=True,
                        original_size=memory_usage,
                        content_hash=self._calculate_hash(content),
                        metadata={
                            "dataframe_shape": [rows, cols],
                            "memory_usage_mb": memory_usage / 1024 / 1024,
                        },
                    )
            else:
                # DataFrame is small enough to return in full
                content = df.to_string()
                return self.handle_output(content, "dataframe")

        except Exception as e:
            logger.error(f"Error handling DataFrame output: {e}")
            return self.handle_output(str(df), "dataframe_fallback")

    def handle_query_results(
        self, results: Any, query: str = "", max_results: int = 10000
    ) -> OutputResult:
        """Handle database query results with smart formatting."""
        try:
            # Handle different result types
            if hasattr(results, 'fetchall'):
                # Database cursor
                rows = results.fetchall()
                if hasattr(results, 'description') and results.description:
                    columns = [desc[0] for desc in results.description]
                    data = [dict(zip(columns, row)) for row in rows]
                else:
                    data = rows
            elif hasattr(results, 'to_dict'):
                # DataFrame-like object
                return self.handle_dataframe_output(results, max_results)
            elif isinstance(results, (list, tuple)):
                data = results
            else:
                data = results

            # Add query context if provided
            if query and isinstance(data, list):
                content = {
                    "query": query,
                    "result_count": len(data),
                    "results": data[:max_results] if len(data) > max_results else data,
                }
                if len(data) > max_results:
                    content["truncated"] = True
                    content["total_results"] = len(data)
            else:
                content = data

            return self.handle_output(content, "query_results")

        except Exception as e:
            logger.error(f"Error handling query results: {e}")
            return self.handle_output(f"Query results error: {e}", "query_error")


# Convenience functions
def safe_output(
    data: OutputData,
    config: OutputConfig | None = None,
    output_type: str = "general",
) -> OutputResult:
    """Create safe output with automatic cleanup."""
    with SafeOutputHandler(config) as handler:
        return handler.handle_output(data, output_type)


def safe_json_output(
    data: dict | list,
    config: OutputConfig | None = None,
    pretty: bool = True,
) -> OutputResult:
    """Create safe JSON output with formatting."""
    if config is None:
        config = OutputConfig()
    config.pretty_json = pretty
    
    with SafeOutputHandler(config) as handler:
        return handler.handle_output(data, "json")


def safe_dataframe_output(
    df: Any,
    config: OutputConfig | None = None,
    max_rows: int = 1000,
) -> OutputResult:
    """Create safe DataFrame output with smart truncation."""
    with SafeOutputHandler(config) as handler:
        return handler.handle_dataframe_output(df, max_rows)


def safe_query_output(
    results: Any,
    query: str = "",
    config: OutputConfig | None = None,
    max_results: int = 10000,
) -> OutputResult:
    """Create safe query results output."""
    with SafeOutputHandler(config) as handler:
        return handler.handle_query_results(results, query, max_results)


# Integration with existing logging system
class SafeOutputLogger:
    """Logger that uses safe output handling for large data."""

    def __init__(self, logger_name: str, config: OutputConfig | None = None):
        self.logger = get_logger(logger_name)
        self.config = config or OutputConfig()

    def log_large_data(
        self,
        data: OutputData,
        level: str = "INFO",
        message: str = "Large data output",
        **kwargs
    ) -> None:
        """Log large data using safe output handling."""
        with SafeOutputHandler(self.config) as handler:
            result = handler.handle_output(data, "log_data")
            
            log_data = {
                "message": message,
                "safe_output": {
                    "is_truncated": result.is_truncated,
                    "original_size": result.original_size,
                    "content_hash": result.content_hash,
                },
                **kwargs,
            }

            if result.file_path:
                log_data["safe_output"]["file_path"] = str(result.file_path)
                log_data["safe_output"]["file_size"] = result.compressed_size

            # Log the safe content
            getattr(self.logger, level.lower())(
                result.content,
                extra=log_data,
            )


def get_safe_output_logger(
    logger_name: str, config: OutputConfig | None = None
) -> SafeOutputLogger:
    """Get a safe output logger instance."""
    return SafeOutputLogger(logger_name, config)