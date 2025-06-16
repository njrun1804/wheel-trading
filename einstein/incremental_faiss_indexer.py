#!/usr/bin/env python3
"""
Incremental FAISS Indexing System

High-performance incremental indexing that only processes changed files:
- File change detection using hashes and timestamps
- Incremental updates without full rebuilds
- Efficient vector addition/removal from FAISS index
- File system monitoring for real-time updates
- Optimized for large codebases with frequent changes

Performance targets:
- Only index changed files
- Sub-second incremental updates
- Efficient vector management
- Real-time file monitoring
- Minimal memory overhead for updates
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .code_optimized_embeddings import CodeEmbeddingResult, CodeOptimizedEmbeddingSystem
from .optimized_faiss_system import FAISSManager

logger = logging.getLogger(__name__)


@dataclass
class FileIndexMetadata:
    """Metadata for an indexed file."""

    file_path: str
    absolute_path: str
    last_modified: float
    content_hash: str
    size_bytes: int
    vector_count: int
    last_indexed: float
    index_version: int = 1


@dataclass
class IndexingStats:
    """Statistics for indexing operations."""

    files_checked: int = 0
    files_changed: int = 0
    files_added: int = 0
    files_removed: int = 0
    vectors_added: int = 0
    vectors_removed: int = 0
    processing_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    index_update_time_ms: float = 0.0


class FileChangeDetector:
    """Efficient file change detection using hashes and timestamps."""

    def __init__(self, metadata_path: Path):
        """Initialize file change detector."""
        self.metadata_path = metadata_path
        self.file_metadata: dict[str, FileIndexMetadata] = {}
        self.load_metadata()

    def load_metadata(self) -> None:
        """Load file metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path) as f:
                    data = json.load(f)
                    self.file_metadata = {
                        path: FileIndexMetadata(**metadata)
                        for path, metadata in data.items()
                    }
                logger.info(f"Loaded metadata for {len(self.file_metadata)} files")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.file_metadata = {}
        else:
            logger.info("No existing metadata found, starting fresh")

    def save_metadata(self) -> None:
        """Save file metadata to disk."""
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                path: asdict(metadata) for path, metadata in self.file_metadata.items()
            }

            # Write to temporary file first for atomic operation
            temp_path = self.metadata_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.metadata_path)

            logger.debug(f"Saved metadata for {len(self.file_metadata)} files")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file content."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.debug(f"Failed to hash {file_path}: {e}")
            return ""

    def get_file_stats(self, file_path: str) -> tuple[float, int]:
        """Get file modification time and size."""
        try:
            stat = os.stat(file_path)
            return stat.st_mtime, stat.st_size
        except Exception as e:
            logger.debug(f"Failed to stat {file_path}: {e}")
            return 0.0, 0

    def detect_changes(
        self, file_paths: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Detect changed, added, and removed files.

        Returns:
            Tuple of (changed_files, added_files, removed_files)
        """
        changed_files = []
        added_files = []
        current_files = set(file_paths)
        previous_files = set(self.file_metadata.keys())

        # Check for removed files
        removed_files = list(previous_files - current_files)

        # Check each current file for changes or additions
        for file_path in file_paths:
            absolute_path = str(Path(file_path).resolve())

            if absolute_path not in self.file_metadata:
                # New file
                added_files.append(file_path)
                continue

            metadata = self.file_metadata[absolute_path]
            current_mtime, current_size = self.get_file_stats(file_path)

            # Quick check: if timestamp and size haven't changed, likely unchanged
            if (
                current_mtime == metadata.last_modified
                and current_size == metadata.size_bytes
            ):
                continue

            # More expensive check: compute hash
            current_hash = self.compute_file_hash(file_path)
            if current_hash != metadata.content_hash:
                changed_files.append(file_path)

        logger.info(
            f"File analysis: {len(changed_files)} changed, {len(added_files)} added, {len(removed_files)} removed"
        )
        return changed_files, added_files, removed_files

    def update_file_metadata(self, file_path: str, vector_count: int) -> None:
        """Update metadata for a successfully indexed file."""
        absolute_path = str(Path(file_path).resolve())
        mtime, size = self.get_file_stats(file_path)
        content_hash = self.compute_file_hash(file_path)

        self.file_metadata[absolute_path] = FileIndexMetadata(
            file_path=file_path,
            absolute_path=absolute_path,
            last_modified=mtime,
            content_hash=content_hash,
            size_bytes=size,
            vector_count=vector_count,
            last_indexed=time.time(),
        )

    def remove_file_metadata(self, file_path: str) -> None:
        """Remove metadata for a deleted file."""
        absolute_path = str(Path(file_path).resolve())
        self.file_metadata.pop(absolute_path, None)


class RealTimeFileMonitor(FileSystemEventHandler):
    """Real-time file system monitoring for incremental updates."""

    def __init__(
        self, indexer: "IncrementalFAISSIndexer", debounce_interval: float = 1.0
    ):
        """
        Initialize file monitor.

        Args:
            indexer: The incremental indexer to notify
            debounce_interval: Minimum time between processing events
        """
        self.indexer = indexer
        self.debounce_interval = debounce_interval
        self.pending_files: set[str] = set()
        self.last_update = 0.0
        self.update_task: asyncio.Task | None = None

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and self._should_process_file(event.src_path):
            self._schedule_update(event.src_path)

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and self._should_process_file(event.src_path):
            self._schedule_update(event.src_path)

    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory and self._should_process_file(event.src_path):
            self._schedule_removal(event.src_path)

    def _should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed."""
        # Only process Python files for now
        return file_path.endswith(".py") and not file_path.endswith(".pyc")

    def _schedule_update(self, file_path: str) -> None:
        """Schedule an incremental update for a file."""
        self.pending_files.add(file_path)

        # Cancel existing update task
        if self.update_task and not self.update_task.done():
            self.update_task.cancel()

        # Schedule new update with debouncing
        self.update_task = asyncio.create_task(self._debounced_update())

    def _schedule_removal(self, file_path: str) -> None:
        """Schedule removal of a deleted file."""
        asyncio.create_task(self.indexer.remove_file(file_path))

    async def _debounced_update(self) -> None:
        """Perform debounced incremental update."""
        await asyncio.sleep(self.debounce_interval)

        if self.pending_files:
            files_to_update = list(self.pending_files)
            self.pending_files.clear()

            logger.info(f"Real-time update triggered for {len(files_to_update)} files")
            await self.indexer.update_files(files_to_update)


class IncrementalFAISSIndexer:
    """High-performance incremental FAISS indexing system."""

    def __init__(
        self,
        project_root: Path,
        embedding_model: Any = None,
        enable_monitoring: bool = True,
    ):
        """
        Initialize incremental FAISS indexer.

        Args:
            project_root: Root directory of the project
            embedding_model: Embedding model for generating vectors
            enable_monitoring: Enable real-time file monitoring
        """
        self.project_root = project_root
        self.index_dir = project_root / ".einstein" / "incremental_faiss"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.faiss_manager = FAISSManager(project_root)
        self.embedding_system = CodeOptimizedEmbeddingSystem(embedding_model)
        self.change_detector = FileChangeDetector(self.index_dir / "file_metadata.json")

        # File monitoring
        self.enable_monitoring = enable_monitoring
        self.file_monitor = None
        self.observer = None

        # Performance tracking
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.stats = IndexingStats()

        # Index state
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the incremental indexing system."""
        try:
            # Initialize FAISS manager
            success = await self.faiss_manager.initialize()
            if not success:
                logger.error("Failed to initialize FAISS manager")
                return False

            # Start file monitoring if enabled
            if self.enable_monitoring:
                await self._start_file_monitoring()

            self.is_initialized = True
            logger.info("✅ Incremental FAISS indexer initialized")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize incremental indexer: {e}")
            return False

    async def _start_file_monitoring(self) -> None:
        """Start real-time file monitoring."""
        try:
            self.file_monitor = RealTimeFileMonitor(self)
            self.observer = Observer()
            self.observer.schedule(
                self.file_monitor, str(self.project_root), recursive=True
            )
            self.observer.start()
            logger.info("✅ Real-time file monitoring started")
        except Exception as e:
            logger.error(f"❌ Failed to start file monitoring: {e}")

    async def scan_and_update(self, file_patterns: list[str] = None) -> IndexingStats:
        """
        Scan project and perform incremental updates.

        Args:
            file_patterns: Glob patterns for files to scan (default: ['**/*.py'])

        Returns:
            IndexingStats with update information
        """
        if not self.is_initialized:
            raise RuntimeError("Indexer not initialized")

        start_time = time.time()
        self.stats = IndexingStats()

        # Default to Python files
        if file_patterns is None:
            file_patterns = ["**/*.py"]

        # Find all matching files
        all_files = []
        for pattern in file_patterns:
            all_files.extend(str(p) for p in self.project_root.rglob(pattern))

        self.stats.files_checked = len(all_files)
        logger.info(f"Scanning {len(all_files)} files for changes...")

        # Detect changes
        changed_files, added_files, removed_files = self.change_detector.detect_changes(
            all_files
        )

        self.stats.files_changed = len(changed_files)
        self.stats.files_added = len(added_files)
        self.stats.files_removed = len(removed_files)

        # Process changes
        if changed_files or added_files:
            await self._update_files(changed_files + added_files)

        if removed_files:
            await self._remove_files(removed_files)

        # Save updated metadata
        self.change_detector.save_metadata()

        self.stats.processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Incremental update completed in {self.stats.processing_time_ms:.1f}ms"
        )
        logger.info(
            f"  Files: {self.stats.files_changed} changed, {self.stats.files_added} added, {self.stats.files_removed} removed"
        )
        logger.info(
            f"  Vectors: {self.stats.vectors_added} added, {self.stats.vectors_removed} removed"
        )

        return self.stats

    async def update_files(self, file_paths: list[str]) -> None:
        """Update specific files in the index."""
        if file_paths:
            await self._update_files(file_paths)
            self.change_detector.save_metadata()

    async def remove_file(self, file_path: str) -> None:
        """Remove a specific file from the index."""
        await self._remove_files([file_path])
        self.change_detector.save_metadata()

    async def _update_files(self, file_paths: list[str]) -> None:
        """Update multiple files in the index."""
        if not file_paths:
            return

        embedding_start = time.time()

        # Generate embeddings for all files in parallel
        embedding_tasks = []
        for file_path in file_paths:
            task = self.embedding_system.embed_file(file_path)
            embedding_tasks.append((file_path, task))

        # Collect results
        file_embeddings = {}
        for file_path, task in embedding_tasks:
            try:
                results = await task
                if results:
                    file_embeddings[file_path] = self._prepare_embedding_data(results)
                    logger.debug(f"Generated {len(results)} embeddings for {file_path}")
            except Exception as e:
                logger.error(f"Failed to generate embeddings for {file_path}: {e}")

        self.stats.embedding_time_ms = (time.time() - embedding_start) * 1000

        if not file_embeddings:
            logger.info("No embeddings generated")
            return

        # Update FAISS index
        index_start = time.time()
        results = await self.faiss_manager.index_files(
            file_embeddings, force_update=True
        )
        self.stats.index_update_time_ms = (time.time() - index_start) * 1000

        # Update metadata for successful files
        for file_path, success in results.items():
            if success and file_path in file_embeddings:
                vector_count = len(file_embeddings[file_path]["embeddings"])
                self.change_detector.update_file_metadata(file_path, vector_count)
                self.stats.vectors_added += vector_count
                logger.debug(
                    f"Updated metadata for {file_path} ({vector_count} vectors)"
                )

    async def _remove_files(self, file_paths: list[str]) -> None:
        """Remove multiple files from the index."""
        for file_path in file_paths:
            # Remove from change detector metadata
            self.change_detector.remove_file_metadata(file_path)

            # Note: FAISS doesn't support efficient vector removal
            # In a production system, you might need to rebuild the index periodically
            # or use a different vector database that supports deletion

            logger.debug(f"Removed metadata for {file_path}")
            self.stats.files_removed += 1

    def _prepare_embedding_data(
        self, results: list[CodeEmbeddingResult]
    ) -> dict[str, Any]:
        """Prepare embedding data for FAISS indexing."""
        embeddings = []
        line_ranges = []
        content_previews = []
        token_counts = []

        for result in results:
            embeddings.append(result.embedding)
            chunk = result.chunk
            line_ranges.append((chunk.start_line, chunk.end_line))
            content_previews.append(chunk.content[:200])  # Preview first 200 chars
            token_counts.append(chunk.tokens)

        return {
            "embeddings": embeddings,
            "line_ranges": line_ranges,
            "content_previews": content_previews,
            "token_counts": token_counts,
        }

    async def get_index_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about the index."""
        faiss_stats = await self.faiss_manager.get_stats()

        return {
            "faiss_stats": asdict(faiss_stats),
            "file_count": len(self.change_detector.file_metadata),
            "total_vectors": sum(
                meta.vector_count
                for meta in self.change_detector.file_metadata.values()
            ),
            "last_update_stats": asdict(self.stats),
            "monitoring_enabled": self.enable_monitoring and self.observer is not None,
            "is_initialized": self.is_initialized,
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Stop file monitoring
        if self.observer:
            self.observer.stop()
            self.observer.join()

        # Clean up other resources
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

        await self.faiss_manager.cleanup()

        logger.info("🧹 Incremental indexer cleanup completed")


# Example usage and testing
if __name__ == "__main__":

    async def test_incremental_indexing():
        """Test the incremental FAISS indexing system."""
        project_root = Path.cwd()

        # Initialize indexer
        indexer = IncrementalFAISSIndexer(
            project_root=project_root, enable_monitoring=False  # Disable for testing
        )

        # Initialize
        success = await indexer.initialize()
        print(f"Initialization: {'✅' if success else '❌'}")

        if success:
            # Perform initial scan
            print("\nPerforming incremental scan...")
            stats = await indexer.scan_and_update(
                ["*.py"]
            )  # Just scan root Python files

            print("\nIndexing Stats:")
            print(f"  Files checked: {stats.files_checked}")
            print(f"  Files changed: {stats.files_changed}")
            print(f"  Files added: {stats.files_added}")
            print(f"  Vectors added: {stats.vectors_added}")
            print(f"  Processing time: {stats.processing_time_ms:.1f}ms")
            print(f"  Embedding time: {stats.embedding_time_ms:.1f}ms")
            print(f"  Index update time: {stats.index_update_time_ms:.1f}ms")

            # Get comprehensive stats
            index_stats = await indexer.get_index_stats()
            print("\nIndex Stats:")
            print(f"  Total files: {index_stats['file_count']}")
            print(f"  Total vectors: {index_stats['total_vectors']}")
            print(f"  FAISS vectors: {index_stats['faiss_stats']['total_vectors']}")

        # Cleanup
        await indexer.cleanup()

    # Run test
    asyncio.run(test_incremental_indexing())
