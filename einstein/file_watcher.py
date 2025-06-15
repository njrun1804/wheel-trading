#!/usr/bin/env python3
"""
Einstein File Watcher - Real-time Index Updates

Provides real-time file monitoring with debounced updates for Einstein indexing.
Optimized for macOS FSEvents and hardware-accelerated performance.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from watchdog.events import (
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from .einstein_config import get_einstein_config

logger = logging.getLogger(__name__)


@dataclass
class FileChangeEvent:
    """Represents a file change event."""
    file_path: Path
    event_type: str  # 'created', 'modified', 'deleted'
    timestamp: float


class EinsteinFileWatcher(FileSystemEventHandler):
    """File system watcher optimized for Einstein indexing."""
    
    def __init__(self, update_callback: Callable[[FileChangeEvent], None]):
        super().__init__()
        self.update_callback = update_callback
        config = get_einstein_config()
        self.debounce_delay = config.monitoring.debounce_delay_ms / 1000.0  # Convert to seconds
        self.pending_updates: dict[str, FileChangeEvent] = {}
        self.debounce_tasks: dict[str, asyncio.Task] = {}
        
        # File patterns to watch
        self.watched_extensions = {'.py', '.md', '.txt', '.json', '.yaml', '.yml'}
        self.ignored_patterns = {
            '__pycache__',
            '.git',
            '.DS_Store',
            'node_modules',
            '.pytest_cache',
            '.mypy_cache'
        }
    
    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed based on patterns."""
        path = Path(file_path)
        
        # Check if any ignored pattern is in the path
        for ignored in self.ignored_patterns:
            if ignored in path.parts:
                return False
        
        # Check extension
        return path.suffix in self.watched_extensions
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and self.should_process_file(event.src_path):
            self._schedule_update(event.src_path, 'modified')
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and self.should_process_file(event.src_path):
            self._schedule_update(event.src_path, 'created')
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory and self.should_process_file(event.src_path):
            self._schedule_update(event.src_path, 'deleted')
    
    def _schedule_update(self, file_path: str, event_type: str):
        """Schedule debounced update for file."""
        
        # Cancel existing debounce task for this file
        if file_path in self.debounce_tasks:
            self.debounce_tasks[file_path].cancel()
        
        # Create new file change event
        change_event = FileChangeEvent(
            file_path=Path(file_path),
            event_type=event_type,
            timestamp=time.time()
        )
        
        self.pending_updates[file_path] = change_event
        
        # Schedule debounced processing
        task = asyncio.create_task(self._debounced_update(file_path))
        self.debounce_tasks[file_path] = task
    
    async def _debounced_update(self, file_path: str):
        """Process file update after debounce delay."""
        
        try:
            # Wait for debounce period
            await asyncio.sleep(self.debounce_delay)
            
            # Get the pending update
            if file_path in self.pending_updates:
                change_event = self.pending_updates.pop(file_path)
                
                # Call the update callback
                if self.update_callback:
                    await self.update_callback(change_event)
                
                logger.debug(f"Processed {change_event.event_type}: {file_path}")
        
        except asyncio.CancelledError:
            # Task was cancelled due to new event
            pass
        except Exception as e:
            logger.error(f"Error processing file update {file_path}: {e}", exc_info=True,
                        extra={
                            'operation': 'debounced_update',
                            'error_type': type(e).__name__,
                            'file_path': file_path,
                            'event_type': change_event.event_type if 'change_event' in locals() else 'unknown',
                            'debounce_delay': self.debounce_delay,
                            'callback_available': self.update_callback is not None,
                            'pending_updates_count': len(self.pending_updates),
                            'active_tasks_count': len(self.debounce_tasks)
                        })
        finally:
            # Clean up task reference
            if file_path in self.debounce_tasks:
                del self.debounce_tasks[file_path]


class EinsteinRealtimeIndexer:
    """Real-time indexing system for Einstein."""
    
    def __init__(self, einstein_hub, watch_paths: list = None):
        self.einstein_hub = einstein_hub
        self.watch_paths = watch_paths or [Path.cwd()]
        self.observer = Observer()
        self.file_handler = EinsteinFileWatcher(self._handle_file_change)
        self.is_watching = False
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'updates_queued': 0,
            'updates_completed': 0,
            'start_time': None
        }
    
    def start_watching(self):
        """Start file system monitoring."""
        
        if self.is_watching:
            return
        
        logger.info("🔍 Starting Einstein real-time indexing...")
        
        # Schedule watching for each path
        for watch_path in self.watch_paths:
            if watch_path.exists():
                self.observer.schedule(
                    self.file_handler,
                    str(watch_path),
                    recursive=True
                )
                logger.info(f"   Watching: {watch_path}")
        
        # Start observer
        self.observer.start()
        self.is_watching = True
        self.stats['start_time'] = time.time()
        
        logger.info("✅ Real-time indexing active")
    
    def stop_watching(self):
        """Stop file system monitoring."""
        
        if not self.is_watching:
            return
        
        logger.info("🛑 Stopping real-time indexing...")
        
        self.observer.stop()
        self.observer.join()
        self.is_watching = False
        
        logger.info("✅ Real-time indexing stopped")
    
    async def _handle_file_change(self, change_event: FileChangeEvent):
        """Handle file change by updating relevant indexes."""
        
        try:
            self.stats['updates_queued'] += 1
            
            logger.debug(f"Processing {change_event.event_type}: {change_event.file_path}")
            
            if change_event.event_type in ['created', 'modified']:
                # Re-analyze the file
                await self.einstein_hub._analyze_file(change_event.file_path)
                
                # Update embeddings if needed
                if hasattr(self.einstein_hub, 'embedding_pipeline'):
                    try:
                        await self.einstein_hub.embedding_pipeline.update_file(
                            str(change_event.file_path)
                        )
                    except Exception as e:
                        logger.error(f"Embedding update failed for {change_event.file_path}: {e}", exc_info=True,
                                   extra={
                                       'operation': 'embedding_update',
                                       'error_type': type(e).__name__,
                                       'file_path': str(change_event.file_path),
                                       'event_type': change_event.event_type,
                                       'has_embedding_pipeline': hasattr(self.einstein_hub, 'embedding_pipeline'),
                                       'stats': self.stats,
                                       'file_exists': change_event.file_path.exists(),
                                       'file_size': change_event.file_path.stat().st_size if change_event.file_path.exists() else 0
                                   })
                
                # Update dependency graph
                if hasattr(self.einstein_hub, 'dependency_graph'):
                    try:
                        await self.einstein_hub.dependency_graph.update_file(
                            str(change_event.file_path)
                        )
                    except Exception as e:
                        logger.error(f"Dependency graph update failed for {change_event.file_path}: {e}", exc_info=True,
                                   extra={
                                       'operation': 'dependency_graph_update',
                                       'error_type': type(e).__name__,
                                       'file_path': str(change_event.file_path),
                                       'event_type': change_event.event_type,
                                       'has_dependency_graph': hasattr(self.einstein_hub, 'dependency_graph'),
                                       'stats': self.stats,
                                       'file_exists': change_event.file_path.exists(),
                                       'file_extension': change_event.file_path.suffix
                                   })
            
            elif change_event.event_type == 'deleted':
                # Remove from all indexes
                await self._remove_from_indexes(change_event.file_path)
            
            self.stats['updates_completed'] += 1
            self.stats['files_processed'] += 1
            
        except Exception as e:
            logger.error(f"Failed to process file change {change_event.file_path}: {e}", exc_info=True,
                        extra={
                            'operation': 'handle_file_change',
                            'error_type': type(e).__name__,
                            'file_path': str(change_event.file_path),
                            'event_type': change_event.event_type,
                            'stats': self.stats,
                            'einstein_hub_available': self.einstein_hub is not None,
                            'watch_paths': [str(p) for p in self.watch_paths],
                            'timestamp': change_event.timestamp,
                            'uptime': time.time() - self.stats.get('start_time', time.time())
                        })
    
    async def _remove_from_indexes(self, file_path: Path):
        """Remove file from all Einstein indexes."""
        
        try:
            # Remove from analytics DB
            await self.einstein_hub.duckdb.execute(
                "DELETE FROM file_analytics WHERE file_path = ?",
                (str(file_path),)
            )
            
            # Remove from embedding index if available
            if hasattr(self.einstein_hub, 'embedding_pipeline'):
                await self.einstein_hub.embedding_pipeline.remove_file(str(file_path))
            
            # Remove from dependency graph
            if hasattr(self.einstein_hub, 'dependency_graph'):
                await self.einstein_hub.dependency_graph.remove_file(str(file_path))
            
            logger.debug(f"Removed {file_path} from all indexes")
            
        except Exception as e:
            logger.error(f"Failed to remove {file_path} from indexes: {e}", exc_info=True,
                        extra={
                            'operation': 'remove_from_indexes',
                            'error_type': type(e).__name__,
                            'file_path': str(file_path),
                            'has_duckdb': hasattr(self.einstein_hub, 'duckdb'),
                            'has_embedding_pipeline': hasattr(self.einstein_hub, 'embedding_pipeline'),
                            'has_dependency_graph': hasattr(self.einstein_hub, 'dependency_graph'),
                            'file_existed': file_path in str(e) if 'not found' in str(e).lower() else 'unknown'
                        })
    
    def get_stats(self) -> dict[str, Any]:
        """Get file watching statistics."""
        
        uptime = 0
        if self.stats['start_time']:
            uptime = time.time() - self.stats['start_time']
        
        return {
            'is_watching': self.is_watching,
            'files_processed': self.stats['files_processed'],
            'updates_queued': self.stats['updates_queued'],
            'updates_completed': self.stats['updates_completed'],
            'uptime_seconds': uptime,
            'watch_paths': [str(p) for p in self.watch_paths]
        }


# Integration with Einstein
def add_realtime_indexing(einstein_hub, watch_paths: list = None):
    """Add real-time indexing capability to Einstein hub."""
    
    indexer = EinsteinRealtimeIndexer(einstein_hub, watch_paths)
    einstein_hub.realtime_indexer = indexer
    
    # Add methods to Einstein hub
    einstein_hub.start_file_watching = indexer.start_watching
    einstein_hub.stop_file_watching = indexer.stop_watching
    einstein_hub.get_file_watching_stats = indexer.get_stats
    
    return indexer


if __name__ == "__main__":
    # Test the file watcher
    async def test_watcher():
        def dummy_callback(event):
            print(f"File {event.event_type}: {event.file_path}")
        
        watcher = EinsteinFileWatcher(dummy_callback)
        
        # Test file filtering
        test_files = [
            "/path/to/script.py",
            "/path/to/__pycache__/cache.pyc", 
            "/path/to/.git/config",
            "/path/to/data.json"
        ]
        
        for file_path in test_files:
            should_process = watcher.should_process_file(file_path)
            print(f"{file_path}: {'✅' if should_process else '❌'}")
    
    asyncio.run(test_watcher())
