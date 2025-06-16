"""Comprehensive resource cleanup and management utilities.

This module provides a centralized resource management system to prevent
file descriptor exhaustion and resource leaks across the codebase.
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import gc
import logging
import os
import resource
import signal
import sys
import threading
import time
import weakref
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from weakref import WeakSet

import psutil

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceMetrics:
    """Metrics for tracking resource usage."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    # File descriptors
    open_files: int = 0
    max_files: int = 0
    
    # Memory usage
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    
    # Database connections
    db_connections: int = 0
    
    # Subprocess/Thread counts
    processes: int = 0
    threads: int = 0
    
    # Network connections
    network_connections: int = 0
    
    # Custom resource counters
    custom_resources: Dict[str, int] = field(default_factory=dict)
    
    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class ResourceTracker:
    """Tracks and manages system resources with automatic cleanup."""
    
    def __init__(self, max_files: int = 1000, max_memory_mb: float = 2000.0):
        self.max_files = max_files
        self.max_memory_mb = max_memory_mb
        
        # Resource tracking
        self._tracked_resources: WeakSet = WeakSet()
        self._resource_counts = defaultdict(int)
        self._resource_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Cleanup registry
        self._cleanup_functions: List[Callable] = []
        self._cleanup_lock = threading.Lock()
        
        # Monitoring
        self._monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_metrics: Optional[ResourceMetrics] = None
        
        # Setup automatic cleanup
        atexit.register(self.cleanup_all)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, cleaning up resources...")
        self.cleanup_all()
        sys.exit(0)
    
    def register_resource(self, resource: Any, resource_type: str, 
                         cleanup_func: Optional[Callable] = None) -> None:
        """Register a resource for tracking and cleanup."""
        self._tracked_resources.add(resource)
        self._resource_counts[resource_type] += 1
        
        if cleanup_func:
            self.register_cleanup(cleanup_func)
            
        logger.debug(f"Registered {resource_type} resource: {self._resource_counts[resource_type]} total")
    
    def unregister_resource(self, resource: Any, resource_type: str) -> None:
        """Unregister a resource from tracking."""
        if resource in self._tracked_resources:
            self._tracked_resources.remove(resource)
            self._resource_counts[resource_type] = max(0, self._resource_counts[resource_type] - 1)
            logger.debug(f"Unregistered {resource_type} resource: {self._resource_counts[resource_type]} remaining")
    
    def register_cleanup(self, cleanup_func: Callable) -> None:
        """Register a cleanup function to be called on shutdown."""
        with self._cleanup_lock:
            self._cleanup_functions.append(cleanup_func)
    
    def register_resource_callback(self, resource_type: str, callback: Callable) -> None:
        """Register a callback to be called when resource limits are exceeded."""
        self._resource_callbacks[resource_type].append(callback)
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics."""
        try:
            process = psutil.Process()
            
            # Get file descriptor count
            try:
                open_files = process.num_fds() if hasattr(process, 'num_fds') else len(process.open_files())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                open_files = 0
            
            # Get memory usage
            try:
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                memory_percent = process.memory_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                memory_mb = 0.0
                memory_percent = 0.0
            
            # Get thread/process counts
            try:
                threads = process.num_threads()
                processes = len(process.children(recursive=True)) + 1  # +1 for current process
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                threads = 0
                processes = 1
            
            # Get network connections
            try:
                network_connections = len(process.connections())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                network_connections = 0
            
            # Get system limits
            try:
                max_files = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
            except (OSError, AttributeError):
                max_files = 1024
            
            metrics = ResourceMetrics(
                open_files=open_files,
                max_files=max_files,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                db_connections=self._resource_counts['database'],
                processes=processes,
                threads=threads,
                network_connections=network_connections,
                custom_resources=dict(self._resource_counts),
            )
            
            # Check for warnings
            if open_files > (max_files * 0.8):
                metrics.warnings.append(f"High file descriptor usage: {open_files}/{max_files}")
            
            if memory_mb > self.max_memory_mb:
                metrics.warnings.append(f"High memory usage: {memory_mb:.1f}MB")
            
            if open_files >= self.max_files:
                metrics.errors.append(f"File descriptor limit exceeded: {open_files}/{max_files}")
                
            self._last_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting resource metrics: {e}")
            return ResourceMetrics(errors=[f"Failed to get metrics: {e}"])
    
    def check_resource_limits(self) -> bool:
        """Check if resource limits are exceeded."""
        metrics = self.get_current_metrics()
        
        # Check file descriptor limit
        if metrics.open_files >= self.max_files:
            logger.error(f"File descriptor limit exceeded: {metrics.open_files}/{metrics.max_files}")
            self._trigger_callbacks('file_descriptors', metrics)
            return False
        
        # Check memory limit
        if metrics.memory_mb > self.max_memory_mb:
            logger.warning(f"Memory limit exceeded: {metrics.memory_mb:.1f}MB")
            self._trigger_callbacks('memory', metrics)
            return False
        
        return True
    
    def _trigger_callbacks(self, resource_type: str, metrics: ResourceMetrics) -> None:
        """Trigger callbacks for resource limit violations."""
        for callback in self._resource_callbacks[resource_type]:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Error in resource callback: {e}")
    
    def cleanup_all(self) -> None:
        """Clean up all registered resources."""
        logger.info("Starting comprehensive resource cleanup...")
        
        with self._cleanup_lock:
            # Run cleanup functions in reverse order
            for cleanup_func in reversed(self._cleanup_functions):
                try:
                    cleanup_func()
                except Exception as e:
                    logger.error(f"Error in cleanup function: {e}")
            
            self._cleanup_functions.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Log final metrics
        final_metrics = self.get_current_metrics()
        logger.info(f"Cleanup complete. Final metrics: {final_metrics.open_files} files, "
                   f"{final_metrics.memory_mb:.1f}MB memory, {final_metrics.threads} threads")
    
    async def start_monitoring(self, interval: float = 30.0) -> None:
        """Start background resource monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitor_resources(interval))
        logger.info(f"Started resource monitoring (interval: {interval}s)")
    
    async def stop_monitoring(self) -> None:
        """Stop background resource monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped resource monitoring")
    
    async def _monitor_resources(self, interval: float) -> None:
        """Background resource monitoring task."""
        while self._monitoring_active:
            try:
                metrics = self.get_current_metrics()
                
                # Log warnings
                for warning in metrics.warnings:
                    logger.warning(f"Resource warning: {warning}")
                
                # Log errors and trigger cleanup if needed
                for error in metrics.errors:
                    logger.error(f"Resource error: {error}")
                    # Trigger emergency cleanup
                    self._emergency_cleanup()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(interval)
    
    def _emergency_cleanup(self) -> None:
        """Emergency cleanup when resource limits are exceeded."""
        logger.warning("Triggering emergency resource cleanup...")
        
        # Force garbage collection
        gc.collect()
        
        # Close any tracked resources that have close methods
        for resource in list(self._tracked_resources):
            try:
                if hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'cleanup'):
                    resource.cleanup()
            except Exception as e:
                logger.debug(f"Error closing resource: {e}")


# Global resource tracker instance
_resource_tracker: Optional[ResourceTracker] = None


def get_resource_tracker() -> ResourceTracker:
    """Get or create the global resource tracker."""
    global _resource_tracker
    if _resource_tracker is None:
        _resource_tracker = ResourceTracker()
    return _resource_tracker


# Context managers for resource management

@contextmanager
def managed_resource(resource: Any, resource_type: str, cleanup_func: Optional[Callable] = None):
    """Context manager for automatic resource tracking and cleanup."""
    tracker = get_resource_tracker()
    
    try:
        tracker.register_resource(resource, resource_type, cleanup_func)
        yield resource
    finally:
        tracker.unregister_resource(resource, resource_type)
        if cleanup_func:
            try:
                cleanup_func()
            except Exception as e:
                logger.error(f"Error in resource cleanup: {e}")


@asynccontextmanager
async def managed_async_resource(resource: Any, resource_type: str, cleanup_func: Optional[Callable] = None):
    """Async context manager for automatic resource tracking and cleanup."""
    tracker = get_resource_tracker()
    
    try:
        tracker.register_resource(resource, resource_type, cleanup_func)
        yield resource
    finally:
        tracker.unregister_resource(resource, resource_type)
        if cleanup_func:
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func()
                else:
                    cleanup_func()
            except Exception as e:
                logger.error(f"Error in async resource cleanup: {e}")


# Decorators for automatic resource management

def track_resources(resource_type: str):
    """Decorator to automatically track resources created by a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if result is not None:
                tracker = get_resource_tracker()
                tracker.register_resource(result, resource_type)
            return result
        return wrapper
    return decorator


def cleanup_on_exit(cleanup_func: Callable):
    """Decorator to register a cleanup function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_resource_tracker()
            tracker.register_cleanup(cleanup_func)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Specific resource managers

class DatabaseConnectionManager:
    """Manages database connections with automatic cleanup."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self._connections: Dict[str, Any] = {}
        self._connection_lock = threading.Lock()
        self._tracker = get_resource_tracker()
    
    @contextmanager
    def get_connection(self, connection_id: str, factory: Callable):
        """Get or create a database connection with automatic cleanup."""
        with self._connection_lock:
            if connection_id not in self._connections:
                if len(self._connections) >= self.max_connections:
                    # Close oldest connection
                    oldest_id = next(iter(self._connections))
                    self._close_connection(oldest_id)
                
                connection = factory()
                self._connections[connection_id] = connection
                self._tracker.register_resource(connection, 'database')
        
        try:
            yield self._connections[connection_id]  
        except Exception as e:
            logger.error(f"Error using database connection {connection_id}: {e}")
            # Remove problematic connection
            self._close_connection(connection_id)
            raise
    
    def _close_connection(self, connection_id: str):
        """Close and remove a database connection."""
        if connection_id in self._connections:
            connection = self._connections.pop(connection_id)
            try:
                if hasattr(connection, 'close'):
                    connection.close()
                self._tracker.unregister_resource(connection, 'database')
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
    
    def cleanup_all(self):
        """Close all database connections."""
        with self._connection_lock:
            for connection_id in list(self._connections.keys()):
                self._close_connection(connection_id)


class FileHandleManager:
    """Manages file handles with automatic cleanup."""
    
    def __init__(self, max_files: int = 100):
        self.max_files = max_files
        self._open_files: Dict[str, Any] = {}
        self._file_lock = threading.Lock()
        self._tracker = get_resource_tracker()
    
    @contextmanager
    def open_file(self, file_path: Union[str, Path], mode: str = 'r', **kwargs):
        """Open a file with automatic cleanup and tracking."""
        file_path = str(file_path)
        
        with self._file_lock:
            if len(self._open_files) >= self.max_files:
                # Close oldest file
                oldest_path = next(iter(self._open_files))
                self._close_file(oldest_path)
        
        try:
            file_handle = open(file_path, mode, **kwargs)
            with self._file_lock:
                self._open_files[file_path] = file_handle
            self._tracker.register_resource(file_handle, 'file')
            
            yield file_handle
            
        finally:
            self._close_file(file_path)
    
    def _close_file(self, file_path: str):
        """Close and remove a file handle."""
        with self._file_lock:
            if file_path in self._open_files:
                file_handle = self._open_files.pop(file_path)
                try:
                    if not file_handle.closed:
                        file_handle.close()
                    self._tracker.unregister_resource(file_handle, 'file')
                except Exception as e:
                    logger.error(f"Error closing file {file_path}: {e}")
    
    def cleanup_all(self):
        """Close all open files."""
        with self._file_lock:
            for file_path in list(self._open_files.keys()):
                self._close_file(file_path)


# Global managers
_db_manager: Optional[DatabaseConnectionManager] = None
_file_manager: Optional[FileHandleManager] = None


def get_db_manager() -> DatabaseConnectionManager:
    """Get or create global database connection manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseConnectionManager()
        get_resource_tracker().register_cleanup(_db_manager.cleanup_all)
    return _db_manager


def get_file_manager() -> FileHandleManager:
    """Get or create global file handle manager."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileHandleManager()
        get_resource_tracker().register_cleanup(_file_manager.cleanup_all)
    return _file_manager


# Utility functions

def log_resource_usage(prefix: str = "Resource usage"):
    """Log current resource usage."""
    tracker = get_resource_tracker()
    metrics = tracker.get_current_metrics()
    
    logger.info(f"{prefix}: {metrics.open_files} files, {metrics.memory_mb:.1f}MB memory, "
               f"{metrics.threads} threads, {metrics.db_connections} DB connections")
    
    if metrics.warnings:
        for warning in metrics.warnings:
            logger.warning(warning)
    
    if metrics.errors:
        for error in metrics.errors:
            logger.error(error)


def force_cleanup():
    """Force cleanup of all resources."""
    logger.info("Forcing cleanup of all resources...")
    
    # Get global managers and clean them up
    if _db_manager:
        _db_manager.cleanup_all()
    
    if _file_manager:
        _file_manager.cleanup_all()
    
    # Clean up global tracker
    tracker = get_resource_tracker()
    tracker.cleanup_all()
    
    # Force garbage collection
    gc.collect()
    
    log_resource_usage("After cleanup")


async def monitor_resources(interval: float = 30.0):
    """Start monitoring resources in the background."""
    tracker = get_resource_tracker()
    await tracker.start_monitoring(interval)


# Initialize resource tracking
def init_resource_management():
    """Initialize the resource management system."""
    tracker = get_resource_tracker()
    
    # Set up resource limit callbacks
    def handle_fd_limit(metrics):
        logger.error("File descriptor limit reached, forcing cleanup...")
        force_cleanup()
    
    def handle_memory_limit(metrics):
        logger.error("Memory limit reached, forcing cleanup...")
        force_cleanup()
    
    tracker.register_resource_callback('file_descriptors', handle_fd_limit)
    tracker.register_resource_callback('memory', handle_memory_limit)
    
    logger.info("Resource management system initialized")


# Auto-initialize on import
init_resource_management()