#!/usr/bin/env python3
"""Real-time memory protection and overflow prevention system."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import resource
import signal
import sys
import time
import traceback
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Union

import psutil

from unity_wheel.monitoring.pressure_gauge import AdaptiveMemoryMonitor, get_pressure_monitor
from unity_wheel.utils import get_logger

logger = get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    
    timestamp: datetime
    process_rss_mb: float
    process_vms_mb: float
    system_available_mb: float
    system_used_percent: float
    swap_used_percent: float
    gc_collected: int
    largest_objects: List[str] = field(default_factory=list)
    
    @property
    def is_critical(self) -> bool:
        """Check if memory usage is in critical range."""
        return (
            self.system_used_percent > 85.0 or
            self.swap_used_percent > 10.0 or
            self.system_available_mb < 2048  # Less than 2GB available
        )
    
    @property
    def is_warning(self) -> bool:
        """Check if memory usage is in warning range."""
        return (
            self.system_used_percent > 75.0 or
            self.swap_used_percent > 5.0 or
            self.system_available_mb < 4096  # Less than 4GB available
        )


@dataclass
class MemoryLimit:
    """Memory limit configuration."""
    
    max_rss_mb: float = 16384  # 16GB max RSS
    max_string_length: int = 100_000_000  # 100MB max string
    max_list_length: int = 10_000_000  # 10M items max
    max_dict_size: int = 5_000_000  # 5M keys max
    emergency_gc_threshold: float = 80.0  # Trigger GC at 80% memory
    critical_shutdown_threshold: float = 90.0  # Emergency shutdown at 90%


class StringOverflowProtector:
    """Prevents string overflow errors with intelligent truncation."""
    
    def __init__(self, max_length: int = 100_000_000):
        self.max_length = max_length
        self.truncation_count = 0
        self.truncated_operations = deque(maxlen=100)
    
    def protect_string(self, data: str, context: str = "unknown") -> str:
        """Protect against string overflow with intelligent truncation."""
        if len(data) <= self.max_length:
            return data
        
        # Truncate with context preservation
        self.truncation_count += 1
        truncated_length = self.max_length - 1000  # Leave buffer for safety message
        
        # Try to truncate at meaningful boundaries
        if truncated_length > 10000:
            # Look for natural break points
            break_points = ['\n\n', '\n', '. ', ', ', ' ']
            for break_point in break_points:
                last_break = data.rfind(break_point, 0, truncated_length)
                if last_break > truncated_length * 0.8:  # At least 80% of desired length
                    truncated_length = last_break + len(break_point)
                    break
        
        truncated = data[:truncated_length]
        safety_message = (
            f"\n\n[MEMORY GUARD: String truncated from {len(data):,} to {len(truncated):,} "
            f"characters in context '{context}' to prevent overflow]"
        )
        
        result = truncated + safety_message
        
        # Log truncation event
        self.truncated_operations.append({
            'timestamp': datetime.now(UTC),
            'context': context,
            'original_length': len(data),
            'truncated_length': len(result),
            'reduction_percent': (1 - len(result) / len(data)) * 100
        })
        
        logger.warning(
            f"String truncated in {context}: {len(data):,} -> {len(result):,} chars "
            f"({(1 - len(result) / len(data)) * 100:.1f}% reduction)"
        )
        
        return result
    
    def protect_collection(self, data: Union[list, dict], context: str = "unknown") -> Union[list, dict]:
        """Protect against collection overflow."""
        if isinstance(data, list):
            if len(data) > 1_000_000:  # 1M items
                truncated = data[:500_000]  # Keep first 500K
                logger.warning(f"List truncated in {context}: {len(data)} -> {len(truncated)} items")
                return truncated
        elif isinstance(data, dict):
            if len(data) > 500_000:  # 500K keys
                truncated = dict(list(data.items())[:250_000])  # Keep first 250K
                logger.warning(f"Dict truncated in {context}: {len(data)} -> {len(truncated)} keys")
                return truncated
        
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get truncation statistics."""
        return {
            'total_truncations': self.truncation_count,
            'recent_truncations': len(self.truncated_operations),
            'max_string_length': self.max_length,
            'recent_operations': list(self.truncated_operations)[-10:]  # Last 10
        }


class MemoryGuard:
    """Real-time memory monitoring and protection system."""
    
    def __init__(self, limits: Optional[MemoryLimit] = None):
        self.limits = limits or MemoryLimit()
        self.string_protector = StringOverflowProtector(self.limits.max_string_length)
        self.pressure_monitor = get_pressure_monitor()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[Thread] = None
        self.stop_event = Event()
        self.lock = Lock()
        
        # Memory snapshots
        self.snapshots = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.alerts = deque(maxlen=500)      # Keep last 500 alerts
        
        # Emergency callbacks
        self.warning_callbacks: List[Callable[[MemorySnapshot], None]] = []
        self.critical_callbacks: List[Callable[[MemorySnapshot], None]] = []
        self.emergency_callbacks: List[Callable[[MemorySnapshot], None]] = []
        
        # Performance counters
        self.gc_collections = 0
        self.emergency_actions = 0
        self.memory_recoveries = 0
        
        # Protected operations tracking
        self.protected_operations = defaultdict(int)
        
        # Setup signal handlers for emergency shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for emergency situations."""
        def emergency_handler(signum, frame):
            logger.critical("Emergency signal received - forcing memory cleanup")
            self.emergency_cleanup()
            sys.exit(1)
        
        signal.signal(signal.SIGUSR1, emergency_handler)
        signal.signal(signal.SIGTERM, emergency_handler)
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start continuous memory monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor_loop, args=(interval_seconds,), daemon=True)
        self.monitor_thread.start()
        logger.info("Memory guard monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory guard monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: float):
        """Main monitoring loop."""
        while not self.stop_event.wait(interval_seconds):
            try:
                snapshot = self._take_snapshot()
                
                with self.lock:
                    self.snapshots.append(snapshot)
                
                # Check thresholds and take action
                self._check_thresholds(snapshot)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                # Continue monitoring even if there's an error
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Get garbage collection stats
        gc_stats = gc.get_stats()
        total_collected = sum(stat.get('collected', 0) for stat in gc_stats)
        
        # Get largest objects (expensive, so only occasionally)
        largest_objects = []
        if len(self.snapshots) % 10 == 0:  # Every 10th snapshot
            largest_objects = self._get_largest_objects()
        
        return MemorySnapshot(
            timestamp=datetime.now(UTC),
            process_rss_mb=memory_info.rss / (1024 * 1024),
            process_vms_mb=memory_info.vms / (1024 * 1024),
            system_available_mb=system_mem.available / (1024 * 1024),
            system_used_percent=system_mem.percent,
            swap_used_percent=swap.percent,
            gc_collected=total_collected,
            largest_objects=largest_objects
        )
    
    def _get_largest_objects(self) -> List[str]:
        """Get information about largest objects in memory."""
        try:
            import sys
            large_objects = []
            
            # Get all objects and find the largest ones
            all_objects = gc.get_objects()
            object_sizes = []
            
            for obj in all_objects[:1000]:  # Limit to first 1000 for performance
                try:
                    size = sys.getsizeof(obj)
                    if size > 1024 * 1024:  # > 1MB
                        obj_type = type(obj).__name__
                        obj_info = f"{obj_type}: {size / (1024*1024):.1f}MB"
                        object_sizes.append((size, obj_info))
                except:
                    continue
            
            # Sort by size and return top 10
            object_sizes.sort(reverse=True)
            large_objects = [info for _, info in object_sizes[:10]]
            
            return large_objects
        except Exception as e:
            logger.debug(f"Error getting largest objects: {e}")
            return []
    
    def _check_thresholds(self, snapshot: MemorySnapshot):
        """Check memory thresholds and take appropriate action."""
        # Emergency threshold - immediate action required
        if snapshot.system_used_percent >= self.limits.critical_shutdown_threshold:
            self._handle_emergency(snapshot)
        
        # Critical threshold - aggressive cleanup
        elif snapshot.is_critical:
            self._handle_critical(snapshot)
        
        # Warning threshold - preventive measures
        elif snapshot.is_warning:
            self._handle_warning(snapshot)
        
        # Automatic GC trigger
        elif snapshot.system_used_percent >= self.limits.emergency_gc_threshold:
            self._trigger_gc("automatic", snapshot)
    
    def _handle_warning(self, snapshot: MemorySnapshot):
        """Handle warning level memory pressure."""
        alert = {
            'level': 'warning',
            'timestamp': snapshot.timestamp,
            'message': f"Memory usage at {snapshot.system_used_percent:.1f}%",
            'actions_taken': []
        }
        
        # Trigger warning callbacks
        for callback in self.warning_callbacks:
            try:
                callback(snapshot)
                alert['actions_taken'].append('callback_executed')
            except Exception as e:
                logger.error(f"Warning callback failed: {e}")
        
        # Optional GC
        if len(self.snapshots) > 5:
            recent_trend = [s.system_used_percent for s in list(self.snapshots)[-5:]]
            if all(recent_trend[i] <= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                # Memory usage is trending up - trigger GC
                self._trigger_gc("warning_trend", snapshot)
                alert['actions_taken'].append('gc_triggered')
        
        with self.lock:
            self.alerts.append(alert)
        
        logger.warning(f"Memory warning: {alert['message']}")
    
    def _handle_critical(self, snapshot: MemorySnapshot):
        """Handle critical level memory pressure."""
        alert = {
            'level': 'critical',
            'timestamp': snapshot.timestamp,
            'message': f"Critical memory usage: {snapshot.system_used_percent:.1f}%",
            'actions_taken': []
        }
        
        # Aggressive garbage collection
        self._trigger_gc("critical", snapshot)
        alert['actions_taken'].append('aggressive_gc')
        
        # Trigger critical callbacks
        for callback in self.critical_callbacks:
            try:
                callback(snapshot)
                alert['actions_taken'].append('critical_callback')
            except Exception as e:
                logger.error(f"Critical callback failed: {e}")
        
        # Clear caches if available
        try:
            # Clear various caches
            if hasattr(gc, 'collect'):
                for generation in range(3):
                    collected = gc.collect(generation)
                    if collected > 0:
                        alert['actions_taken'].append(f'gc_gen_{generation}:{collected}')
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
        
        with self.lock:
            self.alerts.append(alert)
        
        logger.critical(f"Critical memory situation: {alert['message']}")
    
    def _handle_emergency(self, snapshot: MemorySnapshot):
        """Handle emergency memory situation."""
        self.emergency_actions += 1
        
        alert = {
            'level': 'emergency',
            'timestamp': snapshot.timestamp,
            'message': f"EMERGENCY: Memory usage at {snapshot.system_used_percent:.1f}%",
            'actions_taken': []
        }
        
        logger.critical("EMERGENCY MEMORY SITUATION - Taking immediate action")
        
        # Emergency cleanup
        self.emergency_cleanup()
        alert['actions_taken'].append('emergency_cleanup')
        
        # Trigger emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback(snapshot)
                alert['actions_taken'].append('emergency_callback')
            except Exception as e:  
                logger.error(f"Emergency callback failed: {e}")
        
        with self.lock:
            self.alerts.append(alert)
        
        # If still in emergency after cleanup, consider more drastic measures
        time.sleep(1)  # Allow cleanup to take effect
        post_cleanup_snapshot = self._take_snapshot()
        if post_cleanup_snapshot.system_used_percent >= self.limits.critical_shutdown_threshold:
            logger.critical("Emergency cleanup insufficient - system may be unstable")
            # Could trigger system shutdown or process termination here
    
    def _trigger_gc(self, reason: str, snapshot: MemorySnapshot):
        """Trigger garbage collection."""
        start_time = time.time()
        collected_total = 0
        
        try:
            # Full garbage collection
            for generation in range(3):
                collected = gc.collect(generation)
                collected_total += collected
            
            self.gc_collections += 1
            duration_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"GC triggered ({reason}): collected {collected_total} objects "
                f"in {duration_ms:.1f}ms"
            )
            
        except Exception as e:
            logger.error(f"GC trigger failed: {e}")
    
    def emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        logger.critical("Performing emergency memory cleanup")
        
        try:
            # Force aggressive garbage collection
            collected_total = 0
            for generation in range(3):
                collected = gc.collect(generation)
                collected_total += collected
            
            # Clear internal caches
            self._clear_internal_caches()
            
            # Try to free up memory from large objects
            self._emergency_object_cleanup()
            
            logger.info(f"Emergency cleanup completed: {collected_total} objects collected")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def _clear_internal_caches(self):
        """Clear internal monitoring caches."""
        try:
            # Keep only recent snapshots
            with self.lock:
                if len(self.snapshots) > 100:
                    recent_snapshots = list(self.snapshots)[-50:]
                    self.snapshots.clear()
                    self.snapshots.extend(recent_snapshots)
                
                # Keep only recent alerts
                if len(self.alerts) > 100:
                    recent_alerts = list(self.alerts)[-50:]
                    self.alerts.clear()
                    self.alerts.extend(recent_alerts)
            
            # Clear string protector caches
            self.string_protector.truncated_operations.clear()
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
    
    def _emergency_object_cleanup(self):
        """Emergency cleanup of large objects."""
        try:
            # Get current objects and try to identify memory hogs
            import sys
            all_objects = gc.get_objects()
            
            large_strings = []
            large_lists = []
            large_dicts = []
            
            for obj in all_objects[:5000]:  # Check first 5000 objects
                try:
                    size = sys.getsizeof(obj)
                    if size > 10 * 1024 * 1024:  # > 10MB
                        if isinstance(obj, str):
                            large_strings.append(obj)
                        elif isinstance(obj, list):
                            large_lists.append(obj)
                        elif isinstance(obj, dict):
                            large_dicts.append(obj)
                except:
                    continue
            
            # Log findings
            if large_strings or large_lists or large_dicts:
                logger.warning(
                    f"Found large objects: {len(large_strings)} strings, "
                    f"{len(large_lists)} lists, {len(large_dicts)} dicts"
                )
            
        except Exception as e:
            logger.error(f"Emergency object cleanup failed: {e}")
    
    @contextmanager
    def protected_operation(self, operation_name: str, max_memory_mb: Optional[float] = None):
        """Context manager for memory-protected operations."""
        self.protected_operations[operation_name] += 1
        start_snapshot = self._take_snapshot()
        
        try:
            # Check if we have enough memory to start
            if max_memory_mb and start_snapshot.system_available_mb < max_memory_mb:
                raise MemoryError(
                    f"Insufficient memory for {operation_name}: "
                    f"need {max_memory_mb:.1f}MB, available {start_snapshot.system_available_mb:.1f}MB"
                )
            
            logger.debug(f"Starting protected operation: {operation_name}")
            yield self
            
        except MemoryError:
            logger.error(f"Memory error in protected operation: {operation_name}")
            self._trigger_gc(f"protected_operation_{operation_name}", start_snapshot)
            raise
        
        except Exception as e:
            logger.error(f"Error in protected operation {operation_name}: {e}")
            raise
        
        finally:
            end_snapshot = self._take_snapshot()
            memory_delta = end_snapshot.process_rss_mb - start_snapshot.process_rss_mb
            
            if memory_delta > 100:  # Significant memory increase
                logger.warning(
                    f"Operation {operation_name} increased memory by {memory_delta:.1f}MB"
                )
                # Optional: trigger GC if memory increased significantly
                if memory_delta > 500:  # > 500MB increase
                    self._trigger_gc(f"large_increase_{operation_name}", end_snapshot)
    
    def protect_string_operation(self, data: str, operation: str = "unknown") -> str:
        """Protect string operations from overflow."""
        return self.string_protector.protect_string(data, operation)
    
    def protect_collection_operation(self, data: Union[list, dict], operation: str = "unknown") -> Union[list, dict]:
        """Protect collection operations from overflow."""
        return self.string_protector.protect_collection(data, operation)
    
    def register_warning_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Register callback for warning level memory pressure."""
        self.warning_callbacks.append(callback)
    
    def register_critical_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Register callback for critical level memory pressure."""
        self.critical_callbacks.append(callback)
    
    def register_emergency_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Register callback for emergency level memory pressure."""
        self.emergency_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current memory guard status."""
        current_snapshot = self._take_snapshot()
        
        with self.lock:
            recent_alerts = list(self.alerts)[-10:]  # Last 10 alerts
            total_snapshots = len(self.snapshots)
        
        return {
            'monitoring_active': self.is_monitoring,
            'current_memory': {
                'system_used_percent': current_snapshot.system_used_percent,
                'system_available_mb': current_snapshot.system_available_mb,
                'process_rss_mb': current_snapshot.process_rss_mb,
                'swap_used_percent': current_snapshot.swap_used_percent,
                'status': self._get_status_level(current_snapshot)
            },
            'limits': {
                'max_rss_mb': self.limits.max_rss_mb,
                'max_string_length': self.limits.max_string_length,
                'emergency_gc_threshold': self.limits.emergency_gc_threshold,
                'critical_shutdown_threshold': self.limits.critical_shutdown_threshold
            },
            'statistics': {
                'total_snapshots': total_snapshots,
                'total_alerts': len(self.alerts),
                'gc_collections': self.gc_collections,
                'emergency_actions': self.emergency_actions,
                'protected_operations': dict(self.protected_operations),
                'string_truncations': self.string_protector.truncation_count
            },
            'recent_alerts': recent_alerts,
            'string_protector_stats': self.string_protector.get_stats()
        }
    
    def _get_status_level(self, snapshot: MemorySnapshot) -> str:
        """Get status level for memory snapshot."""
        if snapshot.system_used_percent >= self.limits.critical_shutdown_threshold:
            return "emergency"
        elif snapshot.is_critical:
            return "critical"
        elif snapshot.is_warning:
            return "warning"
        else:
            return "normal"
    
    def generate_report(self) -> str:
        """Generate detailed memory guard report."""
        status = self.get_current_status()
        current = status['current_memory']
        stats = status['statistics']
        
        report_lines = [
            "=" * 80,
            "MEMORY GUARD REPORT",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "=" * 80,
            "",
            "CURRENT STATUS:",
            f"  System Memory: {current['system_used_percent']:.1f}% used, "
            f"{current['system_available_mb']:.0f}MB available",
            f"  Process Memory: {current['process_rss_mb']:.1f}MB RSS",
            f"  Swap Usage: {current['swap_used_percent']:.1f}%",
            f"  Status Level: {current['status'].upper()}",
            "",
            "STATISTICS:",
            f"  Monitoring Active: {'Yes' if status['monitoring_active'] else 'No'}",
            f"  Total Snapshots: {stats['total_snapshots']:,}",
            f"  Total Alerts: {stats['total_alerts']:,}",
            f"  GC Collections: {stats['gc_collections']:,}",
            f"  Emergency Actions: {stats['emergency_actions']:,}",
            f"  String Truncations: {stats['string_truncations']:,}",
            "",
            "PROTECTED OPERATIONS:",
        ]
        
        for op, count in stats['protected_operations'].items():
            report_lines.append(f"  {op}: {count:,} times")
        
        if not stats['protected_operations']:
            report_lines.append("  None")
        
        report_lines.extend([
            "",
            "RECENT ALERTS:",
        ])
        
        for alert in status['recent_alerts'][-5:]:
            report_lines.append(
                f"  {alert['timestamp']}: {alert['level'].upper()} - {alert['message']}"
            )
        
        if not status['recent_alerts']:
            report_lines.append("  None")
        
        report_lines.extend([
            "",
            "LIMITS:",
            f"  Max RSS: {status['limits']['max_rss_mb']:,.0f}MB",
            f"  Max String Length: {status['limits']['max_string_length']:,} chars",
            f"  Emergency GC Threshold: {status['limits']['emergency_gc_threshold']:.1f}%",
            f"  Critical Shutdown Threshold: {status['limits']['critical_shutdown_threshold']:.1f}%",
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)


# Global memory guard instance
_memory_guard: Optional[MemoryGuard] = None


def get_memory_guard() -> MemoryGuard:
    """Get or create global memory guard instance."""
    global _memory_guard
    if _memory_guard is None:
        _memory_guard = MemoryGuard()
        _memory_guard.start_monitoring()
    return _memory_guard


def protected_string_operation(data: str, operation: str = "unknown") -> str:
    """Convenience function for string protection."""
    guard = get_memory_guard()
    return guard.protect_string_operation(data, operation)


def protected_operation(operation_name: str, max_memory_mb: Optional[float] = None):
    """Decorator for memory-protected operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            guard = get_memory_guard()
            with guard.protected_operation(operation_name, max_memory_mb):
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo/test mode
    import random
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    
    console = Console()
    
    def create_memory_table(guard: MemoryGuard) -> Table:
        """Create memory status table."""
        status = guard.get_current_status()
        current = status['current_memory']
        stats = status['statistics']
        
        table = Table(title="Memory Guard Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Current status
        table.add_row("System Memory", f"{current['system_used_percent']:.1f}%")
        table.add_row("Available", f"{current['system_available_mb']:.0f}MB")
        table.add_row("Process RSS", f"{current['process_rss_mb']:.1f}MB")
        table.add_row("Swap Usage", f"{current['swap_used_percent']:.1f}%")
        table.add_row("Status", current['status'].upper())
        table.add_row("", "")
        
        # Statistics
        table.add_row("Total Snapshots", f"{stats['total_snapshots']:,}")
        table.add_row("Alerts", f"{stats['total_alerts']:,}")
        table.add_row("GC Collections", f"{stats['gc_collections']:,}")
        table.add_row("Emergency Actions", f"{stats['emergency_actions']:,}")
        table.add_row("String Truncations", f"{stats['string_truncations']:,}")
        
        return table
    
    # Create and start memory guard
    guard = MemoryGuard()
    guard.start_monitoring(interval_seconds=0.5)
    
    console.print("Memory Guard Demo - Press Ctrl+C to exit")
    console.print("Creating memory pressure to test protection...")
    
    try:
        # Create some memory pressure for testing
        test_data = []
        
        with Live(create_memory_table(guard), refresh_per_second=2) as live:
            for i in range(100):
                # Simulate memory usage
                if i % 10 == 0:
                    # Create large string periodically
                    large_string = "x" * random.randint(1000000, 5000000)
                    protected_string = guard.protect_string_operation(
                        large_string, f"test_operation_{i}"
                    )
                    test_data.append(protected_string[:1000])  # Keep only small portion
                
                # Update display
                live.update(create_memory_table(guard))
                time.sleep(0.5)
                
                # Clear some data periodically
                if len(test_data) > 50:
                    test_data = test_data[-25:]
    
    except KeyboardInterrupt:
        console.print("\nShutting down...")
    finally:
        guard.stop_monitoring()
        console.print("\nFinal Report:")
        console.print(guard.generate_report())