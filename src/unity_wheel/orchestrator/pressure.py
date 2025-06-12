"""Memory Pressure Monitor - Tracks RSS/total ratio to prevent OOM."""

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psutil


@dataclass
class MemorySnapshot:
    """Point-in-time memory measurement."""
    timestamp: float
    rss_mb: float
    total_mb: float
    ratio: float
    swap_mb: float
    available_mb: float


class MemoryPressureMonitor:
    """Background monitor for system memory pressure."""
    
    def __init__(self, 
                 threshold_ratio: float = 0.70,
                 sample_interval_ms: int = 250,
                 history_size: int = 240):  # 1 minute of history at 250ms
        
        self.threshold_ratio = threshold_ratio
        self.sample_interval_ms = sample_interval_ms
        self.history_size = history_size
        
        # Memory tracking
        self.history: deque[MemorySnapshot] = deque(maxlen=history_size)
        self.peak_memory_mb: float = 0.0
        self.peak_ratio: float = 0.0
        
        # Pressure tracking
        self.pressure_high: bool = False
        self.pressure_events: list[dict[str, Any]] = []
        
        # Background thread
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        
        # Callbacks
        self._pressure_callbacks: list[Callable[[MemorySnapshot], None]] = []
        
        # System info
        self.total_system_mb = psutil.virtual_memory().total / (1024 * 1024)
        
    def start(self):
        """Start background monitoring."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                snapshot = self._take_snapshot()
                
                with self._lock:
                    self.history.append(snapshot)
                    
                    # Update peaks
                    if snapshot.rss_mb > self.peak_memory_mb:
                        self.peak_memory_mb = snapshot.rss_mb
                    if snapshot.ratio > self.peak_ratio:
                        self.peak_ratio = snapshot.ratio
                        
                    # Check pressure
                    was_high = self.pressure_high
                    self.pressure_high = snapshot.ratio >= self.threshold_ratio
                    
                    # Record pressure events
                    if self.pressure_high and not was_high:
                        self.pressure_events.append({
                            "type": "entered_high_pressure",
                            "timestamp": datetime.now().isoformat(),
                            "ratio": snapshot.ratio,
                            "rss_mb": snapshot.rss_mb
                        })
                    elif not self.pressure_high and was_high:
                        self.pressure_events.append({
                            "type": "exited_high_pressure",
                            "timestamp": datetime.now().isoformat(),
                            "ratio": snapshot.ratio,
                            "rss_mb": snapshot.rss_mb
                        })
                        
                # Notify callbacks
                for callback in self._pressure_callbacks:
                    with contextlib.suppress(Exception):
                        callback(snapshot)
                        
            except Exception as e:
                print(f"Memory monitor error: {e}")
                
            # Sleep until next sample
            time.sleep(self.sample_interval_ms / 1000.0)
            
    def _take_snapshot(self) -> MemorySnapshot:
        """Take memory snapshot."""
        # Process memory
        process = psutil.Process()
        process_info = process.memory_info()
        rss_mb = process_info.rss / (1024 * 1024)
        
        # System memory
        vm = psutil.virtual_memory()
        total_mb = vm.total / (1024 * 1024)
        available_mb = vm.available / (1024 * 1024)
        
        # Swap
        swap = psutil.swap_memory()
        swap_mb = swap.used / (1024 * 1024)
        
        # Calculate ratio
        ratio = rss_mb / total_mb
        
        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss_mb,
            total_mb=total_mb,
            ratio=ratio,
            swap_mb=swap_mb,
            available_mb=available_mb
        )
        
    def is_pressure_high(self) -> bool:
        """Check if memory pressure is currently high."""
        with self._lock:
            return self.pressure_high
            
    def get_current_ratio(self) -> float:
        """Get current memory ratio."""
        with self._lock:
            if self.history:
                return self.history[-1].ratio
        return 0.0
        
    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            if not self.history:
                return {
                    "current_ratio": 0.0,
                    "current_rss_mb": 0.0,
                    "peak_ratio": 0.0,
                    "peak_memory_mb": 0.0,
                    "pressure_high": False,
                    "pressure_events": 0
                }
                
            current = self.history[-1]
            
            # Calculate averages
            recent_snapshots = list(self.history)[-20:]  # Last 5 seconds
            avg_ratio = sum(s.ratio for s in recent_snapshots) / len(recent_snapshots)
            avg_rss = sum(s.rss_mb for s in recent_snapshots) / len(recent_snapshots)
            
            return {
                "current_ratio": round(current.ratio, 3),
                "current_rss_mb": round(current.rss_mb, 1),
                "average_ratio_5s": round(avg_ratio, 3),
                "average_rss_mb_5s": round(avg_rss, 1),
                "peak_ratio": round(self.peak_ratio, 3),
                "peak_memory_mb": round(self.peak_memory_mb, 1),
                "total_system_mb": round(self.total_system_mb, 1),
                "available_mb": round(current.available_mb, 1),
                "swap_mb": round(current.swap_mb, 1),
                "pressure_high": self.pressure_high,
                "pressure_events": len(self.pressure_events),
                "threshold_ratio": self.threshold_ratio
            }
            
    def add_pressure_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Add callback for pressure changes."""
        self._pressure_callbacks.append(callback)
        
    def wait_for_low_pressure(self, timeout: float = 30.0) -> bool:
        """Block until pressure is low or timeout."""
        start = time.time()
        
        while time.time() - start < timeout:
            if not self.is_pressure_high():
                return True
            time.sleep(0.1)
            
        return False
        
    def get_pressure_history(self, seconds: int = 60) -> list[dict[str, Any]]:
        """Get pressure history for specified duration."""
        with self._lock:
            if not self.history:
                return []
                
            cutoff = time.time() - seconds
            return [
                {
                    "timestamp": s.timestamp,
                    "ratio": round(s.ratio, 3),
                    "rss_mb": round(s.rss_mb, 1),
                    "high_pressure": s.ratio >= self.threshold_ratio
                }
                for s in self.history
                if s.timestamp >= cutoff
            ]
            
    def suggest_gc(self) -> bool:
        """Suggest if garbage collection might help."""
        with self._lock:
            if not self.history or len(self.history) < 10:
                return False
                
            # Look for rapid growth
            recent = list(self.history)[-10:]
            growth_rate = (recent[-1].rss_mb - recent[0].rss_mb) / len(recent)
            
            # Suggest GC if growing rapidly and near threshold
            return (growth_rate > 10 and  # Growing >10MB per sample
                   recent[-1].ratio > self.threshold_ratio * 0.9)  # Near threshold