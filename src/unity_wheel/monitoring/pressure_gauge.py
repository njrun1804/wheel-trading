#!/usr/bin/env python3
"""Adaptive memory pressure monitoring with chunk fan-out control."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)



import asyncio
import psutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Callable
from collections import deque
import json

@dataclass
class PressureReading:
    """Memory pressure reading."""
    timestamp: float
    memory_percent: float
    swap_percent: float
    available_gb: float
    rss_gb: float
    pressure_level: float  # 0.0 - 1.0
    
    @property
    def is_high(self) -> bool:
        return self.pressure_level > 0.6
    
    @property
    def is_critical(self) -> bool:
        return self.pressure_level > 0.8

class AdaptiveMemoryMonitor:
    """Real-time memory pressure monitoring with adaptive controls."""
    
    def __init__(self, 
                 sample_interval_ms: int = 250,
                 history_size: int = 240,  # 1 minute of history
                 pressure_callbacks: Optional[List[Callable]] = None):
        self.sample_interval = sample_interval_ms / 1000.0
        self.history = deque(maxlen=history_size)
        self.pressure_callbacks = pressure_callbacks or []
        self._running = False
        self._thread = None
        self._current_reading = None
        self._pressure_events = deque(maxlen=100)
        
        # Thresholds for 24GB system
        self.thresholds = {
            'memory_soft': 0.6,   # 60% = 14.4GB
            'memory_hard': 0.8,   # 80% = 19.2GB
            'swap_warning': 0.1,  # Any swap usage
            'available_min_gb': 4.0  # Keep 4GB free
        }
        
        # Fan-out control parameters
        self._base_fanout = 8
        self._current_fanout = 8
        self._fanout_history = deque(maxlen=20)
    
    def start(self):
        """Start monitoring in background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                reading = self._take_reading()
                self._current_reading = reading
                self.history.append(reading)
                
                # Update fan-out based on pressure
                self._update_fanout(reading)
                
                # Trigger callbacks if pressure changed significantly
                if len(self.history) > 1:
                    prev = self.history[-2]
                    if abs(reading.pressure_level - prev.pressure_level) > 0.1:
                        self._trigger_callbacks(reading)
                
                # Record pressure events
                if reading.is_high:
                    self._pressure_events.append({
                        'timestamp': reading.timestamp,
                        'level': reading.pressure_level,
                        'fanout_reduced_to': self._current_fanout
                    })
                
            except (ValueError, KeyError, AttributeError) as e:
                logger.info("Monitor error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _take_reading(self) -> PressureReading:
        """Take a memory pressure reading."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        process = psutil.Process()
        
        # Calculate composite pressure level
        memory_pressure = mem.percent / 100.0
        swap_pressure = min(swap.percent / 10.0, 1.0) if swap.percent > 0 else 0
        available_pressure = max(0, 1.0 - (mem.available / (self.thresholds['available_min_gb'] * 1024**3)))
        
        # Weighted average
        pressure_level = (memory_pressure * 0.5 + 
                         swap_pressure * 0.3 + 
                         available_pressure * 0.2)
        
        return PressureReading(
            timestamp=time.time(),
            memory_percent=mem.percent,
            swap_percent=swap.percent,
            available_gb=mem.available / (1024**3),
            rss_gb=process.memory_info().rss / (1024**3),
            pressure_level=min(pressure_level, 1.0)
        )
    
    def _update_fanout(self, reading: PressureReading):
        """Dynamically adjust chunk fan-out based on pressure."""
        if reading.pressure_level < 0.4:
            # Low pressure - increase fanout
            self._current_fanout = min(self._base_fanout * 2, 16)
        elif reading.pressure_level < 0.6:
            # Normal pressure
            self._current_fanout = self._base_fanout
        elif reading.pressure_level < 0.8:
            # High pressure - reduce fanout
            self._current_fanout = max(self._base_fanout // 2, 2)
        else:
            # Critical pressure - minimal fanout
            self._current_fanout = 1
        
        self._fanout_history.append(self._current_fanout)
    
    def _trigger_callbacks(self, reading: PressureReading):
        """Trigger registered callbacks."""
        for callback in self.pressure_callbacks:
            try:
                callback(reading)
            except (ValueError, KeyError, AttributeError) as e:
                logger.info("Callback error: {e}")
    
    @property
    def current_pressure(self) -> float:
        """Get current pressure level (0.0 - 1.0)."""
        if self._current_reading:
            return self._current_reading.pressure_level
        return 0.0
    
    @property
    def current_fanout(self) -> int:
        """Get current recommended chunk fan-out."""
        return self._current_fanout
    
    def get_adaptive_config(self) -> dict:
        """Get current adaptive configuration based on pressure."""
        pressure = self.current_pressure
        
        return {
            'chunk_fanout': self._current_fanout,
            'parallel_workers': max(1, int((1 - pressure) * psutil.cpu_count())),
            'cache_aggressive': pressure < 0.5,
            'enable_embeddings': pressure < 0.7,
            'duckdb_memory_limit': f"{int((1 - pressure) * 8)}GB",
            'pressure_level': pressure,
            'throttle_active': pressure > 0.6
        }
    
    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        if not self.history:
            return {}
        
        pressures = [r.pressure_level for r in self.history]
        fanouts = list(self._fanout_history)
        
        return {
            'current_pressure': self.current_pressure,
            'avg_pressure_1min': sum(pressures) / len(pressures),
            'max_pressure_1min': max(pressures),
            'current_fanout': self._current_fanout,
            'avg_fanout': sum(fanouts) / len(fanouts) if fanouts else self._base_fanout,
            'pressure_events': len(self._pressure_events),
            'throttle_events': sum(1 for e in self._pressure_events if e['fanout_reduced_to'] < self._base_fanout)
        }

# Global instance
_monitor = None

def get_pressure_monitor() -> AdaptiveMemoryMonitor:
    """Get or create global pressure monitor."""
    global _monitor
    if _monitor is None:
        _monitor = AdaptiveMemoryMonitor()
        _monitor.start()
    return _monitor

# MCP-compatible channel for reading pressure
class PressureChannel:
    """MCP-compatible pressure gauge channel."""
    
    def __init__(self):
        self.monitor = get_pressure_monitor()
    
    async def read(self) -> dict:
        """Read current pressure state."""
        return {
            'timestamp': time.time(),
            'pressure': self.monitor.current_pressure,
            'config': self.monitor.get_adaptive_config(),
            'stats': self.monitor.get_stats()
        }
    
    async def subscribe(self, callback: Callable):
        """Subscribe to pressure changes."""
        self.monitor.pressure_callbacks.append(callback)

# Export for MCP
async def create_pressure_channel():
    """Create MCP pressure channel at /sys/claude/pressure-gauge."""
    return PressureChannel()


def check_memory_pressure() -> float:
    """Check current memory pressure (0.0 to 1.0)."""
    monitor = get_pressure_monitor()
    return monitor.current_pressure

if __name__ == "__main__":
    # Test/demo mode
    import rich
    from rich.live import Live
    from rich.table import Table
    from rich.console import Console
    
    console = Console()
    
    def create_display(monitor: AdaptiveMemoryMonitor) -> Table:
        """Create display table."""
        table = Table(title="Memory Pressure Monitor")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        config = monitor.get_adaptive_config()
        stats = monitor.get_stats()
        
        table.add_row("Pressure Level", f"{config['pressure_level']:.2%}")
        table.add_row("Chunk Fan-out", str(config['chunk_fanout']))
        table.add_row("Parallel Workers", str(config['parallel_workers']))
        table.add_row("DuckDB Memory", config['duckdb_memory_limit'])
        table.add_row("Throttle Active", "Yes" if config['throttle_active'] else "No")
        table.add_row("", "")
        table.add_row("Avg Pressure (1m)", f"{stats.get('avg_pressure_1min', 0):.2%}")
        table.add_row("Pressure Events", str(stats.get('pressure_events', 0)))
        table.add_row("Throttle Events", str(stats.get('throttle_events', 0)))
        
        return table
    
    monitor = AdaptiveMemoryMonitor(sample_interval_ms=500)
    monitor.start()
    
    try:
        with Live(create_display(monitor), refresh_per_second=2) as live:
            while True:
                time.sleep(0.5)
                live.update(create_display(monitor))
    except KeyboardInterrupt:
        monitor.stop()