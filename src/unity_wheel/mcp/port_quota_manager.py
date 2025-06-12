#!/usr/bin/env python3
"""Port-aware quota manager to prevent file descriptor exhaustion."""

import os
import socket
import threading
import resource
from typing import Dict, Optional, Set, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import time
import psutil

@dataclass
class PortAllocation:
    """Track port allocation."""
    port: int
    service: str
    pid: int
    allocated_at: float
    fd_count: int

class PortQuotaManager:
    """Manages port allocations with file descriptor awareness."""
    
    def __init__(self):
        # Get system limits
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        self.fd_limit = soft_limit
        self.fd_warning_threshold = int(self.fd_limit * 0.8)
        self.fd_critical_threshold = int(self.fd_limit * 0.95)
        
        # Port tracking
        self._allocations: Dict[int, PortAllocation] = {}
        self._service_ports: Dict[str, Set[int]] = {}
        self._lock = threading.Lock()
        
        # FD tracking
        self._fd_count = 0
        self._update_fd_count()
        
        # Queue for pending allocations
        self._pending_queue = []
        self._allocation_event = threading.Event()
        
        # Start monitoring thread
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def _update_fd_count(self):
        """Update current file descriptor count."""
        try:
            # Count open FDs for current process
            proc = psutil.Process()
            self._fd_count = proc.num_fds() if hasattr(proc, 'num_fds') else len(proc.open_files())
        except:
            # Fallback to /proc/self/fd on Linux/Mac
            try:
                self._fd_count = len(os.listdir('/proc/self/fd'))
            except:
                self._fd_count = 100  # Conservative estimate
    
    def _monitor_loop(self):
        """Monitor FD usage and process pending allocations."""
        while self._running:
            self._update_fd_count()
            
            # Process pending allocations if FDs available
            if self._fd_count < self.fd_warning_threshold and self._pending_queue:
                with self._lock:
                    if self._pending_queue:
                        self._allocation_event.set()
            
            time.sleep(1)
    
    @contextmanager
    def allocate_port(self, service: str, preferred_port: Optional[int] = None):
        """Allocate a port with FD quota checking."""
        port = None
        allocated = False
        
        try:
            # Check FD availability
            if self._fd_count >= self.fd_critical_threshold:
                # Queue the request
                with self._lock:
                    queue_entry = {
                        'service': service,
                        'preferred_port': preferred_port,
                        'event': threading.Event()
                    }
                    self._pending_queue.append(queue_entry)
                
                # Wait for FDs to become available
                queue_entry['event'].wait(timeout=30)
                
                if self._fd_count >= self.fd_critical_threshold:
                    raise RuntimeError(
                        f"File descriptor limit reached: {self._fd_count}/{self.fd_limit}"
                    )
            
            # Find available port
            with self._lock:
                if preferred_port and preferred_port not in self._allocations:
                    port = preferred_port
                else:
                    # Find next available port
                    port = self._find_available_port()
                
                if port is None:
                    raise RuntimeError("No available ports")
                
                # Allocate the port
                allocation = PortAllocation(
                    port=port,
                    service=service,
                    pid=os.getpid(),
                    allocated_at=time.time(),
                    fd_count=self._fd_count
                )
                
                self._allocations[port] = allocation
                if service not in self._service_ports:
                    self._service_ports[service] = set()
                self._service_ports[service].add(port)
                allocated = True
            
            yield port
            
        finally:
            # Release the port
            if allocated and port:
                with self._lock:
                    if port in self._allocations:
                        del self._allocations[port]
                    if service in self._service_ports:
                        self._service_ports[service].discard(port)
    
    def _find_available_port(self, start=15000, end=16000) -> Optional[int]:
        """Find an available port in range."""
        for port in range(start, end):
            if port not in self._allocations and self._is_port_free(port):
                return port
        return None
    
    def _is_port_free(self, port: int) -> bool:
        """Check if a port is actually free."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return True
        except:
            return False
    
    def get_status(self) -> dict:
        """Get current quota status."""
        with self._lock:
            return {
                'fd_count': self._fd_count,
                'fd_limit': self.fd_limit,
                'fd_usage_percent': (self._fd_count / self.fd_limit) * 100,
                'warning_threshold': self.fd_warning_threshold,
                'critical_threshold': self.fd_critical_threshold,
                'allocated_ports': len(self._allocations),
                'pending_requests': len(self._pending_queue),
                'service_summary': {
                    service: len(ports) 
                    for service, ports in self._service_ports.items()
                }
            }
    
    def get_metrics(self) -> dict:
        """Get metrics for monitoring."""
        status = self.get_status()
        return {
            'claude_fd_usage': self._fd_count,
            'claude_fd_limit': self.fd_limit,
            'claude_fd_usage_ratio': self._fd_count / self.fd_limit,
            'claude_port_allocations': len(self._allocations),
            'claude_port_queue_length': len(self._pending_queue),
            'claude_fd_warning': self._fd_count >= self.fd_warning_threshold,
            'claude_fd_critical': self._fd_count >= self.fd_critical_threshold
        }
    
    def cleanup_stale_allocations(self):
        """Clean up allocations from dead processes."""
        with self._lock:
            current_pids = {p.pid for p in psutil.process_iter(['pid'])}
            stale_ports = [
                port for port, alloc in self._allocations.items()
                if alloc.pid not in current_pids
            ]
            
            for port in stale_ports:
                alloc = self._allocations[port]
                del self._allocations[port]
                if alloc.service in self._service_ports:
                    self._service_ports[alloc.service].discard(port)
            
            return len(stale_ports)

# Global instance
_quota_manager = None

def get_quota_manager() -> PortQuotaManager:
    """Get or create global quota manager."""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = PortQuotaManager()
    return _quota_manager

# Custom socket wrapper with quota checking
class QuotaAwareSocket(socket.socket):
    """Socket that checks quota before binding."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._quota_mgr = get_quota_manager()
        self._allocated_port = None
    
    def bind(self, address):
        """Bind with quota checking."""
        if isinstance(address, tuple) and len(address) >= 2:
            host, port = address[0], address[1]
            
            # Check quota before binding
            status = self._quota_mgr.get_status()
            if status['fd_usage_percent'] > 95:
                raise OSError(
                    f"File descriptor quota exceeded: {status['fd_count']}/{status['fd_limit']}"
                )
        
        super().bind(address)

# Monkey-patch socket for quota awareness (optional)
def enable_quota_enforcement():
    """Enable system-wide quota enforcement."""
    import socket as sock_module
    sock_module.socket = QuotaAwareSocket

if __name__ == "__main__":
    # Demo/test
    import asyncio
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    manager = get_quota_manager()
    
    async def test_allocation():
        """Test port allocation."""
        console.print("[green]Testing port allocation with quota management...[/green]")
        
        # Allocate some ports
        allocations = []
        for i in range(5):
            try:
                with manager.allocate_port(f"test-service-{i}") as port:
                    console.print(f"✓ Allocated port {port} for test-service-{i}")
                    allocations.append(port)
                    await asyncio.sleep(0.5)
            except Exception as e:
                console.print(f"[red]✗ Failed to allocate port: {e}[/red]")
        
        # Show status
        status = manager.get_status()
        table = Table(title="Port Quota Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("FD Usage", f"{status['fd_count']}/{status['fd_limit']}")
        table.add_row("FD Usage %", f"{status['fd_usage_percent']:.1f}%")
        table.add_row("Allocated Ports", str(status['allocated_ports']))
        table.add_row("Pending Requests", str(status['pending_requests']))
        
        console.print(table)
        
        # Cleanup test
        cleaned = manager.cleanup_stale_allocations()
        console.print(f"\nCleaned up {cleaned} stale allocations")
    
    asyncio.run(test_allocation())