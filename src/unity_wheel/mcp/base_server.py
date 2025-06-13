#!/usr/bin/env python3
"""Base MCP server with health check endpoints and lifecycle management."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)



import asyncio
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import psutil
from mcp.server import FastMCP
import json
import tempfile

class HealthCheckMCP(FastMCP):
    """Enhanced MCP server with health checks and lifecycle management."""
    
    def __init__(self, name: str, workspace_root: Optional[str] = None):
        super().__init__(name)
        self.name = name
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.workspace_root = workspace_root or os.getcwd()
        self._health_file = self._get_health_file_path()
        self._setup_signal_handlers()
        self._register_health_endpoints()
        
    def _get_health_file_path(self) -> Path:
        """Get path for health check file."""
        runtime_dir = Path(self.workspace_root) / ".claude" / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return runtime_dir / f"{self.name}.health"
        
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def handle_shutdown(signum, frame):
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)
        
    def _register_health_endpoints(self):
        """Register health check tools."""
        
        @self.tool()
        def healthz() -> Dict[str, Any]:
            """Health check endpoint - returns server health status."""
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            # Get process info
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)
            
            health_status = {
                "status": "healthy" if self.error_count < 10 else "degraded",
                "server_name": self.name,
                "uptime_seconds": int(uptime_seconds),
                "uptime_human": self._format_uptime(uptime_seconds),
                "request_count": self.request_count,
                "error_count": self.error_count,
                "last_error": self.last_error,
                "memory_mb": round(memory_mb, 2),
                "cpu_percent": round(cpu_percent, 2),
                "pid": os.getpid(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Write health status to file
            self._write_health_status(health_status)
            
            return health_status
            
        @self.tool()
        async def quitquitquit() -> Dict[str, str]:
            """Graceful shutdown endpoint - cleanly stops the server."""
            result = {
                "status": "shutting_down",
                "message": f"Server {self.name} shutting down gracefully",
                "timestamp": datetime.now().isoformat()
            }
            
            # Schedule shutdown after response
            asyncio.create_task(self.shutdown())
            
            return result
            
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m {int(seconds % 60)}s"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
        else:
            days = int(seconds / 86400)
            hours = int((seconds % 86400) / 3600)
            return f"{days}d {hours}h"
            
    def _write_health_status(self, status: Dict[str, Any]):
        """Write health status to file for external monitoring."""
        try:
            # Write atomically using temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=self._health_file.parent,
                delete=False
            ) as tmp:
                json.dump(status, tmp, indent=2)
                tmp_path = tmp.name
                
            # Atomic rename
            Path(tmp_path).replace(self._health_file)
        except (ValueError, KeyError, AttributeError):
            # Ignore errors writing health file
            pass
            
    async def shutdown(self):
        """Graceful shutdown with cleanup."""
        logger.info("\n{self.name}: Shutting down gracefully...")
        
        # Remove health file
        try:
            self._health_file.unlink(missing_ok=True)
        except (ValueError, KeyError, AttributeError):
            pass
            
        # Give ongoing requests time to complete
        await asyncio.sleep(0.5)
        
        # Exit
        sys.exit(0)
        
    def track_request(self):
        """Increment request counter."""
        self.request_count += 1
        
    def track_error(self, error: str):
        """Track error for health monitoring."""
        self.error_count += 1
        self.last_error = error
        
    @classmethod
    def cleanup_stale_health_files(cls, workspace_root: str, max_age_seconds: int = 300):
        """Clean up health files from dead processes."""
        runtime_dir = Path(workspace_root) / ".claude" / "runtime"
        if not runtime_dir.exists():
            return
            
        now = datetime.now()
        for health_file in runtime_dir.glob("*.health"):
            try:
                # Read health file
                with open(health_file) as f:
                    data = json.load(f)
                    
                # Check if process is still running
                pid = data.get("pid")
                if pid and psutil.pid_exists(pid):
                    continue
                    
                # Check age
                timestamp = datetime.fromisoformat(data.get("timestamp", ""))
                age = (now - timestamp).total_seconds()
                
                if age > max_age_seconds:
                    health_file.unlink()
                    logger.info("Cleaned up stale health file: {health_file.name}")
                    
            except (ValueError, KeyError, AttributeError):
                # If we can't read it, remove it
                health_file.unlink(missing_ok=True)


# Example usage for converting existing MCP servers
if __name__ == "__main__":
    # Example server with health checks
    mcp = HealthCheckMCP("example-server")
    
    @mcp.tool()
    def example_tool(input: str) -> str:
        """Example tool that tracks requests."""
        mcp.track_request()
        try:
            # Tool logic here
            return f"Processed: {input}"
        except (ValueError, KeyError, AttributeError) as e:
            mcp.track_error(str(e))
            raise
    
    # Clean up stale files before starting
    HealthCheckMCP.cleanup_stale_health_files(os.getcwd())
    
    # Run server
    mcp.run()