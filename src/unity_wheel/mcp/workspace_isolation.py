"""
Workspace isolation for MCP runtime.
Ensures each VS Code window has its own sandbox.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, List
import hashlib
import psutil
import socket

class WorkspaceIsolation:
    """Manages isolated runtime environments for each workspace."""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = self._generate_workspace_id()
        self.runtime_dir = self._get_runtime_directory()
        self._initialize_runtime()
    
    def _generate_workspace_id(self) -> str:
        """Generate unique ID for this workspace."""
        # Use workspace path hash for consistency
        path_hash = hashlib.md5(str(self.workspace_root).encode()).hexdigest()[:8]
        return f"ws_{path_hash}"
    
    def _get_runtime_directory(self) -> Path:
        """Get isolated runtime directory for this workspace."""
        base_dir = self.workspace_root / ".claude" / "runtime"
        return base_dir / self.workspace_id
    
    def _initialize_runtime(self):
        """Initialize the runtime directory structure."""
        # Create directory structure
        dirs = [
            self.runtime_dir,
            self.runtime_dir / "sockets",
            self.runtime_dir / "locks",
            self.runtime_dir / "logs",
            self.runtime_dir / "cache",
            self.runtime_dir / "temp",
            self.runtime_dir / "state"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create workspace metadata
        metadata = {
            "workspace_id": self.workspace_id,
            "workspace_root": str(self.workspace_root),
            "created_at": os.path.getctime(self.runtime_dir),
            "pid": os.getpid(),
            "python_version": os.sys.version,
            "isolation_version": "1.0.0"
        }
        
        with open(self.runtime_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_socket_path(self, service_name: str) -> str:
        """Get isolated socket path for a service."""
        socket_dir = self.runtime_dir / "sockets"
        return str(socket_dir / f"{service_name}.sock")
    
    def get_lock_path(self, lock_name: str) -> str:
        """Get isolated lock file path."""
        lock_dir = self.runtime_dir / "locks"
        return str(lock_dir / f"{lock_name}.lock")
    
    def get_log_path(self, service_name: str) -> str:
        """Get isolated log file path."""
        log_dir = self.runtime_dir / "logs"
        return str(log_dir / f"{service_name}.log")
    
    def get_cache_dir(self, cache_name: str) -> Path:
        """Get isolated cache directory."""
        cache_dir = self.runtime_dir / "cache" / cache_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_temp_dir(self) -> Path:
        """Get isolated temp directory."""
        return self.runtime_dir / "temp"
    
    def get_state_file(self, state_name: str) -> Path:
        """Get isolated state file path."""
        return self.runtime_dir / "state" / f"{state_name}.json"
    
    def find_free_port(self, start_port: int = 50000, max_port: int = 60000) -> int:
        """Find a free port in the isolated range."""
        # Use workspace-specific port range to avoid conflicts
        port_offset = int(self.workspace_id[-4:], 16) % 1000
        start_port += port_offset
        
        for port in range(start_port, min(start_port + 1000, max_port)):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('127.0.0.1', port))
                sock.close()
                return port
            except OSError:
                continue
        
        raise RuntimeError(f"No free ports found in range {start_port}-{max_port}")
    
    def cleanup_stale_processes(self):
        """Clean up stale processes from previous sessions."""
        state_dir = self.runtime_dir / "state"
        if not state_dir.exists():
            return
        
        # Check PID files
        for pid_file in state_dir.glob("*.pid"):
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                
                # Check if process is still running
                if not psutil.pid_exists(pid):
                    # Remove stale PID file
                    pid_file.unlink()
                    
                    # Clean up associated resources
                    service_name = pid_file.stem
                    socket_path = self.get_socket_path(service_name)
                    if os.path.exists(socket_path):
                        os.unlink(socket_path)
            except Exception:
                # Clean up corrupt PID file
                pid_file.unlink()
    
    def register_service(self, service_name: str, pid: int, port: Optional[int] = None):
        """Register a service in this workspace."""
        state_file = self.runtime_dir / "state" / f"{service_name}.json"
        
        service_info = {
            "name": service_name,
            "pid": pid,
            "port": port,
            "socket": self.get_socket_path(service_name) if not port else None,
            "started_at": os.path.getmtime(self.runtime_dir),
            "workspace_id": self.workspace_id
        }
        
        with open(state_file, 'w') as f:
            json.dump(service_info, f, indent=2)
        
        # Also write PID file for compatibility
        pid_file = self.runtime_dir / "state" / f"{service_name}.pid"
        with open(pid_file, 'w') as f:
            f.write(str(pid))
    
    def list_services(self) -> List[Dict]:
        """List all services in this workspace."""
        services = []
        state_dir = self.runtime_dir / "state"
        
        if state_dir.exists():
            for state_file in state_dir.glob("*.json"):
                if state_file.stem == "metadata":
                    continue
                
                try:
                    with open(state_file) as f:
                        service_info = json.load(f)
                    
                    # Check if still running
                    if psutil.pid_exists(service_info['pid']):
                        service_info['status'] = 'running'
                    else:
                        service_info['status'] = 'stopped'
                    
                    services.append(service_info)
                except Exception:
                    pass
        
        return services
    
    def get_workspace_stats(self) -> Dict:
        """Get statistics about this workspace."""
        stats = {
            "workspace_id": self.workspace_id,
            "workspace_root": str(self.workspace_root),
            "runtime_dir": str(self.runtime_dir),
            "services": len(self.list_services()),
            "disk_usage_mb": 0,
            "cache_entries": 0
        }
        
        # Calculate disk usage
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.runtime_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            stats['disk_usage_mb'] = total_size / (1024 * 1024)
        except Exception:
            pass
        
        # Count cache entries
        cache_dir = self.runtime_dir / "cache"
        if cache_dir.exists():
            stats['cache_entries'] = sum(1 for _ in cache_dir.rglob("*") if _.is_file())
        
        return stats
    
    def cleanup(self, force: bool = False):
        """Clean up workspace runtime directory."""
        if not self.runtime_dir.exists():
            return
        
        # Check for running services
        services = self.list_services()
        running = [s for s in services if s['status'] == 'running']
        
        if running and not force:
            raise RuntimeError(
                f"Cannot cleanup: {len(running)} services still running. "
                "Stop services first or use force=True"
            )
        
        # Stop running services if forced
        if running and force:
            for service in running:
                try:
                    psutil.Process(service['pid']).terminate()
                except Exception:
                    pass
        
        # Remove runtime directory
        shutil.rmtree(self.runtime_dir, ignore_errors=True)


class WorkspaceManager:
    """Manages multiple workspace isolations."""
    
    @staticmethod
    def list_workspaces() -> List[Dict]:
        """List all active workspaces on this machine."""
        workspaces = []
        
        # Search common locations
        search_paths = [
            Path.home() / "Documents",
            Path.home() / "Projects",
            Path.home() / "Development",
            Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike")
        ]
        
        for base_path in search_paths:
            if base_path.exists():
                # Look for .claude/runtime directories
                for claude_dir in base_path.rglob(".claude/runtime/ws_*"):
                    if claude_dir.is_dir():
                        try:
                            metadata_file = claude_dir / "metadata.json"
                            if metadata_file.exists():
                                with open(metadata_file) as f:
                                    metadata = json.load(f)
                                
                                # Get workspace isolation
                                ws = WorkspaceIsolation(metadata['workspace_root'])
                                stats = ws.get_workspace_stats()
                                workspaces.append(stats)
                        except Exception:
                            pass
        
        return workspaces
    
    @staticmethod
    def cleanup_all_stale(dry_run: bool = True) -> List[str]:
        """Clean up stale workspaces across the system."""
        cleaned = []
        
        workspaces = WorkspaceManager.list_workspaces()
        for ws_stats in workspaces:
            try:
                ws = WorkspaceIsolation(ws_stats['workspace_root'])
                services = ws.list_services()
                
                # If no services are running, consider for cleanup
                if not any(s['status'] == 'running' for s in services):
                    if not dry_run:
                        ws.cleanup(force=True)
                    cleaned.append(ws_stats['workspace_id'])
            except Exception:
                pass
        
        return cleaned


# Environment variable helpers
def setup_isolated_environment(workspace_root: str) -> Dict[str, str]:
    """Set up environment variables for isolated workspace."""
    isolation = WorkspaceIsolation(workspace_root)
    
    env = {
        "CLAUDE_WORKSPACE_ID": isolation.workspace_id,
        "CLAUDE_RUNTIME_DIR": str(isolation.runtime_dir),
        "CLAUDE_SOCKET_DIR": str(isolation.runtime_dir / "sockets"),
        "CLAUDE_LOCK_DIR": str(isolation.runtime_dir / "locks"),
        "CLAUDE_LOG_DIR": str(isolation.runtime_dir / "logs"),
        "CLAUDE_CACHE_DIR": str(isolation.runtime_dir / "cache"),
        "CLAUDE_TEMP_DIR": str(isolation.runtime_dir / "temp"),
    }
    
    # Update standard temp directories
    env["TMPDIR"] = str(isolation.get_temp_dir())
    env["TEMP"] = str(isolation.get_temp_dir())
    env["TMP"] = str(isolation.get_temp_dir())
    
    return env