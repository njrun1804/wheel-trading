"""Backup and disaster recovery system for wheel trading platform.

Provides automated backup of:
- Trading databases
- Configuration files  
- Performance metrics
- Alert history
- System state
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

from unity_wheel.utils import get_logger

logger = get_logger(__name__)


class BackupSystem:
    """Automated backup and disaster recovery system."""

    def __init__(
        self,
        backup_dir: Path = Path.home() / ".wheel_trading" / "backups",
        retention_days: int = 30,
        backup_interval_hours: int = 6
    ):
        """Initialize backup system."""
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.backup_interval_hours = backup_interval_hours
        
        # Backup state
        self.is_running = False
        self.backup_thread: Optional[threading.Thread] = None
        self.last_backup_time: Optional[datetime] = None
        self.backup_history: List[Dict[str, Any]] = []
        
        # Critical paths to backup
        self.critical_paths = [
            Path.home() / ".wheel_trading" / "cache",
            Path("data"),
            Path("config.yaml"),
            Path("logging_config.json"),
            Path("logs"),
        ]

    def start_automated_backups(self) -> None:
        """Start automated backup process."""
        if self.is_running:
            logger.warning("Backup system already running")
            return

        self.is_running = True
        self.backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
        self.backup_thread.start()
        
        logger.info(
            "Automated backup system started",
            extra={
                "backup_interval_hours": self.backup_interval_hours,
                "retention_days": self.retention_days,
                "backup_dir": str(self.backup_dir)
            }
        )

    def stop_automated_backups(self) -> None:
        """Stop automated backup process."""
        if not self.is_running:
            return

        self.is_running = False
        if self.backup_thread:
            self.backup_thread.join(timeout=10)
        
        logger.info("Automated backup system stopped")

    def _backup_loop(self) -> None:
        """Main backup loop."""
        while self.is_running:
            try:
                current_time = datetime.now(UTC)
                
                # Check if backup is needed
                if (self.last_backup_time is None or 
                    current_time - self.last_backup_time > timedelta(hours=self.backup_interval_hours)):
                    
                    logger.info("Starting scheduled backup")
                    result = self.create_full_backup()
                    
                    if result["success"]:
                        self.last_backup_time = current_time
                        self.backup_history.append(result)
                        logger.info(f"Scheduled backup completed: {result['backup_file']}")
                    else:
                        logger.error(f"Scheduled backup failed: {result['error']}")
                
                # Cleanup old backups
                self._cleanup_old_backups()
                
            except Exception as e:
                logger.error(f"Backup loop error: {e}", exc_info=True)
                
            # Sleep for 1 hour between checks
            time.sleep(3600)

    def create_full_backup(self) -> Dict[str, Any]:
        """Create a full system backup."""
        backup_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"wheel_trading_backup_{backup_id}.tar.gz"
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                backup_manifest = {
                    "backup_id": backup_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "version": "1.0",
                    "components": []
                }
                
                # Backup databases
                db_backup_result = self._backup_databases(temp_path / "databases")
                backup_manifest["components"].append(db_backup_result)
                
                # Backup configuration files
                config_backup_result = self._backup_configurations(temp_path / "config")
                backup_manifest["components"].append(config_backup_result)
                
                # Backup logs (recent only)
                logs_backup_result = self._backup_logs(temp_path / "logs")
                backup_manifest["components"].append(logs_backup_result)
                
                # Backup system state
                state_backup_result = self._backup_system_state(temp_path / "state")
                backup_manifest["components"].append(state_backup_result)
                
                # Create manifest file
                manifest_file = temp_path / "backup_manifest.json"
                with open(manifest_file, "w") as f:
                    json.dump(backup_manifest, f, indent=2)
                
                # Create compressed archive
                with tarfile.open(backup_file, "w:gz") as tar:
                    tar.add(temp_path, arcname=".")
                
                # Calculate checksum
                checksum = self._calculate_file_checksum(backup_file)
                
                # Update manifest with final details
                backup_manifest.update({
                    "backup_file": str(backup_file),
                    "file_size_bytes": backup_file.stat().st_size,
                    "checksum_sha256": checksum,
                    "success": True
                })
                
                logger.info(
                    "Full backup completed successfully",
                    extra={
                        "backup_id": backup_id,
                        "backup_file": str(backup_file),
                        "file_size_mb": backup_file.stat().st_size / (1024 * 1024),
                        "checksum": checksum[:16]
                    }
                )
                
                return backup_manifest
                
        except Exception as e:
            logger.error(f"Backup creation failed: {e}", exc_info=True)
            return {
                "backup_id": backup_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "success": False,
                "error": str(e)
            }

    def _backup_databases(self, backup_path: Path) -> Dict[str, Any]:
        """Backup all database files."""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            db_files = []
            
            # Find all DuckDB files
            cache_dir = Path.home() / ".wheel_trading" / "cache"
            if cache_dir.exists():
                for db_file in cache_dir.glob("*.db"):
                    # Copy database file
                    dest_file = backup_path / db_file.name
                    shutil.copy2(db_file, dest_file)
                    db_files.append({
                        "file": db_file.name,
                        "size_bytes": dest_file.stat().st_size,
                        "checksum": self._calculate_file_checksum(dest_file)[:16]
                    })
            
            # Also backup any data directory databases
            data_dir = Path("data")
            if data_dir.exists():
                for db_file in data_dir.glob("*.db"):
                    dest_file = backup_path / f"data_{db_file.name}"
                    shutil.copy2(db_file, dest_file)
                    db_files.append({
                        "file": f"data_{db_file.name}",
                        "size_bytes": dest_file.stat().st_size,
                        "checksum": self._calculate_file_checksum(dest_file)[:16]
                    })
            
            return {
                "component": "databases",
                "success": True,
                "files_backed_up": len(db_files),
                "files": db_files
            }
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return {
                "component": "databases",
                "success": False,
                "error": str(e)
            }

    def _backup_configurations(self, backup_path: Path) -> Dict[str, Any]:
        """Backup configuration files."""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            config_files = []
            
            # List of important config files
            important_configs = [
                "config.yaml",
                "logging_config.json",
                "pyproject.toml",
                "requirements.txt",
                ".env.example"
            ]
            
            for config_name in important_configs:
                config_file = Path(config_name)
                if config_file.exists():
                    dest_file = backup_path / config_name
                    shutil.copy2(config_file, dest_file)
                    config_files.append({
                        "file": config_name,
                        "size_bytes": dest_file.stat().st_size
                    })
            
            # Backup config directory if it exists
            config_dir = Path("config")
            if config_dir.exists():
                dest_config_dir = backup_path / "config"
                shutil.copytree(config_dir, dest_config_dir)
                
                # Count files in config directory
                config_count = len(list(dest_config_dir.rglob("*")))
                config_files.append({
                    "directory": "config",
                    "files_count": config_count
                })
            
            return {
                "component": "configurations",
                "success": True,
                "files_backed_up": len(config_files),
                "files": config_files
            }
            
        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            return {
                "component": "configurations",
                "success": False,
                "error": str(e)
            }

    def _backup_logs(self, backup_path: Path) -> Dict[str, Any]:
        """Backup recent log files."""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            log_files = []
            logs_dir = Path("logs")
            
            if logs_dir.exists():
                # Only backup recent logs (last 7 days)
                cutoff_time = datetime.now() - timedelta(days=7)
                
                for log_file in logs_dir.glob("*.log*"):
                    # Check file modification time
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if mtime > cutoff_time:
                        dest_file = backup_path / log_file.name
                        shutil.copy2(log_file, dest_file)
                        log_files.append({
                            "file": log_file.name,
                            "size_bytes": dest_file.stat().st_size,
                            "modified_date": mtime.isoformat()
                        })
            
            return {
                "component": "logs",
                "success": True,
                "files_backed_up": len(log_files),
                "files": log_files
            }
            
        except Exception as e:
            logger.error(f"Logs backup failed: {e}")
            return {
                "component": "logs",
                "success": False,
                "error": str(e)
            }

    def _backup_system_state(self, backup_path: Path) -> Dict[str, Any]:
        """Backup current system state and metrics."""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get current system state
            import psutil
            
            system_state = {
                "timestamp": datetime.now(UTC).isoformat(),
                "system_info": {
                    "platform": psutil.Process().name(),
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "disk_total_gb": psutil.disk_usage('/').total / (1024**3)
                },
                "process_info": {
                    "pid": psutil.Process().pid,
                    "memory_percent": psutil.Process().memory_percent(),
                    "cpu_percent": psutil.Process().cpu_percent()
                }
            }
            
            # Save system state
            state_file = backup_path / "system_state.json"
            with open(state_file, "w") as f:
                json.dump(system_state, f, indent=2)
            
            # Export current metrics if production monitor is available
            try:
                from unity_wheel.observability.production_monitor import get_production_monitor
                monitor = get_production_monitor()
                
                # Export current status
                status_data = monitor.get_current_status()
                status_file = backup_path / "monitoring_status.json"
                with open(status_file, "w") as f:
                    json.dump(status_data, f, indent=2)
                
                # Export dashboard data
                dashboard_data = monitor.export_dashboard_data()
                dashboard_file = backup_path / "dashboard_data.json"
                with open(dashboard_file, "w") as f:
                    json.dump(dashboard_data, f, indent=2)
                    
            except ImportError:
                logger.debug("Production monitor not available for state backup")
            
            return {
                "component": "system_state",
                "success": True,
                "files_created": len(list(backup_path.glob("*.json")))
            }
            
        except Exception as e:
            logger.error(f"System state backup failed: {e}")
            return {
                "component": "system_state",
                "success": False,
                "error": str(e)
            }

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _cleanup_old_backups(self) -> None:
        """Remove backups older than retention period."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            
            for backup_file in self.backup_dir.glob("wheel_trading_backup_*.tar.gz"):
                # Extract timestamp from filename
                try:
                    timestamp_str = backup_file.stem.split("_")[-2] + "_" + backup_file.stem.split("_")[-1]
                    backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if backup_time < cutoff_time:
                        backup_file.unlink()
                        logger.info(f"Removed old backup: {backup_file.name}")
                        
                except (ValueError, IndexError):
                    # Skip files that don't match our naming pattern
                    continue
                    
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    def restore_from_backup(self, backup_file: Path, restore_path: Path) -> Dict[str, Any]:
        """Restore system from backup file."""
        try:
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            # Create restore directory
            restore_path.mkdir(parents=True, exist_ok=True)
            
            # Extract backup
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(restore_path)
            
            # Read manifest
            manifest_file = restore_path / "backup_manifest.json"
            if not manifest_file.exists():
                raise ValueError("Backup manifest not found")
            
            with open(manifest_file) as f:
                manifest = json.load(f)
            
            logger.info(
                "Backup restored successfully",
                extra={
                    "backup_id": manifest.get("backup_id"),
                    "restore_path": str(restore_path),
                    "components": len(manifest.get("components", []))
                }
            )
            
            return {
                "success": True,
                "manifest": manifest,
                "restore_path": str(restore_path)
            }
            
        except Exception as e:
            logger.error(f"Backup restore failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def list_available_backups(self) -> List[Dict[str, Any]]:
        """List all available backup files."""
        backups = []
        
        for backup_file in self.backup_dir.glob("wheel_trading_backup_*.tar.gz"):
            try:
                # Extract timestamp from filename
                timestamp_str = backup_file.stem.split("_")[-2] + "_" + backup_file.stem.split("_")[-1]
                backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                backups.append({
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "timestamp": backup_time.isoformat(),
                    "size_mb": backup_file.stat().st_size / (1024 * 1024),
                    "age_days": (datetime.now() - backup_time).days
                })
                
            except (ValueError, IndexError):
                continue
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        return backups

    def get_backup_status(self) -> Dict[str, Any]:
        """Get current backup system status."""
        return {
            "is_running": self.is_running,
            "backup_dir": str(self.backup_dir),
            "retention_days": self.retention_days,
            "backup_interval_hours": self.backup_interval_hours,
            "last_backup_time": self.last_backup_time.isoformat() if self.last_backup_time else None,
            "available_backups": len(self.list_available_backups()),
            "backup_history_count": len(self.backup_history),
            "total_backup_size_mb": sum(
                backup_file.stat().st_size / (1024 * 1024)
                for backup_file in self.backup_dir.glob("*.tar.gz")
            )
        }


# Global backup system instance
_backup_system: Optional[BackupSystem] = None


def get_backup_system() -> BackupSystem:
    """Get or create global backup system instance."""
    global _backup_system
    if _backup_system is None:
        _backup_system = BackupSystem()
    return _backup_system


def create_emergency_backup() -> Dict[str, Any]:
    """Create an emergency backup immediately."""
    backup_system = get_backup_system()
    return backup_system.create_full_backup()


def start_backup_monitoring() -> None:
    """Start automated backup monitoring."""
    backup_system = get_backup_system()
    backup_system.start_automated_backups()


def stop_backup_monitoring() -> None:
    """Stop automated backup monitoring."""
    backup_system = get_backup_system()
    backup_system.stop_automated_backups()