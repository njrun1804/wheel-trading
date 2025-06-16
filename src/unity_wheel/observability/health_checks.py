"""Comprehensive health check system for wheel trading platform.

Provides automated health monitoring of:
- Database connectivity and performance
- API endpoints and data freshness
- System resources and performance
- Trading system components
- Configuration validation
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import psutil
import requests

from unity_wheel.utils import get_logger

logger = get_logger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    check_name: str
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    response_time_ms: float
    message: str
    details: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @property
    def is_healthy(self) -> bool:
        """Check if result indicates healthy status."""
        return self.status == "healthy"


class BaseHealthCheck(ABC):
    """Base class for health checks."""

    def __init__(self, name: str, timeout_seconds: float = 10.0):
        self.name = name
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass

    def _create_result(
        self,
        status: str,
        message: str,
        response_time_ms: float,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> HealthCheckResult:
        """Create a health check result."""
        return HealthCheckResult(
            check_name=self.name,
            status=status,
            timestamp=datetime.now(UTC),
            response_time_ms=response_time_ms,
            message=message,
            details=details or {},
            error=error
        )


class DatabaseHealthCheck(BaseHealthCheck):
    """Health check for database connectivity and performance."""

    def __init__(self, db_path: Optional[Path] = None):
        super().__init__("database_connectivity")
        self.db_path = db_path or (
            Path.home() / ".wheel_trading" / "cache" / "wheel_trading_master.duckdb"
        )

    async def check(self) -> HealthCheckResult:
        """Check database health."""
        start_time = time.time()
        
        try:
            if not self.db_path.exists():
                return self._create_result(
                    "unhealthy",
                    f"Database file not found: {self.db_path}",
                    0.0,
                    error="Database file missing"
                )

            # Test connection and basic query
            conn = duckdb.connect(str(self.db_path), read_only=True)
            
            # Test basic query
            result = conn.execute("SELECT 1 as test").fetchone()
            
            # Get database info
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [row[0] for row in tables]
            
            # Check database size
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
            
            conn.close()
            
            response_time = (time.time() - start_time) * 1000
            
            return self._create_result(
                "healthy",
                f"Database accessible with {len(table_names)} tables",
                response_time,
                {
                    "database_path": str(self.db_path),
                    "database_size_mb": round(db_size_mb, 2),
                    "table_count": len(table_names),
                    "tables": table_names[:10],  # First 10 tables
                    "test_query_result": result[0] if result else None
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return self._create_result(
                "unhealthy",
                f"Database connection failed: {str(e)}",
                response_time,
                error=str(e)
            )


class SystemResourcesHealthCheck(BaseHealthCheck):
    """Health check for system resources."""

    def __init__(self):
        super().__init__("system_resources")

    async def check(self) -> HealthCheckResult:
        """Check system resource health."""
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get load averages (Unix only)
            load_avg = [0.0, 0.0, 0.0]
            if hasattr(psutil, 'getloadavg'):
                load_avg = list(psutil.getloadavg())
            
            # Determine status based on thresholds
            issues = []
            if cpu_percent > 80:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 85:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            if disk.percent > 90:
                issues.append(f"Low disk space: {disk.percent:.1f}%")
            if load_avg[0] > 8.0:  # M4 Pro has 12 cores
                issues.append(f"High system load: {load_avg[0]:.2f}")
            
            status = "unhealthy" if len(issues) > 2 else "degraded" if issues else "healthy"
            message = "; ".join(issues) if issues else "System resources within normal ranges"
            
            response_time = (time.time() - start_time) * 1000
            
            return self._create_result(
                status,
                message,
                response_time,
                {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_usage_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "load_avg_1m": load_avg[0],
                    "load_avg_5m": load_avg[1],
                    "load_avg_15m": load_avg[2],
                    "issues_detected": len(issues)
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return self._create_result(
                "unhealthy",
                f"Failed to check system resources: {str(e)}",
                response_time,
                error=str(e)
            )


class ConfigurationHealthCheck(BaseHealthCheck):
    """Health check for configuration files and settings."""

    def __init__(self):
        super().__init__("configuration")

    async def check(self) -> HealthCheckResult:
        """Check configuration health."""
        start_time = time.time()
        
        try:
            config_issues = []
            config_details = {}
            
            # Check main config file
            config_file = Path("config.yaml")
            if config_file.exists():
                config_details["config_yaml"] = {
                    "exists": True,
                    "size_bytes": config_file.stat().st_size,
                    "modified": datetime.fromtimestamp(config_file.stat().st_mtime).isoformat()
                }
            else:
                config_issues.append("config.yaml not found")
                config_details["config_yaml"] = {"exists": False}
            
            # Check logging config
            logging_config = Path("logging_config.json")
            if logging_config.exists():
                try:
                    with open(logging_config) as f:
                        logging_data = json.load(f)
                    config_details["logging_config"] = {
                        "exists": True,
                        "valid_json": True,
                        "handlers": len(logging_data.get("handlers", {})),
                        "loggers": len(logging_data.get("loggers", {}))
                    }
                except json.JSONDecodeError:
                    config_issues.append("logging_config.json invalid JSON")
                    config_details["logging_config"] = {"exists": True, "valid_json": False}
            else:
                config_issues.append("logging_config.json not found")
                config_details["logging_config"] = {"exists": False}
            
            # Check logs directory
            logs_dir = Path("logs")
            if logs_dir.exists():
                log_files = list(logs_dir.glob("*.log*"))
                config_details["logs_directory"] = {
                    "exists": True,
                    "log_files_count": len(log_files),
                    "total_size_mb": sum(f.stat().st_size for f in log_files) / (1024 * 1024)
                }
            else:
                config_issues.append("logs directory not found")
                config_details["logs_directory"] = {"exists": False}
            
            # Check environment variables
            required_env_vars = ["HOME"]  # Add any required env vars
            missing_env_vars = []
            for var in required_env_vars:
                if var not in os.environ:
                    missing_env_vars.append(var)
            
            if missing_env_vars:
                config_issues.extend([f"Missing env var: {var}" for var in missing_env_vars])
            
            config_details["environment"] = {
                "required_vars_present": len(missing_env_vars) == 0,
                "missing_vars": missing_env_vars
            }
            
            status = "unhealthy" if len(config_issues) > 2 else "degraded" if config_issues else "healthy"
            message = "; ".join(config_issues) if config_issues else "Configuration files are valid"
            
            response_time = (time.time() - start_time) * 1000
            
            return self._create_result(
                status,
                message,
                response_time,
                config_details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return self._create_result(
                "unhealthy",
                f"Configuration check failed: {str(e)}",
                response_time,
                error=str(e)
            )


class TradingSystemHealthCheck(BaseHealthCheck):
    """Health check for trading system components."""

    def __init__(self):
        super().__init__("trading_system")

    async def check(self) -> HealthCheckResult:
        """Check trading system health."""
        start_time = time.time()
        
        try:
            trading_details = {}
            issues = []
            
            # Check if we can import core trading modules
            try:
                from unity_wheel.api import advisor
                from unity_wheel.strategy import wheel
                from unity_wheel.risk import analytics
                trading_details["core_modules"] = {
                    "advisor": True,
                    "wheel_strategy": True,
                    "risk_analytics": True
                }
            except ImportError as e:
                issues.append(f"Core module import failed: {str(e)}")
                trading_details["core_modules"] = {"import_error": str(e)}
            
            # Check performance monitor
            try:
                from unity_wheel.monitoring import get_performance_monitor
                perf_monitor = get_performance_monitor()
                recent_stats = perf_monitor.get_all_stats(window_minutes=5)
                trading_details["performance_monitor"] = {
                    "available": True,
                    "recent_operations": len(recent_stats),
                    "operations": list(recent_stats.keys())[:5]
                }
            except Exception as e:
                issues.append(f"Performance monitor unavailable: {str(e)}")
                trading_details["performance_monitor"] = {"available": False, "error": str(e)}
            
            # Check data providers
            try:
                # Test basic data access
                cache_path = Path.home() / ".wheel_trading" / "cache"
                if cache_path.exists():
                    db_files = list(cache_path.glob("*.db"))
                    trading_details["data_access"] = {
                        "cache_directory": True,
                        "database_files": len(db_files),
                        "databases": [f.name for f in db_files[:3]]
                    }
                else:
                    issues.append("Data cache directory not found")
                    trading_details["data_access"] = {"cache_directory": False}
            except Exception as e:
                issues.append(f"Data access check failed: {str(e)}")
                trading_details["data_access"] = {"error": str(e)}
            
            status = "unhealthy" if len(issues) > 1 else "degraded" if issues else "healthy"
            message = "; ".join(issues) if issues else "Trading system components operational"
            
            response_time = (time.time() - start_time) * 1000
            
            return self._create_result(
                status,
                message,
                response_time,
                trading_details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return self._create_result(
                "unhealthy",
                f"Trading system check failed: {str(e)}",
                response_time,
                error=str(e)
            )


class DataFreshnessHealthCheck(BaseHealthCheck):
    """Health check for data freshness and quality."""

    def __init__(self, db_path: Optional[Path] = None):
        super().__init__("data_freshness")
        self.db_path = db_path or (
            Path.home() / ".wheel_trading" / "cache" / "wheel_trading_master.duckdb"
        )

    async def check(self) -> HealthCheckResult:
        """Check data freshness."""
        start_time = time.time()
        
        try:
            if not self.db_path.exists():
                return self._create_result(
                    "unhealthy",
                    "Database not found for data freshness check",
                    0.0,
                    error="Database file missing"
                )

            conn = duckdb.connect(str(self.db_path), read_only=True)
            data_details = {}
            issues = []
            
            # Check various data tables for freshness
            tables_to_check = [
                ("option_chains", "created_at"),
                ("position_snapshots", "created_at"),
                ("predictions_cache", "created_at")
            ]
            
            for table_name, timestamp_col in tables_to_check:
                try:
                    # Check if table exists
                    table_exists = conn.execute(
                        f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
                    ).fetchone()[0] > 0
                    
                    if table_exists:
                        # Get latest record timestamp
                        latest_record = conn.execute(
                            f"SELECT MAX({timestamp_col}) FROM {table_name}"
                        ).fetchone()[0]
                        
                        if latest_record:
                            latest_time = latest_record if isinstance(latest_record, datetime) else datetime.fromisoformat(latest_record)
                            age_hours = (datetime.now(UTC) - latest_time.replace(tzinfo=UTC)).total_seconds() / 3600
                            
                            data_details[table_name] = {
                                "exists": True,
                                "latest_record": latest_time.isoformat(),
                                "age_hours": round(age_hours, 2)
                            }
                            
                            # Check if data is stale (older than 24 hours for most data)
                            if age_hours > 24:
                                issues.append(f"{table_name} data is {age_hours:.1f} hours old")
                        else:
                            data_details[table_name] = {"exists": True, "latest_record": None}
                            issues.append(f"{table_name} has no records")
                    else:
                        data_details[table_name] = {"exists": False}
                        
                except Exception as e:
                    data_details[table_name] = {"error": str(e)}
                    issues.append(f"{table_name} check failed: {str(e)}")
            
            conn.close()
            
            status = "unhealthy" if len(issues) > 2 else "degraded" if issues else "healthy"
            message = "; ".join(issues) if issues else "Data is fresh and up-to-date"
            
            response_time = (time.time() - start_time) * 1000
            
            return self._create_result(
                status,
                message,
                response_time,
                data_details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return self._create_result(
                "unhealthy",
                f"Data freshness check failed: {str(e)}",
                response_time,
                error=str(e)
            )


class HealthCheckOrchestrator:
    """Orchestrates and manages multiple health checks."""

    def __init__(self):
        self.health_checks: List[BaseHealthCheck] = [
            DatabaseHealthCheck(),
            SystemResourcesHealthCheck(),
            ConfigurationHealthCheck(),
            TradingSystemHealthCheck(),
            DataFreshnessHealthCheck()
        ]
        self.last_check_time: Optional[datetime] = None
        self.last_results: List[HealthCheckResult] = []

    async def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        logger.info("Starting comprehensive health checks")
        
        results = []
        tasks = []
        
        # Run all checks concurrently
        for check in self.health_checks:
            tasks.append(check.check())
        
        # Wait for all checks to complete
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create error result for failed check
                    error_result = HealthCheckResult(
                        check_name=self.health_checks[i].name,
                        status="unhealthy",
                        timestamp=datetime.now(UTC),
                        response_time_ms=0.0,
                        message=f"Health check failed with exception: {str(result)}",
                        details={},
                        error=str(result)
                    )
                    final_results.append(error_result)
                else:
                    final_results.append(result)
            
            self.last_results = final_results
            self.last_check_time = datetime.now(UTC)
            
            # Log summary
            healthy_count = sum(1 for r in final_results if r.is_healthy)
            logger.info(
                f"Health checks completed: {healthy_count}/{len(final_results)} healthy",
                extra={
                    "total_checks": len(final_results),
                    "healthy_checks": healthy_count,
                    "check_duration_ms": sum(r.response_time_ms for r in final_results)
                }
            )
            
            return final_results
            
        except Exception as e:
            logger.error(f"Health check orchestration failed: {e}", exc_info=True)
            return []

    def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.last_results:
            return {
                "status": "unknown",
                "message": "No health checks have been performed",
                "last_check": None
            }

        # Calculate overall status
        healthy_count = sum(1 for r in self.last_results if r.status == "healthy")
        degraded_count = sum(1 for r in self.last_results if r.status == "degraded")
        unhealthy_count = sum(1 for r in self.last_results if r.status == "unhealthy")
        
        total_checks = len(self.last_results)
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = "unhealthy"
            message = f"{unhealthy_count} critical issues detected"
        elif degraded_count > total_checks // 2:
            overall_status = "degraded"
            message = f"{degraded_count} components degraded"
        elif degraded_count > 0:
            overall_status = "degraded"
            message = f"{degraded_count} minor issues detected"
        else:
            overall_status = "healthy"
            message = "All systems operational"
        
        return {
            "status": overall_status,
            "message": message,
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "summary": {
                "total_checks": total_checks,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            },
            "checks": [result.to_dict() for result in self.last_results]
        }

    def get_failing_checks(self) -> List[HealthCheckResult]:
        """Get list of failing health checks."""
        return [r for r in self.last_results if r.status in ["degraded", "unhealthy"]]


# Global health check orchestrator
_health_orchestrator: Optional[HealthCheckOrchestrator] = None


def get_health_orchestrator() -> HealthCheckOrchestrator:
    """Get or create global health check orchestrator."""
    global _health_orchestrator
    if _health_orchestrator is None:
        _health_orchestrator = HealthCheckOrchestrator()
    return _health_orchestrator


async def run_health_checks() -> List[HealthCheckResult]:
    """Run all health checks."""
    orchestrator = get_health_orchestrator()
    return await orchestrator.run_all_checks()


def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    orchestrator = get_health_orchestrator()
    return orchestrator.get_overall_health_status()