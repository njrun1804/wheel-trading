"""
Einstein Diagnostics and Health Monitoring

Comprehensive system health checking and diagnostics for Einstein components,
providing detailed information about system state, performance, and issues.
"""

import asyncio
import logging
import psutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of Einstein components."""
    INDEX = "index"
    SEARCH = "search"
    EMBEDDING = "embedding"
    FILEWATCHER = "file_watcher"
    DATABASE = "database"
    FAISS = "faiss"
    NEURAL = "neural"
    DEPENDENCIES = "dependencies"
    SYSTEM = "system"


@dataclass
class HealthCheck:
    """Result of a health check."""
    component: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'component': self.component,
            'component_type': self.component_type.value,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'recommendations': self.recommendations
        }


@dataclass
class SystemDiagnostics:
    """Complete system diagnostics."""
    overall_status: HealthStatus
    health_checks: List[HealthCheck]
    system_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    error_summary: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_status': self.overall_status.value,
            'health_checks': [check.to_dict() for check in self.health_checks],
            'system_info': self.system_info,
            'performance_metrics': self.performance_metrics,
            'error_summary': self.error_summary,
            'timestamp': self.timestamp
        }


class SystemHealthChecker:
    """Comprehensive system health checker for Einstein components."""
    
    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.logger = logging.getLogger(f"{__name__}.SystemHealthChecker")
        
        # Health check functions
        self.health_checkers = {
            ComponentType.SYSTEM: self._check_system_health,
            ComponentType.INDEX: self._check_index_health,
            ComponentType.SEARCH: self._check_search_health,
            ComponentType.EMBEDDING: self._check_embedding_health,
            ComponentType.FILEWATCHER: self._check_filewatcher_health,
            ComponentType.DATABASE: self._check_database_health,
            ComponentType.FAISS: self._check_faiss_health,
            ComponentType.NEURAL: self._check_neural_health,
            ComponentType.DEPENDENCIES: self._check_dependencies_health,
        }
    
    async def run_full_diagnostics(self) -> SystemDiagnostics:
        """Run complete system diagnostics."""
        start_time = time.time()
        health_checks = []
        
        # Run all health checks
        for component_type, checker in self.health_checkers.items():
            try:
                check = await checker()
                health_checks.append(check)
            except Exception as e:
                self.logger.error(f"Health check failed for {component_type.value}: {e}")
                health_checks.append(HealthCheck(
                    component=component_type.value,
                    component_type=component_type,
                    status=HealthStatus.FAILED,
                    message=f"Health check failed: {e}",
                    details={'error': str(e)}
                ))
        
        # Determine overall status
        overall_status = self._determine_overall_status(health_checks)
        
        # Collect system info and metrics
        system_info = await self._collect_system_info()
        performance_metrics = await self._collect_performance_metrics()
        error_summary = self._generate_error_summary(health_checks)
        
        diagnostics_time = time.time() - start_time
        self.logger.info(f"Full diagnostics completed in {diagnostics_time:.2f}s")
        
        return SystemDiagnostics(
            overall_status=overall_status,
            health_checks=health_checks,
            system_info=system_info,
            performance_metrics=performance_metrics,
            error_summary=error_summary
        )
    
    async def quick_health_check(self) -> HealthStatus:
        """Quick health check returning overall status."""
        critical_checks = [
            self._check_system_health(),
            self._check_dependencies_health(),
        ]
        
        results = await asyncio.gather(*critical_checks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                return HealthStatus.FAILED
            elif result.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                return result.status
        
        return HealthStatus.HEALTHY
    
    def _determine_overall_status(self, health_checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system status from individual checks."""
        if not health_checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in health_checks]
        
        if HealthStatus.FAILED in statuses:
            return HealthStatus.FAILED
        elif HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    async def _check_system_health(self) -> HealthCheck:
        """Check overall system health."""
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1.0)
            
            # Disk check
            disk = psutil.disk_usage(str(self.project_root))
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine status
            status = HealthStatus.HEALTHY
            message = "System resources are within normal limits"
            recommendations = []
            
            if memory_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory_percent:.1f}%"
                recommendations.extend([
                    "Restart system or free memory immediately",
                    "Check for memory leaks",
                    "Reduce system load"
                ])
            elif memory_percent > 85:
                status = HealthStatus.WARNING
                message = f"High memory usage: {memory_percent:.1f}%"
                recommendations.extend([
                    "Monitor memory usage closely",
                    "Consider reducing batch sizes",
                    "Clear caches if possible"
                ])
            
            if cpu_percent > 90:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                    message = f"High CPU usage: {cpu_percent:.1f}%"
                recommendations.extend([
                    "Reduce concurrent operations",
                    "Check for runaway processes",
                    "Consider CPU throttling"
                ])
            
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical disk usage: {disk_percent:.1f}%"
                recommendations.extend([
                    "Free disk space immediately",
                    "Clean up temporary files",
                    "Archive old data"
                ])
            elif disk_percent > 85:
                if status in [HealthStatus.HEALTHY, HealthStatus.WARNING]:
                    status = HealthStatus.WARNING
                    message = f"High disk usage: {disk_percent:.1f}%"
                recommendations.append("Monitor disk usage and clean up files")
            
            return HealthCheck(
                component="system_resources",
                component_type=ComponentType.SYSTEM,
                status=status,
                message=message,
                details={
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'cpu_percent': cpu_percent,
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk.free / (1024**3)
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return HealthCheck(
                component="system_resources",
                component_type=ComponentType.SYSTEM,
                status=HealthStatus.FAILED,
                message=f"System health check failed: {e}",
                details={'error': str(e)}
            )
    
    async def _check_index_health(self) -> HealthCheck:
        """Check Einstein index health."""
        try:
            einstein_dir = self.project_root / ".einstein"
            
            if not einstein_dir.exists():
                return HealthCheck(
                    component="einstein_index",
                    component_type=ComponentType.INDEX,
                    status=HealthStatus.WARNING,
                    message="Einstein directory does not exist",
                    details={'einstein_dir': str(einstein_dir)},
                    recommendations=["Initialize Einstein index", "Run index build"]
                )
            
            # Check for index files
            index_files = list(einstein_dir.glob("*.idx"))
            faiss_files = list(einstein_dir.glob("*.faiss"))
            db_files = list(einstein_dir.glob("*.db"))
            
            status = HealthStatus.HEALTHY
            message = "Index files present and accessible"
            details = {
                'einstein_dir_exists': True,
                'index_files_count': len(index_files),
                'faiss_files_count': len(faiss_files),
                'db_files_count': len(db_files),
                'total_index_size_mb': sum(f.stat().st_size for f in index_files) / (1024*1024)
            }
            recommendations = []
            
            if len(index_files) == 0 and len(faiss_files) == 0:
                status = HealthStatus.WARNING
                message = "No index files found"
                recommendations.extend([
                    "Build initial index",
                    "Check indexing permissions",
                    "Verify source files exist"
                ])
            
            # Check index freshness
            if index_files:
                newest_index = max(index_files, key=lambda f: f.stat().st_mtime)
                age_hours = (time.time() - newest_index.stat().st_mtime) / 3600
                details['newest_index_age_hours'] = age_hours
                
                if age_hours > 24:
                    status = HealthStatus.WARNING
                    message = f"Index files are {age_hours:.1f} hours old"
                    recommendations.append("Consider rebuilding index")
            
            return HealthCheck(
                component="einstein_index",
                component_type=ComponentType.INDEX,
                status=status,
                message=message,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            return HealthCheck(
                component="einstein_index",
                component_type=ComponentType.INDEX,
                status=HealthStatus.FAILED,
                message=f"Index health check failed: {e}",
                details={'error': str(e)}
            )
    
    async def _check_search_health(self) -> HealthCheck:
        """Check search functionality health."""
        try:
            # Try to perform a basic search test
            test_query = "function"
            search_available = False
            search_methods = []
            
            # Test ripgrep availability
            try:
                import subprocess
                result = subprocess.run(['rg', '--version'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    search_methods.append("ripgrep")
                    search_available = True
            except Exception:
                pass
            
            # Test Python search capability
            try:
                python_files = list(self.project_root.rglob("*.py"))
                if python_files:
                    search_methods.append("python_search")
                    search_available = True
            except Exception:
                pass
            
            status = HealthStatus.HEALTHY if search_available else HealthStatus.WARNING
            message = f"Search methods available: {', '.join(search_methods)}" if search_methods else "No search methods available"
            
            return HealthCheck(
                component="search_functionality",
                component_type=ComponentType.SEARCH,
                status=status,
                message=message,
                details={
                    'search_methods': search_methods,
                    'search_available': search_available,
                    'test_query': test_query
                },
                recommendations=["Install ripgrep for better search performance"] if "ripgrep" not in search_methods else []
            )
            
        except Exception as e:
            return HealthCheck(
                component="search_functionality",
                component_type=ComponentType.SEARCH,
                status=HealthStatus.FAILED,
                message=f"Search health check failed: {e}",
                details={'error': str(e)}
            )
    
    async def _check_embedding_health(self) -> HealthCheck:
        """Check embedding system health."""
        try:
            embedding_available = False
            embedding_backends = []
            
            # Test sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                embedding_backends.append("sentence_transformers")
                embedding_available = True
            except ImportError:
                pass
            
            # Test MLX (Apple Silicon)
            try:
                import mlx.core as mx
                if mx.metal.is_available():
                    embedding_backends.append("mlx")
                    embedding_available = True
            except ImportError:
                pass
            
            # Test PyTorch
            try:
                import torch
                embedding_backends.append("pytorch")
                embedding_available = True
            except ImportError:
                pass
            
            status = HealthStatus.HEALTHY if embedding_available else HealthStatus.WARNING
            message = f"Embedding backends: {', '.join(embedding_backends)}" if embedding_backends else "No embedding backends available"
            
            return HealthCheck(
                component="embedding_system",
                component_type=ComponentType.EMBEDDING,
                status=status,
                message=message,
                details={
                    'embedding_backends': embedding_backends,
                    'embedding_available': embedding_available
                },
                recommendations=["Install sentence-transformers for embedding support"] if not embedding_available else []
            )
            
        except Exception as e:
            return HealthCheck(
                component="embedding_system",
                component_type=ComponentType.EMBEDDING,
                status=HealthStatus.FAILED,
                message=f"Embedding health check failed: {e}",
                details={'error': str(e)}
            )
    
    async def _check_filewatcher_health(self) -> HealthCheck:
        """Check file watcher health."""
        try:
            # Test watchdog availability
            try:
                from watchdog.observers import Observer
                from watchdog.events import FileSystemEventHandler
                filewatcher_available = True
            except ImportError:
                filewatcher_available = False
            
            status = HealthStatus.HEALTHY if filewatcher_available else HealthStatus.WARNING
            message = "File watcher available" if filewatcher_available else "File watcher not available"
            
            return HealthCheck(
                component="file_watcher",
                component_type=ComponentType.FILEWATCHER,
                status=status,
                message=message,
                details={'filewatcher_available': filewatcher_available},
                recommendations=["Install watchdog for file monitoring"] if not filewatcher_available else []
            )
            
        except Exception as e:
            return HealthCheck(
                component="file_watcher",
                component_type=ComponentType.FILEWATCHER,
                status=HealthStatus.FAILED,
                message=f"File watcher health check failed: {e}",
                details={'error': str(e)}
            )
    
    async def _check_database_health(self) -> HealthCheck:
        """Check database health."""
        try:
            databases_available = []
            
            # Test SQLite
            try:
                import sqlite3
                databases_available.append("sqlite3")
            except ImportError:
                pass
            
            # Test DuckDB
            try:
                import duckdb
                databases_available.append("duckdb")
            except ImportError:
                pass
            
            status = HealthStatus.HEALTHY if databases_available else HealthStatus.CRITICAL
            message = f"Databases available: {', '.join(databases_available)}" if databases_available else "No databases available"
            
            return HealthCheck(
                component="database_systems",
                component_type=ComponentType.DATABASE,
                status=status,
                message=message,
                details={'databases_available': databases_available},
                recommendations=["Install DuckDB for better performance"] if "duckdb" not in databases_available else []
            )
            
        except Exception as e:
            return HealthCheck(
                component="database_systems",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.FAILED,
                message=f"Database health check failed: {e}",
                details={'error': str(e)}
            )
    
    async def _check_faiss_health(self) -> HealthCheck:
        """Check FAISS health."""
        try:
            faiss_available = False
            faiss_version = None
            
            try:
                import faiss
                import numpy as np
                
                # Test basic FAISS functionality
                test_index = faiss.IndexFlatL2(128)
                test_vector = np.array([[1.0] * 128], dtype=np.float32)
                test_index.add(test_vector)
                
                if test_index.ntotal == 1:
                    faiss_available = True
                    faiss_version = getattr(faiss, '__version__', 'unknown')
                
            except (ImportError, Exception):
                pass
            
            status = HealthStatus.HEALTHY if faiss_available else HealthStatus.WARNING
            message = f"FAISS available (version: {faiss_version})" if faiss_available else "FAISS not available"
            
            return HealthCheck(
                component="faiss_system",
                component_type=ComponentType.FAISS,
                status=status,
                message=message,
                details={
                    'faiss_available': faiss_available,
                    'faiss_version': faiss_version
                },
                recommendations=["Install FAISS for vector search capabilities"] if not faiss_available else []
            )
            
        except Exception as e:
            return HealthCheck(
                component="faiss_system",
                component_type=ComponentType.FAISS,
                status=HealthStatus.FAILED,
                message=f"FAISS health check failed: {e}",
                details={'error': str(e)}
            )
    
    async def _check_neural_health(self) -> HealthCheck:
        """Check neural backend health."""
        try:
            neural_backends = []
            
            # Test MLX
            try:
                import mlx.core as mx
                if mx.metal.is_available():
                    neural_backends.append("mlx")
            except ImportError:
                pass
            
            # Test PyTorch MPS
            try:
                import torch
                if torch.backends.mps.is_available():
                    neural_backends.append("pytorch_mps")
                elif torch.cuda.is_available():
                    neural_backends.append("pytorch_cuda")
                else:
                    neural_backends.append("pytorch_cpu")
            except ImportError:
                pass
            
            status = HealthStatus.HEALTHY if neural_backends else HealthStatus.WARNING
            message = f"Neural backends: {', '.join(neural_backends)}" if neural_backends else "No neural backends available"
            
            return HealthCheck(
                component="neural_backends",
                component_type=ComponentType.NEURAL,
                status=status,
                message=message,
                details={'neural_backends': neural_backends},
                recommendations=["Install PyTorch or MLX for neural acceleration"] if not neural_backends else []
            )
            
        except Exception as e:
            return HealthCheck(
                component="neural_backends",
                component_type=ComponentType.NEURAL,
                status=HealthStatus.FAILED,
                message=f"Neural backend health check failed: {e}",
                details={'error': str(e)}
            )
    
    async def _check_dependencies_health(self) -> HealthCheck:
        """Check critical dependencies health."""
        try:
            missing_deps = []
            available_deps = []
            
            # Check critical dependencies
            critical_deps = [
                'numpy',
                'pathlib',
                'asyncio',
                'logging',
                'json',
                'time',
                'os',
                'subprocess'
            ]
            
            for dep in critical_deps:
                try:
                    __import__(dep)
                    available_deps.append(dep)
                except ImportError:
                    missing_deps.append(dep)
            
            status = HealthStatus.CRITICAL if missing_deps else HealthStatus.HEALTHY
            message = f"Dependencies check: {len(available_deps)}/{len(critical_deps)} available"
            
            if missing_deps:
                message += f", missing: {', '.join(missing_deps)}"
            
            return HealthCheck(
                component="dependencies",
                component_type=ComponentType.DEPENDENCIES,
                status=status,
                message=message,
                details={
                    'available_deps': available_deps,
                    'missing_deps': missing_deps,
                    'total_checked': len(critical_deps)
                },
                recommendations=[f"Install missing dependency: {dep}" for dep in missing_deps]
            )
            
        except Exception as e:
            return HealthCheck(
                component="dependencies",
                component_type=ComponentType.DEPENDENCIES,
                status=HealthStatus.FAILED,
                message=f"Dependencies health check failed: {e}",
                details={'error': str(e)}
            )
    
    async def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        try:
            import platform
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage(str(self.project_root)).total / (1024**3),
                'project_root': str(self.project_root),
                'einstein_dir_exists': (self.project_root / ".einstein").exists()
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        try:
            # Basic performance timing tests
            start_time = time.time()
            
            # File I/O test
            test_files = list(self.project_root.rglob("*.py"))[:10]
            file_read_time = 0.0
            
            for file_path in test_files:
                file_start = time.time()
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        f.read()
                    file_read_time += time.time() - file_start
                except Exception:
                    continue
            
            total_time = time.time() - start_time
            
            return {
                'health_check_duration_ms': total_time * 1000,
                'file_read_duration_ms': file_read_time * 1000,
                'files_tested': len(test_files),
                'avg_file_read_ms': (file_read_time / max(1, len(test_files))) * 1000
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_error_summary(self, health_checks: List[HealthCheck]) -> Dict[str, Any]:
        """Generate error summary from health checks."""
        failed_checks = [check for check in health_checks if check.status == HealthStatus.FAILED]
        critical_checks = [check for check in health_checks if check.status == HealthStatus.CRITICAL]
        warning_checks = [check for check in health_checks if check.status == HealthStatus.WARNING]
        
        all_recommendations = []
        for check in health_checks:
            all_recommendations.extend(check.recommendations)
        
        return {
            'total_checks': len(health_checks),
            'failed_count': len(failed_checks),
            'critical_count': len(critical_checks),
            'warning_count': len(warning_checks),
            'healthy_count': len([check for check in health_checks if check.status == HealthStatus.HEALTHY]),
            'failed_components': [check.component for check in failed_checks],
            'critical_components': [check.component for check in critical_checks],
            'warning_components': [check.component for check in warning_checks],
            'total_recommendations': len(all_recommendations),
            'unique_recommendations': list(set(all_recommendations))
        }


class EinsteinDiagnostics:
    """Main diagnostics interface for Einstein system."""
    
    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.health_checker = SystemHealthChecker(self.project_root)
        self.logger = logging.getLogger(f"{__name__}.EinsteinDiagnostics")
    
    async def run_diagnostics(self, quick: bool = False) -> SystemDiagnostics:
        """Run Einstein system diagnostics."""
        if quick:
            status = await self.health_checker.quick_health_check()
            return SystemDiagnostics(
                overall_status=status,
                health_checks=[],
                system_info={},
                performance_metrics={},
                error_summary={}
            )
        else:
            return await self.health_checker.run_full_diagnostics()
    
    async def check_component_health(self, component_type: ComponentType) -> HealthCheck:
        """Check health of a specific component."""
        checker = self.health_checker.health_checkers.get(component_type)
        if checker:
            return await checker()
        else:
            return HealthCheck(
                component=component_type.value,
                component_type=component_type,
                status=HealthStatus.UNKNOWN,
                message="No health checker available for this component"
            )
    
    def generate_report(self, diagnostics: SystemDiagnostics) -> str:
        """Generate human-readable diagnostics report."""
        lines = [
            "=" * 60,
            "EINSTEIN SYSTEM DIAGNOSTICS REPORT",
            "=" * 60,
            f"Overall Status: {diagnostics.overall_status.value.upper()}",
            f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(diagnostics.timestamp))}",
            "",
        ]
        
        # System info
        if diagnostics.system_info:
            lines.extend([
                "SYSTEM INFORMATION:",
                "-" * 30,
            ])
            for key, value in diagnostics.system_info.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Health checks
        if diagnostics.health_checks:
            lines.extend([
                "COMPONENT HEALTH CHECKS:",
                "-" * 30,
            ])
            
            for check in diagnostics.health_checks:
                status_symbol = {
                    HealthStatus.HEALTHY: "✅",
                    HealthStatus.WARNING: "⚠️",
                    HealthStatus.CRITICAL: "🔴",
                    HealthStatus.FAILED: "❌",
                    HealthStatus.UNKNOWN: "❓"
                }.get(check.status, "❓")
                
                lines.append(f"{status_symbol} {check.component}: {check.message}")
                
                if check.recommendations:
                    lines.append(f"    Recommendations:")
                    for rec in check.recommendations:
                        lines.append(f"      • {rec}")
                lines.append("")
        
        # Error summary
        if diagnostics.error_summary:
            lines.extend([
                "ERROR SUMMARY:",
                "-" * 30,
                f"  Total Checks: {diagnostics.error_summary.get('total_checks', 0)}",
                f"  Failed: {diagnostics.error_summary.get('failed_count', 0)}",
                f"  Critical: {diagnostics.error_summary.get('critical_count', 0)}",
                f"  Warnings: {diagnostics.error_summary.get('warning_count', 0)}",
                f"  Healthy: {diagnostics.error_summary.get('healthy_count', 0)}",
                "",
            ])
            
            if diagnostics.error_summary.get('unique_recommendations'):
                lines.extend([
                    "TOP RECOMMENDATIONS:",
                    "-" * 30,
                ])
                for rec in diagnostics.error_summary['unique_recommendations'][:5]:
                    lines.append(f"  • {rec}")
                lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Global diagnostics instance
_diagnostics: EinsteinDiagnostics | None = None


def get_einstein_diagnostics(project_root: Path | None = None) -> EinsteinDiagnostics:
    """Get or create the global Einstein diagnostics instance."""
    global _diagnostics
    if _diagnostics is None:
        _diagnostics = EinsteinDiagnostics(project_root)
    return _diagnostics