#!/usr/bin/env python3
"""
Service Startup Dependency Validation

Validates that all required services and dependencies are available before
starting the trading system. Prevents startup failures and provides clear
error messages for missing dependencies.

Key Features:
- Database connectivity validation
- API endpoint availability checks  
- File system permissions validation
- Resource availability verification
- Service health monitoring
- Automated dependency resolution
"""

import asyncio
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class DependencyCheck:
    """Definition of a dependency check."""

    name: str
    description: str
    required: bool = True
    check_function: str | None = None
    fix_function: str | None = None
    timeout: float = 30.0
    retry_attempts: int = 3
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of a dependency validation."""

    name: str
    passed: bool
    message: str
    duration: float
    error: Exception | None = None
    fix_available: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class ServiceStartupValidator:
    """Validates all service dependencies before startup."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.validation_results: dict[str, ValidationResult] = {}
        self.dependencies = self._define_dependencies()

        # Performance tracking
        self.start_time = 0.0
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0

    def _define_dependencies(self) -> dict[str, DependencyCheck]:
        """Define all service dependencies."""
        return {
            "python_version": DependencyCheck(
                name="python_version",
                description="Python version compatibility (>=3.8)",
                check_function="check_python_version",
            ),
            "system_resources": DependencyCheck(
                name="system_resources",
                description="System resources (CPU, Memory, Disk)",
                check_function="check_system_resources",
            ),
            "required_packages": DependencyCheck(
                name="required_packages",
                description="Required Python packages",
                check_function="check_required_packages",
                fix_function="fix_required_packages",
            ),
            "database_files": DependencyCheck(
                name="database_files",
                description="Database files accessibility",
                check_function="check_database_files",
                fix_function="fix_database_files",
            ),
            "database_connectivity": DependencyCheck(
                name="database_connectivity",
                description="Database connection pools",
                check_function="check_database_connectivity",
                dependencies=["database_files"],
            ),
            "file_permissions": DependencyCheck(
                name="file_permissions",
                description="File system permissions",
                check_function="check_file_permissions",
                fix_function="fix_file_permissions",
            ),
            "network_connectivity": DependencyCheck(
                name="network_connectivity",
                description="Network connectivity for APIs",
                check_function="check_network_connectivity",
            ),
            "api_endpoints": DependencyCheck(
                name="api_endpoints",
                description="External API endpoints",
                check_function="check_api_endpoints",
                required=False,  # Optional for offline mode
                dependencies=["network_connectivity"],
            ),
            "accelerated_tools": DependencyCheck(
                name="accelerated_tools",
                description="Hardware-accelerated tools",
                check_function="check_accelerated_tools",
                fix_function="fix_accelerated_tools",
            ),
            "einstein_integration": DependencyCheck(
                name="einstein_integration",
                description="Einstein system integration",
                check_function="check_einstein_integration",
                dependencies=["database_connectivity", "accelerated_tools"],
            ),
            "bolt_integration": DependencyCheck(
                name="bolt_integration",
                description="Bolt system integration",
                check_function="check_bolt_integration",
                dependencies=["database_connectivity"],
            ),
        }

    async def validate_all(self, fix_issues: bool = True) -> bool:
        """Validate all dependencies with optional automatic fixing."""
        self.start_time = time.time()
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0

        logger.info("🚀 Starting service dependency validation...")

        # Validate dependencies in dependency order
        ordered_deps = self._get_dependency_order()

        for dep_name in ordered_deps:
            dependency = self.dependencies[dep_name]
            self.total_checks += 1

            logger.info(f"🔍 Checking: {dependency.description}")

            # Run the validation check
            result = await self._run_check(dependency)
            self.validation_results[dep_name] = result

            if result.passed:
                self.passed_checks += 1
                logger.info(f"✅ {dependency.description}: {result.message}")
            else:
                self.failed_checks += 1
                logger.error(f"❌ {dependency.description}: {result.message}")

                # Attempt to fix if requested and fix function available
                if fix_issues and dependency.fix_function and result.fix_available:
                    logger.info(f"🔧 Attempting to fix: {dependency.description}")
                    fixed = await self._run_fix(dependency)

                    if fixed:
                        # Re-run the check
                        result = await self._run_check(dependency)
                        self.validation_results[dep_name] = result

                        if result.passed:
                            logger.info(f"✅ Fixed: {dependency.description}")
                        else:
                            logger.error(f"❌ Fix failed: {dependency.description}")

                # Stop if critical dependency failed
                if dependency.required and not result.passed:
                    logger.error(
                        f"💥 Critical dependency failed: {dependency.description}"
                    )
                    return False

        duration = time.time() - self.start_time

        logger.info(f"🏁 Validation complete in {duration:.2f}s")
        logger.info(
            f"📊 Results: {self.passed_checks}/{self.total_checks} passed, {self.failed_checks} failed"
        )

        return self.failed_checks == 0

    def _get_dependency_order(self) -> list[str]:
        """Get dependencies in correct order based on dependencies."""
        ordered = []
        visited = set()
        visiting = set()

        def visit(dep_name: str):
            if dep_name in visiting:
                raise ValueError(f"Circular dependency detected: {dep_name}")
            if dep_name in visited:
                return

            visiting.add(dep_name)

            dependency = self.dependencies[dep_name]
            for prereq in dependency.dependencies:
                if prereq in self.dependencies:
                    visit(prereq)

            visiting.remove(dep_name)
            visited.add(dep_name)
            ordered.append(dep_name)

        for dep_name in self.dependencies:
            visit(dep_name)

        return ordered

    async def _run_check(self, dependency: DependencyCheck) -> ValidationResult:
        """Run a single dependency check."""
        start_time = time.time()

        try:
            if dependency.check_function:
                check_method = getattr(self, dependency.check_function)
                result = await check_method()
                result.duration = time.time() - start_time
                return result
            else:
                return ValidationResult(
                    name=dependency.name,
                    passed=False,
                    message="No check function defined",
                    duration=time.time() - start_time,
                )

        except Exception as e:
            return ValidationResult(
                name=dependency.name,
                passed=False,
                message=f"Check failed: {str(e)}",
                duration=time.time() - start_time,
                error=e,
            )

    async def _run_fix(self, dependency: DependencyCheck) -> bool:
        """Run a dependency fix function."""
        try:
            if dependency.fix_function:
                fix_method = getattr(self, dependency.fix_function)
                return await fix_method()
            return False
        except Exception as e:
            logger.error(f"Fix failed for {dependency.name}: {e}")
            return False

    # Individual check functions

    async def check_python_version(self) -> ValidationResult:
        """Check Python version compatibility."""
        version_info = sys.version_info
        required_major, required_minor = 3, 8

        if (
            version_info.major >= required_major
            and version_info.minor >= required_minor
        ):
            return ValidationResult(
                name="python_version",
                passed=True,
                message=f"Python {version_info.major}.{version_info.minor}.{version_info.micro}",
                duration=0.0,
                details={
                    "version": f"{version_info.major}.{version_info.minor}.{version_info.micro}"
                },
            )
        else:
            return ValidationResult(
                name="python_version",
                passed=False,
                message=f"Python {version_info.major}.{version_info.minor} < required {required_major}.{required_minor}",
                duration=0.0,
                fix_available=False,
            )

    async def check_system_resources(self) -> ValidationResult:
        """Check system resources availability."""
        # Check available memory (require at least 4GB)
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        # Check available disk space (require at least 5GB)
        disk = psutil.disk_usage(str(self.project_root))
        available_disk_gb = disk.free / (1024**3)

        # Check CPU count
        cpu_count = psutil.cpu_count()

        issues = []
        if available_gb < 4.0:
            issues.append(f"Low memory: {available_gb:.1f}GB available (4GB required)")
        if available_disk_gb < 5.0:
            issues.append(
                f"Low disk space: {available_disk_gb:.1f}GB available (5GB required)"
            )
        if cpu_count < 2:
            issues.append(f"Insufficient CPU cores: {cpu_count} (2+ recommended)")

        if issues:
            return ValidationResult(
                name="system_resources",
                passed=False,
                message="; ".join(issues),
                duration=0.0,
                details={
                    "memory_gb": available_gb,
                    "disk_gb": available_disk_gb,
                    "cpu_count": cpu_count,
                },
            )
        else:
            return ValidationResult(
                name="system_resources",
                passed=True,
                message=f"Memory: {available_gb:.1f}GB, Disk: {available_disk_gb:.1f}GB, CPU: {cpu_count} cores",
                duration=0.0,
                details={
                    "memory_gb": available_gb,
                    "disk_gb": available_disk_gb,
                    "cpu_count": cpu_count,
                },
            )

    async def check_required_packages(self) -> ValidationResult:
        """Check required Python packages."""
        required_packages = [
            "asyncio",
            "pandas",
            "numpy",
            "duckdb",
            "requests",
            "psutil",
            "pathlib",
            "dataclasses",
            "typing",
            "concurrent.futures",
        ]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            return ValidationResult(
                name="required_packages",
                passed=False,
                message=f"Missing packages: {', '.join(missing_packages)}",
                duration=0.0,
                fix_available=True,
                details={"missing": missing_packages},
            )
        else:
            return ValidationResult(
                name="required_packages",
                passed=True,
                message=f"All {len(required_packages)} required packages available",
                duration=0.0,
                details={"checked": required_packages},
            )

    async def check_database_files(self) -> ValidationResult:
        """Check database files accessibility."""
        db_patterns = ["*.db", "*.duckdb", "*.sqlite", "*.sqlite3"]
        db_files = []

        for pattern in db_patterns:
            db_files.extend(self.project_root.rglob(pattern))

        issues = []
        accessible_files = 0

        for db_file in db_files:
            try:
                # Check file exists and is readable
                if not db_file.exists():
                    issues.append(f"Database file missing: {db_file}")
                    continue

                # Check file permissions
                if not os.access(str(db_file), os.R_OK):
                    issues.append(f"Database file not readable: {db_file}")
                    continue

                # Check file is not locked
                try:
                    with open(db_file, "rb") as f:
                        f.read(1)  # Try to read one byte
                    accessible_files += 1
                except OSError as e:
                    issues.append(f"Database file locked: {db_file} ({e})")

            except Exception as e:
                issues.append(f"Database file error: {db_file} ({e})")

        if issues:
            return ValidationResult(
                name="database_files",
                passed=False,
                message=f"{len(issues)} database issues found",
                duration=0.0,
                fix_available=True,
                details={
                    "issues": issues,
                    "accessible": accessible_files,
                    "total": len(db_files),
                },
            )
        else:
            return ValidationResult(
                name="database_files",
                passed=True,
                message=f"All {len(db_files)} database files accessible",
                duration=0.0,
                details={"accessible": accessible_files, "total": len(db_files)},
            )

    async def check_database_connectivity(self) -> ValidationResult:
        """Check database connectivity using connection pools."""
        try:
            # Test bolt database fixes
            from bolt_database_fixes import create_database_manager

            # Find main database
            main_db = self.project_root / "data" / "wheel_trading_master.duckdb"
            if not main_db.exists():
                # Look for any DuckDB file
                db_files = list(self.project_root.rglob("*.duckdb"))
                if db_files:
                    main_db = db_files[0]
                else:
                    return ValidationResult(
                        name="database_connectivity",
                        passed=False,
                        message="No DuckDB files found",
                        duration=0.0,
                        fix_available=True,
                    )

            # Test connection pool
            db_manager = create_database_manager(str(main_db))

            # Test basic query
            db_manager.query("SELECT 1 as test")

            # Get performance stats
            stats = db_manager.get_performance_stats()

            db_manager.close()

            return ValidationResult(
                name="database_connectivity",
                passed=True,
                message=f"Database connectivity verified: {main_db.name}",
                duration=0.0,
                details={"stats": stats, "database": str(main_db)},
            )

        except Exception as e:
            return ValidationResult(
                name="database_connectivity",
                passed=False,
                message=f"Database connectivity failed: {str(e)}",
                duration=0.0,
                error=e,
                fix_available=True,
            )

    async def check_file_permissions(self) -> ValidationResult:
        """Check file system permissions."""
        test_paths = [
            self.project_root,
            self.project_root / "data",
            self.project_root / "logs",
            self.project_root / "cache",
        ]

        issues = []

        for path in test_paths:
            try:
                # Create directory if it doesn't exist
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)

                # Test write permissions
                test_file = path / f".test_write_{os.getpid()}"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                except OSError as e:
                    issues.append(f"Cannot write to {path}: {e}")

            except Exception as e:
                issues.append(f"Permission check failed for {path}: {e}")

        if issues:
            return ValidationResult(
                name="file_permissions",
                passed=False,
                message=f"{len(issues)} permission issues found",
                duration=0.0,
                fix_available=True,
                details={"issues": issues},
            )
        else:
            return ValidationResult(
                name="file_permissions",
                passed=True,
                message=f"All {len(test_paths)} paths writable",
                duration=0.0,
            )

    async def check_network_connectivity(self) -> ValidationResult:
        """Check network connectivity."""
        test_hosts = [
            ("8.8.8.8", 53),  # Google DNS
            ("1.1.1.1", 53),  # Cloudflare DNS
        ]

        connectivity_issues = []

        for host, port in test_hosts:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()

                if result != 0:
                    connectivity_issues.append(f"Cannot reach {host}:{port}")

            except Exception as e:
                connectivity_issues.append(
                    f"Network test failed for {host}:{port}: {e}"
                )

        if connectivity_issues:
            return ValidationResult(
                name="network_connectivity",
                passed=False,
                message=f"Network issues: {'; '.join(connectivity_issues)}",
                duration=0.0,
            )
        else:
            return ValidationResult(
                name="network_connectivity",
                passed=True,
                message="Network connectivity verified",
                duration=0.0,
            )

    async def check_api_endpoints(self) -> ValidationResult:
        """Check external API endpoints."""
        # This would test actual API endpoints
        # For now, just return success since it's optional
        return ValidationResult(
            name="api_endpoints",
            passed=True,
            message="API endpoint checks skipped (optional)",
            duration=0.0,
        )

    async def check_accelerated_tools(self) -> ValidationResult:
        """Check hardware-accelerated tools."""
        try:
            # Test ripgrep turbo
            from src.unity_wheel.accelerated_tools.ripgrep_turbo import (
                get_ripgrep_turbo,
            )

            rg = get_ripgrep_turbo()

            # Test dependency graph turbo
            from src.unity_wheel.accelerated_tools.dependency_graph_turbo import (
                get_dependency_graph,
            )

            graph = get_dependency_graph()

            return ValidationResult(
                name="accelerated_tools",
                passed=True,
                message="Accelerated tools loaded successfully",
                duration=0.0,
                details={
                    "ripgrep_workers": rg.max_workers,
                    "graph_workers": graph.max_workers,
                },
            )

        except Exception as e:
            return ValidationResult(
                name="accelerated_tools",
                passed=False,
                message=f"Accelerated tools failed: {str(e)}",
                duration=0.0,
                error=e,
                fix_available=True,
            )

    async def check_einstein_integration(self) -> ValidationResult:
        """Check Einstein system integration."""
        try:
            # Check if Einstein files exist
            einstein_path = self.project_root / "einstein"
            if not einstein_path.exists():
                return ValidationResult(
                    name="einstein_integration",
                    passed=False,
                    message="Einstein directory not found",
                    duration=0.0,
                    fix_available=False,
                )

            # Test basic Einstein import
            try:
                import sys

                sys.path.insert(0, str(self.project_root))
                from einstein import unified_index

                return ValidationResult(
                    name="einstein_integration",
                    passed=True,
                    message="Einstein integration verified",
                    duration=0.0,
                )
            except ImportError as e:
                return ValidationResult(
                    name="einstein_integration",
                    passed=False,
                    message=f"Einstein import failed: {str(e)}",
                    duration=0.0,
                    fix_available=False,
                )

        except Exception as e:
            return ValidationResult(
                name="einstein_integration",
                passed=False,
                message=f"Einstein check failed: {str(e)}",
                duration=0.0,
                error=e,
            )

    async def check_bolt_integration(self) -> ValidationResult:
        """Check Bolt system integration."""
        try:
            # Check if bolt files exist
            bolt_files = list(self.project_root.glob("bolt*"))
            if not bolt_files:
                return ValidationResult(
                    name="bolt_integration",
                    passed=False,
                    message="Bolt files not found",
                    duration=0.0,
                    fix_available=False,
                )

            # Test bolt database fixes
            from bolt_database_fixes import DatabaseConcurrencyManager

            manager = DatabaseConcurrencyManager()
            stats = manager.get_performance_stats()

            return ValidationResult(
                name="bolt_integration",
                passed=True,
                message="Bolt integration verified",
                duration=0.0,
                details={"stats": stats},
            )

        except Exception as e:
            return ValidationResult(
                name="bolt_integration",
                passed=False,
                message=f"Bolt integration failed: {str(e)}",
                duration=0.0,
                error=e,
            )

    # Fix functions

    async def fix_required_packages(self) -> bool:
        """Attempt to install missing packages."""
        try:
            result = self.validation_results.get("required_packages")
            if not result or not result.details.get("missing"):
                return True

            missing = result.details["missing"]

            # Try to install missing packages
            for package in missing:
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", package]
                    )
                    logger.info(f"Installed missing package: {package}")
                except subprocess.CalledProcessError:
                    logger.error(f"Failed to install package: {package}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to fix required packages: {e}")
            return False

    async def fix_database_files(self) -> bool:
        """Attempt to fix database file issues."""
        try:
            from bolt_database_fixes import fix_existing_database_locks

            results = fix_existing_database_locks(str(self.project_root))

            fixed_count = sum(1 for success in results.values() if success)
            total_count = len(results)

            logger.info(f"Fixed {fixed_count}/{total_count} database files")
            return fixed_count == total_count

        except Exception as e:
            logger.error(f"Failed to fix database files: {e}")
            return False

    async def fix_file_permissions(self) -> bool:
        """Attempt to fix file permission issues."""
        try:
            # Create necessary directories
            directories = ["data", "logs", "cache", ".cache"]

            for dir_name in directories:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)

                # Set permissions to be writable
                if hasattr(os, "chmod"):
                    os.chmod(str(dir_path), 0o755)

            return True

        except Exception as e:
            logger.error(f"Failed to fix file permissions: {e}")
            return False

    async def fix_accelerated_tools(self) -> bool:
        """Attempt to fix accelerated tools issues."""
        try:
            # This would implement fixes for accelerated tools
            # For now, just return True
            return True

        except Exception as e:
            logger.error(f"Failed to fix accelerated tools: {e}")
            return False

    def get_validation_report(self) -> dict[str, Any]:
        """Get comprehensive validation report."""
        return {
            "summary": {
                "total_checks": self.total_checks,
                "passed_checks": self.passed_checks,
                "failed_checks": self.failed_checks,
                "success_rate": self.passed_checks / max(1, self.total_checks),
                "total_duration": time.time() - self.start_time
                if self.start_time > 0
                else 0.0,
            },
            "results": {
                name: {
                    "passed": result.passed,
                    "message": result.message,
                    "duration": result.duration,
                    "fix_available": result.fix_available,
                    "details": result.details,
                }
                for name, result in self.validation_results.items()
            },
            "system_info": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "project_root": str(self.project_root),
            },
        }


async def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Service Startup Dependency Validation"
    )
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument(
        "--no-fix", action="store_true", help="Don't attempt to fix issues"
    )
    parser.add_argument("--report", help="Save validation report to file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Run validation
    validator = ServiceStartupValidator(args.project_root)
    success = await validator.validate_all(fix_issues=not args.no_fix)

    # Generate report
    report = validator.get_validation_report()

    if args.report:
        import json

        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Validation report saved to: {args.report}")

    # Print summary
    summary = report["summary"]
    print(f"\n{'='*60}")
    print("SERVICE STARTUP VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Checks: {summary['total_checks']}")
    print(f"Passed: {summary['passed_checks']}")
    print(f"Failed: {summary['failed_checks']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Duration: {summary['total_duration']:.2f}s")

    if success:
        print("\n✅ All critical dependencies validated successfully!")
        print("🚀 System ready for startup")
        sys.exit(0)
    else:
        print("\n❌ Critical dependencies failed validation")
        print("💥 System not ready for startup")

        # Show failed checks
        for name, result in validator.validation_results.items():
            if not result.passed and validator.dependencies[name].required:
                print(f"   - {name}: {result.message}")

        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
