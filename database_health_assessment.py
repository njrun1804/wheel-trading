#!/usr/bin/env python3
"""
Database Health Assessment - Complete integrity and performance validation.

Agent 7 deliverable for comprehensive database analysis.
"""

import asyncio
import json
import logging
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Import our database fixes
try:
    from bolt_database_fixes import (
        ConcurrentDatabase,
        DatabaseConfig,
        fix_existing_database_locks,
    )

    CONCURRENT_DB_AVAILABLE = True
except ImportError:
    CONCURRENT_DB_AVAILABLE = False

try:
    from bolt.database_connection_manager import (
        DatabaseConnectionPool,
        get_database_pool,
    )

    CONNECTION_POOL_AVAILABLE = True
except ImportError:
    CONNECTION_POOL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseInfo:
    """Information about a database file."""

    path: str
    size_mb: float
    modified: datetime
    db_type: str
    accessible: bool = False
    tables: list[str] = field(default_factory=list)
    schema_info: dict[str, Any] = field(default_factory=dict)
    performance_stats: dict[str, Any] = field(default_factory=dict)
    integrity_check: dict[str, Any] = field(default_factory=dict)
    error_msg: str | None = None


@dataclass
class SchemaConsistencyResult:
    """Results of schema consistency analysis."""

    consistent: bool
    primary_schema: dict[str, Any] | None
    inconsistencies: list[dict[str, Any]] = field(default_factory=list)
    missing_tables: list[str] = field(default_factory=list)
    extra_tables: list[str] = field(default_factory=list)


@dataclass
class PerformanceTestResult:
    """Results of database performance testing."""

    db_path: str
    query_latency_ms: float
    concurrent_access_success: bool
    connection_pool_test: bool
    vacuum_time_ms: float
    insert_throughput_ops_per_sec: float
    error_msg: str | None = None


@dataclass
class DatabaseHealthReport:
    """Complete database health assessment report."""

    timestamp: datetime
    databases_found: int
    databases_accessible: int
    total_size_mb: float
    schema_consistency: SchemaConsistencyResult
    performance_results: list[PerformanceTestResult]
    security_assessment: dict[str, Any]
    backup_status: dict[str, Any]
    recommendations: list[str]
    critical_issues: list[str]


class DatabaseHealthAssessor:
    """Comprehensive database health assessment tool."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.databases: list[DatabaseInfo] = []
        self.report_dir = self.project_root / "database_reports"
        self.report_dir.mkdir(exist_ok=True)

    async def run_assessment(self) -> DatabaseHealthReport:
        """Run complete database health assessment."""
        logger.info("Starting comprehensive database health assessment...")

        # 1. Discovery phase
        await self._discover_databases()

        # 2. Basic health checks
        await self._check_database_accessibility()

        # 3. Schema analysis
        schema_consistency = await self._analyze_schema_consistency()

        # 4. Performance testing
        performance_results = await self._run_performance_tests()

        # 5. Security assessment
        security_assessment = await self._assess_security()

        # 6. Backup validation
        backup_status = await self._validate_backups()

        # 7. Generate recommendations
        recommendations, critical_issues = await self._generate_recommendations()

        # Create comprehensive report
        report = DatabaseHealthReport(
            timestamp=datetime.now(),
            databases_found=len(self.databases),
            databases_accessible=sum(1 for db in self.databases if db.accessible),
            total_size_mb=sum(db.size_mb for db in self.databases),
            schema_consistency=schema_consistency,
            performance_results=performance_results,
            security_assessment=security_assessment,
            backup_status=backup_status,
            recommendations=recommendations,
            critical_issues=critical_issues,
        )

        await self._save_report(report)
        return report

    async def _discover_databases(self):
        """Discover all database files in the project."""
        logger.info("Discovering database files...")

        patterns = ["*.db", "*.duckdb", "*.sqlite", "*.sqlite3"]

        for pattern in patterns:
            for db_path in self.project_root.rglob(pattern):
                if db_path.is_file():
                    try:
                        stat = db_path.stat()
                        db_type = self._detect_db_type(db_path)

                        db_info = DatabaseInfo(
                            path=str(db_path),
                            size_mb=stat.st_size / (1024 * 1024),
                            modified=datetime.fromtimestamp(stat.st_mtime),
                            db_type=db_type,
                        )

                        self.databases.append(db_info)

                    except Exception as e:
                        logger.warning(f"Error processing {db_path}: {e}")

        logger.info(f"Found {len(self.databases)} database files")

    def _detect_db_type(self, db_path: Path) -> str:
        """Detect database type from file path and content."""
        if db_path.suffix.lower() in [".duckdb"]:
            return "duckdb"
        elif db_path.suffix.lower() in [".db", ".sqlite", ".sqlite3"]:
            return "sqlite"
        else:
            # Try to detect from file content
            try:
                with open(db_path, "rb") as f:
                    header = f.read(16)
                    if header.startswith(b"SQLite format 3"):
                        return "sqlite"
                    elif b"DUCK" in header:
                        return "duckdb"
            except Exception:
                pass

        return "unknown"

    async def _check_database_accessibility(self):
        """Check if databases are accessible and gather basic info."""
        logger.info("Checking database accessibility...")

        tasks = []
        for db_info in self.databases:
            task = asyncio.create_task(self._check_single_database(db_info))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_single_database(self, db_info: DatabaseInfo):
        """Check accessibility of a single database."""
        try:
            if db_info.db_type == "duckdb" and DUCKDB_AVAILABLE:
                await self._check_duckdb(db_info)
            elif db_info.db_type == "sqlite":
                await self._check_sqlite(db_info)
            else:
                db_info.error_msg = f"Unsupported database type: {db_info.db_type}"

        except Exception as e:
            db_info.error_msg = str(e)
            logger.error(f"Error checking {db_info.path}: {e}")

    async def _check_duckdb(self, db_info: DatabaseInfo):
        """Check DuckDB database."""
        loop = asyncio.get_event_loop()

        def _sync_check():
            try:
                conn = duckdb.connect(db_info.path, read_only=True)

                # Get tables
                tables_result = conn.execute("SHOW TABLES").fetchall()
                db_info.tables = [row[0] for row in tables_result]

                # Get schema info
                db_info.schema_info = {}
                for table in db_info.tables:
                    try:
                        schema = conn.execute(f"DESCRIBE {table}").fetchall()
                        db_info.schema_info[table] = [
                            {"name": row[0], "type": row[1], "null": row[2]}
                            for row in schema
                        ]
                    except Exception as e:
                        logger.warning(f"Could not describe table {table}: {e}")

                # Basic integrity check
                try:
                    conn.execute("SELECT 1").fetchone()
                    db_info.integrity_check["basic_query"] = True
                except Exception as e:
                    db_info.integrity_check["basic_query"] = False
                    db_info.integrity_check["basic_query_error"] = str(e)

                conn.close()
                db_info.accessible = True

            except Exception as e:
                db_info.error_msg = str(e)

        await loop.run_in_executor(None, _sync_check)

    async def _check_sqlite(self, db_info: DatabaseInfo):
        """Check SQLite database."""
        loop = asyncio.get_event_loop()

        def _sync_check():
            try:
                # Use concurrent database if available
                if CONCURRENT_DB_AVAILABLE:
                    db = ConcurrentDatabase(db_info.path)

                    # Get tables
                    tables_result = db.query(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    db_info.tables = [row["name"] for row in tables_result]

                    # Get schema info
                    db_info.schema_info = db.get_schema_info()

                    # Basic integrity check
                    try:
                        db.query("SELECT 1")
                        db_info.integrity_check["basic_query"] = True
                    except Exception as e:
                        db_info.integrity_check["basic_query"] = False
                        db_info.integrity_check["basic_query_error"] = str(e)

                    # SQLite-specific integrity check
                    try:
                        integrity_result = db.query("PRAGMA integrity_check")
                        db_info.integrity_check["pragma_check"] = (
                            integrity_result[0] if integrity_result else "Unknown"
                        )
                    except Exception as e:
                        db_info.integrity_check["pragma_check_error"] = str(e)

                    db.close()

                else:
                    # Fallback to direct connection
                    conn = sqlite3.connect(db_info.path, timeout=30)
                    conn.row_factory = sqlite3.Row

                    # Get tables
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    db_info.tables = [row[0] for row in cursor.fetchall()]

                    # Get schema info
                    db_info.schema_info = {}
                    for table in db_info.tables:
                        try:
                            cursor = conn.execute(f"PRAGMA table_info({table})")
                            db_info.schema_info[table] = [
                                dict(row) for row in cursor.fetchall()
                            ]
                        except Exception as e:
                            logger.warning(
                                f"Could not get schema for table {table}: {e}"
                            )

                    conn.close()

                db_info.accessible = True

            except Exception as e:
                db_info.error_msg = str(e)

        await loop.run_in_executor(None, _sync_check)

    async def _analyze_schema_consistency(self) -> SchemaConsistencyResult:
        """Analyze schema consistency across similar databases."""
        logger.info("Analyzing schema consistency...")

        # Group databases by type and purpose
        trading_dbs = [
            db
            for db in self.databases
            if "trading" in db.path.lower() and db.accessible
        ]
        [db for db in self.databases if "cache" in db.path.lower() and db.accessible]

        inconsistencies = []

        # Check trading databases consistency
        if len(trading_dbs) > 1:
            primary_schema = trading_dbs[0].schema_info
            for db in trading_dbs[1:]:
                for table in primary_schema:
                    if table not in db.schema_info:
                        inconsistencies.append(
                            {
                                "type": "missing_table",
                                "database": db.path,
                                "table": table,
                            }
                        )
                    elif primary_schema[table] != db.schema_info[table]:
                        inconsistencies.append(
                            {
                                "type": "schema_mismatch",
                                "database": db.path,
                                "table": table,
                                "expected": primary_schema[table],
                                "actual": db.schema_info[table],
                            }
                        )

        return SchemaConsistencyResult(
            consistent=len(inconsistencies) == 0,
            primary_schema=trading_dbs[0].schema_info if trading_dbs else None,
            inconsistencies=inconsistencies,
        )

    async def _run_performance_tests(self) -> list[PerformanceTestResult]:
        """Run performance tests on accessible databases."""
        logger.info("Running performance tests...")

        results = []

        # Test a sample of databases to avoid overwhelming the system
        test_databases = [db for db in self.databases if db.accessible][:10]

        tasks = []
        for db_info in test_databases:
            task = asyncio.create_task(self._test_database_performance(db_info))
            tasks.append(task)

        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed_results:
            if isinstance(result, PerformanceTestResult):
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Performance test failed: {result}")

        return results

    async def _test_database_performance(
        self, db_info: DatabaseInfo
    ) -> PerformanceTestResult:
        """Test performance of a single database."""
        result = PerformanceTestResult(
            db_path=db_info.path,
            query_latency_ms=0,
            concurrent_access_success=False,
            connection_pool_test=False,
            vacuum_time_ms=0,
            insert_throughput_ops_per_sec=0,
        )

        try:
            # Test basic query latency
            start_time = time.time()
            if db_info.db_type == "duckdb" and DUCKDB_AVAILABLE:
                conn = duckdb.connect(db_info.path, read_only=True)
                conn.execute("SELECT 1").fetchone()
                conn.close()
            else:
                conn = sqlite3.connect(db_info.path, timeout=10)
                conn.execute("SELECT 1").fetchone()
                conn.close()

            result.query_latency_ms = (time.time() - start_time) * 1000

            # Test concurrent access if concurrent database is available
            if CONCURRENT_DB_AVAILABLE and db_info.db_type == "sqlite":
                try:
                    await self._test_concurrent_access(db_info.path)
                    result.concurrent_access_success = True
                except Exception as e:
                    logger.warning(
                        f"Concurrent access test failed for {db_info.path}: {e}"
                    )

            # Test connection pool if available
            if CONNECTION_POOL_AVAILABLE:
                try:
                    await self._test_connection_pool(db_info.path, db_info.db_type)
                    result.connection_pool_test = True
                except Exception as e:
                    logger.warning(
                        f"Connection pool test failed for {db_info.path}: {e}"
                    )

        except Exception as e:
            result.error_msg = str(e)

        return result

    async def _test_concurrent_access(self, db_path: str):
        """Test concurrent database access."""

        async def concurrent_query():
            db = ConcurrentDatabase(db_path)
            try:
                db.query("SELECT 1")
            finally:
                db.close()

        # Run 5 concurrent queries
        tasks = [concurrent_query() for _ in range(5)]
        await asyncio.gather(*tasks)

    async def _test_connection_pool(self, db_path: str, db_type: str):
        """Test connection pool functionality."""
        pool = get_database_pool(db_path, pool_size=4, db_type=db_type)
        await pool.initialize()

        try:
            # Test multiple concurrent queries through pool
            async def pool_query():
                return await pool.execute_query("SELECT 1")

            tasks = [pool_query() for _ in range(10)]
            await asyncio.gather(*tasks)

        finally:
            await pool.close()

    async def _assess_security(self) -> dict[str, Any]:
        """Assess database security configurations."""
        logger.info("Assessing database security...")

        security_report = {
            "file_permissions": [],
            "access_controls": [],
            "encryption_status": "Not implemented",
            "backup_security": "Unknown",
        }

        for db_info in self.databases:
            db_path = Path(db_info.path)

            # Check file permissions
            try:
                stat = db_path.stat()
                mode = oct(stat.st_mode)[-3:]

                security_report["file_permissions"].append(
                    {
                        "path": str(db_path),
                        "permissions": mode,
                        "owner_readable": bool(stat.st_mode & 0o400),
                        "owner_writable": bool(stat.st_mode & 0o200),
                        "group_readable": bool(stat.st_mode & 0o040),
                        "group_writable": bool(stat.st_mode & 0o020),
                        "other_readable": bool(stat.st_mode & 0o004),
                        "other_writable": bool(stat.st_mode & 0o002),
                    }
                )

            except Exception as e:
                logger.warning(f"Could not check permissions for {db_path}: {e}")

        return security_report

    async def _validate_backups(self) -> dict[str, Any]:
        """Validate backup procedures and status."""
        logger.info("Validating backup status...")

        backup_dirs = [
            self.project_root / "backups",
            self.project_root / "data" / "archive",
            self.project_root / "data" / "backups",
        ]

        backup_status = {
            "backup_dirs_found": [],
            "recent_backups": [],
            "backup_coverage": 0,
            "oldest_backup": None,
            "newest_backup": None,
        }

        for backup_dir in backup_dirs:
            if backup_dir.exists():
                backup_status["backup_dirs_found"].append(str(backup_dir))

                # Find backup files
                for backup_file in backup_dir.rglob("*.db*"):
                    if backup_file.is_file():
                        stat = backup_file.stat()
                        backup_info = {
                            "path": str(backup_file),
                            "size_mb": stat.st_size / (1024 * 1024),
                            "modified": datetime.fromtimestamp(
                                stat.st_mtime
                            ).isoformat(),
                        }
                        backup_status["recent_backups"].append(backup_info)

        # Calculate backup coverage
        if self.databases and backup_status["recent_backups"]:
            backup_status["backup_coverage"] = min(
                len(backup_status["recent_backups"]) / len(self.databases), 1.0
            )

        return backup_status

    async def _generate_recommendations(self) -> tuple[list[str], list[str]]:
        """Generate recommendations and identify critical issues."""
        recommendations = []
        critical_issues = []

        # Check for inaccessible databases
        inaccessible_dbs = [db for db in self.databases if not db.accessible]
        if inaccessible_dbs:
            critical_issues.append(
                f"{len(inaccessible_dbs)} databases are inaccessible"
            )
            recommendations.append("Fix database accessibility issues immediately")

        # Check for large databases
        large_dbs = [db for db in self.databases if db.size_mb > 100]
        if large_dbs:
            recommendations.append(
                f"Consider archiving or optimizing {len(large_dbs)} large databases (>100MB)"
            )

        # Check for old databases
        old_threshold = datetime.now() - timedelta(days=30)
        old_dbs = [db for db in self.databases if db.modified < old_threshold]
        if old_dbs:
            recommendations.append(
                f"Review {len(old_dbs)} databases not modified in 30+ days"
            )

        # Check total disk usage
        total_size = sum(db.size_mb for db in self.databases)
        if total_size > 1000:  # >1GB
            recommendations.append(
                f"Total database size is {total_size:.1f}MB - consider cleanup"
            )

        # Connection pool recommendations
        if not CONNECTION_POOL_AVAILABLE:
            recommendations.append(
                "Enable connection pooling for better concurrent access"
            )

        # Security recommendations
        world_readable_dbs = []
        for db in self.databases:
            try:
                stat = Path(db.path).stat()
                if stat.st_mode & 0o004:  # World readable
                    world_readable_dbs.append(db.path)
            except Exception:
                pass

        if world_readable_dbs:
            critical_issues.append(
                f"{len(world_readable_dbs)} databases are world-readable"
            )
            recommendations.append("Fix database file permissions for security")

        return recommendations, critical_issues

    async def _save_report(self, report: DatabaseHealthReport):
        """Save the comprehensive report."""
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        report_file = self.report_dir / f"database_health_report_{timestamp}.json"

        # Convert to serializable format
        report_dict = asdict(report)
        report_dict["timestamp"] = report.timestamp.isoformat()

        with open(report_file, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        logger.info(f"Health report saved to {report_file}")

        # Also save a summary report
        summary_file = self.report_dir / f"database_health_summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write("DATABASE HEALTH ASSESSMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Assessment Time: {report.timestamp}\n")
            f.write(f"Databases Found: {report.databases_found}\n")
            f.write(f"Databases Accessible: {report.databases_accessible}\n")
            f.write(f"Total Size: {report.total_size_mb:.1f} MB\n")
            f.write(f"Schema Consistent: {report.schema_consistency.consistent}\n\n")

            if report.critical_issues:
                f.write("CRITICAL ISSUES:\n")
                for issue in report.critical_issues:
                    f.write(f"  ❌ {issue}\n")
                f.write("\n")

            if report.recommendations:
                f.write("RECOMMENDATIONS:\n")
                for rec in report.recommendations:
                    f.write(f"  💡 {rec}\n")
                f.write("\n")

            f.write("PERFORMANCE SUMMARY:\n")
            if report.performance_results:
                avg_latency = sum(
                    r.query_latency_ms
                    for r in report.performance_results
                    if r.query_latency_ms > 0
                ) / len(report.performance_results)
                concurrent_success_rate = sum(
                    1 for r in report.performance_results if r.concurrent_access_success
                ) / len(report.performance_results)
                f.write(f"  Average Query Latency: {avg_latency:.2f}ms\n")
                f.write(
                    f"  Concurrent Access Success Rate: {concurrent_success_rate:.1%}\n"
                )
            else:
                f.write("  No performance tests completed\n")

        logger.info(f"Summary report saved to {summary_file}")


async def main():
    """Main function to run database health assessment."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."

    assessor = DatabaseHealthAssessor(project_root)

    try:
        report = await assessor.run_assessment()

        print("\n" + "=" * 60)
        print("DATABASE HEALTH ASSESSMENT COMPLETE")
        print("=" * 60)
        print(f"Databases Found: {report.databases_found}")
        print(f"Databases Accessible: {report.databases_accessible}")
        print(f"Total Size: {report.total_size_mb:.1f} MB")
        print(f"Schema Consistent: {report.schema_consistency.consistent}")

        if report.critical_issues:
            print(f"\n❌ CRITICAL ISSUES ({len(report.critical_issues)}):")
            for issue in report.critical_issues:
                print(f"   {issue}")

        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(report.recommendations)}):")
            for rec in report.recommendations[:5]:  # Show top 5
                print(f"   {rec}")
            if len(report.recommendations) > 5:
                print(f"   ... and {len(report.recommendations) - 5} more")

        print(f"\nDetailed reports saved to: {assessor.report_dir}")
        print("=" * 60)

        # Return appropriate exit code
        return 1 if report.critical_issues else 0

    except Exception as e:
        logger.error(f"Assessment failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
