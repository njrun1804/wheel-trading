#!/usr/bin/env python3
"""
Standalone Database Integrity and Performance Assessment
Agent 7 - Database Analysis Tool
"""

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
class DatabaseHealthReport:
    """Complete database health assessment report."""

    timestamp: datetime
    databases_found: int
    databases_accessible: int
    total_size_mb: float
    recommendations: list[str]
    critical_issues: list[str]
    database_details: list[dict[str, Any]]


class SimpleDatabaseAssessor:
    """Simplified database health assessment tool."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.databases: list[DatabaseInfo] = []
        self.report_dir = self.project_root / "database_reports"
        self.report_dir.mkdir(exist_ok=True)

    def discover_databases(self):
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
        """Detect database type from file path."""
        if db_path.suffix.lower() in [".duckdb"]:
            return "duckdb"
        elif db_path.suffix.lower() in [".db", ".sqlite", ".sqlite3"]:
            return "sqlite"
        return "unknown"

    def check_database_accessibility(self):
        """Check if databases are accessible and gather basic info."""
        logger.info("Checking database accessibility...")

        for db_info in self.databases:
            try:
                if db_info.db_type == "duckdb" and DUCKDB_AVAILABLE:
                    self._check_duckdb(db_info)
                elif db_info.db_type == "sqlite":
                    self._check_sqlite(db_info)
                else:
                    db_info.error_msg = f"Unsupported database type: {db_info.db_type}"

            except Exception as e:
                db_info.error_msg = str(e)
                logger.error(f"Error checking {db_info.path}: {e}")

    def _check_duckdb(self, db_info: DatabaseInfo):
        """Check DuckDB database."""
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
                start_time = time.time()
                conn.execute("SELECT 1").fetchone()
                query_time = (time.time() - start_time) * 1000
                db_info.integrity_check["basic_query"] = True
                db_info.performance_stats["query_latency_ms"] = query_time
            except Exception as e:
                db_info.integrity_check["basic_query"] = False
                db_info.integrity_check["basic_query_error"] = str(e)

            # Get table row counts
            db_info.performance_stats["table_counts"] = {}
            for table in db_info.tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    db_info.performance_stats["table_counts"][table] = count
                except Exception as e:
                    logger.warning(f"Could not count rows in {table}: {e}")

            conn.close()
            db_info.accessible = True

        except Exception as e:
            db_info.error_msg = str(e)

    def _check_sqlite(self, db_info: DatabaseInfo):
        """Check SQLite database."""
        try:
            conn = sqlite3.connect(db_info.path, timeout=30)
            conn.row_factory = sqlite3.Row

            # Get tables
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
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
                    logger.warning(f"Could not get schema for table {table}: {e}")

            # Basic integrity check
            try:
                start_time = time.time()
                conn.execute("SELECT 1").fetchone()
                query_time = (time.time() - start_time) * 1000
                db_info.integrity_check["basic_query"] = True
                db_info.performance_stats["query_latency_ms"] = query_time
            except Exception as e:
                db_info.integrity_check["basic_query"] = False
                db_info.integrity_check["basic_query_error"] = str(e)

            # SQLite-specific integrity check
            try:
                integrity_result = conn.execute("PRAGMA integrity_check").fetchone()
                db_info.integrity_check["pragma_check"] = (
                    integrity_result[0] if integrity_result else "Unknown"
                )
            except Exception as e:
                db_info.integrity_check["pragma_check_error"] = str(e)

            # Get table row counts
            db_info.performance_stats["table_counts"] = {}
            for table in db_info.tables:
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    db_info.performance_stats["table_counts"][table] = count
                except Exception as e:
                    logger.warning(f"Could not count rows in {table}: {e}")

            conn.close()
            db_info.accessible = True

        except Exception as e:
            db_info.error_msg = str(e)

    def generate_recommendations(self) -> tuple[list[str], list[str]]:
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

        # Check for databases with integrity issues
        integrity_issues = [
            db
            for db in self.databases
            if not db.integrity_check.get("basic_query", True)
        ]
        if integrity_issues:
            critical_issues.append(
                f"{len(integrity_issues)} databases have integrity issues"
            )

        # Check for duplicate databases (same name in different locations)
        db_names = {}
        for db in self.databases:
            name = Path(db.path).name
            if name in db_names:
                db_names[name].append(db.path)
            else:
                db_names[name] = [db.path]

        duplicates = {name: paths for name, paths in db_names.items() if len(paths) > 1}
        if duplicates:
            recommendations.append(
                f"Found duplicate databases: {list(duplicates.keys())}"
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

    def run_assessment(self) -> DatabaseHealthReport:
        """Run complete database health assessment."""
        logger.info("Starting database integrity assessment...")

        # 1. Discovery phase
        self.discover_databases()

        # 2. Basic health checks
        self.check_database_accessibility()

        # 3. Generate recommendations
        recommendations, critical_issues = self.generate_recommendations()

        # Create comprehensive report
        report = DatabaseHealthReport(
            timestamp=datetime.now(),
            databases_found=len(self.databases),
            databases_accessible=sum(1 for db in self.databases if db.accessible),
            total_size_mb=sum(db.size_mb for db in self.databases),
            recommendations=recommendations,
            critical_issues=critical_issues,
            database_details=[asdict(db) for db in self.databases],
        )

        self.save_report(report)
        return report

    def save_report(self, report: DatabaseHealthReport):
        """Save the comprehensive report."""
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        report_file = self.report_dir / f"database_integrity_report_{timestamp}.json"

        # Convert to serializable format
        report_dict = asdict(report)
        report_dict["timestamp"] = report.timestamp.isoformat()

        with open(report_file, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        logger.info(f"Integrity report saved to {report_file}")

        # Also save a summary report
        summary_file = self.report_dir / f"database_integrity_summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write("DATABASE INTEGRITY ASSESSMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Assessment Time: {report.timestamp}\n")
            f.write(f"Databases Found: {report.databases_found}\n")
            f.write(f"Databases Accessible: {report.databases_accessible}\n")
            f.write(f"Total Size: {report.total_size_mb:.1f} MB\n\n")

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

            # Database details
            f.write("DATABASE DETAILS:\n")
            for db in report.database_details:
                status = "✅ OK" if db["accessible"] else "❌ ERROR"
                f.write(
                    f"  {status} {db['path']} ({db['size_mb']:.1f}MB, {db['db_type']})\n"
                )
                if db["error_msg"]:
                    f.write(f"      Error: {db['error_msg']}\n")
                if db["tables"]:
                    f.write(f"      Tables: {', '.join(db['tables'][:5])}\n")
                    if len(db["tables"]) > 5:
                        f.write(f"      ... and {len(db['tables']) - 5} more\n")

        logger.info(f"Summary report saved to {summary_file}")


def main():
    """Main function to run database integrity assessment."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."

    assessor = SimpleDatabaseAssessor(project_root)

    try:
        report = assessor.run_assessment()

        print("\n" + "=" * 60)
        print("DATABASE INTEGRITY ASSESSMENT COMPLETE")
        print("=" * 60)
        print(f"Databases Found: {report.databases_found}")
        print(f"Databases Accessible: {report.databases_accessible}")
        print(f"Total Size: {report.total_size_mb:.1f} MB")

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
    exit_code = main()
    sys.exit(exit_code)
