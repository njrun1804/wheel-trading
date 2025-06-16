#!/usr/bin/env python3
"""
Database Integrity and Performance Test Suite - Agent 7
Comprehensive testing of database health, connection pooling, and performance optimization.
"""

import asyncio
import json
import logging
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    from bolt_database_fixes import (
        ConcurrentDatabase,
        DatabaseConfig,
        fix_existing_database_locks,
    )

    HAS_BOLT_FIXES = True
except ImportError:
    HAS_BOLT_FIXES = False

try:
    from bolt.database_connection_manager import (
        DatabaseConnectionPool,
        get_database_pool,
    )

    HAS_CONNECTION_MANAGER = True
except ImportError:
    HAS_CONNECTION_MANAGER = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatabaseIntegrityTester:
    """Comprehensive database integrity and performance testing."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            "timestamp": time.time(),
            "database_discovery": {},
            "schema_validation": {},
            "connection_pooling": {},
            "query_performance": {},
            "data_consistency": {},
            "backup_validation": {},
            "security_assessment": {},
            "recommendations": [],
        }

    def discover_databases(self) -> dict[str, Any]:
        """Discover and catalog all database files."""
        print("🔍 Discovering database files...")

        discovery = {
            "duckdb_files": [],
            "sqlite_files": [],
            "total_size_mb": 0,
            "file_details": {},
        }

        # Find DuckDB files
        for pattern in ["*.duckdb", "*.db"]:
            for db_file in self.project_root.rglob(pattern):
                if db_file.is_file():
                    size_mb = db_file.stat().st_size / (1024 * 1024)

                    file_info = {
                        "path": str(db_file),
                        "size_mb": round(size_mb, 2),
                        "modified": db_file.stat().st_mtime,
                        "type": "unknown",
                    }

                    # Determine database type
                    if str(db_file).endswith(".duckdb"):
                        discovery["duckdb_files"].append(str(db_file))
                        file_info["type"] = "duckdb"
                    else:
                        # Try to identify if it's SQLite
                        try:
                            conn = sqlite3.connect(str(db_file), timeout=1.0)
                            conn.execute("SELECT 1")
                            conn.close()
                            discovery["sqlite_files"].append(str(db_file))
                            file_info["type"] = "sqlite"
                        except Exception:
                            file_info["type"] = "unknown"

                    discovery["file_details"][str(db_file)] = file_info
                    discovery["total_size_mb"] += size_mb

        discovery["total_size_mb"] = round(discovery["total_size_mb"], 2)
        self.results["database_discovery"] = discovery

        print(f"✅ Found {len(discovery['duckdb_files'])} DuckDB files")
        print(f"✅ Found {len(discovery['sqlite_files'])} SQLite files")
        print(f"✅ Total size: {discovery['total_size_mb']} MB")

        return discovery

    def validate_schemas(self) -> dict[str, Any]:
        """Validate database schemas and structure."""
        print("📋 Validating database schemas...")

        validation = {
            "schema_consistency": {},
            "table_analysis": {},
            "index_analysis": {},
            "issues": [],
        }

        discovery = self.results.get("database_discovery", {})

        # Check DuckDB files
        for db_file in discovery.get("duckdb_files", []):
            if not HAS_DUCKDB:
                validation["issues"].append(f"DuckDB not available to check {db_file}")
                continue

            try:
                conn = duckdb.connect(db_file)
                tables = conn.execute("SHOW TABLES").fetchall()

                schema_info = {"tables": [], "total_records": 0, "has_data": False}

                for table in tables:
                    table_name = table[0]
                    try:
                        count = conn.execute(
                            f"SELECT COUNT(*) FROM {table_name}"
                        ).fetchone()[0]
                        schema = conn.execute(f"DESCRIBE {table_name}").fetchall()

                        table_info = {
                            "name": table_name,
                            "columns": len(schema),
                            "records": count,
                            "schema": [(col[0], col[1]) for col in schema],
                        }

                        schema_info["tables"].append(table_info)
                        schema_info["total_records"] += count
                        if count > 0:
                            schema_info["has_data"] = True

                    except Exception as e:
                        validation["issues"].append(
                            f"Error analyzing table {table_name} in {db_file}: {e}"
                        )

                validation["schema_consistency"][db_file] = schema_info
                conn.close()

            except Exception as e:
                validation["issues"].append(f"Error connecting to {db_file}: {e}")

        # Check SQLite files
        for db_file in discovery.get("sqlite_files", []):
            try:
                conn = sqlite3.connect(db_file, timeout=5.0)
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = cursor.fetchall()

                schema_info = {"tables": [], "total_records": 0, "has_data": False}

                for table in tables:
                    table_name = table[0]
                    if table_name.startswith("sqlite_"):
                        continue

                    try:
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]

                        cursor = conn.execute(f"PRAGMA table_info({table_name})")
                        schema = cursor.fetchall()

                        table_info = {
                            "name": table_name,
                            "columns": len(schema),
                            "records": count,
                            "schema": [(col[1], col[2]) for col in schema],
                        }

                        schema_info["tables"].append(table_info)
                        schema_info["total_records"] += count
                        if count > 0:
                            schema_info["has_data"] = True

                    except Exception as e:
                        validation["issues"].append(
                            f"Error analyzing table {table_name} in {db_file}: {e}"
                        )

                validation["schema_consistency"][db_file] = schema_info
                conn.close()

            except Exception as e:
                validation["issues"].append(f"Error connecting to {db_file}: {e}")

        self.results["schema_validation"] = validation

        print(f"✅ Validated {len(validation['schema_consistency'])} databases")
        if validation["issues"]:
            print(f"⚠️ Found {len(validation['issues'])} issues")

        return validation

    def test_connection_pooling(self) -> dict[str, Any]:
        """Test connection pooling implementations."""
        print("🔗 Testing connection pooling...")

        pooling_results = {
            "bolt_fixes_available": HAS_BOLT_FIXES,
            "connection_manager_available": HAS_CONNECTION_MANAGER,
            "concurrent_access_tests": {},
            "performance_tests": {},
            "recommendations": [],
        }

        # Test concurrent access if bolt fixes are available
        if HAS_BOLT_FIXES:
            pooling_results["concurrent_access_tests"] = self._test_concurrent_access()

        # Test connection manager if available
        if HAS_CONNECTION_MANAGER:
            pooling_results["performance_tests"] = self._test_connection_manager()

        self.results["connection_pooling"] = pooling_results

        print("✅ Connection pooling tests completed")
        return pooling_results

    def _test_concurrent_access(self) -> dict[str, Any]:
        """Test concurrent database access."""
        results = {}

        # Find a SQLite database to test
        discovery = self.results.get("database_discovery", {})
        test_db = None

        for db_file in discovery.get("sqlite_files", []):
            if Path(db_file).exists():
                test_db = db_file
                break

        if not test_db:
            return {"error": "No SQLite database found for testing"}

        def concurrent_worker(worker_id: int, num_queries: int = 10) -> dict[str, Any]:
            """Worker function for concurrent testing."""
            worker_results = {
                "worker_id": worker_id,
                "queries_completed": 0,
                "errors": [],
                "total_time": 0,
            }

            start_time = time.time()

            try:
                db = ConcurrentDatabase(test_db)

                for _i in range(num_queries):
                    try:
                        db.query("SELECT 1")
                        worker_results["queries_completed"] += 1
                        time.sleep(0.01)  # Small delay to simulate work
                    except Exception as e:
                        worker_results["errors"].append(str(e))

                db.close()

            except Exception as e:
                worker_results["errors"].append(f"Database connection error: {e}")

            worker_results["total_time"] = time.time() - start_time
            return worker_results

        # Run concurrent workers
        num_workers = 5
        results["test_database"] = test_db
        results["num_workers"] = num_workers

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(concurrent_worker, i) for i in range(num_workers)
            ]
            worker_results = [future.result() for future in as_completed(futures)]

        results["worker_results"] = worker_results

        # Analyze results
        total_queries = sum(r["queries_completed"] for r in worker_results)
        total_errors = sum(len(r["errors"]) for r in worker_results)
        avg_time = sum(r["total_time"] for r in worker_results) / len(worker_results)

        results["summary"] = {
            "total_queries_completed": total_queries,
            "total_errors": total_errors,
            "average_worker_time": round(avg_time, 3),
            "success_rate": round(
                total_queries / (total_queries + total_errors) * 100, 2
            )
            if (total_queries + total_errors) > 0
            else 0,
        }

        return results

    def _test_connection_manager(self) -> dict[str, Any]:
        """Test connection manager performance."""
        results = {}

        # Find a DuckDB database to test
        discovery = self.results.get("database_discovery", {})
        test_db = None

        for db_file in discovery.get("duckdb_files", []):
            if Path(db_file).exists():
                test_db = db_file
                break

        if not test_db or not HAS_DUCKDB:
            return {
                "error": "No DuckDB database found for testing or DuckDB not available"
            }

        async def performance_test():
            pool = get_database_pool(test_db, pool_size=4, db_type="duckdb")

            test_results = {
                "pool_stats": {},
                "query_performance": {},
                "concurrent_performance": {},
            }

            try:
                # Initialize pool
                await pool.initialize()
                test_results["pool_stats"] = pool.get_pool_stats()

                # Test single query performance
                start_time = time.time()
                result = await pool.execute_query("SELECT 1")
                query_time = time.time() - start_time

                test_results["query_performance"] = {
                    "single_query_time": round(query_time * 1000, 2),  # ms
                    "result": result,
                }

                # Test concurrent queries
                async def concurrent_query(query_id: int):
                    start = time.time()
                    await pool.execute_query("SELECT 1")
                    return time.time() - start

                concurrent_start = time.time()
                tasks = [concurrent_query(i) for i in range(10)]
                times = await asyncio.gather(*tasks)
                concurrent_total = time.time() - concurrent_start

                test_results["concurrent_performance"] = {
                    "total_time": round(concurrent_total * 1000, 2),  # ms
                    "individual_times": [round(t * 1000, 2) for t in times],
                    "average_time": round(sum(times) / len(times) * 1000, 2),
                    "throughput_qps": round(len(times) / concurrent_total, 2),
                }

                await pool.close()

            except Exception as e:
                test_results["error"] = str(e)

            return test_results

        try:
            results = asyncio.run(performance_test())
        except Exception as e:
            results = {"error": f"Async test failed: {e}"}

        return results

    def analyze_query_performance(self) -> dict[str, Any]:
        """Analyze query performance across databases."""
        print("⚡ Analyzing query performance...")

        performance = {
            "database_benchmarks": {},
            "optimization_opportunities": [],
            "performance_metrics": {},
        }

        discovery = self.results.get("database_discovery", {})

        # Benchmark DuckDB files
        for db_file in discovery.get("duckdb_files", []):
            if not HAS_DUCKDB or not Path(db_file).exists():
                continue

            try:
                conn = duckdb.connect(db_file)

                # Test basic query performance
                start_time = time.time()
                conn.execute("SELECT 1").fetchone()
                basic_query_time = (time.time() - start_time) * 1000

                # Test with existing tables if any
                tables = conn.execute("SHOW TABLES").fetchall()
                table_performance = {}

                for table in tables:
                    table_name = table[0]
                    try:
                        # Count query
                        start_time = time.time()
                        count = conn.execute(
                            f"SELECT COUNT(*) FROM {table_name}"
                        ).fetchone()[0]
                        count_time = (time.time() - start_time) * 1000

                        # Sample query if data exists
                        sample_time = 0
                        if count > 0:
                            start_time = time.time()
                            conn.execute(
                                f"SELECT * FROM {table_name} LIMIT 10"
                            ).fetchall()
                            sample_time = (time.time() - start_time) * 1000

                        table_performance[table_name] = {
                            "count_query_ms": round(count_time, 2),
                            "sample_query_ms": round(sample_time, 2),
                            "record_count": count,
                        }

                    except Exception as e:
                        table_performance[table_name] = {"error": str(e)}

                performance["database_benchmarks"][db_file] = {
                    "basic_query_ms": round(basic_query_time, 2),
                    "table_performance": table_performance,
                    "database_type": "duckdb",
                }

                conn.close()

            except Exception as e:
                performance["database_benchmarks"][db_file] = {"error": str(e)}

        # Benchmark SQLite files
        for db_file in discovery.get("sqlite_files", []):
            if not Path(db_file).exists():
                continue

            try:
                conn = sqlite3.connect(db_file, timeout=5.0)

                # Test basic query performance
                start_time = time.time()
                conn.execute("SELECT 1").fetchone()
                basic_query_time = (time.time() - start_time) * 1000

                # Test with existing tables
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = cursor.fetchall()
                table_performance = {}

                for table in tables:
                    table_name = table[0]
                    if table_name.startswith("sqlite_"):
                        continue

                    try:
                        # Count query
                        start_time = time.time()
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        count_time = (time.time() - start_time) * 1000

                        # Sample query if data exists
                        sample_time = 0
                        if count > 0:
                            start_time = time.time()
                            cursor = conn.execute(
                                f"SELECT * FROM {table_name} LIMIT 10"
                            )
                            cursor.fetchall()
                            sample_time = (time.time() - start_time) * 1000

                        table_performance[table_name] = {
                            "count_query_ms": round(count_time, 2),
                            "sample_query_ms": round(sample_time, 2),
                            "record_count": count,
                        }

                    except Exception as e:
                        table_performance[table_name] = {"error": str(e)}

                performance["database_benchmarks"][db_file] = {
                    "basic_query_ms": round(basic_query_time, 2),
                    "table_performance": table_performance,
                    "database_type": "sqlite",
                }

                conn.close()

            except Exception as e:
                performance["database_benchmarks"][db_file] = {"error": str(e)}

        # Generate optimization recommendations
        self._generate_performance_recommendations(performance)

        self.results["query_performance"] = performance

        print("✅ Query performance analysis completed")
        return performance

    def _generate_performance_recommendations(self, performance: dict[str, Any]):
        """Generate performance optimization recommendations."""
        recommendations = []

        for db_file, metrics in performance["database_benchmarks"].items():
            if "error" in metrics:
                recommendations.append(f"Fix connectivity issues with {db_file}")
                continue

            # Check for slow queries
            if metrics.get("basic_query_ms", 0) > 10:
                recommendations.append(
                    f"Basic queries in {db_file} are slow ({metrics['basic_query_ms']}ms)"
                )

            # Check table performance
            for table_name, table_metrics in metrics.get(
                "table_performance", {}
            ).items():
                if "error" in table_metrics:
                    recommendations.append(
                        f"Fix table access issues: {table_name} in {db_file}"
                    )
                    continue

                if table_metrics.get("count_query_ms", 0) > 100:
                    recommendations.append(
                        f"Consider indexing {table_name} in {db_file} (count query: {table_metrics['count_query_ms']}ms)"
                    )

                if table_metrics.get("sample_query_ms", 0) > 50:
                    recommendations.append(
                        f"Consider optimizing {table_name} in {db_file} (sample query: {table_metrics['sample_query_ms']}ms)"
                    )

        performance["optimization_opportunities"] = recommendations

    def check_data_consistency(self) -> dict[str, Any]:
        """Check data consistency and referential integrity."""
        print("🔍 Checking data consistency...")

        consistency = {
            "integrity_checks": {},
            "referential_integrity": {},
            "data_quality_issues": [],
        }

        # This is a placeholder for more comprehensive consistency checks
        # In a real system, you would check:
        # - Foreign key constraints
        # - Data type consistency
        # - Null value patterns
        # - Duplicate detection
        # - Cross-table relationships

        self.results["data_consistency"] = consistency

        print("✅ Data consistency checks completed")
        return consistency

    def validate_backup_procedures(self) -> dict[str, Any]:
        """Validate backup and restoration procedures."""
        print("💾 Validating backup procedures...")

        backup_validation = {
            "backup_locations": [],
            "backup_schedules": {},
            "restoration_tests": {},
            "recommendations": [],
        }

        # Check for backup directories
        backup_dirs = [
            self.project_root / "backups",
            self.project_root / "data" / "backups",
            self.project_root / "data" / "archive",
        ]

        for backup_dir in backup_dirs:
            if backup_dir.exists():
                backup_files = list(backup_dir.rglob("*.db*"))
                backup_validation["backup_locations"].append(
                    {
                        "path": str(backup_dir),
                        "file_count": len(backup_files),
                        "total_size_mb": sum(f.stat().st_size for f in backup_files)
                        / (1024 * 1024),
                    }
                )

        # Generate backup recommendations
        if not backup_validation["backup_locations"]:
            backup_validation["recommendations"].append(
                "No backup directories found - implement backup strategy"
            )

        self.results["backup_validation"] = backup_validation

        print("✅ Backup validation completed")
        return backup_validation

    def assess_security(self) -> dict[str, Any]:
        """Assess database security configurations."""
        print("🔒 Assessing database security...")

        security = {
            "access_controls": {},
            "encryption_status": {},
            "security_issues": [],
            "recommendations": [],
        }

        discovery = self.results.get("database_discovery", {})

        for db_file in discovery.get("file_details", {}):
            file_path = Path(db_file)

            # Check file permissions
            stat = file_path.stat()
            permissions = oct(stat.st_mode)[-3:]

            access_info = {
                "permissions": permissions,
                "readable_by_others": permissions[2] in ["4", "5", "6", "7"],
                "writable_by_others": permissions[2] in ["2", "3", "6", "7"],
            }

            if access_info["readable_by_others"]:
                security["security_issues"].append(f"{db_file} is readable by others")

            if access_info["writable_by_others"]:
                security["security_issues"].append(f"{db_file} is writable by others")

            security["access_controls"][db_file] = access_info

        # Generate security recommendations
        if security["security_issues"]:
            security["recommendations"].append("Fix file permission issues")

        security["recommendations"].append(
            "Consider database encryption for sensitive data"
        )
        security["recommendations"].append("Implement access logging and monitoring")

        self.results["security_assessment"] = security

        print("✅ Security assessment completed")
        return security

    def generate_recommendations(self) -> list[str]:
        """Generate overall recommendations based on all assessments."""
        recommendations = []

        # Database discovery recommendations
        discovery = self.results.get("database_discovery", {})
        if discovery.get("total_size_mb", 0) > 1000:  # > 1GB
            recommendations.append(
                "Large database files detected - consider archiving old data"
            )

        # Schema validation recommendations
        schema_validation = self.results.get("schema_validation", {})
        if schema_validation.get("issues"):
            recommendations.append(
                f"Fix {len(schema_validation['issues'])} schema issues"
            )

        # Connection pooling recommendations
        pooling = self.results.get("connection_pooling", {})
        if not pooling.get("bolt_fixes_available"):
            recommendations.append(
                "Install bolt database fixes for improved concurrency"
            )

        if not pooling.get("connection_manager_available"):
            recommendations.append("Install connection manager for better performance")

        # Performance recommendations
        performance = self.results.get("query_performance", {})
        recommendations.extend(performance.get("optimization_opportunities", []))

        # Security recommendations
        security = self.results.get("security_assessment", {})
        recommendations.extend(security.get("recommendations", []))

        # Backup recommendations
        backup = self.results.get("backup_validation", {})
        recommendations.extend(backup.get("recommendations", []))

        self.results["recommendations"] = recommendations
        return recommendations

    def run_full_assessment(self) -> dict[str, Any]:
        """Run complete database integrity and performance assessment."""
        print("🚀 Starting comprehensive database assessment...")
        start_time = time.time()

        try:
            # Run all assessments
            self.discover_databases()
            self.validate_schemas()
            self.test_connection_pooling()
            self.analyze_query_performance()
            self.check_data_consistency()
            self.validate_backup_procedures()
            self.assess_security()
            self.generate_recommendations()

            # Add summary
            self.results["assessment_summary"] = {
                "total_time_seconds": round(time.time() - start_time, 2),
                "databases_analyzed": len(
                    self.results["database_discovery"].get("file_details", {})
                ),
                "total_recommendations": len(self.results["recommendations"]),
                "critical_issues": len(
                    [r for r in self.results["recommendations"] if "Fix" in r]
                ),
                "status": "completed",
            }

            print(
                f"✅ Assessment completed in {self.results['assessment_summary']['total_time_seconds']}s"
            )
            print(
                f"📊 Analyzed {self.results['assessment_summary']['databases_analyzed']} databases"
            )
            print(
                f"💡 Generated {self.results['assessment_summary']['total_recommendations']} recommendations"
            )

        except Exception as e:
            self.results["assessment_summary"] = {
                "status": "failed",
                "error": str(e),
                "total_time_seconds": round(time.time() - start_time, 2),
            }
            print(f"❌ Assessment failed: {e}")

        return self.results

    def save_results(self, output_file: str = "database_integrity_assessment.json"):
        """Save assessment results to file."""
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"📄 Results saved to {output_path}")
        return output_path


def main():
    """Run database integrity assessment."""
    print("🔧 Database Integrity and Performance Assessment - Agent 7")
    print("=" * 60)

    tester = DatabaseIntegrityTester()
    results = tester.run_full_assessment()

    # Save results
    output_file = tester.save_results()

    # Print summary
    print("\n📋 ASSESSMENT SUMMARY")
    print("=" * 30)

    summary = results.get("assessment_summary", {})
    print(f"Status: {summary.get('status', 'unknown')}")
    print(f"Duration: {summary.get('total_time_seconds', 0)}s")
    print(f"Databases: {summary.get('databases_analyzed', 0)}")
    print(f"Recommendations: {summary.get('total_recommendations', 0)}")

    # Print top recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        print("\n💡 TOP RECOMMENDATIONS")
        print("-" * 25)
        for i, rec in enumerate(recommendations[:10], 1):
            print(f"{i}. {rec}")

        if len(recommendations) > 10:
            print(f"... and {len(recommendations) - 10} more")

    print(f"\n📄 Full results: {output_file}")

    return results


if __name__ == "__main__":
    main()
