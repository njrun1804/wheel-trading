"""
Fallback DuckDB implementation with connection management.

Provides real DuckDB functionality when accelerated tools are not available.
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pandas as pd

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None
    pd = None


class DuckDBFallback:
    """Fallback DuckDB implementation with connection pooling."""

    def __init__(self, database_path: str = ":memory:", max_connections: int = 10):
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is not available")

        self.database_path = database_path
        self.max_connections = max_connections
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self.connection_count = 0

        # Initialize with one connection
        self._create_connection()

    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new DuckDB connection."""
        try:
            conn = duckdb.connect(self.database_path)

            # Configure for performance
            conn.execute(
                "SET memory_limit='4GB'"
            )  # Fixed: use explicit GB instead of %
            conn.execute("SET threads TO 4")
            conn.execute("SET enable_progress_bar=false")

            return conn
        except Exception as e:
            logger.error(f"Failed to create DuckDB connection: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = None
        try:
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                elif self.connection_count < self.max_connections:
                    conn = self._create_connection()
                    self.connection_count += 1
                else:
                    # Wait for a connection to become available
                    pass

            if conn is None:
                # Create temporary connection if pool is exhausted
                conn = self._create_connection()
                temp_connection = True
            else:
                temp_connection = False

            yield conn

        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                try:
                    if not temp_connection:
                        with self.pool_lock:
                            if len(self.connection_pool) < self.max_connections:
                                self.connection_pool.append(conn)
                            else:
                                conn.close()
                                self.connection_count -= 1
                    else:
                        conn.close()
                except Exception as e:
                    logger.debug(f"Error returning connection to pool: {e}")

    @asynccontextmanager
    async def get_connection_async(self):
        """Async version of get_connection."""
        loop = asyncio.get_event_loop()

        def get_conn():
            return self.get_connection()

        # Run the synchronous context manager in executor
        conn_cm = await loop.run_in_executor(None, get_conn)

        try:
            conn = await loop.run_in_executor(None, conn_cm.__enter__)
            yield conn
        except Exception as e:
            await loop.run_in_executor(
                None, conn_cm.__exit__, type(e), e, e.__traceback__
            )
            raise
        else:
            await loop.run_in_executor(None, conn_cm.__exit__, None, None, None)

    async def query_to_pandas(
        self, query: str, parameters: dict = None
    ) -> pd.DataFrame:
        """Execute query and return results as pandas DataFrame."""
        if not pd:
            raise ImportError("pandas is not available")

        loop = asyncio.get_event_loop()

        def _execute_query():
            with self.get_connection() as conn:
                if parameters:
                    # Simple parameter substitution (for safety, use DuckDB's prepared statements in production)
                    for key, value in parameters.items():
                        if isinstance(value, str):
                            query_with_params = query.replace(f"${key}", f"'{value}'")
                        else:
                            query_with_params = query.replace(f"${key}", str(value))
                    result = conn.execute(query_with_params).fetchdf()
                else:
                    result = conn.execute(query).fetchdf()
                return result

        return await loop.run_in_executor(None, _execute_query)

    async def execute(self, query: str, parameters: dict = None) -> list[tuple]:
        """Execute query and return raw results."""
        loop = asyncio.get_event_loop()

        def _execute_query():
            with self.get_connection() as conn:
                if parameters:
                    for key, value in parameters.items():
                        if isinstance(value, str):
                            query_with_params = query.replace(f"${key}", f"'{value}'")
                        else:
                            query_with_params = query.replace(f"${key}", str(value))
                    result = conn.execute(query_with_params).fetchall()
                else:
                    result = conn.execute(query).fetchall()
                return result

        return await loop.run_in_executor(None, _execute_query)

    async def execute_many(self, queries: list[str]) -> list[Any]:
        """Execute multiple queries in a single connection."""
        loop = asyncio.get_event_loop()

        def _execute_queries():
            results = []
            with self.get_connection() as conn:
                for query in queries:
                    try:
                        result = conn.execute(query).fetchall()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Query failed: {query[:100]}... Error: {e}")
                        results.append([])
            return results

        return await loop.run_in_executor(None, _execute_queries)

    async def create_table_from_csv(self, table_name: str, csv_path: str) -> bool:
        """Create table from CSV file."""
        try:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                logger.error(f"CSV file not found: {csv_path}")
                return False

            query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} AS 
                SELECT * FROM read_csv_auto('{csv_path}')
            """

            await self.execute(query)
            logger.info(f"Created table {table_name} from {csv_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create table from CSV: {e}")
            return False

    async def create_table_from_parquet(
        self, table_name: str, parquet_path: str
    ) -> bool:
        """Create table from Parquet file."""
        try:
            parquet_path = Path(parquet_path)
            if not parquet_path.exists():
                logger.error(f"Parquet file not found: {parquet_path}")
                return False

            query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} AS 
                SELECT * FROM read_parquet('{parquet_path}')
            """

            await self.execute(query)
            logger.info(f"Created table {table_name} from {parquet_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create table from Parquet: {e}")
            return False

    async def get_table_info(self, table_name: str) -> dict[str, Any]:
        """Get information about a table."""
        try:
            # Get table schema
            schema_query = f"DESCRIBE {table_name}"
            schema_result = await self.query_to_pandas(schema_query)

            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            count_result = await self.execute(count_query)
            row_count = count_result[0][0] if count_result else 0

            return {
                "table_name": table_name,
                "row_count": row_count,
                "columns": schema_result.to_dict("records") if pd else [],
                "column_count": len(schema_result) if pd else 0,
            }

        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            return {"error": str(e)}

    async def list_tables(self) -> list[str]:
        """List all tables in the database."""
        try:
            query = "SHOW TABLES"
            result = await self.execute(query)
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []

    async def optimize_table(self, table_name: str) -> bool:
        """Optimize table performance."""
        try:
            # DuckDB doesn't have explicit OPTIMIZE, but we can run VACUUM
            query = "VACUUM"
            await self.execute(query)
            logger.info("Optimized database (vacuumed)")
            return True
        except Exception as e:
            logger.error(f"Failed to optimize table {table_name}: {e}")
            return False

    async def backup_database(self, backup_path: str) -> bool:
        """Backup database to file."""
        try:
            query = f"EXPORT DATABASE '{backup_path}'"
            await self.execute(query)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False

    def close_all_connections(self):
        """Close all connections in the pool."""
        with self.pool_lock:
            for conn in self.connection_pool:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug(f"Error closing connection: {e}")
            self.connection_pool.clear()
            self.connection_count = 0

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        with self.pool_lock:
            return {
                "backend": "duckdb_fallback",
                "database_path": self.database_path,
                "max_connections": self.max_connections,
                "active_connections": self.connection_count,
                "pooled_connections": len(self.connection_pool),
                "parallel_capable": True,
                "pandas_integration": pd is not None,
            }


# Convenience function for easy access
def get_duckdb_turbo(database_path: str = None) -> DuckDBFallback:
    """Get DuckDB fallback instance."""
    if not DUCKDB_AVAILABLE:
        raise ImportError("DuckDB is not available. Please install duckdb package.")

    if database_path is None:
        database_path = ":memory:"

    return DuckDBFallback(database_path)
