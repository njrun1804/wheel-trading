"""Native DuckDB wrapper optimized for M4 Pro - no MCP overhead."""

import duckdb
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import pyarrow as pa
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager


class DuckDBTurbo:
    """Hardware-accelerated DuckDB operations for M4 Pro."""
    
    def __init__(self, db_path: Optional[str] = None):
        # Load optimization config
        with open("optimization_config.json") as f:
            self.config = json.load(f)
        
        self.db_path = db_path or ":memory:"
        
        # For in-memory databases, use a single shared connection
        # For file databases, use connection pool
        if self.db_path == ":memory:":
            self.pool_size = 1
            self.shared_memory = True
        else:
            self.pool_size = self.config["io"]["concurrent_reads"]  # 24 concurrent
            self.shared_memory = False
            
        self.connections = []
        self._executor = ThreadPoolExecutor(max_workers=max(self.pool_size, 8))
        
        # Configure DuckDB for M4 Pro
        self._init_connections()
        
    def _init_connections(self):
        """Initialize connection pool with M4 Pro optimizations."""
        config = {
            'threads': self.config["cpu"]["max_workers"],  # 8 performance cores
            'memory_limit': f"{self.config['memory']['max_allocation_gb']}GB",  # 19GB
            'max_memory': f"{self.config['memory']['max_allocation_gb']}GB",
            'temp_directory': '/tmp/duckdb_temp',
            'preserve_insertion_order': False,  # Better performance
            'enable_object_cache': True
        }
        
        # Create connection pool
        if self.shared_memory:
            # Single shared connection for in-memory database
            self.main_conn = duckdb.connect(self.db_path, config=config)
            # All connections share the same in-memory database
            for _ in range(8):  # Still create workers for parallel execution
                conn = self.main_conn
            
            # Enable M4 Pro specific extensions
            conn.install_extension('httpfs')
            conn.install_extension('parquet')
            conn.install_extension('json')
            conn.load_extension('httpfs')
            conn.load_extension('parquet')
            conn.load_extension('json')
            
            # Set pragmas for performance (only valid pragmas)
            conn.execute(f"PRAGMA threads={self.config['cpu']['max_workers']}")
            conn.execute(f"PRAGMA memory_limit='{self.config['memory']['max_allocation_gb']}GB'")
            
            self.connections.append(conn)
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        # Simple round-robin (could be enhanced)
        conn = self.connections[hash(asyncio.current_task()) % len(self.connections)]
        try:
            yield conn
        finally:
            pass  # Connection stays in pool
    
    async def execute(self, query: str, params: Optional[tuple] = None) -> pa.Table:
        """Execute query with full parallelization."""
        async with self.get_connection() as conn:
            loop = asyncio.get_event_loop()
            
            def run_query():
                if params:
                    result = conn.execute(query, params)
                else:
                    result = conn.execute(query)
                return result.arrow()
            
            # Run in thread pool to avoid blocking
            arrow_table = await loop.run_in_executor(self._executor, run_query)
            return arrow_table
    
    async def execute_many(self, queries: List[Union[str, tuple]]) -> List[pa.Table]:
        """Execute multiple queries in parallel."""
        tasks = []
        
        for query in queries:
            if isinstance(query, tuple):
                sql, params = query
                tasks.append(self.execute(sql, params))
            else:
                tasks.append(self.execute(query))
        
        return await asyncio.gather(*tasks)
    
    async def insert_batch(self, table_name: str, data: Union[pd.DataFrame, pa.Table, List[Dict]],
                          chunk_size: Optional[int] = None) -> int:
        """Insert data in optimized batches."""
        if chunk_size is None:
            chunk_size = self.config["cpu"]["chunk_size"]  # 65536 rows
        
        async with self.get_connection() as conn:
            loop = asyncio.get_event_loop()
            
            def do_insert():
                # Convert to DataFrame if needed
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, pa.Table):
                    df = data.to_pandas()
                else:
                    df = data
                
                # Insert in chunks for better memory usage
                total_rows = 0
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i + chunk_size]
                    conn.register('temp_insert', chunk)
                    conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_insert")
                    total_rows += len(chunk)
                    conn.unregister('temp_insert')
                
                return total_rows
            
            return await loop.run_in_executor(self._executor, do_insert)
    
    async def query_to_pandas(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute query and return pandas DataFrame."""
        arrow_table = await self.execute(query, params)
        return arrow_table.to_pandas()
    
    async def create_table_from_parquet(self, table_name: str, parquet_path: str) -> None:
        """Create table from Parquet file with parallel loading."""
        async with self.get_connection() as conn:
            query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} AS 
            SELECT * FROM read_parquet('{parquet_path}', 
                parallel=true,
                hive_partitioning=true,
                union_by_name=true)
            """
            await self.execute(query)
    
    async def export_to_parquet(self, query: str, output_path: str, 
                               compression: str = 'snappy') -> None:
        """Export query results to Parquet with parallel writing."""
        export_query = f"""
        COPY ({query}) TO '{output_path}' 
        WITH (FORMAT 'parquet', COMPRESSION '{compression}', ROW_GROUP_SIZE 100000)
        """
        await self.execute(export_query)
    
    async def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """Analyze table statistics using parallel aggregation."""
        queries = [
            f"SELECT COUNT(*) as row_count FROM {table_name}",
            f"SELECT * FROM {table_name} LIMIT 0",  # Get schema
            f"SELECT approx_count_distinct(*) FROM {table_name}",
            f"SUMMARIZE {table_name}"
        ]
        
        results = await self.execute_many(queries)
        
        return {
            "row_count": results[0].to_pandas()['row_count'][0],
            "columns": results[1].schema.names,
            "schema": str(results[1].schema),
            "summary": results[3].to_pandas().to_dict('records')
        }
    
    async def parallel_aggregate(self, table_name: str, group_by: List[str],
                               aggregations: Dict[str, str]) -> pd.DataFrame:
        """Perform parallel aggregation optimized for M4 Pro."""
        # Build aggregation query
        agg_exprs = [f"{func}({col}) as {col}_{func}" 
                    for col, func in aggregations.items()]
        
        query = f"""
        SELECT {', '.join(group_by)}, {', '.join(agg_exprs)}
        FROM {table_name}
        GROUP BY {', '.join(group_by)}
        ORDER BY {', '.join(group_by)}
        """
        
        return await self.query_to_pandas(query)
    
    async def optimize_table(self, table_name: str) -> None:
        """Optimize table for M4 Pro's unified memory."""
        queries = [
            f"CHECKPOINT {table_name}",
            f"ANALYZE {table_name}",
            f"PRAGMA optimize"
        ]
        
        await self.execute_many(queries)
    
    def cleanup(self):
        """Close all connections and cleanup."""
        for conn in self.connections:
            conn.close()
        self._executor.shutdown(wait=False)


# Singleton instance
_duckdb_instance: Optional[DuckDBTurbo] = None


def get_duckdb_turbo(db_path: Optional[str] = None) -> DuckDBTurbo:
    """Get or create the turbo DuckDB instance."""
    global _duckdb_instance
    if _duckdb_instance is None:
        _duckdb_instance = DuckDBTurbo(db_path)
    return _duckdb_instance


# Drop-in replacements for MCP functions
async def query(sql: str, db_path: Optional[str] = None) -> str:
    """Drop-in replacement for MCP duckdb.query."""
    db = get_duckdb_turbo(db_path)
    df = await db.query_to_pandas(sql)
    
    # Format as string output like MCP would
    if len(df) > 100:
        return f"{df.head(50).to_string()}\n...\n{df.tail(50).to_string()}\n\n[{len(df)} rows x {len(df.columns)} columns]"
    else:
        return df.to_string()


async def execute(sql: str, db_path: Optional[str] = None) -> str:
    """Drop-in replacement for MCP duckdb.execute."""
    db = get_duckdb_turbo(db_path)
    await db.execute(sql)
    return "Query executed successfully"


async def describe_table(table_name: str, db_path: Optional[str] = None) -> str:
    """Drop-in replacement for MCP duckdb.describe_table."""
    db = get_duckdb_turbo(db_path)
    analysis = await db.analyze_table(table_name)
    
    output = [
        f"Table: {table_name}",
        f"Rows: {analysis['row_count']:,}",
        f"Columns: {len(analysis['columns'])}",
        "",
        "Schema:"
    ]
    
    for col_info in analysis['summary']:
        output.append(f"  - {col_info}")
    
    return "\n".join(output)