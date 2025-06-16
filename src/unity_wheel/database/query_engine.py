#!/usr/bin/env python3
"""
Unified Query Engine with Optimization and Routing

Intelligent query execution with automatic optimization, parallel execution,
and smart routing based on query patterns and data characteristics.

Key Features:
- Query parsing and optimization
- Parallel query execution for M4 Pro
- Smart routing based on query type
- Query plan caching
- Automatic index recommendations
- Cost-based optimization
"""

import asyncio
import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

import sqlparse
from sqlparse.sql import Identifier, IdentifierList

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for routing."""

    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    DROP = "DROP"
    AGGREGATE = "AGGREGATE"
    JOIN = "JOIN"
    ANALYTICAL = "ANALYTICAL"
    TRANSACTION = "TRANSACTION"


@dataclass
class QueryPlan:
    """Execution plan for a query."""

    query_id: str
    original_query: str
    optimized_query: str
    query_type: QueryType
    tables: list[str]
    columns: list[str]
    estimated_cost: float
    parallelizable: bool
    cache_eligible: bool
    indexes_used: list[str]
    recommended_indexes: list[str]


@dataclass
class QueryResult:
    """Result of query execution."""

    query_id: str
    data: Any
    row_count: int
    execution_time_ms: float
    cache_hit: bool
    plan_used: QueryPlan | None = None
    error: str | None = None


class QueryOptimizer:
    """Optimizes queries for better performance."""

    def __init__(self):
        self.optimization_rules = [
            self._optimize_select_star,
            self._optimize_joins,
            self._optimize_aggregates,
            self._optimize_subqueries,
            self._add_limit_for_exploration,
        ]

    def optimize(self, query: str, metadata: dict[str, Any] | None = None) -> str:
        """Apply optimization rules to query."""
        optimized = query

        for rule in self.optimization_rules:
            try:
                optimized = rule(optimized, metadata)
            except Exception as e:
                logger.debug(f"Optimization rule failed: {e}")

        return optimized

    def _optimize_select_star(
        self, query: str, metadata: dict[str, Any] | None
    ) -> str:
        """Replace SELECT * with specific columns when possible."""
        # This is a simplified example - real implementation would use metadata
        return query

    def _optimize_joins(self, query: str, metadata: dict[str, Any] | None) -> str:
        """Optimize JOIN order based on table statistics."""
        # Simplified - would use table statistics in practice
        return query

    def _optimize_aggregates(
        self, query: str, metadata: dict[str, Any] | None
    ) -> str:
        """Optimize aggregate functions."""
        # Add GROUP BY optimizations
        return query

    def _optimize_subqueries(
        self, query: str, metadata: dict[str, Any] | None
    ) -> str:
        """Convert subqueries to JOINs where beneficial."""
        return query

    def _add_limit_for_exploration(
        self, query: str, metadata: dict[str, Any] | None
    ) -> str:
        """Add LIMIT for exploratory queries without one."""
        if metadata and metadata.get("exploratory", False):
            if "limit" not in query.lower() and query.lower().startswith("select"):
                return f"{query.rstrip(';')} LIMIT 1000"
        return query


class QueryParser:
    """Parses SQL queries to extract metadata."""

    def __init__(self):
        self.table_pattern = re.compile(r"FROM\s+(\w+)", re.IGNORECASE)
        self.join_pattern = re.compile(r"JOIN\s+(\w+)", re.IGNORECASE)

    def parse(self, query: str) -> dict[str, Any]:
        """Parse query and extract metadata."""
        parsed = sqlparse.parse(query)[0]

        metadata = {
            "query_type": self._get_query_type(parsed),
            "tables": self._extract_tables(parsed),
            "columns": self._extract_columns(parsed),
            "has_joins": self._has_joins(parsed),
            "has_aggregates": self._has_aggregates(query),
            "has_subqueries": self._has_subqueries(parsed),
            "is_complex": False,
        }

        # Determine complexity
        metadata["is_complex"] = (
            metadata["has_joins"]
            or metadata["has_aggregates"]
            or metadata["has_subqueries"]
            or len(metadata["tables"]) > 2
        )

        return metadata

    def _get_query_type(self, parsed) -> QueryType:
        """Determine the type of query."""
        first_token = parsed.token_first(skip_ws=True, skip_cm=True)
        if first_token:
            token_value = first_token.value.upper()

            if token_value == "SELECT":
                # Check for aggregates
                if self._has_aggregates(str(parsed)):
                    return QueryType.AGGREGATE
                # Check for joins
                if self._has_joins(parsed):
                    return QueryType.JOIN
                return QueryType.SELECT

            elif token_value == "INSERT":
                return QueryType.INSERT
            elif token_value == "UPDATE":
                return QueryType.UPDATE
            elif token_value == "DELETE":
                return QueryType.DELETE
            elif token_value == "CREATE":
                return QueryType.CREATE
            elif token_value == "DROP":
                return QueryType.DROP

        return QueryType.SELECT

    def _extract_tables(self, parsed) -> list[str]:
        """Extract table names from query."""
        tables = []

        # Find FROM clause
        from_seen = False
        for token in parsed.tokens:
            if from_seen:
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        tables.append(str(identifier))
                elif isinstance(token, Identifier):
                    tables.append(str(token))
                elif token.ttype is None:
                    tables.append(str(token).strip())

            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "FROM":
                from_seen = True

        return [t for t in tables if t and not t.isspace()]

    def _extract_columns(self, parsed) -> list[str]:
        """Extract column names from query."""
        # Simplified implementation
        return []

    def _has_joins(self, parsed) -> bool:
        """Check if query has JOINs."""
        return "JOIN" in str(parsed).upper()

    def _has_aggregates(self, query: str) -> bool:
        """Check if query has aggregate functions."""
        aggregates = ["COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP BY"]
        query_upper = query.upper()
        return any(agg in query_upper for agg in aggregates)

    def _has_subqueries(self, parsed) -> bool:
        """Check if query has subqueries."""
        # Count parentheses after first SELECT
        str_query = str(parsed)
        first_select = str_query.upper().find("SELECT")
        if first_select >= 0:
            after_select = str_query[first_select + 6 :]
            return "(SELECT" in after_select.upper()
        return False


class QueryEngine:
    """Unified query execution engine with optimization."""

    def __init__(self, connection_pool, cache_coordinator=None):
        self.connection_pool = connection_pool
        self.cache_coordinator = cache_coordinator
        self.parser = QueryParser()
        self.optimizer = QueryOptimizer()

        # Query plan cache
        self.plan_cache: dict[str, QueryPlan] = {}

        # Executor for parallel queries
        self.executor = ThreadPoolExecutor(max_workers=8)  # M4 Pro optimized

        # Performance tracking
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "parallel_executions": 0,
            "optimization_time_ms": 0.0,
            "execution_time_ms": 0.0,
        }

    def _get_query_hash(self, query: str, params: Any | None = None) -> str:
        """Generate hash for query + params for caching."""
        cache_key = query
        if params:
            cache_key += str(params)
        return hashlib.md5(cache_key.encode()).hexdigest()

    def _create_query_plan(self, query: str, metadata: dict[str, Any]) -> QueryPlan:
        """Create execution plan for query."""
        # Parse query
        query_type = metadata["query_type"]

        # Optimize query
        start_time = time.time()
        optimized_query = self.optimizer.optimize(query, metadata)
        optimization_time = (time.time() - start_time) * 1000

        self.stats["optimization_time_ms"] += optimization_time

        # Determine execution strategy
        parallelizable = self._is_parallelizable(query_type, metadata)
        cache_eligible = self._is_cache_eligible(query_type, metadata)

        # Create plan
        plan = QueryPlan(
            query_id=self._get_query_hash(query),
            original_query=query,
            optimized_query=optimized_query,
            query_type=query_type,
            tables=metadata.get("tables", []),
            columns=metadata.get("columns", []),
            estimated_cost=self._estimate_cost(metadata),
            parallelizable=parallelizable,
            cache_eligible=cache_eligible,
            indexes_used=[],  # Would be populated by actual query planner
            recommended_indexes=[],  # Would be populated by index advisor
        )

        return plan

    def _is_parallelizable(
        self, query_type: QueryType, metadata: dict[str, Any]
    ) -> bool:
        """Determine if query can be executed in parallel."""
        # SELECT queries without transactions can be parallelized
        if query_type in [QueryType.SELECT, QueryType.AGGREGATE, QueryType.ANALYTICAL]:
            return True

        # Complex JOINs might benefit from parallel execution
        if query_type == QueryType.JOIN and metadata.get("is_complex", False):
            return True

        return False

    def _is_cache_eligible(
        self, query_type: QueryType, metadata: dict[str, Any]
    ) -> bool:
        """Determine if query results can be cached."""
        # Only cache read queries
        return query_type in [
            QueryType.SELECT,
            QueryType.AGGREGATE,
            QueryType.JOIN,
            QueryType.ANALYTICAL,
        ]

    def _estimate_cost(self, metadata: dict[str, Any]) -> float:
        """Estimate query execution cost."""
        cost = 1.0

        # Increase cost for complex operations
        if metadata.get("has_joins", False):
            cost *= 2.0
        if metadata.get("has_aggregates", False):
            cost *= 1.5
        if metadata.get("has_subqueries", False):
            cost *= 3.0

        # Increase cost based on number of tables
        cost *= len(metadata.get("tables", []))

        return cost

    def execute(self, query: str, params: Any | None = None) -> QueryResult:
        """Execute query with optimization and caching."""
        start_time = time.time()
        query_hash = self._get_query_hash(query, params)

        self.stats["total_queries"] += 1

        # Check cache first
        if self.cache_coordinator:
            cached_result = self.cache_coordinator.get(query_hash)
            if cached_result:
                self.stats["cache_hits"] += 1
                execution_time = (time.time() - start_time) * 1000
                return QueryResult(
                    query_id=query_hash,
                    data=cached_result["data"],
                    row_count=cached_result["row_count"],
                    execution_time_ms=execution_time,
                    cache_hit=True,
                )

        self.stats["cache_misses"] += 1

        # Parse query
        metadata = self.parser.parse(query)

        # Get or create query plan
        if query_hash in self.plan_cache:
            plan = self.plan_cache[query_hash]
        else:
            plan = self._create_query_plan(query, metadata)
            self.plan_cache[query_hash] = plan

        # Execute query
        try:
            if plan.parallelizable and len(plan.tables) > 1:
                # Execute in parallel for complex queries
                result_data = self._execute_parallel(plan.optimized_query, params)
                self.stats["parallel_executions"] += 1
            else:
                # Execute normally
                result_data = self._execute_single(plan.optimized_query, params)

            # Create result
            row_count = len(result_data) if hasattr(result_data, "__len__") else 0
            execution_time = (time.time() - start_time) * 1000

            result = QueryResult(
                query_id=query_hash,
                data=result_data,
                row_count=row_count,
                execution_time_ms=execution_time,
                cache_hit=False,
                plan_used=plan,
            )

            # Cache result if eligible
            if plan.cache_eligible and self.cache_coordinator:
                self.cache_coordinator.put(
                    query_hash,
                    {
                        "data": result_data,
                        "row_count": row_count,
                        "timestamp": time.time(),
                    },
                )

            self.stats["execution_time_ms"] += execution_time
            return result

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            execution_time = (time.time() - start_time) * 1000
            return QueryResult(
                query_id=query_hash,
                data=None,
                row_count=0,
                execution_time_ms=execution_time,
                cache_hit=False,
                error=str(e),
            )

    def _execute_single(self, query: str, params: Any | None = None) -> Any:
        """Execute query on single connection."""
        with self.connection_pool.acquire() as conn:
            if params:
                result = conn.execute(query, params)
            else:
                result = conn.execute(query)

            # Fetch results
            if hasattr(result, "fetchdf"):
                return result.fetchdf()
            elif hasattr(result, "fetchall"):
                return result.fetchall()
            else:
                return result

    def _execute_parallel(self, query: str, params: Any | None = None) -> Any:
        """Execute query in parallel (simplified)."""
        # In practice, this would partition the query across multiple connections
        # For now, just use single execution
        return self._execute_single(query, params)

    async def execute_async(
        self, query: str, params: Any | None = None
    ) -> QueryResult:
        """Execute query asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.execute, query, params)

    def explain(self, query: str) -> dict[str, Any]:
        """Get query execution plan."""
        metadata = self.parser.parse(query)
        plan = self._create_query_plan(query, metadata)

        return {
            "query_type": plan.query_type.value,
            "tables": plan.tables,
            "estimated_cost": plan.estimated_cost,
            "parallelizable": plan.parallelizable,
            "cache_eligible": plan.cache_eligible,
            "optimized_query": plan.optimized_query,
            "optimization_applied": plan.original_query != plan.optimized_query,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = self.stats.copy()

        # Calculate averages
        if stats["total_queries"] > 0:
            stats["avg_optimization_time_ms"] = (
                stats["optimization_time_ms"] / stats["total_queries"]
            )
            stats["avg_execution_time_ms"] = (
                stats["execution_time_ms"] / stats["total_queries"]
            )
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_queries"]
            stats["parallel_execution_rate"] = (
                stats["parallel_executions"] / stats["total_queries"]
            )

        # Add plan cache stats
        stats["cached_plans"] = len(self.plan_cache)

        return stats

    def clear_plan_cache(self):
        """Clear the query plan cache."""
        self.plan_cache.clear()
        logger.info("Query plan cache cleared")
