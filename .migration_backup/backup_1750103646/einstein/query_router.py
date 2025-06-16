#!/usr/bin/env python3
"""
Einstein Query Router

Intelligently routes queries to optimal search modalities based on:
- Query type analysis
- Performance characteristics  
- Expected result quality
- Historical search patterns
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

from einstein.einstein_config import get_einstein_config

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries Einstein can handle."""

    LITERAL_TEXT = "literal_text"  # Exact text search
    SEMANTIC_CODE = "semantic_code"  # Code similarity search
    STRUCTURAL = "structural"  # Symbol/dependency search
    ANALYTICAL = "analytical"  # Metrics/complexity search
    HYBRID = "hybrid"  # Multi-modal search


@dataclass
class QueryPlan:
    """Plan for executing a query across search modalities."""

    query: str
    query_type: QueryType
    search_modalities: list[str]
    confidence: float
    estimated_time_ms: float
    reasoning: str


class QueryRouter:
    """Routes queries to optimal search strategies."""

    def __init__(self) -> None:
        # Query patterns for classification
        self.patterns = {
            QueryType.LITERAL_TEXT: [
                r'^".*"$',  # Quoted strings
                r"\b(error|exception|traceback)\b",  # Error messages
                r"\b(TODO|FIXME|NOTE)\b",  # Code comments
            ],
            QueryType.SEMANTIC_CODE: [
                r"\b(similar to|like|related to)\b",  # Similarity requests
                r"\b(pattern|example|implementation)\b",  # Pattern searches
                r"\b(algorithm|function|method)\b",  # Code constructs
            ],
            QueryType.STRUCTURAL: [
                r"\b(class|function|method|import)\s+\w+",  # Symbol names
                r"\b(depends on|uses|calls)\b",  # Dependency queries
                r"\w+\.[a-zA-Z_]\w*",  # Qualified names
            ],
            QueryType.ANALYTICAL: [
                r"\b(complexity|performance|metrics)\b",  # Performance queries
                r"\b(large|small|slow|fast)\s+(file|function)\b",  # Size queries
                r"\b(most|least|highest|lowest)\b",  # Ranking queries
            ],
        }

        # Performance characteristics (ms) for each modality from config
        config = get_einstein_config()
        self.modality_performance = {
            "text": config.performance.target_text_search_ms,
            "semantic": config.performance.target_semantic_search_ms,
            "structural": config.performance.target_structural_search_ms,
            "analytical": config.performance.target_analytical_search_ms,
        }

    def analyze_query(self, query: str) -> QueryPlan:
        """Analyze query and determine optimal search strategy."""

        try:
            if not query or not isinstance(query, str):
                logger.warning(
                    f"Invalid query input: {type(query).__name__} with value {query!r}"
                )
                query = str(query) if query else ""

            query_lower = query.lower().strip()

            # Classify query type
            query_type = self._classify_query(query_lower)

            # Determine search modalities based on type
            modalities = self._select_modalities(query_type, query_lower)

            # Estimate performance
            estimated_time = sum(self.modality_performance[m] for m in modalities)

            # Calculate confidence
            confidence = self._calculate_confidence(query_type, modalities)

            # Generate reasoning
            reasoning = self._generate_reasoning(query_type, modalities)

            plan = QueryPlan(
                query=query,
                query_type=query_type,
                search_modalities=modalities,
                confidence=confidence,
                estimated_time_ms=estimated_time,
                reasoning=reasoning,
            )

            logger.debug(
                f"Query analysis completed: {query_type.value} with {len(modalities)} modalities"
            )
            return plan

        except Exception as e:
            logger.error(
                f"Failed to analyze query: {e}",
                exc_info=True,
                extra={
                    "operation": "analyze_query",
                    "error_type": type(e).__name__,
                    "query": query[:100]
                    if isinstance(query, str)
                    else str(query)[:100],
                    "query_length": len(query) if isinstance(query, str) else 0,
                    "query_type": type(query).__name__,
                    "modality_performance_available": bool(self.modality_performance),
                    "patterns_available": bool(self.patterns),
                },
            )
            # Return fallback plan
            return QueryPlan(
                query=query or "",
                query_type=QueryType.LITERAL_TEXT,
                search_modalities=["text"],
                confidence=0.5,
                estimated_time_ms=self.modality_performance.get("text", 5.0),
                reasoning="Fallback plan due to analysis error",
            )

    def _classify_query(self, query: str) -> QueryType:
        """Classify query type based on patterns."""

        try:
            # Check each query type pattern
            for query_type, patterns in self.patterns.items():
                for pattern in patterns:
                    try:
                        if re.search(pattern, query, re.IGNORECASE):
                            return query_type
                    except re.error as e:
                        logger.warning(
                            f"Invalid regex pattern '{pattern}' for query type {query_type}: {e}",
                            extra={
                                "operation": "classify_query_pattern_match",
                                "error_type": type(e).__name__,
                                "pattern": pattern,
                                "query_type": query_type.value,
                                "query_sample": query[:50],
                            },
                        )
                        continue

            # Default classification logic
            if any(char in query for char in ['"', "'"]):
                return QueryType.LITERAL_TEXT
            elif any(word in query for word in ["class", "function", "def ", "import"]):
                return QueryType.STRUCTURAL
            elif len(query.split()) > 5:
                return QueryType.SEMANTIC_CODE
            else:
                return QueryType.HYBRID

        except Exception as e:
            logger.error(
                f"Failed to classify query: {e}",
                exc_info=True,
                extra={
                    "operation": "classify_query",
                    "error_type": type(e).__name__,
                    "query": query[:100],
                    "query_length": len(query),
                    "patterns_count": len(self.patterns),
                    "available_query_types": [qt.value for qt in self.patterns],
                },
            )
            return QueryType.LITERAL_TEXT  # Safe fallback

    def _select_modalities(self, query_type: QueryType, query: str) -> list[str]:
        """Select optimal search modalities for query type."""

        try:
            modality_map = {
                QueryType.LITERAL_TEXT: ["text"],
                QueryType.SEMANTIC_CODE: ["text", "semantic"],
                QueryType.STRUCTURAL: ["structural", "text"],
                QueryType.ANALYTICAL: ["analytical", "structural"],
                QueryType.HYBRID: ["text", "structural", "semantic"],
            }

            base_modalities = modality_map.get(query_type, ["text"])

            # Add analytical search for queries about metrics
            if (
                any(
                    word in query
                    for word in ["complex", "large", "performance", "slow"]
                )
                and "analytical" not in base_modalities
            ):
                base_modalities.append("analytical")

            # Validate modalities are available in performance config
            valid_modalities = [
                m for m in base_modalities if m in self.modality_performance
            ]
            if not valid_modalities:
                logger.warning(
                    f"No valid modalities found for query type {query_type.value}, using text fallback"
                )
                valid_modalities = ["text"]

            return valid_modalities

        except Exception as e:
            logger.error(
                f"Failed to select modalities: {e}",
                exc_info=True,
                extra={
                    "operation": "select_modalities",
                    "error_type": type(e).__name__,
                    "query_type": query_type.value if query_type else "unknown",
                    "query": query[:100],
                    "available_modalities": list(self.modality_performance.keys()),
                    "modality_performance_config": bool(self.modality_performance),
                },
            )
            return ["text"]  # Safe fallback

    def _calculate_confidence(
        self, query_type: QueryType, modalities: list[str]
    ) -> float:
        """Calculate confidence in the search strategy."""

        # Base confidence by query type
        base_confidence = {
            QueryType.LITERAL_TEXT: 0.95,  # Text search is very reliable
            QueryType.SEMANTIC_CODE: 0.75,  # Depends on embeddings quality
            QueryType.STRUCTURAL: 0.85,  # Dependency graphs are accurate
            QueryType.ANALYTICAL: 0.80,  # DuckDB queries are precise
            QueryType.HYBRID: 0.70,  # Multi-modal has more uncertainty
        }

        confidence = base_confidence.get(query_type, 0.60)

        # Boost confidence for multi-modal searches
        if len(modalities) > 1:
            confidence += 0.1

        # Cap at 0.95
        return min(confidence, 0.95)

    def _generate_reasoning(self, query_type: QueryType, modalities: list[str]) -> str:
        """Generate human-readable reasoning for the search strategy."""

        reasoning_templates = {
            QueryType.LITERAL_TEXT: "Using fast text search for exact matches",
            QueryType.SEMANTIC_CODE: "Using semantic search to find similar code patterns",
            QueryType.STRUCTURAL: "Using dependency analysis to find related symbols",
            QueryType.ANALYTICAL: "Using analytics to find files matching complexity criteria",
            QueryType.HYBRID: "Using multi-modal search for comprehensive results",
        }

        base_reasoning = reasoning_templates.get(
            query_type, "Using default search strategy"
        )

        if len(modalities) > 1:
            modality_list = ", ".join(modalities)
            return (
                f"{base_reasoning}. Combining {modality_list} search for best coverage."
            )
        else:
            return f"{base_reasoning}. Single modality search for optimal speed."

    def optimize_for_latency(self, plan: QueryPlan) -> QueryPlan:
        """Optimize query plan for minimum latency."""

        try:
            original_time = plan.estimated_time_ms

            # For latency optimization, prefer text search
            if plan.estimated_time_ms > 20.0:
                # Remove slower modalities
                fast_modalities = [
                    m
                    for m in plan.search_modalities
                    if self.modality_performance.get(m, float("inf")) <= 5.0
                ]

                if fast_modalities:
                    plan.search_modalities = fast_modalities
                    plan.estimated_time_ms = sum(
                        self.modality_performance.get(m, 5.0) for m in fast_modalities
                    )
                    plan.reasoning += " Optimized for low latency."

                    logger.debug(
                        f"Latency optimization: {original_time:.1f}ms -> {plan.estimated_time_ms:.1f}ms"
                    )
                else:
                    logger.warning(
                        "No fast modalities available for latency optimization, keeping original plan"
                    )

            return plan

        except Exception as e:
            logger.error(
                f"Failed to optimize for latency: {e}",
                exc_info=True,
                extra={
                    "operation": "optimize_for_latency",
                    "error_type": type(e).__name__,
                    "original_modalities": plan.search_modalities,
                    "original_time_ms": plan.estimated_time_ms,
                    "query": plan.query[:100],
                    "query_type": plan.query_type.value,
                },
            )
            return plan  # Return original plan on error

    def optimize_for_recall(self, plan: QueryPlan) -> QueryPlan:
        """Optimize query plan for maximum recall."""

        try:
            original_modalities = plan.search_modalities[:]

            # For recall optimization, use all modalities
            all_modalities = ["text", "semantic", "structural", "analytical"]

            # Only include modalities that have performance data
            available_modalities = [
                m for m in all_modalities if m in self.modality_performance
            ]

            if available_modalities:
                plan.search_modalities = available_modalities
                plan.estimated_time_ms = sum(
                    self.modality_performance[m] for m in available_modalities
                )
                plan.reasoning += " Optimized for maximum recall across all modalities."

                logger.debug(
                    f"Recall optimization: {len(original_modalities)} -> {len(available_modalities)} modalities"
                )
            else:
                logger.warning(
                    "No modalities available for recall optimization, keeping original plan"
                )

            return plan

        except Exception as e:
            logger.error(
                f"Failed to optimize for recall: {e}",
                exc_info=True,
                extra={
                    "operation": "optimize_for_recall",
                    "error_type": type(e).__name__,
                    "original_modalities": plan.search_modalities,
                    "query": plan.query[:100],
                    "query_type": plan.query_type.value,
                    "available_performance_keys": list(
                        self.modality_performance.keys()
                    ),
                },
            )
            return plan  # Return original plan on error


if __name__ == "__main__":
    # Test the query router
    router = QueryRouter()

    test_queries = [
        "WheelStrategy class",
        "complex functions with high cyclomatic complexity",
        '"options pricing calculation"',
        "similar to Black-Scholes implementation",
        "import pandas",
    ]

    for query in test_queries:
        plan = router.analyze_query(query)
        print(f"Query: {query}")
        print(f"  Type: {plan.query_type.value}")
        print(f"  Modalities: {plan.search_modalities}")
        print(f"  Estimated time: {plan.estimated_time_ms:.1f}ms")
        print(f"  Confidence: {plan.confidence:.1%}")
        print(f"  Reasoning: {plan.reasoning}")
        print()
