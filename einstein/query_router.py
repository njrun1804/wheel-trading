#!/usr/bin/env python3
"""
Einstein Query Router

Intelligently routes queries to optimal search modalities based on:
- Query type analysis
- Performance characteristics  
- Expected result quality
- Historical search patterns
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """Types of queries Einstein can handle."""
    LITERAL_TEXT = "literal_text"           # Exact text search
    SEMANTIC_CODE = "semantic_code"         # Code similarity search
    STRUCTURAL = "structural"               # Symbol/dependency search
    ANALYTICAL = "analytical"               # Metrics/complexity search
    HYBRID = "hybrid"                       # Multi-modal search


@dataclass
class QueryPlan:
    """Plan for executing a query across search modalities."""
    query: str
    query_type: QueryType
    search_modalities: List[str]
    confidence: float
    estimated_time_ms: float
    reasoning: str


class QueryRouter:
    """Routes queries to optimal search strategies."""
    
    def __init__(self):
        # Query patterns for classification
        self.patterns = {
            QueryType.LITERAL_TEXT: [
                r'^".*"$',  # Quoted strings
                r'\b(error|exception|traceback)\b',  # Error messages
                r'\b(TODO|FIXME|NOTE)\b',  # Code comments
            ],
            QueryType.SEMANTIC_CODE: [
                r'\b(similar to|like|related to)\b',  # Similarity requests
                r'\b(pattern|example|implementation)\b',  # Pattern searches
                r'\b(algorithm|function|method)\b',  # Code constructs
            ],
            QueryType.STRUCTURAL: [
                r'\b(class|function|method|import)\s+\w+',  # Symbol names
                r'\b(depends on|uses|calls)\b',  # Dependency queries
                r'\w+\.[a-zA-Z_]\w*',  # Qualified names
            ],
            QueryType.ANALYTICAL: [
                r'\b(complexity|performance|metrics)\b',  # Performance queries
                r'\b(large|small|slow|fast)\s+(file|function)\b',  # Size queries
                r'\b(most|least|highest|lowest)\b',  # Ranking queries
            ]
        }
        
        # Performance characteristics (ms) for each modality
        self.modality_performance = {
            'text': 2.0,      # Ripgrep is blazing fast
            'semantic': 15.0,  # Neural embedding lookup
            'structural': 5.0, # Dependency graph traversal
            'analytical': 8.0  # DuckDB query
        }
    
    def analyze_query(self, query: str) -> QueryPlan:
        """Analyze query and determine optimal search strategy."""
        
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
        
        return QueryPlan(
            query=query,
            query_type=query_type,
            search_modalities=modalities,
            confidence=confidence,
            estimated_time_ms=estimated_time,
            reasoning=reasoning
        )
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify query type based on patterns."""
        
        # Check each query type pattern
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return query_type
        
        # Default classification logic
        if any(char in query for char in ['"', "'"]):
            return QueryType.LITERAL_TEXT
        elif any(word in query for word in ['class', 'function', 'def ', 'import']):
            return QueryType.STRUCTURAL
        elif len(query.split()) > 5:
            return QueryType.SEMANTIC_CODE
        else:
            return QueryType.HYBRID
    
    def _select_modalities(self, query_type: QueryType, query: str) -> List[str]:
        """Select optimal search modalities for query type."""
        
        modality_map = {
            QueryType.LITERAL_TEXT: ['text'],
            QueryType.SEMANTIC_CODE: ['text', 'semantic'],
            QueryType.STRUCTURAL: ['structural', 'text'],
            QueryType.ANALYTICAL: ['analytical', 'structural'],
            QueryType.HYBRID: ['text', 'structural', 'semantic']
        }
        
        base_modalities = modality_map.get(query_type, ['text'])
        
        # Add analytical search for queries about metrics
        if any(word in query for word in ['complex', 'large', 'performance', 'slow']):
            if 'analytical' not in base_modalities:
                base_modalities.append('analytical')
        
        return base_modalities
    
    def _calculate_confidence(self, query_type: QueryType, modalities: List[str]) -> float:
        """Calculate confidence in the search strategy."""
        
        # Base confidence by query type
        base_confidence = {
            QueryType.LITERAL_TEXT: 0.95,    # Text search is very reliable
            QueryType.SEMANTIC_CODE: 0.75,   # Depends on embeddings quality
            QueryType.STRUCTURAL: 0.85,      # Dependency graphs are accurate
            QueryType.ANALYTICAL: 0.80,      # DuckDB queries are precise
            QueryType.HYBRID: 0.70          # Multi-modal has more uncertainty
        }
        
        confidence = base_confidence.get(query_type, 0.60)
        
        # Boost confidence for multi-modal searches
        if len(modalities) > 1:
            confidence += 0.1
        
        # Cap at 0.95
        return min(confidence, 0.95)
    
    def _generate_reasoning(self, query_type: QueryType, modalities: List[str]) -> str:
        """Generate human-readable reasoning for the search strategy."""
        
        reasoning_templates = {
            QueryType.LITERAL_TEXT: "Using fast text search for exact matches",
            QueryType.SEMANTIC_CODE: "Using semantic search to find similar code patterns",
            QueryType.STRUCTURAL: "Using dependency analysis to find related symbols",
            QueryType.ANALYTICAL: "Using analytics to find files matching complexity criteria",
            QueryType.HYBRID: "Using multi-modal search for comprehensive results"
        }
        
        base_reasoning = reasoning_templates.get(query_type, "Using default search strategy")
        
        if len(modalities) > 1:
            modality_list = ', '.join(modalities)
            return f"{base_reasoning}. Combining {modality_list} search for best coverage."
        else:
            return f"{base_reasoning}. Single modality search for optimal speed."
    
    def optimize_for_latency(self, plan: QueryPlan) -> QueryPlan:
        """Optimize query plan for minimum latency."""
        
        # For latency optimization, prefer text search
        if plan.estimated_time_ms > 20.0:
            # Remove slower modalities
            fast_modalities = [m for m in plan.search_modalities 
                             if self.modality_performance[m] <= 5.0]
            
            if fast_modalities:
                plan.search_modalities = fast_modalities
                plan.estimated_time_ms = sum(self.modality_performance[m] 
                                            for m in fast_modalities)
                plan.reasoning += " Optimized for low latency."
        
        return plan
    
    def optimize_for_recall(self, plan: QueryPlan) -> QueryPlan:
        """Optimize query plan for maximum recall."""
        
        # For recall optimization, use all modalities
        all_modalities = ['text', 'semantic', 'structural', 'analytical']
        
        plan.search_modalities = all_modalities
        plan.estimated_time_ms = sum(self.modality_performance[m] 
                                   for m in all_modalities)
        plan.reasoning += " Optimized for maximum recall across all modalities."
        
        return plan


if __name__ == "__main__":
    # Test the query router
    router = QueryRouter()
    
    test_queries = [
        "WheelStrategy class",
        "complex functions with high cyclomatic complexity",
        '"options pricing calculation"',
        "similar to Black-Scholes implementation",
        "import pandas"
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
