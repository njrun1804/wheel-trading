"""Diversity Engine for generating varied code solutions.

Implements AlphaCode 2-style diversity generation through different
architectural, stylistic, and optimization approaches.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..core.solution import SearchNode

logger = logging.getLogger(__name__)


class DiversityDimension(Enum):
    """Dimensions along which to generate diversity."""
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"
    ERROR_HANDLING = "error_handling"
    TESTING = "testing"
    STYLE = "style"
    ALGORITHM = "algorithm"
    PARALLELISM = "parallelism"
    MEMORY = "memory"


@dataclass
class DiversityStrategy:
    """Strategy for generating diverse solutions."""
    dimension: DiversityDimension
    name: str
    description: str
    transformations: List[str]
    weight: float = 1.0


class DiversityEngine:
    """Generates diverse code solutions."""
    
    def __init__(self):
        self.strategies = self._init_strategies()
        self.generation_count = 0
        self.diversity_scores = []
    
    def _init_strategies(self) -> Dict[DiversityDimension, List[DiversityStrategy]]:
        """Initialize diversity strategies."""
        return {
            DiversityDimension.ARCHITECTURE: [
                DiversityStrategy(
                    dimension=DiversityDimension.ARCHITECTURE,
                    name="functional",
                    description="Pure functional approach",
                    transformations=["immutable_data", "pure_functions", "composition"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.ARCHITECTURE,
                    name="object_oriented",
                    description="Object-oriented design",
                    transformations=["classes", "inheritance", "encapsulation"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.ARCHITECTURE,
                    name="event_driven",
                    description="Event-driven architecture",
                    transformations=["event_handlers", "callbacks", "observers"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.ARCHITECTURE,
                    name="microservice",
                    description="Microservice pattern",
                    transformations=["service_boundaries", "api_design", "loose_coupling"]
                ),
            ],
            
            DiversityDimension.OPTIMIZATION: [
                DiversityStrategy(
                    dimension=DiversityDimension.OPTIMIZATION,
                    name="latency",
                    description="Optimize for low latency",
                    transformations=["caching", "precomputation", "fast_paths"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.OPTIMIZATION,
                    name="throughput",
                    description="Optimize for high throughput",
                    transformations=["batching", "pipelining", "parallelization"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.OPTIMIZATION,
                    name="memory",
                    description="Optimize for memory efficiency",
                    transformations=["streaming", "compression", "lazy_evaluation"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.OPTIMIZATION,
                    name="readability",
                    description="Optimize for code clarity",
                    transformations=["explicit_names", "small_functions", "documentation"]
                ),
            ],
            
            DiversityDimension.ERROR_HANDLING: [
                DiversityStrategy(
                    dimension=DiversityDimension.ERROR_HANDLING,
                    name="exceptions",
                    description="Exception-based error handling",
                    transformations=["try_except", "custom_exceptions", "exception_hierarchy"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.ERROR_HANDLING,
                    name="result_types",
                    description="Result type error handling",
                    transformations=["optional", "either", "result_monads"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.ERROR_HANDLING,
                    name="defensive",
                    description="Defensive programming",
                    transformations=["assertions", "preconditions", "invariants"]
                ),
            ],
            
            DiversityDimension.ALGORITHM: [
                DiversityStrategy(
                    dimension=DiversityDimension.ALGORITHM,
                    name="iterative",
                    description="Iterative algorithms",
                    transformations=["loops", "state_machines", "accumulation"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.ALGORITHM,
                    name="recursive",
                    description="Recursive algorithms",
                    transformations=["recursion", "divide_conquer", "memoization"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.ALGORITHM,
                    name="dynamic_programming",
                    description="Dynamic programming approach",
                    transformations=["subproblems", "optimal_substructure", "tabulation"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.ALGORITHM,
                    name="greedy",
                    description="Greedy algorithms",
                    transformations=["local_optimal", "heuristics", "approximation"]
                ),
            ],
            
            DiversityDimension.PARALLELISM: [
                DiversityStrategy(
                    dimension=DiversityDimension.PARALLELISM,
                    name="sequential",
                    description="Sequential execution",
                    transformations=["single_thread", "ordered", "deterministic"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.PARALLELISM,
                    name="threaded",
                    description="Multi-threaded execution",
                    transformations=["thread_pool", "locks", "concurrent_data"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.PARALLELISM,
                    name="async",
                    description="Asynchronous execution",
                    transformations=["coroutines", "event_loop", "futures"]
                ),
                DiversityStrategy(
                    dimension=DiversityDimension.PARALLELISM,
                    name="distributed",
                    description="Distributed processing",
                    transformations=["map_reduce", "message_passing", "consensus"]
                ),
            ],
        }
    
    async def generate(self, solution_tree: SearchNode,
                      num_variants: int = 100,
                      clustering_method: str = "behavioral",
                      hardware_executor: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Generate diverse solutions from search tree."""
        logger.info(f"Generating {num_variants} diverse solutions")
        
        # Extract base solutions from tree
        base_solutions = self._extract_solutions_from_tree(solution_tree, max_solutions=20)
        
        # Generate variants
        all_variants = []
        
        for base in base_solutions:
            # Apply different strategies
            variants = await self._generate_variants(base, num_variants // len(base_solutions))
            all_variants.extend(variants)
        
        # Cluster by behavior
        if clustering_method == "behavioral":
            clusters = await self._cluster_by_behavior(all_variants)
            selected = self._select_diverse_set(clusters, num_variants)
        else:
            # Simple diversity based on code similarity
            selected = self._select_by_code_diversity(all_variants, num_variants)
        
        self.generation_count += len(selected)
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(selected)
        self.diversity_scores.append(diversity_score)
        logger.info(f"Generated {len(selected)} variants with diversity score: {diversity_score:.3f}")
        
        return selected
    
    def _extract_solutions_from_tree(self, root: SearchNode, 
                                   max_solutions: int = 20) -> List[Dict[str, Any]]:
        """Extract promising solutions from search tree."""
        solutions = []
        
        # If root has code, use it as base solution
        if root.code:
            solutions.append({
                'code': root.code,
                'value': root.average_value if hasattr(root, 'average_value') else 0.8,
                'visits': root.visits,
                'approach': 'base'
            })
            logger.debug(f"Using root solution with {len(root.code)} chars")
        
        # BFS to find complete solutions
        queue = list(root.children) if hasattr(root, 'children') else []
        while queue and len(solutions) < max_solutions:
            node = queue.pop(0)
            
            # Check if this is a complete solution
            if self._is_complete_solution(node):
                solutions.append({
                    'code': node.code,
                    'value': node.average_value if hasattr(node, 'average_value') else 0.8,
                    'visits': node.visits,
                    'path': [n.action_taken for n in node.get_path_to_root()]
                })
            
            # Add children to queue (prioritize high-value nodes)
            children = sorted(node.children, key=lambda c: c.average_value, reverse=True)
            queue.extend(children[:3])  # Only top 3 children
        
        return solutions
    
    def _is_complete_solution(self, node: SearchNode) -> bool:
        """Check if node represents a complete solution."""
        # Simple heuristic: deep enough and has been visited
        return node.depth >= 3 and node.visits > 0 and len(node.code) > 50
    
    async def _generate_variants(self, base_solution: Dict[str, Any],
                               num_variants: int) -> List[Dict[str, Any]]:
        """Generate variants of a base solution."""
        variants = [base_solution]  # Include original
        
        # Select strategies to apply
        strategies_to_apply = self._select_strategies(num_variants)
        
        for strategy in strategies_to_apply:
            variant = await self._apply_strategy(base_solution, strategy)
            if variant:
                variants.append(variant)
        
        return variants[:num_variants]
    
    def _select_strategies(self, num_needed: int) -> List[DiversityStrategy]:
        """Select diverse strategies to apply."""
        selected = []
        
        # Ensure we get at least one from each dimension
        for dimension, strategies in self.strategies.items():
            if strategies:
                # Weighted random selection
                weights = [s.weight for s in strategies]
                idx = np.random.choice(len(strategies), p=np.array(weights)/sum(weights))
                selected.append(strategies[idx])
        
        # Fill remaining slots
        while len(selected) < num_needed:
            # Random dimension
            dim = np.random.choice(list(self.strategies.keys()))
            strategies = self.strategies[dim]
            if strategies:
                idx = np.random.randint(len(strategies))
                selected.append(strategies[idx])
        
        return selected[:num_needed]
    
    async def _apply_strategy(self, solution: Dict[str, Any],
                            strategy: DiversityStrategy) -> Optional[Dict[str, Any]]:
        """Apply a diversity strategy to generate variant."""
        code = solution['code']
        
        # Apply transformations
        transformed_code = code
        for transformation in strategy.transformations:
            transformed_code = self._apply_transformation(transformed_code, transformation)
        
        # Create variant
        variant = {
            'code': transformed_code,
            'approach': f"{strategy.dimension.value}:{strategy.name}",
            'transformations': strategy.transformations,
            'base_value': solution.get('value', 0),
            'strategy': strategy.name
        }
        
        return variant
    
    def _apply_transformation(self, code: str, transformation: str) -> str:
        """Apply a specific code transformation."""
        # Simplified transformations for demonstration
        # In practice, would use AST manipulation
        
        transformations = {
            # Architecture
            "immutable_data": lambda c: c.replace("self.", "frozen_"),
            "pure_functions": lambda c: c.replace("def ", "def pure_"),
            "classes": lambda c: f"class Solution:\n{self._indent(c)}",
            
            # Optimization
            "caching": lambda c: f"@lru_cache(maxsize=128)\n{c}",
            "batching": lambda c: c.replace("for item in", "for batch in chunked("),
            "streaming": lambda c: c.replace("return list(", "yield from ("),
            
            # Error handling
            "try_except": lambda c: f"try:\n{self._indent(c)}\nexcept Exception as e:\n    logger.error(e)\n    raise",
            "assertions": lambda c: f"assert input is not None\n{c}",
            
            # Parallelism
            "async": lambda c: c.replace("def ", "async def ").replace("(", "async ("),
            "thread_pool": lambda c: f"with ThreadPoolExecutor() as executor:\n{self._indent(c)}",
        }
        
        transform_func = transformations.get(transformation, lambda c: c)
        return transform_func(code)
    
    def _indent(self, code: str, spaces: int = 4) -> str:
        """Indent code block."""
        lines = code.split('\n')
        return '\n'.join(' ' * spaces + line if line.strip() else line for line in lines)
    
    async def _cluster_by_behavior(self, variants: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Cluster variants by behavioral similarity."""
        # Generate behavioral signatures
        signatures = []
        for variant in variants:
            sig = await self._generate_behavioral_signature(variant)
            signatures.append(sig)
        
        # Simple clustering based on signature similarity
        clusters = []
        used = set()
        
        for i, sig1 in enumerate(signatures):
            if i in used:
                continue
            
            cluster = [variants[i]]
            used.add(i)
            
            # Find similar variants
            for j, sig2 in enumerate(signatures):
                if j in used or i == j:
                    continue
                
                similarity = self._signature_similarity(sig1, sig2)
                if similarity > 0.8:  # High similarity threshold
                    cluster.append(variants[j])
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    async def _generate_behavioral_signature(self, variant: Dict[str, Any]) -> str:
        """Generate behavioral signature for variant."""
        code = variant['code']
        
        # Simple signature based on code characteristics
        features = [
            f"len:{len(code)}",
            f"lines:{code.count('\\n')}",
            f"imports:{code.count('import')}",
            f"functions:{code.count('def ')}",
            f"classes:{code.count('class ')}",
            f"approach:{variant.get('approach', 'default')}",
        ]
        
        # Hash features for compact signature
        signature = hashlib.md5('|'.join(features).encode()).hexdigest()[:16]
        return signature
    
    def _signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between signatures."""
        # Simple character-based similarity
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(1 for c1, c2 in zip(sig1, sig2) if c1 == c2)
        return matches / len(sig1)
    
    def _select_diverse_set(self, clusters: List[List[Dict[str, Any]]],
                          target_count: int) -> List[Dict[str, Any]]:
        """Select diverse set from clusters."""
        selected = []
        
        # Take best from each cluster
        for cluster in clusters:
            if cluster:
                # Sort by value if available
                best = max(cluster, key=lambda x: x.get('base_value', 0))
                selected.append(best)
        
        # If we need more, take additional variants from larger clusters
        remaining = target_count - len(selected)
        if remaining > 0:
            # Sort clusters by size
            sorted_clusters = sorted(clusters, key=len, reverse=True)
            
            for cluster in sorted_clusters:
                if len(cluster) > 1:
                    # Take additional variants (skip the first which we already took)
                    for variant in cluster[1:min(1 + remaining, len(cluster))]:
                        selected.append(variant)
                        remaining -= 1
                        if remaining <= 0:
                            break
                
                if remaining <= 0:
                    break
        
        return selected[:target_count]
    
    def _select_by_code_diversity(self, variants: List[Dict[str, Any]],
                                target_count: int) -> List[Dict[str, Any]]:
        """Select variants based on code diversity."""
        if len(variants) <= target_count:
            return variants
        
        selected = []
        remaining = list(variants)
        
        # Greedy selection: always pick the most different variant
        while len(selected) < target_count and remaining:
            if not selected:
                # Pick first variant
                selected.append(remaining.pop(0))
            else:
                # Pick most different from selected
                max_diff = -1
                max_idx = 0
                
                for i, variant in enumerate(remaining):
                    min_sim = min(
                        self._code_similarity(variant['code'], s['code'])
                        for s in selected
                    )
                    if min_sim > max_diff:
                        max_diff = min_sim
                        max_idx = i
                
                selected.append(remaining.pop(max_idx))
        
        return selected
    
    def _code_similarity(self, code1: str, code2: str) -> float:
        """Calculate code similarity."""
        # Simple token-based similarity
        tokens1 = set(code1.split())
        tokens2 = set(code2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_diversity_score(self, variants: List[Dict[str, Any]]) -> float:
        """Calculate overall diversity score."""
        if len(variants) < 2:
            return 0.0
        
        # Average pairwise dissimilarity
        dissimilarities = []
        
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                sim = self._code_similarity(variants[i]['code'], variants[j]['code'])
                dissimilarities.append(1.0 - sim)
        
        return np.mean(dissimilarities) if dissimilarities else 0.0


class DiversityMetrics:
    """Track and analyze diversity metrics."""
    
    def __init__(self):
        self.approach_counts = {}
        self.dimension_coverage = {}
        self.behavioral_clusters = 0
    
    def analyze_batch(self, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze diversity metrics for a batch."""
        # Count approaches
        approaches = {}
        for v in variants:
            approach = v.get('approach', 'unknown')
            approaches[approach] = approaches.get(approach, 0) + 1
        
        # Dimension coverage
        dimensions_seen = set()
        for v in variants:
            if 'approach' in v and ':' in v['approach']:
                dim = v['approach'].split(':')[0]
                dimensions_seen.add(dim)
        
        # Calculate metrics
        metrics = {
            'num_variants': len(variants),
            'unique_approaches': len(approaches),
            'dimension_coverage': len(dimensions_seen) / len(DiversityDimension),
            'approach_distribution': approaches,
            'most_common_approach': max(approaches.items(), key=lambda x: x[1])[0] if approaches else None,
            'approach_entropy': self._calculate_entropy(list(approaches.values()))
        }
        
        return metrics
    
    def _calculate_entropy(self, counts: List[int]) -> float:
        """Calculate Shannon entropy."""
        if not counts or sum(counts) == 0:
            return 0.0
        
        total = sum(counts)
        probs = [c / total for c in counts]
        
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy