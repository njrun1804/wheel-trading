"""
Hardware-accelerated sequential thinking for M4 Pro.
Replaces slow MCP server with native implementation using Metal GPU and multiprocessing.
"""

import asyncio
import hashlib
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import lmdb
import numpy as np

# Try to use uvloop for faster async
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None  # type: ignore
    nn = None  # type: ignore

try:
    from ..optimization.hardware_detector import HardwareCapabilities
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    try:
        from src.unity_wheel.optimization.hardware_detector import HardwareCapabilities
    except ImportError:
        # Simple fallback implementation
        class HardwareCapabilities:
            @staticmethod
            def get_cpu_cores():
                import os
                return os.cpu_count() or 4
            
            @staticmethod 
            def get_available_memory():
                return 16 * 1024 * 1024 * 1024  # 16GB fallback


@dataclass
class ThinkingStep:
    """A single step in sequential reasoning."""
    step_number: int
    action: str
    reasoning: str
    result: Any | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class ThinkingContext:
    """Context maintained across thinking steps."""
    goal: str
    constraints: list[str]
    steps_completed: list[ThinkingStep]
    current_state: dict[str, Any]
    max_steps: int = 100
    timeout: float = 300.0
    

@dataclass
class ThinkingConfig:
    """Configuration for sequential thinking."""
    # Scoring weights
    goal_relevance_weight: float = 0.3
    feasibility_weight: float = 0.2
    efficiency_weight: float = 0.2
    novelty_weight: float = 0.15
    constraint_weight: float = 0.15
    
    # Thresholds
    min_confidence_for_completion: float = 0.8
    constraint_satisfaction_ratio: float = 0.8
    min_steps_for_completion: int = 3
    
    # Threading
    thread_multiplier: int = 2  # threads = cpu_cores * multiplier
    
    # Memory and performance
    cache_size_mb: int = 1024  # LMDB cache size in MB
    memory_pool_size: int = 1000  # Pre-allocated memory pool size
    gpu_batch_size: int = 64  # Batch size for GPU operations


# Try to import numba for JIT compilation
try:
    import numexpr as ne
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(f):  # No-op decorator
        return f
    def njit(f):  # No-op decorator
        return f
    ne = None


# JIT-compiled functions for hot paths
@njit(fastmath=True, cache=True)
def compute_cosine_similarity_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """JIT-compiled batch cosine similarity computation."""
    n_candidates = candidates.shape[0]
    similarities = np.zeros(n_candidates, dtype=np.float32)
    
    query_norm = np.sqrt(np.sum(query * query))
    
    for i in prange(n_candidates):
        candidate_norm = np.sqrt(np.sum(candidates[i] * candidates[i]))
        dot_product = np.sum(query * candidates[i])
        similarities[i] = dot_product / (query_norm * candidate_norm + 1e-8)
    
    return similarities


@njit(fastmath=True, cache=True)
def weighted_feature_aggregation(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """JIT-compiled weighted feature aggregation."""
    n_samples, n_features = features.shape
    scores = np.zeros(n_samples, dtype=np.float32)
    
    for i in prange(n_samples):
        score = 0.0
        for j in range(n_features):
            score += features[i, j] * weights[j]
        scores[i] = score
    
    return scores


class SemanticEmbeddings:
    """Lightweight semantic embeddings using pre-computed word vectors."""
    
    def __init__(self):
        # Pre-computed embeddings for common task-related words
        self.word_vectors = {
            # Action verbs (normalized 3D vectors)
            'optimize': np.array([0.8, 0.3, 0.2], dtype=np.float32),
            'improve': np.array([0.7, 0.4, 0.2], dtype=np.float32),
            'enhance': np.array([0.7, 0.3, 0.3], dtype=np.float32),
            'debug': np.array([0.2, 0.8, 0.3], dtype=np.float32),
            'fix': np.array([0.2, 0.7, 0.4], dtype=np.float32),
            'resolve': np.array([0.3, 0.7, 0.3], dtype=np.float32),
            'implement': np.array([0.5, 0.2, 0.8], dtype=np.float32),
            'create': np.array([0.4, 0.2, 0.8], dtype=np.float32),
            'build': np.array([0.5, 0.3, 0.7], dtype=np.float32),
            'analyze': np.array([0.3, 0.5, 0.5], dtype=np.float32),
            'profile': np.array([0.8, 0.4, 0.1], dtype=np.float32),
            'parallelize': np.array([0.9, 0.2, 0.3], dtype=np.float32),
            'cache': np.array([0.7, 0.2, 0.4], dtype=np.float32),
            'vectorize': np.array([0.8, 0.1, 0.4], dtype=np.float32),
            
            # Domain words
            'performance': np.array([0.9, 0.2, 0.1], dtype=np.float32),
            'latency': np.array([0.8, 0.3, 0.1], dtype=np.float32),
            'throughput': np.array([0.8, 0.2, 0.2], dtype=np.float32),
            'memory': np.array([0.6, 0.3, 0.3], dtype=np.float32),
            'algorithm': np.array([0.4, 0.5, 0.5], dtype=np.float32),
            'trading': np.array([0.2, 0.3, 0.9], dtype=np.float32),
            'system': np.array([0.3, 0.4, 0.6], dtype=np.float32),
            'database': np.array([0.2, 0.6, 0.5], dtype=np.float32),
        }
        
        # Normalize vectors
        for word, vec in self.word_vectors.items():
            self.word_vectors[word] = vec / np.linalg.norm(vec)
            
        # Pre-allocate batch processing arrays
        self.batch_embeddings = np.zeros((100, 3), dtype=np.float32)
        
    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """Get semantic embedding for a sentence using word averaging."""
        words = text.lower().split()
        embeddings = []
        
        for word in words:
            if word in self.word_vectors:
                embeddings.append(self.word_vectors[word])
            # Check for partial matches
            else:
                for key in self.word_vectors:
                    if key in word or word in key:
                        embeddings.append(self.word_vectors[key])
                        break
                        
        if embeddings:
            # Average word embeddings
            return np.mean(embeddings, axis=0)
        else:
            # Random embedding for unknown words
            return np.random.randn(3).astype(np.float32) * 0.1
            
    def compute_similarity(self, embed1: np.ndarray, embed2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(embed1, embed2))
        
    def batch_similarities(self, query_embed: np.ndarray, candidate_embeds: np.ndarray) -> np.ndarray:
        """Compute similarities for a batch of candidates using JIT compilation and GPU."""
        if NUMBA_AVAILABLE:
            # Use JIT-compiled function for maximum performance
            return compute_cosine_similarity_batch(query_embed, candidate_embeds)
        elif MLX_AVAILABLE and mx:
            # Use Metal GPU for batch computation
            mx_query = mx.array(query_embed)
            mx_candidates = mx.array(candidate_embeds)
            similarities = mx.matmul(mx_candidates, mx_query)
            return np.array(similarities)
        else:
            # NumPy fallback
            return np.dot(candidate_embeds, query_embed)


class MemoryPool:
    """Pre-allocated memory pool for feature arrays."""
    def __init__(self, max_candidates: int):
        self.max_candidates = max_candidates
        self.feature_pool = np.zeros((max_candidates, 5), dtype=np.float32)
        self.score_pool = np.zeros(max_candidates, dtype=np.float32)
        self.embedding_pool = np.zeros((max_candidates, 3), dtype=np.float32)
        
    def get_feature_buffer(self, n_candidates: int) -> np.ndarray:
        """Get a view into the feature pool."""
        return self.feature_pool[:n_candidates]
        
    def get_score_buffer(self, n_candidates: int) -> np.ndarray:
        """Get a view into the score pool."""
        return self.score_pool[:n_candidates]
        
    def get_embedding_buffer(self, n_candidates: int) -> np.ndarray:
        """Get a view into the embedding pool."""
        return self.embedding_pool[:n_candidates]


class SequentialThinkingTurbo:
    """Hardware-accelerated sequential thinking engine."""
    
    def __init__(self, config: ThinkingConfig | None = None, cache_dir: str = ".thinking_cache"):
        self.config = config or ThinkingConfig()
        self.hw = HardwareCapabilities()
        self.cpu_cores = self.hw.cpu_cores
        self.gpu_cores = self.hw.gpu_cores
        
        # Memory pool for zero allocation
        self.memory_pool = MemoryPool(self.config.memory_pool_size)
        
        # Semantic embeddings for better relevance scoring
        self.embeddings = SemanticEmbeddings()
        
        # Initialize LMDB cache
        import os
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_env = lmdb.open(cache_dir, map_size=self.config.cache_size_mb*1024*1024)
        
        # Process pools for CPU parallelism
        self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_cores)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.cpu_cores * self.config.thread_multiplier)
        
        # MLX for GPU acceleration
        if MLX_AVAILABLE and mx:
            self.device = mx.gpu if hasattr(mx, 'metal') and mx.metal.is_available() else mx.cpu
        else:
            self.device = None
        
        # Thinking strategies
        self.strategies = {
            'breadth_first': self._breadth_first_strategy,
            'depth_first': self._depth_first_strategy,
            'beam_search': self._beam_search_strategy,
            'monte_carlo': self._monte_carlo_strategy,
            'parallel_explore': self._parallel_explore_strategy
        }
        
        # Performance tracking
        self.stats = {
            'total_steps': 0,
            'gpu_accelerated_steps': 0,
            'parallel_branches': 0,
            'avg_step_time': 0.0
        }
        
        # Profiling
        self.profiling_enabled = False
        self.profile = None
        self.profile_data = []
        
    def _generate_cache_key(self, goal: str, constraints: list[str], strategy: str) -> bytes:
        """Generate deterministic cache key."""
        key_data = f"{goal}:{sorted(constraints)}:{strategy}:{self.config}"
        return hashlib.sha256(key_data.encode()).digest()
        
    async def think(self, 
                   goal: str,
                   constraints: list[str] | None = None,
                   initial_state: dict[str, Any] | None = None,
                   strategy: str = 'parallel_explore',
                   max_steps: int = 100,
                   timeout: float = 300.0) -> list[ThinkingStep]:
        """
        Execute sequential thinking with hardware acceleration.
        
        Args:
            goal: The objective to achieve
            constraints: Limitations or requirements
            initial_state: Starting context
            strategy: Thinking strategy to use
            max_steps: Maximum thinking steps
            timeout: Maximum time in seconds
            
        Returns:
            List of thinking steps taken
        """
        # Track request
        self.stats['total_requests'] = self.stats.get('total_requests', 0) + 1
        
        # Check cache first
        with self.profile_context('cache_lookup'):
            cache_key = self._generate_cache_key(goal, constraints or [], strategy)
            with self.cache_env.begin() as txn:
                cached_result = txn.get(cache_key)
                if cached_result:
                    # Deserialize and return cached result
                    import pickle
                    self.stats['cache_hits'] = self.stats.get('cache_hits', 0) + 1
                    return pickle.loads(cached_result)
        
        context = ThinkingContext(
            goal=goal,
            constraints=constraints or [],
            steps_completed=[],
            current_state=initial_state or {},
            max_steps=max_steps,
            timeout=timeout
        )
        
        start_time = time.time()
        
        if strategy not in self.strategies:
            strategy = 'parallel_explore'
            
        thinking_fn = self.strategies[strategy]
        
        try:
            steps = await thinking_fn(context)
            
            # Update stats
            elapsed = time.time() - start_time
            self.stats['total_steps'] += len(steps)
            self.stats['avg_step_time'] = elapsed / len(steps) if steps else 0
            
            # Cache the result
            if len(steps) > 0:
                with self.profile_context('cache_write'), self.cache_env.begin(write=True) as txn:
                        import pickle
                        txn.put(cache_key, pickle.dumps(steps))
            
            return steps
            
        except TimeoutError:
            # Return partial results on timeout
            return context.steps_completed
            
    async def _parallel_explore_strategy(self, context: ThinkingContext) -> list[ThinkingStep]:
        """
        Explore multiple reasoning paths in parallel using all CPU cores.
        """
        steps: list[ThinkingStep] = []
        
        while len(steps) < context.max_steps:
            # Generate multiple candidate next steps in parallel
            candidates = await self._generate_parallel_candidates(context, num_candidates=self.cpu_cores)
            
            # Score candidates using GPU
            scores = await self._score_candidates_gpu(candidates, context)
            
            # Select best candidate
            best_idx = np.argmax(scores)
            best_candidate = candidates[best_idx]
            
            # Execute the step
            result = await self._execute_step(best_candidate, context)
            
            step = ThinkingStep(
                step_number=len(steps) + 1,
                action=best_candidate['action'],
                reasoning=best_candidate['reasoning'],
                result=result,
                confidence=float(scores[best_idx]),
                metadata={'strategy': 'parallel_explore', 'candidates_evaluated': len(candidates)}
            )
            
            steps.append(step)
            context.steps_completed.append(step)
            
            # Check if goal achieved
            if await self._is_goal_achieved(context):
                break
                
        return steps
        
    async def _generate_parallel_candidates(self, context: ThinkingContext, num_candidates: int) -> list[dict[str, Any]]:
        """Generate candidate steps with semantic understanding."""
        with self.profile_context('generate_parallel_candidates'):
            candidates = []
        
        # Analyze goal semantics
        goal_verbs = self._extract_action_verbs(context.goal)
        # goal_objects = self._extract_objects(context.goal)  # Currently unused
        
        # Smart templates based on goal analysis
        if any(v in goal_verbs for v in ['audit', 'review', 'assess', 'analyze']):
            # Meta system audit specific templates
            if 'meta' in context.goal.lower() or 'audit' in context.goal.lower():
                templates = [
                    ("Identify all meta system issues from", "Systematically catalog problems found in"),
                    ("Prioritize meta system fixes by", "Rank issues by impact and urgency in"),
                    ("Create comprehensive action plan for", "Develop step-by-step remediation strategy for"),
                    ("Address configuration inconsistencies in", "Standardize and consolidate config across"),
                    ("Fix duplicate functionality in", "Eliminate redundant code and merge components in"),
                    ("Improve error handling patterns in", "Strengthen resilience and recovery in"),
                    ("Enhance integration between", "Improve coordination and data flow in"),
                    ("Validate system architecture of", "Ensure design principles are followed in")
                ]
            else:
                templates = [
                    ("Conduct thorough analysis of", "Systematically examine all aspects of"),
                    ("Identify key issues in", "Find critical problems and gaps in"),
                    ("Assess current state of", "Evaluate existing implementation of"),
                    ("Document findings about", "Create comprehensive report on"),
                    ("Validate assumptions regarding", "Test hypotheses and verify claims about")
                ]
        elif any(v in goal_verbs for v in ['plan', 'organize', 'prioritize']):
            templates = [
                ("Create detailed roadmap for", "Develop step-by-step plan to achieve"),
                ("Prioritize action items for", "Rank tasks by importance and urgency for"),
                ("Organize components of", "Structure and categorize elements of"),
                ("Sequence implementation steps for", "Order activities in logical progression for"),
                ("Allocate resources for", "Determine what's needed to accomplish")
            ]
        elif any(v in goal_verbs for v in ['optimize', 'improve', 'enhance']):
            templates = [
                ("Profile and measure", "Identify performance bottlenecks in"),
                ("Parallelize operations for", "Use all CPU cores to accelerate"),
                ("Cache results of", "Reduce redundant computation in"),
                ("Vectorize calculations in", "Use SIMD operations for"),
                ("Remove bottlenecks from", "Eliminate slow operations in")
            ]
        elif any(v in goal_verbs for v in ['debug', 'fix', 'resolve']):
            templates = [
                ("Reproduce issue in", "Create minimal test case for"),
                ("Add logging to", "Trace execution flow in"),
                ("Isolate problem in", "Narrow down root cause of"),
                ("Validate assumptions about", "Check preconditions for"),
                ("Test edge cases of", "Find boundary conditions in")
            ]
        elif any(v in goal_verbs for v in ['implement', 'create', 'build']):
            templates = [
                ("Design architecture for", "Plan component structure of"),
                ("Implement core logic of", "Build essential functionality for"),
                ("Add error handling to", "Make robust implementation of"),
                ("Write tests for", "Ensure correctness of"),
                ("Document implementation of", "Explain design decisions for")
            ]
        else:
            # Generic templates
            templates = [
                ("Analyze requirements for", "Understand the problem space of"),
                ("Research approaches to", "Find best practices for"),
                ("Plan implementation of", "Create roadmap for"),
                ("Evaluate options for", "Compare alternatives for"),
                ("Prototype solution for", "Quick proof of concept for")
            ]
            
        # Generate candidates with context awareness
        for i in range(num_candidates):
            template_idx = i % len(templates) if len(templates) > 0 else 0
            action_prefix, reasoning_template = templates[template_idx]
            
            # Build constraint-aware reasoning
            constraint_mentions = []
            for constraint in context.constraints[:2]:  # Top 2 constraints
                key_words = [w for w in constraint.lower().split() 
                           if len(w) > 3 and w not in ['with', 'must', 'should']]
                if key_words:
                    constraint_mentions.append(key_words[0])
                    
            reasoning = f"{reasoning_template} {context.goal}"
            if constraint_mentions:
                reasoning += f" while ensuring {', '.join(constraint_mentions)}"
                
            candidates.append({
                'action': f"{action_prefix} {context.goal}",
                'reasoning': reasoning,
                'variation': i,
                'template_idx': template_idx,
                'semantic_match': 0.8 if i < len(templates) else 0.5
            })
        
        self.stats['parallel_branches'] += len(candidates)
        
        return candidates
        
    def _extract_action_verbs(self, text: str) -> list[str]:
        """Extract action verbs from goal text."""
        words = text.lower().split()
        common_verbs = {
            'optimize', 'improve', 'enhance', 'debug', 'fix', 'resolve',
            'implement', 'create', 'build', 'design', 'develop', 'refactor',
            'analyze', 'test', 'deploy', 'integrate', 'migrate', 'update',
            'plan', 'address', 'audit', 'review', 'assess', 'identify',
            'consolidate', 'organize', 'prioritize', 'execute', 'validate'
        }
        return [w for w in words if w in common_verbs]
        
    def _extract_objects(self, text: str) -> list[str]:
        """Extract object nouns from goal text."""
        words = text.lower().split()
        # Simple heuristic - words that are likely objects
        objects = []
        for i, word in enumerate(words):
            if i > 0 and words[i-1] in ['the', 'a', 'an', 'optimize', 'implement', 'fix']:
                objects.append(word)
        return objects
        
    async def _generate_candidate(self, context: ThinkingContext, variation: int) -> dict[str, Any]:
        """Generate a single candidate step with variation."""
        # Use different heuristics based on variation
        heuristics = [
            self._greedy_heuristic,
            self._exploratory_heuristic,
            self._constraint_focused_heuristic,
            self._efficiency_heuristic,
            self._creative_heuristic
        ]
        
        heuristic = heuristics[variation % len(heuristics) if len(heuristics) > 0 else 0]
        
        return await heuristic(context)
        
    async def _score_candidates_gpu(self, candidates: list[dict[str, Any]], context: ThinkingContext) -> np.ndarray:
        """Score candidates using GPU acceleration with batching."""
        with self.profile_context('score_candidates_gpu'):
            # Convert candidates to numerical features
            features: np.ndarray = self._extract_features(candidates, context)
        
        # Create weights array from config
        weights = np.array([
            self.config.goal_relevance_weight,
            self.config.feasibility_weight,
            self.config.efficiency_weight,
            self.config.novelty_weight,
            self.config.constraint_weight
        ])
        
        if NUMBA_AVAILABLE:
            # Use JIT-compiled function for maximum performance
            scores = weighted_feature_aggregation(features, weights)
            self.stats['gpu_accelerated_steps'] += len(candidates)
        elif MLX_AVAILABLE and mx:
            # Batch processing for better GPU utilization
            batch_size = self.config.gpu_batch_size
            n_candidates = len(candidates)
            scores = np.zeros(n_candidates, dtype=np.float32)
            
            for i in range(0, n_candidates, batch_size):
                end_idx = min(i + batch_size, n_candidates)
                batch_features = features[i:end_idx]
                
                # Convert to MLX arrays
                mx_features = mx.array(batch_features)
                mx_weights = mx.array(weights)
                
                # Batch matrix multiplication
                batch_scores = mx.matmul(mx_features, mx_weights)
                scores[i:end_idx] = np.array(batch_scores)
            self.stats['gpu_accelerated_steps'] += len(candidates)
        else:
            # Fallback to NumPy with numexpr if available
            if ne is not None:
                # Use numexpr for SIMD optimization
                scores = ne.evaluate('sum(features * weights, axis=1)')
            else:
                scores = np.matmul(features, weights)
        
        return scores
        
    def _extract_features(self, candidates: list[dict[str, Any]], context: ThinkingContext) -> np.ndarray:
        """Extract numerical features from candidates - vectorized with semantic embeddings."""
        with self.profile_context('extract_features'):
            n_candidates = len(candidates)
            # Use pre-allocated memory pool
            features = self.memory_pool.get_feature_buffer(n_candidates)
            embeddings = self.memory_pool.get_embedding_buffer(n_candidates)
        
            # Get goal embedding once
            goal_embedding = self.embeddings.get_sentence_embedding(context.goal)
            
            # Batch compute candidate embeddings
            for i, candidate in enumerate(candidates):
                candidate_text = f"{candidate['action']} {candidate['reasoning']}"
                embeddings[i] = self.embeddings.get_sentence_embedding(candidate_text)
            
            # Batch compute semantic similarities using GPU
            semantic_scores = self.embeddings.batch_similarities(goal_embedding, embeddings[:n_candidates])
            features[:n_candidates, 0] = semantic_scores
            
        # Vectorized feasibility scoring
        features[:, 1] = 0.5  # Base score
        if context.constraints:
            constraint_words = [set(c.lower().split()) for c in context.constraints]
            for i, candidate in enumerate(candidates):
                action_words = set(candidate['action'].lower().split())
                matches = sum(1 for cw in constraint_words if action_words & cw)
                features[i, 1] += 0.1 * matches
                
        # Progress penalty
        progress = len(context.steps_completed) / context.max_steps
        features[:, 1] *= (1.0 - progress * 0.3)
        
        # Vectorized efficiency scoring
        features[:, 2] = 1.0 / (1.0 + np.arange(n_candidates) * 0.1)
        
        # Vectorized novelty scoring
        if context.steps_completed:
            recent_actions = [s.action for s in context.steps_completed[-5:]]
            for i, candidate in enumerate(candidates):
                if candidate['action'] in recent_actions:
                    features[i, 3] = 0.2
                else:
                    features[i, 3] = 0.9
        else:
            features[:, 3] = 1.0
            
        # Vectorized constraint satisfaction
        if context.constraints:
            for i, candidate in enumerate(candidates):
                satisfied = sum(1 for c in context.constraints 
                              if any(w in candidate['reasoning'].lower() 
                                    for w in c.lower().split()))
                features[i, 4] = satisfied / len(context.constraints)
        else:
                features[:, 4] = 1.0
                
        return features
        
    def _goal_relevance_score(self, candidate: dict[str, Any], context: ThinkingContext) -> float:
        """Score how relevant the candidate is to the goal using semantic embeddings."""
        # Get semantic embeddings
        goal_embedding = self.embeddings.get_sentence_embedding(context.goal)
        candidate_text = f"{candidate['action']} {candidate['reasoning']}"
        candidate_embedding = self.embeddings.get_sentence_embedding(candidate_text)
        
        # Compute semantic similarity
        semantic_score = self.embeddings.compute_similarity(goal_embedding, candidate_embedding)
        
        # Combine with keyword matching for robustness
        goal_words = set(context.goal.lower().split())
        candidate_words = set(candidate['action'].lower().split() + candidate['reasoning'].lower().split())
        keyword_score = len(goal_words & candidate_words) / len(goal_words) if goal_words else 0.0
        
        # Weighted combination
        return 0.7 * semantic_score + 0.3 * keyword_score
        
    def _feasibility_score(self, candidate: dict[str, Any], context: ThinkingContext) -> float:
        """Score how feasible the candidate action is."""
        score = 0.5  # Base score
        
        # Check if action respects constraints
        action_words = set(candidate['action'].lower().split())
        for constraint in context.constraints:
            constraint_words = set(constraint.lower().split())
            # Higher score if action mentions constraint concepts
            if action_words & constraint_words:
                score += 0.1
                
        # Check complexity based on current state
        if context.current_state:
            # Penalize if too many steps already taken
            progress = len(context.steps_completed) / context.max_steps
            score *= (1.0 - progress * 0.3)  # Reduce feasibility as we progress
            
        # Check for action feasibility indicators
        if any(word in candidate['action'].lower() for word in ['implement', 'create', 'build']):
            score *= 0.9  # Slightly lower for complex actions
        elif any(word in candidate['action'].lower() for word in ['analyze', 'check', 'review']):
            score *= 1.1  # Higher for analysis actions
            
        return min(1.0, max(0.0, score))
        
    def _efficiency_score(self, candidate: dict[str, Any], context: ThinkingContext) -> float:
        """Score the efficiency of the candidate."""
        # Prefer actions that make progress with fewer steps
        estimated_steps = candidate.get('estimated_steps', 1)
        return 1.0 / (1.0 + estimated_steps)
        
    def _novelty_score(self, candidate: dict[str, Any], context: ThinkingContext) -> float:
        """Score how novel the candidate is compared to previous steps."""
        if not context.steps_completed:
            return 1.0
            
        # Check similarity to previous actions
        previous_actions = [step.action for step in context.steps_completed[-5:]]
        
        if candidate['action'] in previous_actions:
            return 0.2
            
        return 0.9
        
    def _constraint_satisfaction_score(self, candidate: dict[str, Any], context: ThinkingContext) -> float:
        """Score how well the candidate satisfies constraints."""
        if not context.constraints:
            return 1.0
            
        # Check each constraint
        satisfied = 0
        for constraint in context.constraints:
            # Simple keyword check (enhance with proper constraint parsing)
            if any(word in candidate['reasoning'].lower() for word in constraint.lower().split()):
                satisfied += 1
                
        return satisfied / len(context.constraints)
        
    async def _execute_step(self, candidate: dict[str, Any], context: ThinkingContext) -> Any:
        """Execute the chosen step."""
        # Update context state
        context.current_state['last_action'] = candidate['action']
        context.current_state['step_count'] = len(context.steps_completed) + 1
        
        # Simulate execution (replace with actual execution logic)
        await asyncio.sleep(0.01)  # Minimal delay
        
        return {'status': 'completed', 'changes': candidate.get('expected_changes', {})}
        
    async def _is_goal_achieved(self, context: ThinkingContext) -> bool:
        """Check if the goal has been achieved."""
        if not context.steps_completed:
            return False
            
        # Check if we've taken enough meaningful steps
        if len(context.steps_completed) < self.config.min_steps_for_completion:
            return False
            
        # Check if recent steps show completion patterns
        last_step = context.steps_completed[-1]
        completion_indicators = [
            'complete', 'finished', 'done', 'achieved', 'ready',
            'implemented', 'optimized', 'resolved', 'final'
        ]
        
        if any(indicator in last_step.action.lower() for indicator in completion_indicators) and last_step.confidence > self.config.min_confidence_for_completion:
                return True
                
        # Check if we've addressed all constraints
        if context.constraints:
            addressed_constraints = set()
            for step in context.steps_completed:
                for constraint in context.constraints:
                    if any(word in step.action.lower() for word in constraint.lower().split()):
                        addressed_constraints.add(constraint)
                        
            # If we've addressed most constraints, consider it done
            if len(addressed_constraints) >= len(context.constraints) * self.config.constraint_satisfaction_ratio:
                return True
                
        # Default: continue if under max steps
        return len(context.steps_completed) >= context.max_steps
        
    # Heuristic functions for candidate generation
    async def _greedy_heuristic(self, context: ThinkingContext) -> dict[str, Any]:
        """Generate candidate that directly approaches the goal."""
        return {
            'action': f"Direct approach to {context.goal}",
            'reasoning': "Taking the most direct path to achieve the goal",
            'estimated_steps': 3
        }
        
    async def _exploratory_heuristic(self, context: ThinkingContext) -> dict[str, Any]:
        """Generate candidate that explores new possibilities."""
        return {
            'action': f"Explore alternative for {context.goal}",
            'reasoning': "Investigating less obvious approaches that might be more effective",
            'estimated_steps': 5
        }
        
    async def _constraint_focused_heuristic(self, context: ThinkingContext) -> dict[str, Any]:
        """Generate candidate that prioritizes constraints."""
        constraint_focus = context.constraints[0] if context.constraints else "requirements"
        return {
            'action': f"Address constraint: {constraint_focus}",
            'reasoning': f"Ensuring we satisfy {constraint_focus} before proceeding",
            'estimated_steps': 2
        }
        
    async def _efficiency_heuristic(self, context: ThinkingContext) -> dict[str, Any]:
        """Generate candidate that optimizes for efficiency."""
        return {
            'action': f"Optimize approach to {context.goal}",
            'reasoning': "Finding the most resource-efficient path",
            'estimated_steps': 4
        }
        
    async def _creative_heuristic(self, context: ThinkingContext) -> dict[str, Any]:
        """Generate creative/unconventional candidate."""
        return {
            'action': f"Creative solution for {context.goal}",
            'reasoning': "Applying lateral thinking to find innovative solutions",
            'estimated_steps': 6
        }
        
    # Additional thinking strategies
    async def _breadth_first_strategy(self, context: ThinkingContext) -> list[ThinkingStep]:
        """Explore all possibilities at each level before going deeper."""
        steps: list[ThinkingStep] = []
        level_candidates = await self._generate_parallel_candidates(context, self.cpu_cores * 2)
        
        while len(steps) < context.max_steps and level_candidates:
            # Score all candidates at this level
            scores = await self._score_candidates_gpu(level_candidates, context)
            
            # Add best from this level
            best_idx = np.argmax(scores)
            best = level_candidates[best_idx]
            
            step = ThinkingStep(
                step_number=len(steps) + 1,
                action=best['action'],
                reasoning=best['reasoning'],
                confidence=float(scores[best_idx]),
                metadata={'strategy': 'breadth_first'}
            )
            steps.append(step)
            context.steps_completed.append(step)
            
            # Generate next level
            level_candidates = await self._generate_parallel_candidates(context, self.cpu_cores * 2)
            
        return steps
        
    async def _depth_first_strategy(self, context: ThinkingContext) -> list[ThinkingStep]:
        """Follow one path deeply before backtracking."""
        steps: list[ThinkingStep] = []
        
        while len(steps) < context.max_steps:
            # Generate fewer candidates, go deeper
            candidates = await self._generate_parallel_candidates(context, 3)
            if not candidates:
                break
                
            scores = await self._score_candidates_gpu(candidates, context)
            best_idx = np.argmax(scores)
            
            step = ThinkingStep(
                step_number=len(steps) + 1,
                action=candidates[best_idx]['action'],
                reasoning=candidates[best_idx]['reasoning'],
                confidence=float(scores[best_idx]),
                metadata={'strategy': 'depth_first'}
            )
            steps.append(step)
            context.steps_completed.append(step)
            
            # Check if we should backtrack (low confidence)
            if step.confidence < 0.3 and len(steps) > 1:
                # Backtrack by removing last step
                steps.pop()
                context.steps_completed.pop()
                
        return steps
        
    async def _beam_search_strategy(self, context: ThinkingContext) -> list[ThinkingStep]:
        """Keep top-k candidates at each step."""
        beam_width = min(5, self.cpu_cores)
        steps: list[ThinkingStep] = []
        
        # Initialize beam with diverse candidates
        beam = await self._generate_parallel_candidates(context, beam_width)
        
        while len(steps) < context.max_steps and beam:
            # Expand beam
            all_candidates = []
            for _candidate in beam:
                # Generate successors for each beam candidate
                new_candidates = await self._generate_parallel_candidates(context, 2)
                all_candidates.extend(new_candidates)
                
            if not all_candidates:
                break
                
            # Score and select top-k
            scores = await self._score_candidates_gpu(all_candidates, context)
            top_k_indices = np.argsort(scores)[-beam_width:][::-1]
            
            # Add best to steps
            best_idx = top_k_indices[0]
            step = ThinkingStep(
                step_number=len(steps) + 1,
                action=all_candidates[best_idx]['action'],
                reasoning=all_candidates[best_idx]['reasoning'],
                confidence=float(scores[best_idx]),
                metadata={'strategy': 'beam_search', 'beam_width': beam_width}
            )
            steps.append(step)
            context.steps_completed.append(step)
            
            # Update beam
            beam = [all_candidates[i] for i in top_k_indices]
            
        return steps
        
    async def _monte_carlo_strategy(self, context: ThinkingContext) -> list[ThinkingStep]:
        """Use Monte Carlo tree search for exploration."""
        steps: list[ThinkingStep] = []
        n_simulations = 100
        
        while len(steps) < context.max_steps:
            # Run multiple simulations
            simulation_results = []
            
            for _ in range(n_simulations):
                # Simulate a path
                sim_candidates = await self._generate_parallel_candidates(context, 1)
                if sim_candidates:
                    score = (await self._score_candidates_gpu(sim_candidates, context))[0]
                    simulation_results.append((sim_candidates[0], score))
                    
            if not simulation_results:
                break
                
            # Select action based on simulation results
            best_candidate, best_score = max(simulation_results, key=lambda x: x[1])
            
            step = ThinkingStep(
                step_number=len(steps) + 1,
                action=best_candidate['action'],
                reasoning=best_candidate['reasoning'],
                confidence=float(best_score),
                metadata={'strategy': 'monte_carlo', 'simulations': n_simulations}
            )
            steps.append(step)
            context.steps_completed.append(step)
            
        return steps
        
    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        cache_size = 0
        try:
            with self.cache_env.begin() as txn:
                cache_size = txn.stat()['entries']
        except Exception:
            pass
            
        return {
            **self.stats,
            'cpu_cores_used': self.cpu_cores,
            'gpu_available': MLX_AVAILABLE and mx and hasattr(mx, 'metal') and mx.metal.is_available(),
            'hardware': self.hw.platform_info,
            'cache_entries': cache_size,
            'cache_hit_rate': self.stats.get('cache_hits', 0) / max(1, self.stats.get('total_requests', 0))
        }
        
    def close(self):
        """Clean up resources."""
        self.process_pool.shutdown(wait=True)
        self.thread_pool.shutdown(wait=True)
        self.cache_env.close()
        
    # Profiling methods
    def enable_profiling(self):
        """Enable performance profiling."""
        self.profiling_enabled = True
        
    def disable_profiling(self):
        """Disable performance profiling."""
        self.profiling_enabled = False
        
    @contextmanager
    def profile_context(self, name: str):
        """Context manager for profiling specific sections."""
        if not self.profiling_enabled:
            yield
            return
            
        start_time = time.perf_counter()
        start_memory = 0
        try:
            import tracemalloc
            if tracemalloc.is_tracing():
                start_memory = tracemalloc.get_traced_memory()[0]
        except Exception:
            pass
            
        yield
        
        elapsed = time.perf_counter() - start_time
        end_memory = 0
        try:
            if tracemalloc.is_tracing():
                end_memory = tracemalloc.get_traced_memory()[0]
        except Exception:
            pass
            
        self.profile_data.append({
            'name': name,
            'time': elapsed,
            'memory_delta': end_memory - start_memory,
            'timestamp': time.time()
        })
        
    def profile_function(self, func: Callable) -> Callable:
        """Decorator for profiling functions."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not self.profiling_enabled:
                return await func(*args, **kwargs)
                
            with self.profile_context(func.__name__):
                return await func(*args, **kwargs)
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not self.profiling_enabled:
                return func(*args, **kwargs)
                
            with self.profile_context(func.__name__):
                return func(*args, **kwargs)
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
        
    def get_profile_report(self) -> dict[str, Any]:
        """Get detailed profiling report."""
        if not self.profile_data:
            return {'message': 'No profiling data available'}
            
        # Aggregate by function name
        aggregated = {}
        for entry in self.profile_data:
            name = entry['name']
            if name not in aggregated:
                aggregated[name] = {
                    'count': 0,
                    'total_time': 0,
                    'avg_time': 0,
                    'max_time': 0,
                    'min_time': float('inf'),
                    'total_memory': 0
                }
            
            agg = aggregated[name]
            agg['count'] += 1
            agg['total_time'] += entry['time']
            agg['max_time'] = max(agg['max_time'], entry['time'])
            agg['min_time'] = min(agg['min_time'], entry['time'])
            agg['total_memory'] += entry['memory_delta']
            
        # Calculate averages
        for _name, agg in aggregated.items():
            agg['avg_time'] = agg['total_time'] / agg['count']
            agg['avg_memory'] = agg['total_memory'] / agg['count']
            
        # Sort by total time
        sorted_funcs = sorted(aggregated.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        return {
            'summary': {
                'total_entries': len(self.profile_data),
                'unique_functions': len(aggregated),
                'total_time': sum(e['time'] for e in self.profile_data),
                'profiling_overhead': len(self.profile_data) * 0.0001  # Estimated overhead
            },
            'top_by_time': dict(sorted_funcs[:10]),
            'raw_data': self.profile_data[-100:]  # Last 100 entries
        }
        
    def clear_profile_data(self):
        """Clear profiling data."""
        self.profile_data = []


# Singleton instance
_sequential_thinking_instance = None


def get_sequential_thinking(config: ThinkingConfig | None = None) -> SequentialThinkingTurbo:
    """Get or create the sequential thinking instance."""
    global _sequential_thinking_instance
    if _sequential_thinking_instance is None:
        _sequential_thinking_instance = SequentialThinkingTurbo(config)
    return _sequential_thinking_instance


# Example usage
async def demo():
    """Demonstrate hardware-accelerated sequential thinking."""
    thinking = get_sequential_thinking()
    
    # Enable profiling for performance analysis
    thinking.enable_profiling()
    
    # Complex multi-step reasoning task
    steps = await thinking.think(
        goal="Implement a high-performance trading system",
        constraints=[
            "Must handle real-time data",
            "Risk management required",
            "Use all available hardware"
        ],
        strategy='parallel_explore',
        max_steps=20
    )
    
    print(f"Completed {len(steps)} thinking steps")
    for step in steps:
        print(f"Step {step.step_number}: {step.action} (confidence: {step.confidence:.2f})")
        
    print(f"\nPerformance stats: {thinking.get_stats()}")
    
    # Show profiling report
    profile_report = thinking.get_profile_report()
    print("\nProfiling Report:")
    print(f"Total time: {profile_report['summary']['total_time']:.3f}s")
    print("Top functions by time:")
    for func, data in list(profile_report['top_by_time'].items())[:5]:
        print(f"  {func}: {data['total_time']:.3f}s ({data['count']} calls, avg: {data['avg_time']:.3f}s)")


if __name__ == "__main__":
    asyncio.run(demo())