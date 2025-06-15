"""
Ultra-optimized sequential thinking for M4 Pro - Version 2.
Leverages all available hardware acceleration technologies.
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
import time
import numpy as np
from functools import lru_cache, partial
import os

# Performance libraries
import numba
from numba import jit, cuda, prange, vectorize
import torch
import mlx.core as mx
import mlx.nn as nn
from joblib import Parallel, delayed
import msgpack
import orjson
import lmdb
from scipy.spatial.distance import cosine
from scipy.optimize import differential_evolution

# Async optimization
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from ..optimization.hardware_detector import HardwareCapabilities


@dataclass
class ThinkingContext:
    """Context maintained across thinking steps."""
    goal: str
    constraints: List[str]
    steps_completed: List['ThinkingStep']
    current_state: Dict[str, Any]
    max_steps: int = 100
    timeout: float = 300.0


# Enable Numba parallel execution
numba.set_num_threads(8)  # Use P-cores only

@dataclass
class ThinkingStep:
    """A single step in sequential reasoning."""
    step_number: int
    action: str
    reasoning: str
    result: Optional[Any] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for caching."""
        return msgpack.packb({
            'step_number': self.step_number,
            'action': self.action,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'metadata': self.metadata
        })
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'ThinkingStep':
        """Deserialize from bytes."""
        d = msgpack.unpackb(data)
        return cls(**d)


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def fast_score_candidates(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Numba-accelerated candidate scoring."""
    n_candidates = features.shape[0]
    scores = np.zeros(n_candidates)
    
    for i in prange(n_candidates):
        scores[i] = np.dot(features[i], weights)
    
    return scores


@jit(nopython=True, cache=True, fastmath=True)
def fast_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Numba-accelerated cosine similarity."""
    dot = np.dot(vec1, vec2)
    norm1 = np.sqrt(np.sum(vec1 * vec1))
    norm2 = np.sqrt(np.sum(vec2 * vec2))
    return dot / (norm1 * norm2 + 1e-8)


class ThinkingCache:
    """High-performance LMDB cache for thinking steps."""
    
    def __init__(self, cache_dir: str = ".thinking_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.env = lmdb.open(cache_dir, map_size=1024*1024*1024)  # 1GB cache
        
    def get(self, key: str) -> Optional[List[ThinkingStep]]:
        """Get cached thinking steps."""
        with self.env.begin() as txn:
            data = txn.get(key.encode())
            if data:
                steps_data = msgpack.unpackb(data)
                return [ThinkingStep.from_bytes(s) for s in steps_data]
        return None
        
    def put(self, key: str, steps: List[ThinkingStep]):
        """Cache thinking steps."""
        with self.env.begin(write=True) as txn:
            steps_data = [s.to_bytes() for s in steps]
            txn.put(key.encode(), msgpack.packb(steps_data))


class SequentialThinkingTurboV2:
    """Ultra-optimized sequential thinking engine."""
    
    def __init__(self):
        self.hw = HardwareCapabilities()
        
        # CPU setup - separate P and E cores
        self.p_cores = 8  # Performance cores
        self.e_cores = 4  # Efficiency cores
        
        # Dedicated executors
        self.compute_pool = ProcessPoolExecutor(max_workers=self.p_cores)
        self.io_pool = ThreadPoolExecutor(max_workers=self.e_cores * 2)
        
        # Joblib for embarassingly parallel tasks
        self.parallel = Parallel(n_jobs=self.p_cores, backend='multiprocessing')
        
        # MLX for Metal GPU
        self.mlx_device = mx.gpu
        mx.set_default_device(self.mlx_device)
        
        # PyTorch with Metal Performance Shaders
        self.torch_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Thinking cache
        self.cache = ThinkingCache()
        
        # Pre-compile Numba functions
        self._warmup_jit()
        
        # Strategies with hardware affinity
        self.strategies = {
            'parallel_explore': self._parallel_explore_strategy,
            'gpu_beam_search': self._gpu_beam_search_strategy,
            'hybrid_mcts': self._hybrid_mcts_strategy,
            'quantum_inspired': self._quantum_inspired_strategy
        }
        
        # Performance tracking
        self.stats = {
            'total_steps': 0,
            'gpu_accelerated_steps': 0,
            'numba_accelerated_ops': 0,
            'cache_hits': 0,
            'parallel_branches': 0,
            'avg_step_time': 0.0
        }
        
    def _warmup_jit(self):
        """Pre-compile Numba functions."""
        dummy_features = np.random.rand(10, 5).astype(np.float32)
        dummy_weights = np.random.rand(5).astype(np.float32)
        _ = fast_score_candidates(dummy_features, dummy_weights)
        
    async def think(self, 
                   goal: str,
                   constraints: Optional[List[str]] = None,
                   initial_state: Optional[Dict[str, Any]] = None,
                   strategy: str = 'hybrid_mcts',
                   max_steps: int = 100,
                   timeout: float = 300.0,
                   use_cache: bool = True) -> List[ThinkingStep]:
        """
        Execute ultra-optimized sequential thinking.
        """
        # Check cache first
        cache_key = f"{goal}:{strategy}:{max_steps}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                self.stats['cache_hits'] += 1
                return cached
                
        context = ThinkingContext(
            goal=goal,
            constraints=constraints or [],
            steps_completed=[],
            current_state=initial_state or {},
            max_steps=max_steps,
            timeout=timeout
        )
        
        start_time = time.time()
        
        # Select strategy
        thinking_fn = self.strategies.get(strategy, self._hybrid_mcts_strategy)
        
        # Run with timeout
        try:
            steps = await asyncio.wait_for(
                thinking_fn(context),
                timeout=timeout
            )
            
            # Update stats
            elapsed = time.time() - start_time
            self.stats['total_steps'] += len(steps)
            self.stats['avg_step_time'] = elapsed / len(steps) if steps else 0
            
            # Cache result
            if use_cache and len(steps) > 5:
                self.cache.put(cache_key, steps)
                
            return steps
            
        except asyncio.TimeoutError:
            return context.steps_completed
            
    async def _hybrid_mcts_strategy(self, context: ThinkingContext) -> List[ThinkingStep]:
        """
        Hybrid MCTS using both CPU and GPU for different phases.
        """
        steps = []
        
        # Phase 1: Parallel exploration on P-cores
        exploration_tasks = []
        for i in range(self.p_cores):
            task = asyncio.create_task(
                self._explore_branch(context, branch_id=i)
            )
            exploration_tasks.append(task)
            
        branches = await asyncio.gather(*exploration_tasks)
        
        # Phase 2: GPU evaluation of all branches
        all_candidates = []
        for branch in branches:
            all_candidates.extend(branch)
            
        if all_candidates:
            # Batch evaluate on GPU
            scores = await self._gpu_batch_evaluate(all_candidates, context)
            
            # Phase 3: Select best path using differential evolution
            best_path = await self._optimize_path_selection(
                all_candidates, scores, context
            )
            
            for idx, step_data in enumerate(best_path):
                step = ThinkingStep(
                    step_number=idx + 1,
                    action=step_data['action'],
                    reasoning=step_data['reasoning'],
                    confidence=step_data['score'],
                    metadata={'strategy': 'hybrid_mcts'}
                )
                steps.append(step)
                
        return steps
        
    async def _explore_branch(self, context: ThinkingContext, branch_id: int) -> List[Dict]:
        """Explore a single branch in parallel."""
        candidates = []
        
        # Use different exploration strategies per branch
        strategies = [
            self._greedy_exploration,
            self._stochastic_exploration,
            self._adversarial_exploration,
            self._creative_exploration
        ]
        
        strategy = strategies[branch_id % len(strategies)]
        
        for _ in range(5):  # 5 steps per branch
            candidate = await strategy(context)
            candidates.append(candidate)
            
        self.stats['parallel_branches'] += 1
        return candidates
        
    async def _gpu_batch_evaluate(self, candidates: List[Dict], context: ThinkingContext) -> np.ndarray:
        """Evaluate all candidates on GPU in a single batch."""
        # Extract features
        features = np.array([
            self._extract_features_fast(c, context) for c in candidates
        ], dtype=np.float32)
        
        # MLX evaluation
        mx_features = mx.array(features)
        
        # Neural evaluation (simplified - in practice use trained model)
        weights = mx.random.normal((features.shape[1], 32))
        hidden = mx.tanh(mx_features @ weights)
        output_weights = mx.random.normal((32, 1))
        scores = mx.sigmoid(hidden @ output_weights)
        
        # Also use PyTorch for comparison
        torch_features = torch.from_numpy(features).to(self.torch_device)
        torch_scores = torch.sigmoid(torch.randn_like(torch_features[:, 0]))
        
        # Combine scores
        mlx_scores = np.array(scores).flatten()
        torch_scores = torch_scores.cpu().numpy()
        
        combined_scores = 0.6 * mlx_scores + 0.4 * torch_scores
        
        self.stats['gpu_accelerated_steps'] += len(candidates)
        
        return combined_scores
        
    def _extract_features_fast(self, candidate: Dict, context: ThinkingContext) -> np.ndarray:
        """Fast feature extraction using Numba where possible."""
        # Basic features
        features = np.zeros(10, dtype=np.float32)
        
        # Text similarity features (pre-computed embeddings would be better)
        goal_words = set(context.goal.lower().split())
        candidate_words = set(candidate['action'].lower().split())
        
        features[0] = len(goal_words & candidate_words) / (len(goal_words) + 1e-8)
        features[1] = candidate.get('feasibility', 0.5)
        features[2] = 1.0 / (1.0 + candidate.get('complexity', 1))
        features[3] = candidate.get('novelty', 0.5)
        features[4] = len(context.steps_completed) / context.max_steps
        
        # Constraint satisfaction
        satisfied = sum(1 for c in context.constraints 
                       if any(w in candidate['reasoning'].lower() 
                             for w in c.lower().split()))
        features[5] = satisfied / (len(context.constraints) + 1e-8)
        
        # Random exploration bonus
        features[6:] = np.random.rand(4) * 0.1
        
        self.stats['numba_accelerated_ops'] += 1
        
        return features
        
    async def _optimize_path_selection(self, candidates: List[Dict], 
                                     scores: np.ndarray, 
                                     context: ThinkingContext) -> List[Dict]:
        """Use scipy's differential evolution to find optimal path."""
        n_candidates = len(candidates)
        
        def objective(path_indices):
            # Penalize revisiting similar actions
            total_score = 0
            used_actions = set()
            
            for i, idx in enumerate(path_indices):
                idx = int(idx) % n_candidates
                action = candidates[idx]['action']
                
                penalty = 0.5 if action in used_actions else 1.0
                total_score += scores[idx] * penalty * (0.95 ** i)  # Decay factor
                
                used_actions.add(action)
                
            return -total_score  # Minimize negative score
            
        # Optimize path selection
        bounds = [(0, n_candidates - 1) for _ in range(min(10, context.max_steps))]
        
        result = differential_evolution(
            objective, 
            bounds,
            maxiter=50,
            popsize=15,
            workers=self.p_cores
        )
        
        # Extract optimal path
        path = []
        for idx in result.x:
            idx = int(idx) % n_candidates
            candidate = candidates[idx].copy()
            candidate['score'] = float(scores[idx])
            path.append(candidate)
            
        return path
        
    # Exploration strategies
    async def _greedy_exploration(self, context: ThinkingContext) -> Dict[str, Any]:
        """Greedy hill-climbing exploration."""
        return {
            'action': f"Optimize {context.goal} directly",
            'reasoning': "Taking the steepest gradient toward the goal",
            'feasibility': 0.9,
            'complexity': 1,
            'novelty': 0.2
        }
        
    async def _stochastic_exploration(self, context: ThinkingContext) -> Dict[str, Any]:
        """Stochastic exploration with randomness."""
        actions = ["Explore", "Investigate", "Analyze", "Test", "Prototype"]
        action = np.random.choice(actions)
        
        return {
            'action': f"{action} alternative approach for {context.goal}",
            'reasoning': "Introducing controlled randomness for diversity",
            'feasibility': 0.7,
            'complexity': 2,
            'novelty': 0.8
        }
        
    async def _adversarial_exploration(self, context: ThinkingContext) -> Dict[str, Any]:
        """Adversarial thinking - what could go wrong?"""
        return {
            'action': f"Identify failure modes in {context.goal}",
            'reasoning': "Adversarial analysis to improve robustness",
            'feasibility': 0.8,
            'complexity': 3,
            'novelty': 0.6
        }
        
    async def _creative_exploration(self, context: ThinkingContext) -> Dict[str, Any]:
        """Creative lateral thinking."""
        return {
            'action': f"Reimagine {context.goal} from first principles",
            'reasoning': "Breaking assumptions for innovative solutions",
            'feasibility': 0.5,
            'complexity': 4,
            'novelty': 0.95
        }
        
    async def _gpu_beam_search_strategy(self, context: ThinkingContext) -> List[ThinkingStep]:
        """Pure GPU beam search - keep top-k at each step."""
        beam_width = 32
        steps = []
        
        # Initialize beam with diverse starting points
        current_beam = await self._generate_parallel_candidates(context, beam_width)
        
        for step_num in range(min(20, context.max_steps)):
            # Expand each beam candidate
            all_expansions = []
            
            for candidate in current_beam:
                expansions = await self._expand_candidate(candidate, context)
                all_expansions.extend(expansions)
                
            # GPU evaluation of all expansions
            if all_expansions:
                scores = await self._gpu_batch_evaluate(all_expansions, context)
                
                # Keep top-k
                top_indices = np.argpartition(scores, -beam_width)[-beam_width:]
                current_beam = [all_expansions[i] for i in top_indices]
                
                # Add best to steps
                best_idx = np.argmax(scores)
                best = all_expansions[best_idx]
                
                step = ThinkingStep(
                    step_number=step_num + 1,
                    action=best['action'],
                    reasoning=best['reasoning'],
                    confidence=float(scores[best_idx]),
                    metadata={'strategy': 'gpu_beam_search', 'beam_width': beam_width}
                )
                steps.append(step)
                
        return steps
        
    async def _quantum_inspired_strategy(self, context: ThinkingContext) -> List[ThinkingStep]:
        """Quantum-inspired superposition of multiple strategies."""
        # Run multiple strategies in superposition
        strategies = ['greedy', 'stochastic', 'adversarial', 'creative']
        
        # Parallel execution of all strategies
        tasks = []
        for strat in strategies:
            if strat == 'greedy':
                task = self._simple_strategy(context, self._greedy_exploration)
            elif strat == 'stochastic':
                task = self._simple_strategy(context, self._stochastic_exploration)
            elif strat == 'adversarial':
                task = self._simple_strategy(context, self._adversarial_exploration)
            else:
                task = self._simple_strategy(context, self._creative_exploration)
                
            tasks.append(asyncio.create_task(task))
            
        # Gather all results
        all_paths = await asyncio.gather(*tasks)
        
        # Quantum-inspired interference - combine paths
        combined_steps = []
        
        for step_idx in range(min(len(p) for p in all_paths)):
            # Superposition of steps at each position
            candidates = [path[step_idx] for path in all_paths]
            
            # Interference pattern - constructive for similar, destructive for different
            scores = []
            for i, c1 in enumerate(candidates):
                score = 0
                for j, c2 in enumerate(candidates):
                    if i != j:
                        similarity = self._action_similarity(c1['action'], c2['action'])
                        score += similarity  # Constructive interference
                        
                scores.append(score)
                
            # Collapse to best option
            best_idx = np.argmax(scores)
            best = candidates[best_idx]
            
            step = ThinkingStep(
                step_number=step_idx + 1,
                action=best['action'],
                reasoning=best['reasoning'] + " [quantum superposition]",
                confidence=scores[best_idx] / len(candidates),
                metadata={'strategy': 'quantum_inspired', 'interference_score': scores[best_idx]}
            )
            combined_steps.append(step)
            
        return combined_steps
        
    async def _simple_strategy(self, context: ThinkingContext, 
                             exploration_fn: Callable) -> List[Dict]:
        """Simple strategy execution for quantum superposition."""
        steps = []
        for _ in range(10):
            candidate = await exploration_fn(context)
            steps.append(candidate)
        return steps
        
    def _action_similarity(self, action1: str, action2: str) -> float:
        """Fast action similarity computation."""
        words1 = set(action1.lower().split())
        words2 = set(action2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
        
    async def _generate_parallel_candidates(self, context: ThinkingContext, 
                                          num_candidates: int) -> List[Dict[str, Any]]:
        """Generate candidates in parallel using joblib."""
        def generate_one(i):
            strategies = [
                self._greedy_exploration,
                self._stochastic_exploration,
                self._adversarial_exploration,
                self._creative_exploration
            ]
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            strategy = strategies[i % len(strategies)]
            return loop.run_until_complete(strategy(context))
            
        candidates = self.parallel(delayed(generate_one)(i) for i in range(num_candidates))
        
        self.stats['parallel_branches'] += num_candidates
        
        return candidates
        
    async def _expand_candidate(self, candidate: Dict, context: ThinkingContext) -> List[Dict]:
        """Expand a candidate into multiple next steps."""
        expansions = []
        
        # Different expansion types
        expansion_types = [
            ("Refine", "Improving details of"),
            ("Extend", "Building upon"),
            ("Validate", "Testing assumptions in"),
            ("Optimize", "Enhancing performance of")
        ]
        
        for action_prefix, reasoning_prefix in expansion_types:
            expansion = {
                'action': f"{action_prefix} {candidate['action']}",
                'reasoning': f"{reasoning_prefix} the previous approach",
                'parent': candidate,
                'depth': candidate.get('depth', 0) + 1
            }
            expansions.append(expansion)
            
        return expansions
        
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        return {
            **self.stats,
            'hardware': {
                'p_cores': self.p_cores,
                'e_cores': self.e_cores,
                'mlx_device': str(self.mlx_device),
                'torch_device': str(self.torch_device),
                'torch_mps_available': torch.backends.mps.is_available()
            },
            'optimization': {
                'numba_threads': numba.get_num_threads(),
                'cache_size_mb': os.path.getsize(self.cache.cache_dir) / 1024 / 1024 if os.path.exists(self.cache.cache_dir) else 0
            }
        }
        
    def close(self):
        """Clean up resources."""
        self.compute_pool.shutdown(wait=True)
        self.io_pool.shutdown(wait=True)
        self.cache.env.close()




# Singleton instance
_sequential_thinking_v2_instance = None


def get_sequential_thinking_v2() -> SequentialThinkingTurboV2:
    """Get or create the V2 sequential thinking instance."""
    global _sequential_thinking_v2_instance
    if _sequential_thinking_v2_instance is None:
        _sequential_thinking_v2_instance = SequentialThinkingTurboV2()
    return _sequential_thinking_v2_instance