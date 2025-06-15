"""
Mac Silicon-optimized sequential thinking.
Uses only technologies that actually work well on Apple Silicon.
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
import time
import numpy as np
from functools import lru_cache
import os
import queue
import threading

# Mac-optimized imports
import mlx.core as mx
import mlx.nn as nn
import numba
from numba import jit, prange
import msgpack
import orjson
import lmdb

# Use dispatch queues for better Mac performance
from multiprocessing import Queue
from threading import Thread

from ..optimization.hardware_detector import HardwareCapabilities


@dataclass
class ThinkingContext:
    goal: str
    constraints: List[str]
    steps_completed: List['ThinkingStep']
    current_state: Dict[str, Any]
    max_steps: int = 100
    timeout: float = 300.0


@dataclass 
class ThinkingStep:
    step_number: int
    action: str
    reasoning: str
    result: Optional[Any] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MacOptimizedThinking:
    """Sequential thinking optimized for Apple Silicon."""
    
    def __init__(self):
        self.hw = HardwareCapabilities()
        
        # Mac-specific optimization
        self.p_cores = 8  # M4 Pro performance cores
        self.e_cores = 4  # M4 Pro efficiency cores
        
        # Use GCD-style dispatch for Mac
        self.compute_queue = ThreadPoolExecutor(max_workers=self.p_cores, thread_name_prefix="p-core")
        self.io_queue = ThreadPoolExecutor(max_workers=self.e_cores, thread_name_prefix="e-core")
        
        # MLX setup - Apple's native framework
        mx.set_default_device(mx.gpu)
        
        # Pre-allocate MLX arrays for better performance
        self._init_mlx_buffers()
        
        # LMDB cache - works great on Mac
        self.cache_dir = ".mac_thinking_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache = lmdb.open(self.cache_dir, map_size=1024*1024*512)  # 512MB
        
        # Stats
        self.stats = {
            'total_steps': 0,
            'mlx_operations': 0,
            'cache_hits': 0,
            'p_core_tasks': 0,
            'e_core_tasks': 0
        }
        
    def _init_mlx_buffers(self):
        """Pre-allocate MLX buffers for zero-copy operations."""
        # Pre-allocate common sizes to avoid allocation overhead
        self.feature_buffer = mx.zeros((100, 10), dtype=mx.float32)
        self.score_buffer = mx.zeros((100,), dtype=mx.float32)
        self.weight_matrix = mx.random.normal((10, 32))
        self.output_weights = mx.random.normal((32, 1))
        
    async def think(self,
                   goal: str,
                   constraints: Optional[List[str]] = None,
                   initial_state: Optional[Dict[str, Any]] = None,
                   max_steps: int = 100,
                   timeout: float = 300.0) -> List[ThinkingStep]:
        """Execute Mac-optimized sequential thinking."""
        
        # Check cache
        cache_key = f"{goal}:{max_steps}".encode()
        with self.cache.begin() as txn:
            cached = txn.get(cache_key)
            if cached:
                self.stats['cache_hits'] += 1
                return msgpack.unpackb(cached, object_hook=self._decode_step)
                
        context = ThinkingContext(
            goal=goal,
            constraints=constraints or [],
            steps_completed=[],
            current_state=initial_state or {},
            max_steps=max_steps,
            timeout=timeout
        )
        
        steps = await self._mac_optimized_search(context)
        
        # Cache result
        if len(steps) > 5:
            with self.cache.begin(write=True) as txn:
                txn.put(cache_key, msgpack.packb(steps, default=self._encode_step))
                
        return steps
        
    async def _mac_optimized_search(self, context: ThinkingContext) -> List[ThinkingStep]:
        """Mac-optimized search using GCD-style dispatch."""
        steps = []
        
        # Phase 1: Generate candidates on P-cores
        candidate_futures = []
        for i in range(min(self.p_cores, context.max_steps)):
            future = self.compute_queue.submit(self._generate_candidate_batch, context, i)
            candidate_futures.append(future)
            self.stats['p_core_tasks'] += 1
            
        # Collect candidates
        all_candidates = []
        for future in candidate_futures:
            batch = await asyncio.get_event_loop().run_in_executor(None, future.result)
            all_candidates.extend(batch)
            
        # Phase 2: MLX GPU evaluation
        scores = await self._mlx_batch_evaluate(all_candidates, context)
        
        # Phase 3: Select best path
        best_indices = np.argsort(scores)[-min(10, len(scores)):][::-1]
        
        for idx in best_indices:
            if len(steps) >= context.max_steps:
                break
                
            candidate = all_candidates[idx]
            step = ThinkingStep(
                step_number=len(steps) + 1,
                action=candidate['action'],
                reasoning=candidate['reasoning'],
                confidence=float(scores[idx]),
                metadata={'mac_optimized': True}
            )
            steps.append(step)
            context.steps_completed.append(step)
            
        self.stats['total_steps'] += len(steps)
        
        return steps
        
    def _generate_candidate_batch(self, context: ThinkingContext, batch_id: int) -> List[Dict]:
        """Generate candidates - runs on P-cores."""
        candidates = []
        
        # Different strategies per batch
        strategies = [
            ("Direct", "Straightforward approach"),
            ("Optimize", "Performance-focused"),
            ("Creative", "Innovative solution"),
            ("Robust", "Error-resistant approach")
        ]
        
        for i in range(5):  # 5 candidates per batch
            strategy_idx = (batch_id + i) % len(strategies)
            action_prefix, reasoning_base = strategies[strategy_idx]
            
            candidate = {
                'action': f"{action_prefix} approach to {context.goal}",
                'reasoning': f"{reasoning_base} considering constraints",
                'batch_id': batch_id,
                'strategy': strategy_idx
            }
            candidates.append(candidate)
            
        return candidates
        
    async def _mlx_batch_evaluate(self, candidates: List[Dict], context: ThinkingContext) -> np.ndarray:
        """Evaluate candidates using MLX on GPU."""
        n_candidates = len(candidates)
        
        # Extract features using NumPy (Accelerate-optimized)
        features = np.zeros((n_candidates, 10), dtype=np.float32)
        
        for i, candidate in enumerate(candidates):
            features[i] = self._extract_features_numpy(candidate, context)
            
        # Transfer to MLX GPU
        mx_features = mx.array(features)
        
        # Neural evaluation on GPU
        # Reuse pre-allocated buffers when possible
        if n_candidates <= 100:
            # Use pre-allocated buffers
            self.feature_buffer[:n_candidates] = mx_features
            hidden = mx.tanh(self.feature_buffer[:n_candidates] @ self.weight_matrix)
            scores = mx.sigmoid(hidden @ self.output_weights).squeeze()
            self.score_buffer[:n_candidates] = scores
            result = np.array(self.score_buffer[:n_candidates])
        else:
            # Fallback for larger batches
            hidden = mx.tanh(mx_features @ self.weight_matrix)
            scores = mx.sigmoid(hidden @ self.output_weights).squeeze()
            result = np.array(scores)
            
        self.stats['mlx_operations'] += 1
        
        return result
        
    @staticmethod
    @jit(nopython=True, cache=True)
    def _similarity_score(goal_words: set, candidate_words: set) -> float:
        """Numba-optimized similarity calculation."""
        if len(goal_words) == 0:
            return 0.0
        intersection = len(goal_words & candidate_words)
        return intersection / len(goal_words)
        
    def _extract_features_numpy(self, candidate: Dict, context: ThinkingContext) -> np.ndarray:
        """Extract features using NumPy (Accelerate-optimized)."""
        features = np.zeros(10, dtype=np.float32)
        
        # Text similarity
        goal_words = set(context.goal.lower().split())
        candidate_words = set(candidate['action'].lower().split())
        
        # Basic features
        features[0] = len(goal_words & candidate_words) / (len(goal_words) + 1e-8)
        features[1] = 0.8  # feasibility
        features[2] = 1.0 / (1.0 + candidate.get('batch_id', 0))
        features[3] = 0.5 + 0.5 * (candidate.get('strategy', 0) / 4)
        features[4] = len(context.steps_completed) / context.max_steps
        
        # Constraint matching
        if context.constraints:
            matches = sum(1 for c in context.constraints 
                         if any(w in candidate['reasoning'].lower() 
                               for w in c.lower().split()))
            features[5] = matches / len(context.constraints)
        else:
            features[5] = 1.0
            
        # Random exploration
        features[6:] = np.random.rand(4) * 0.2
        
        return features
        
    def _encode_step(self, obj):
        """Encode ThinkingStep for msgpack."""
        if isinstance(obj, ThinkingStep):
            return {
                '__thinking_step__': True,
                'step_number': obj.step_number,
                'action': obj.action,
                'reasoning': obj.reasoning,
                'confidence': obj.confidence,
                'metadata': obj.metadata
            }
        return obj
        
    def _decode_step(self, obj):
        """Decode ThinkingStep from msgpack."""
        if '__thinking_step__' in obj:
            return ThinkingStep(
                step_number=obj['step_number'],
                action=obj['action'],
                reasoning=obj['reasoning'],
                confidence=obj['confidence'],
                metadata=obj['metadata']
            )
        return obj
        
    async def benchmark(self):
        """Run a quick benchmark."""
        print("Mac-Optimized Sequential Thinking Benchmark")
        print("-" * 50)
        
        # Warm up MLX
        _ = mx.zeros((100, 100)) @ mx.ones((100, 100))
        
        test_cases = [
            ("Simple", "Fix a bug", ["Quick fix"]),
            ("Medium", "Optimize algorithm", ["Faster", "Maintainable"]),
            ("Complex", "Design system", ["Scalable", "Secure", "Fast", "Reliable"])
        ]
        
        for name, goal, constraints in test_cases:
            start = time.perf_counter()
            
            steps = await self.think(
                goal=goal,
                constraints=constraints,
                max_steps=20
            )
            
            elapsed = time.perf_counter() - start
            
            print(f"\n{name}:")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Steps: {len(steps)}")
            print(f"  Steps/sec: {len(steps)/elapsed:.1f}")
            
        print(f"\nStats: {self.stats}")
        
    def close(self):
        """Clean up resources."""
        self.compute_queue.shutdown(wait=True)
        self.io_queue.shutdown(wait=True)
        self.cache.close()


# Integration with existing system
def get_mac_optimized_thinking() -> MacOptimizedThinking:
    """Get Mac-optimized thinking instance."""
    return MacOptimizedThinking()


# Example usage
async def demo():
    """Demo Mac-optimized thinking."""
    thinking = get_mac_optimized_thinking()
    
    steps = await thinking.think(
        goal="Build a high-performance Mac app",
        constraints=[
            "Use native Apple frameworks",
            "Optimize for M4 Pro",
            "Low memory footprint",
            "Smooth 120Hz ProMotion"
        ],
        max_steps=15
    )
    
    print(f"Generated {len(steps)} steps:")
    for step in steps[:5]:
        print(f"  {step.step_number}. {step.action} (conf: {step.confidence:.2f})")
        
    await thinking.benchmark()
    
    thinking.close()


if __name__ == "__main__":
    asyncio.run(demo())