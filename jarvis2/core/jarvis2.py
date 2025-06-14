"""Jarvis 2.0 - Main Coordinator.

The brain of the intelligent meta-coder that leverages M4 Pro's full potential
to explore, learn, and evolve code generation strategies.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from ..diversity.diversity_engine import DiversityEngine
from ..evaluation.evaluator import MultiObjectiveEvaluator
from ..experience.experience_buffer import ExperienceReplaySystem
from ..hardware.hardware_optimizer import HardwareAwareExecutor
from ..index.index_manager import DeepIndexManager
from .solution import CodeSolution, SolutionMetrics

logger = logging.getLogger(__name__)


@dataclass
class Jarvis2Config:
    """Configuration for Jarvis 2.0."""
    max_parallel_simulations: int = 2000
    gpu_batch_size: int = 256
    use_all_cores: bool = True
    mcts_exploration_constant: float = 1.414
    min_simulations_simple: int = 100
    min_simulations_complex: int = 1000
    num_diverse_solutions: int = 100
    diversity_threshold: float = 0.7
    experience_buffer_size: int = 10000
    background_learning_interval: int = 100
    model_update_threshold: int = 1000
    index_update_interval: int = 300
    use_incremental_indexing: bool = True
    index_path: Path = field(default_factory=lambda : Path('.jarvis/indexes'))
    model_path: Path = field(default_factory=lambda : Path('.jarvis/models'))
    experience_path: Path = field(default_factory=lambda : Path(
        '.jarvis/experience'))


class ComplexityEstimator:
    """Estimates task complexity to guide resource allocation."""

    def __init__(self):
        self.complexity_patterns = {'simple': ['fix', 'typo', 'rename',
            'comment', 'format'], 'medium': ['refactor', 'optimize', 'test',
            'debug', 'update'], 'complex': ['implement', 'design',
            'architect', 'integrate', 'migrate']}
        self.historical_data: Dict[str, float] = {}

    async def estimate_complexity(self, query: str, context: Dict[str, Any]
        ) ->ComplexityEstimate:
        """Estimate complexity of the task."""
        await asyncio.sleep(0)
        query_lower = query.lower()
        complexity_score = 0.3
        for level, patterns in self.complexity_patterns.items():
            if any(p in query_lower for p in patterns):
                if level == 'simple':
                    complexity_score = 0.2
                elif level == 'medium':
                    complexity_score = 0.5
                elif level == 'complex':
                    complexity_score = 0.8
                break
        if context:
            num_files = len(context.get('files', []))
            complexity_score += min(0.2, num_files * 0.02)
            num_deps = len(context.get('dependencies', []))
            complexity_score += min(0.1, num_deps * 0.01)
        similar_tasks = self._find_similar_historical(query)
        if similar_tasks:
            avg_historical = np.mean([t[1] for t in similar_tasks])
            complexity_score = 0.7 * complexity_score + 0.3 * avg_historical
        if complexity_score < 0.3:
            simulations = 100
            variants = 10
            search_depth = 3
        elif complexity_score < 0.6:
            simulations = 500
            variants = 50
            search_depth = 5
        else:
            simulations = 2000
            variants = 100
            search_depth = 8
        return ComplexityEstimate(score=complexity_score, is_complex=
            complexity_score > 0.6, suggested_simulations=simulations,
            suggested_variants=variants, suggested_search_depth=
            search_depth, estimated_time_ms=simulations * 0.5 + variants * 10)

    def _find_similar_historical(self, query: str) ->List[Tuple[str, float]]:
        """Find similar historical tasks."""
        query_words = set(query.lower().split())
        similar = []
        for hist_query, complexity in self.historical_data.items():
            hist_words = set(hist_query.lower().split())
            similarity = len(query_words & hist_words) / max(len(
                query_words), 1)
            if similarity > 0.5:
                similar.append((hist_query, complexity))
        return sorted(similar, key=lambda x: x[1], reverse=True)[:5]

    def record_actual_complexity(self, query: str, actual_complexity: float):
        """Record actual complexity for learning."""
        self.historical_data[query] = actual_complexity
        if len(self.historical_data) > 1000:
            oldest = sorted(self.historical_data.keys())[:100]
            for key in oldest:
                del self.historical_data[key]


@dataclass
class ComplexityEstimate:
    """Complexity estimation result."""
    score: float
    is_complex: bool
    suggested_simulations: int
    suggested_variants: int
    suggested_search_depth: int
    estimated_time_ms: float


class Jarvis2:
    """Main Jarvis 2.0 coordinator."""

    def __init__(self, config: Optional[Jarvis2Config]=None):
        self.config = config or Jarvis2Config()
        self._current_memory_pressure = 0.0
        self._adaptive_batch_size = self.config.gpu_batch_size
        self.hardware_executor = HardwareAwareExecutor()
        self.index_manager = DeepIndexManager(self.config.index_path)
        self.complexity_estimator = ComplexityEstimator()
        from ..search.mcts_simple import SimplifiedMCTS
        self.mcts = SimplifiedMCTS()
        self.diversity_engine = DiversityEngine()
        self.evaluator = MultiObjectiveEvaluator()
        self.experience_buffer = ExperienceReplaySystem(self.config.
            experience_path, buffer_size=self.config.experience_buffer_size)
        self._initialized = False
        self._background_tasks: List[asyncio.Task] = []
        self._last_index_update = 0
        self._total_assists = 0
        self._performance_history: List[float] = []

    async def initialize(self):
        """Initialize all components with proper integration."""
        if self._initialized:
            return
        logger.info('Initializing Jarvis 2.0...')
        start_time = time.time()
        self.config.index_path.mkdir(parents=True, exist_ok=True)
        self.config.model_path.mkdir(parents=True, exist_ok=True)
        self.config.experience_path.mkdir(parents=True, exist_ok=True)
        await self.hardware_executor.initialize()
        self._configure_components_for_hardware()
        logger.info('Initializing index manager...')
        try:
            await asyncio.wait_for(self.index_manager.initialize(), timeout=5.0
                )
            logger.info('Index manager initialized')
        except asyncio.TimeoutError:
            logger.warning(
                'Index manager initialization timed out, continuing...')
        except Exception as e:
            logger.error(f'Index manager initialization failed: {e}')
        logger.info('Initializing MCTS...')
        try:
            await asyncio.wait_for(self.mcts.initialize(), timeout=5.0)
            logger.info('MCTS initialized')
        except asyncio.TimeoutError:
            logger.warning('MCTS initialization timed out, continuing...')
        except Exception as e:
            logger.error(f'MCTS initialization failed: {e}')
        logger.info('Initializing experience buffer...')
        try:
            await asyncio.wait_for(self.experience_buffer.initialize(),
                timeout=5.0)
            logger.info('Experience buffer initialized')
        except asyncio.TimeoutError:
            logger.warning(
                'Experience buffer initialization timed out, continuing...')
        except Exception as e:
            logger.error(f'Experience buffer initialization failed: {e}')
        self._wire_components()
        self._start_background_tasks()
        init_time = time.time() - start_time
        logger.info(f'Jarvis 2.0 initialized in {init_time:.2f}s')
        self._initialized = True

    async def assist(self, query: str, context: Optional[Dict[str, Any]]=None
        ) ->CodeSolution:
        """Main entry point for code assistance."""
        if not self._initialized:
            await self.initialize()
        start_time = time.time()
        self._total_assists += 1
        logger.info(f'Assist #{self._total_assists}: {query}')
        try:
            indexed_context = await self.index_manager.get_context(query)
            if context:
                indexed_context.update(context)
            complexity = await self.complexity_estimator.estimate_complexity(
                query, indexed_context)
            logger.info(
                f'Complexity: {complexity.score:.2f} (simulations: {complexity.suggested_simulations})'
                )
            current_memory = psutil.virtual_memory()
            self._current_memory_pressure = current_memory.percent / 100.0
            if self._current_memory_pressure > 0.8:
                batch_size = max(32, self.config.gpu_batch_size // 4)
                logger.warning(
                    f'High memory pressure ({self._current_memory_pressure:.1%}), reducing batch size to {batch_size}'
                    )
            else:
                batch_size = self.hardware_executor.get_optimal_batch_size(
                    'mcts', item_size_mb=2)
            if complexity.is_complex:
                solution_tree = await self.mcts.explore(query,
                    indexed_context, simulations=complexity.
                    suggested_simulations, parallel_batch_size=batch_size,
                    hardware_executor=self.hardware_executor)
            else:
                solution_tree = await self.mcts.fast_search(query,
                    indexed_context, simulations=complexity.
                    suggested_simulations, hardware_executor=self.
                    hardware_executor)
            logger.debug(
                f'Generating {complexity.suggested_variants} diverse solutions...'
                )
            diverse_solutions = await self.diversity_engine.generate(
                solution_tree, num_variants=complexity.suggested_variants,
                clustering_method='behavioral', hardware_executor=self.
                hardware_executor)
            logger.info(f'Generated {len(diverse_solutions)} diverse solutions'
                )
            eval_batch_size = self.hardware_executor.get_optimal_batch_size(
                'gpu', item_size_mb=1)
            evaluations = await self.evaluator.batch_evaluate(diverse_solutions
                , metrics=['performance', 'readability', 'correctness',
                'resource_usage'], context=indexed_context, batch_size=
                eval_batch_size, hardware_executor=self.hardware_executor)
            best_solution = self._select_best_solution(evaluations)
            total_time = (time.time() - start_time) * 1000
            metrics = SolutionMetrics(generation_time_ms=total_time,
                simulations_performed=complexity.suggested_simulations,
                variants_generated=len(diverse_solutions), confidence_score
                =best_solution.get('confidence', 0.0), complexity_score=
                complexity.score, gpu_utilization=self.hardware_executor.
                get_gpu_utilization(), memory_used_mb=self.
                hardware_executor.get_memory_usage_mb())
            await self.experience_buffer.record(query=query, solution=
                best_solution, metrics=metrics, evaluations=evaluations)
            actual_complexity = min(1.0, total_time / 10000)
            self.complexity_estimator.record_actual_complexity(query,
                actual_complexity)
            self._performance_history.append(total_time)
            if len(self._performance_history) > 100:
                self._performance_history = self._performance_history[-100:]
            return CodeSolution(query=query, code=best_solution.get('code',
                ''), explanation=best_solution.get('explanation', ''),
                confidence=best_solution.get('confidence', 0.0),
                alternatives=self._get_top_alternatives(evaluations, n=3),
                metrics=metrics)
        except Exception as e:
            logger.error(f'Error in assist: {e}', exc_info=True)
            return CodeSolution(query=query, code='', explanation=
                f'Error generating solution: {str(e)}', confidence=0.0,
                alternatives=[], metrics=SolutionMetrics(generation_time_ms
                =(time.time() - start_time) * 1000, simulations_performed=0,
                variants_generated=0, confidence_score=0.0,
                complexity_score=0.0, gpu_utilization=0.0, memory_used_mb=0.0))

    def _select_best_solution(self, evaluations: List[Dict[str, Any]]) ->Dict[
        str, Any]:
        """Select best solution from evaluations."""
        if not evaluations:
            return {'code': '', 'explanation': 'No solutions generated',
                'confidence': 0.0}
        best_score = -float('inf')
        best_solution = None
        for eval_result in evaluations:
            score = 0.4 * eval_result.get('performance', 0
                ) + 0.3 * eval_result.get('correctness', 0
                ) + 0.2 * eval_result.get('readability', 0
                ) + 0.1 * eval_result.get('resource_usage', 0)
            if score > best_score:
                best_score = score
                best_solution = eval_result.get('solution', {})
        if best_solution:
            best_solution['confidence'] = best_score
        return best_solution or evaluations[0].get('solution', {})

    def _get_top_alternatives(self, evaluations: List[Dict[str, Any]], n: int=3
        ) ->List[Dict[str, Any]]:
        """Get top N alternative solutions."""
        sorted_evals = sorted(evaluations, key=lambda x: sum(x.get(m, 0) for
            m in ['performance', 'correctness', 'readability']), reverse=True)
        alternatives = []
        for eval_result in sorted_evals[1:n + 1]:
            solution = eval_result.get('solution', {})
            alternatives.append({'code': solution.get('code', ''),
                'explanation': solution.get('explanation', ''),
                'confidence': eval_result.get('confidence', 0.0),
                'differentiator': solution.get('approach',
                'Alternative approach')})
        return alternatives

    def _start_background_tasks(self):
        """Start background learning and maintenance tasks."""
        if self.config.index_update_interval < 1000:
            logger.info('Skipping background tasks for faster initialization')
            return

        async def update_indexes():
            while True:
                await asyncio.sleep(self.config.index_update_interval)
                if (time.time() - self._last_index_update > self.config.
                    index_update_interval):
                    try:
                        await self.index_manager.update_incremental()
                        self._last_index_update = time.time()
                        logger.info('Index update completed')
                    except Exception as e:
                        logger.error(f'Index update failed: {e}')

        async def background_learning():
            while True:
                await asyncio.sleep(60)
                try:
                    buffer_size = await self.experience_buffer.size()
                    if buffer_size >= self.config.model_update_threshold:
                        logger.info(
                            f'Triggering model update (buffer size: {buffer_size})'
                            )
                        await self._update_models()
                except Exception as e:
                    logger.error(f'Background learning failed: {e}')

        async def monitor_performance():
            while True:
                await asyncio.sleep(300)
                if self._performance_history:
                    avg_time = np.mean(self._performance_history)
                    logger.info(
                        f'Performance stats - Assists: {self._total_assists}, Avg time: {avg_time:.0f}ms, GPU: {self.hardware_executor.get_gpu_utilization():.1f}%'
                        )
        try:
            self._background_tasks = [asyncio.create_task(update_indexes()),
                asyncio.create_task(background_learning()), asyncio.
                create_task(monitor_performance())]
        except Exception as e:
            logger.warning(f'Failed to start background tasks: {e}')

    async def _update_models(self):
        """Update neural models from experience buffer."""
        experiences = await self.experience_buffer.sample(1000)
        if not experiences:
            return
        value_data = [(exp['context'], exp['metrics']['confidence_score']) for
            exp in experiences]
        value_loss = await self.mcts.value_net.train_batch(value_data)
        policy_data = [(exp['query'], exp['solution']) for exp in experiences]
        policy_loss = await self.mcts.policy_net.train_batch(policy_data)
        logger.info(
            f'Model update complete - Value loss: {value_loss:.4f}, Policy loss: {policy_loss:.4f}'
            )
        await self.mcts.save_models(self.config.model_path)

    async def shutdown(self):
        """Clean shutdown."""
        logger.info('Shutting down Jarvis 2.0...')
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions
                =True)
        if self._initialized:
            await self.mcts.save_models(self.config.model_path)
            await self.experience_buffer.flush()
        logger.info('Jarvis 2.0 shutdown complete')

    def get_stats(self) ->Dict[str, Any]:
        """Get current statistics."""
        return {'total_assists': self._total_assists, 'average_time_ms': np
            .mean(self._performance_history) if self._performance_history else
            0, 'gpu_utilization': self.hardware_executor.
            get_gpu_utilization(), 'memory_usage_mb': self.
            hardware_executor.get_memory_usage_mb(), 'index_size': self.
            index_manager.get_index_stats(), 'experience_buffer_size': 0,
            'models_updated': self.mcts.get_update_count() if hasattr(self.
            mcts, 'get_update_count') else 0}

    def _configure_components_for_hardware(self):
        """Configure components based on hardware capabilities."""
        memory_allocs = self.hardware_executor.memory_allocations
        if hasattr(self.mcts, 'config'):
            self.mcts.config.batch_size = (self.hardware_executor.
                get_optimal_batch_size('gpu'))
        if hasattr(self.index_manager, 'set_memory_limit'):
            self.index_manager.set_memory_limit(memory_allocs.get(
                'index_cache', 4096))
        self.config.experience_buffer_size = min(self.config.
            experience_buffer_size, memory_allocs.get('experience_buffer', 
            4096) * 100)

    def _wire_components(self):
        """Wire components together for seamless integration."""
        if hasattr(self.mcts, 'set_hardware_executor'):
            self.mcts.set_hardware_executor(self.hardware_executor)
        if hasattr(self.evaluator, 'set_hardware_executor'):
            self.evaluator.set_hardware_executor(self.hardware_executor)
        if hasattr(self.mcts, 'set_experience_buffer'):
            self.mcts.set_experience_buffer(self.experience_buffer)
        for component in [self.mcts, self.diversity_engine, self.evaluator]:
            if hasattr(component, 'set_index_manager'):
                component.set_index_manager(self.index_manager)
        logger.info('Components wired together successfully')
