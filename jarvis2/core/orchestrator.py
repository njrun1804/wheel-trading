"""Main orchestrator for Jarvis2 on M4 Pro.

Coordinates all components without blocking the async event loop.
"""
import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from ..search.vector_index import HybridVectorIndex
from ..storage.experience_buffer import ExperienceBuffer, GenerationExperience
from ..workers.learning_worker import Experience, LearningWorker
from ..workers.neural_worker import NeuralWorkerPool
from ..workers.search_worker import SearchWorkerPool
from .device_router import OperationType, get_router
from .error_handling import ErrorSeverity, ResourceGuard, with_error_handling, with_timeout
from .memory_manager import get_memory_manager
from src.unity_wheel.accelerated_tools.sequential_thinking_config import SequentialThinkingEngine

logger = logging.getLogger(__name__)


class Jarvis2Config:
    """Configuration for Jarvis2."""

    def __init__(self):
        self.num_neural_workers = 2
        self.num_search_workers = 8
        self.num_learning_workers = 2
        self.default_simulations = 2000
        self.exploration_constant = 1.414
        self.max_memory_gb = 18
        self.buffer_pool_gb = 4
        self.index_path = '.jarvis/indexes'
        self.experience_path = '.jarvis/experience.db'
        self.model_path = '.jarvis/models'


class CodeRequest:
    """Request for code generation."""

    def __init__(self, query: str, context: Optional[Dict]=None):
        self.query = query
        self.context = context or {}
        self.timestamp = time.time()


class CodeSolution:
    """Generated code solution."""

    def __init__(self, code: str, confidence: float, alternatives: List[
        Dict], metrics: Dict):
        self.code = code
        self.confidence = confidence
        self.alternatives = alternatives
        self.metrics = metrics


class Jarvis2Orchestrator:
    """Main coordinator for Jarvis2 meta-coding system."""

    def __init__(self, config: Optional[Jarvis2Config]=None):
        self.config = config or Jarvis2Config()
        self.initialized = False
        self.device_router = get_router()
        self.memory_manager = get_memory_manager()
        self.neural_pool = None
        self.search_pool = None
        self.learning_worker = None
        self.vector_index = None
        self.experience_buffer = None
        self.sequential_thinking = SequentialThinkingEngine(use_mcp=False)
        self.request_count = 0
        self.total_time = 0
        self._background_tasks = set()
        self.resource_guard = ResourceGuard(max_memory_gb = self.config.
            max_memory_gb, max_cpu_percent = 85.0)

    @with_error_handling('orchestrator', 'initialize', ErrorSeverity.HIGH)
    @with_timeout(60.0)
    async def initialize(self):
        """Initialize all components with error handling."""
        if self.initialized:
            return
        logger.info('Initializing Jarvis2 Orchestrator...')
        start_time = time.time()
        logger.info('Starting neural workers...')
        self.neural_pool = NeuralWorkerPool(self.config.num_neural_workers)
        logger.info('Starting search workers...')
        self.search_pool = SearchWorkerPool(self.config.num_search_workers)
        logger.info('Starting learning worker...')
        self.learning_worker = LearningWorker(self.config.model_path)
        self.learning_worker.start()
        logger.info('Initializing storage...')
        self.vector_index = HybridVectorIndex(self.config.index_path)
        await self.vector_index.initialize()
        self.experience_buffer = ExperienceBuffer(self.config.experience_path)
        await self.experience_buffer.initialize()
        await self._run_initialization_benchmarks()
        self.initialized = True
        init_time = time.time() - start_time
        logger.info(f"Jarvis2 initialized in {init_time:.1f}s")

    async def generate_code(self, request: CodeRequest) ->CodeSolution:
        """Generate code using AI search with comprehensive error handling.
        
        This is the main entry point that:
        1. Gets context from indexes (instant)
        2. Runs parallel MCTS exploration (P-cores)
        3. Evaluates with neural networks (GPU)
        4. Learns from the interaction (E-cores)
        """
        if not self.initialized:
            await self.initialize()
        if not await self.resource_guard.wait_for_resources(timeout = 30):
            logger.error('Insufficient resources for code generation')
            return CodeSolution(code=
                """# Error: Insufficient system resources
# Please try again later"""
                , confidence = 0.0, alternatives=[], metrics={'error':
                'resource_exhaustion'})
        start_time = time.time()
        self.request_count += 1
        logger.info(
            f"Processing request #{self.request_count}: {request.query[:50]}..."
            )
        context = await self._get_context(request)
        
        # Use sequential thinking for complex planning
        thinking_plan = await self.sequential_thinking.plan_implementation(
            feature=request.query,
            requirements=[
                "Must be efficient and use hardware acceleration",
                "Follow existing code patterns",
                "Include error handling",
                "Be maintainable and testable"
            ],
            existing_code=context.get('related_code', {})
        )
        
        # Add thinking plan to context for search
        context['thinking_plan'] = thinking_plan
        context['implementation_steps'] = thinking_plan.get('steps', [])
        
        guidance_task = asyncio.create_task(self._get_neural_guidance(
            request.query, context))
        default_guidance = {'value': np.array([[0.5]]), 'policy': np.ones(
            50) / 50}
        search_task = asyncio.create_task(self.search_pool.parallel_search(
            request.query, context, default_guidance, simulations = self.
            config.default_simulations))
        try:
            guidance, search_result = await asyncio.gather(guidance_task,
                search_task, return_exceptions = True)
            if isinstance(guidance, Exception):
                logger.error(f"Neural guidance failed: {guidance}")
                guidance = default_guidance
            if isinstance(search_result, Exception):
                logger.error(f"Search failed: {search_result}")
                return CodeSolution(code = f"""# Error during search: {str(search_result)[:100]}
def {request.query.replace(' ', '_')[:20]}():
    # TODO: Implement {request.query}
    pass"""
                    , confidence = 0.1, alternatives=[], metrics={'error':
                    'search_failure', 'error_msg': str(search_result)})
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            guidance = default_guidance
            search_result = {'best_code':
                '# Error during generation\ndef solution():\n    pass',
                'confidence': 0.0, 'alternatives': [], 'stats': {'error':
                str(e)}}
        solution = await self._evaluate_and_select(search_result, guidance,
            context)
        task = asyncio.create_task(self._record_experience(request, solution))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        elapsed = time.time() - start_time
        self.total_time += elapsed
        solution.metrics.update({'generation_time_ms': elapsed * 1000,
            'request_number': self.request_count, 'backend_used': self.
            device_router.route(OperationType.NEURAL_FORWARD).value})
        logger.info(
            f"Generated solution in {elapsed * 1000:.0f}ms (confidence: {solution.confidence:.0%})"
            )
        return solution

    @with_error_handling('orchestrator', 'get_context', ErrorSeverity.LOW)
    @with_timeout(10.0, default_value={})
    async def _get_context(self, request: CodeRequest) ->Dict[str, Any]:
        """Get context from pre-computed indexes with error handling."""
        context = request.context.copy()
        if self.vector_index:
            similar_code = await self.vector_index.search(request.query, k = 10)
            context['similar_code'] = similar_code
        if self.experience_buffer:
            past_experiences = (await self.experience_buffer.
                get_similar_experiences(request.query, limit = 5))
            context['past_experiences'] = [{'query': exp.query,
                'confidence': exp.confidence, 'code_preview': exp.
                generated_code[:200]} for exp in past_experiences]
        context.update({'platform': 'M4 Pro', 'available_memory_gb': self.
            memory_manager.get_stats()['system_available_gb'],
            'request_count': self.request_count})
        return context

    @with_error_handling('orchestrator', 'neural_guidance', ErrorSeverity.
        MEDIUM)
    @with_timeout(15.0, default_value={'value': np.array([[0.5]]), 'policy':
        np.ones(50) / 50})
    async def _get_neural_guidance(self, query: str, context: Dict[str, Any]
        ) ->Dict[str, np.ndarray]:
        """Get neural network guidance for search with error handling."""
        from .code_embeddings import LightweightCodeEmbedder, MLXCodeEmbedder
        if not hasattr(self, '_embedder'):
            try:
                self._embedder = MLXCodeEmbedder()
                logger.info('Using MLX embedder for neural guidance')
            except Exception as e:
                self._embedder = LightweightCodeEmbedder(vector_dim = 768)
                logger.info('Using lightweight embedder for neural guidance')
        embedding = self._embedder.embed(query)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        value_task = self.neural_pool.compute_async('value', embedding)
        policy_task = self.neural_pool.compute_async('policy', embedding)
        value, policy = await asyncio.gather(value_task, policy_task)
        return {'embedding': embedding, 'value': value, 'policy': policy}

    async def _evaluate_and_select(self, search_result: Dict, guidance:
        Dict, context: Dict) ->CodeSolution:
        """Evaluate and select best solution."""
        await asyncio.sleep(0)
        best_code = search_result.get('best_code', '')
        confidence = search_result.get('confidence', 0.5)
        if guidance['value'].size > 0:
            neural_confidence = float(guidance['value'][0][0])
            confidence = 0.7 * confidence + 0.3 * neural_confidence
        return CodeSolution(code = best_code, confidence = confidence,
            alternatives = search_result.get('alternatives', []), metrics={
            'simulations': search_result['stats'].get('total_simulations',
            self.config.default_simulations), 'neural_evaluations': 1,
            'nodes_explored': search_result['stats'].get(
            'total_nodes_explored', 0), 'avg_search_time_ms': search_result
            ['stats'].get('avg_search_time_ms', 0)})

    async def _record_experience(self, request: CodeRequest, solution:
        CodeSolution):
        """Record experience for learning (non-blocking)."""
        try:
            exp = GenerationExperience(id = f"exp_{uuid.uuid4()}", timestamp = time.time(), query = request.query, generated_code = solution.
                code, confidence = solution.confidence, context = request.
                context, alternatives = solution.alternatives, metrics = solution.metrics)
            await self.experience_buffer.add_experience(exp)
            policy_actions = []
            if isinstance(search_result, dict
                ) and 'search_tree' in search_result:
                tree = search_result.get('search_tree')
                if tree and hasattr(tree, 'children'):
                    node = tree
                    while hasattr(node, 'children') and node.children:
                        best_child = max(node.children.values(), key = lambda
                            n: getattr(n, 'visits', 0))
                        if hasattr(best_child, 'action') and best_child.action:
                            policy_actions.append(best_child.action)
                        node = best_child
            learning_exp = Experience(query = request.query, code = solution.
                code, context = request.context, value = solution.confidence,
                policy_actions = policy_actions[:10], timestamp = time.time(),
                metadata = solution.metrics)
            self.learning_worker.add_experience(learning_exp)
            logger.debug(f"Recorded experience for: {request.query[:30]}...")
        except Exception as e:
            logger.error(f"Failed to record experience: {e}")

    def _generate_dummy_code(self, query: str) ->str:
        """Generate dummy code for testing."""
        query_lower = query.lower()
        if 'hello' in query_lower:
            return """def hello_world():
    ""\"Print Hello World.""\"
    print("Hello, World!")
    
if __name__ == "__main__":
    hello_world()"""
        elif 'factorial' in query_lower:
            return """def factorial(n: int) -> int:
    ""\"Calculate factorial of n.""\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""
        else:
            return f"""def solution():
    ""\"Solution for: {query}""\"
    # TODO: Implement
    pass"""

    async def _run_initialization_benchmarks(self):
        """Run benchmarks to select optimal backends."""
        await asyncio.sleep(0)  # Make properly async
        logger.info('Running initialization benchmarks...')
        tree_backend = self.device_router.select_optimal_backend(OperationType
            .TREE_SEARCH, input_size=(1000, 64))
        logger.info(f"Selected {tree_backend.value} for tree search")
        neural_backend = self.device_router.select_optimal_backend(
            OperationType.NEURAL_FORWARD, input_size=(256, 768))
        logger.info(f"Selected {neural_backend.value} for neural ops")

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info('Shutting down Jarvis2...')
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions = True)
        if self.neural_pool:
            self.neural_pool.shutdown()
        if self.search_pool:
            self.search_pool.shutdown()
        if self.learning_worker:
            self.learning_worker.stop()
        if self.experience_buffer:
            await self.experience_buffer.close()
        self.memory_manager.cleanup()
        logger.info('Jarvis2 shutdown complete')

    def get_stats(self) ->Dict[str, Any]:
        """Get system statistics."""
        stats = {'request_count': self.request_count, 'average_time_ms': 
            self.total_time / self.request_count * 1000 if self.
            request_count > 0 else 0, 'memory_stats': self.memory_manager.
            get_stats(), 'backends': {op.value: self.device_router.route(op
            ).value for op in OperationType}}
        return stats


async def main():
    """Example usage of Jarvis2."""
    jarvis = Jarvis2Orchestrator()
    await jarvis.initialize()
    request = CodeRequest('Create a function to calculate fibonacci numbers')
    solution = await jarvis.generate_code(request)
    print(f"\nGenerated code:")
    print(solution.code)
    print(f"\nConfidence: {solution.confidence:.0%}")
    print(f"Time: {solution.metrics['generation_time_ms']:.0f}ms")
    stats = jarvis.get_stats()
    print(f"\nStats: {stats}")
    await jarvis.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
