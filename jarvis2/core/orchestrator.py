"""Main orchestrator for Jarvis2 on M4 Pro.

Coordinates all components without blocking the async event loop.
"""
import asyncio
import logging
import time
import uuid
from typing import List, Optional, Dict, Any
import numpy as np

from .device_router import get_router, OperationType
from .memory_manager import get_memory_manager
from ..workers.neural_worker import NeuralWorkerPool
from ..workers.search_worker import SearchWorkerPool
from ..workers.learning_worker import LearningWorker, Experience
from ..search.vector_index import HybridVectorIndex
from ..storage.experience_buffer import ExperienceBuffer, GenerationExperience

logger = logging.getLogger(__name__)


class Jarvis2Config:
    """Configuration for Jarvis2."""
    def __init__(self):
        # Hardware
        self.num_neural_workers = 2
        self.num_search_workers = 8  # P-cores
        self.num_learning_workers = 2  # E-cores
        
        # Search
        self.default_simulations = 2000
        self.exploration_constant = 1.414
        
        # Memory
        self.max_memory_gb = 18  # Metal limit
        self.buffer_pool_gb = 4
        
        # Paths
        self.index_path = ".jarvis/indexes"
        self.experience_path = ".jarvis/experience.db"
        self.model_path = ".jarvis/models"


class CodeRequest:
    """Request for code generation."""
    def __init__(self, query: str, context: Optional[Dict] = None):
        self.query = query
        self.context = context or {}
        self.timestamp = time.time()


class CodeSolution:
    """Generated code solution."""
    def __init__(self, code: str, confidence: float, 
                 alternatives: List[Dict], metrics: Dict):
        self.code = code
        self.confidence = confidence
        self.alternatives = alternatives
        self.metrics = metrics


class Jarvis2Orchestrator:
    """Main coordinator for Jarvis2 meta-coding system."""
    
    def __init__(self, config: Optional[Jarvis2Config] = None):
        self.config = config or Jarvis2Config()
        self.initialized = False
        
        # Core components
        self.device_router = get_router()
        self.memory_manager = get_memory_manager()
        
        # Worker pools
        self.neural_pool = None
        self.search_pool = None
        self.learning_worker = None
        
        # Storage
        self.vector_index = None
        self.experience_buffer = None
        
        # Metrics
        self.request_count = 0
        self.total_time = 0
        
        # Track background tasks for cleanup
        self._background_tasks = set()
        
    async def initialize(self):
        """Initialize all components."""
        if self.initialized:
            return
            
        logger.info("Initializing Jarvis2 Orchestrator...")
        start_time = time.time()
        
        # Start neural workers
        logger.info("Starting neural workers...")
        self.neural_pool = NeuralWorkerPool(self.config.num_neural_workers)
        
        # Start search workers
        logger.info("Starting search workers...")
        self.search_pool = SearchWorkerPool(self.config.num_search_workers)
        
        # Start learning worker
        logger.info("Starting learning worker...")
        self.learning_worker = LearningWorker(self.config.model_path)
        self.learning_worker.start()
        
        # Initialize storage
        logger.info("Initializing storage...")
        self.vector_index = HybridVectorIndex(self.config.index_path)
        await self.vector_index.initialize()
        
        self.experience_buffer = ExperienceBuffer(self.config.experience_path)
        await self.experience_buffer.initialize()
        
        # Run benchmarks to select optimal backends
        await self._run_initialization_benchmarks()
        
        self.initialized = True
        init_time = time.time() - start_time
        logger.info(f"Jarvis2 initialized in {init_time:.1f}s")
        
    async def generate_code(self, request: CodeRequest) -> CodeSolution:
        """Generate code using AI search.
        
        This is the main entry point that:
        1. Gets context from indexes (instant)
        2. Runs parallel MCTS exploration (P-cores)
        3. Evaluates with neural networks (GPU)
        4. Learns from the interaction (E-cores)
        """
        if not self.initialized:
            await self.initialize()
            
        start_time = time.time()
        self.request_count += 1
        
        logger.info(f"Processing request #{self.request_count}: {request.query[:50]}...")
        
        # Phase 1: Context retrieval (instant from pre-computed indexes)
        context = await self._get_context(request)
        
        # Phase 2 & 3: Start neural guidance and search in parallel
        # This prevents bottlenecks when handling multiple requests
        
        # Start neural guidance (non-blocking)
        guidance_task = asyncio.create_task(
            self._get_neural_guidance(request.query, context)
        )
        
        # Start search immediately with default guidance
        # This allows search to begin without waiting for neural computation
        default_guidance = {
            'value': np.array([[0.5]]),  # Neutral value
            'policy': np.ones(50) / 50    # Uniform policy
        }
        
        search_task = asyncio.create_task(
            self.search_pool.parallel_search(
                request.query, context, default_guidance,
                simulations=self.config.default_simulations
            )
        )
        
        # Phase 4: Wait for both to complete
        # This ensures true parallelism between neural and search operations
        guidance, search_result = await asyncio.gather(guidance_task, search_task)
        
        # Phase 5: Combine and evaluate
        solution = await self._evaluate_and_select(
            search_result, guidance, context
        )
        
        # Phase 6: Background learning (non-blocking)
        task = asyncio.create_task(
            self._record_experience(request, solution)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        # Metrics
        elapsed = time.time() - start_time
        self.total_time += elapsed
        
        solution.metrics.update({
            'generation_time_ms': elapsed * 1000,
            'request_number': self.request_count,
            'backend_used': self.device_router.route(OperationType.NEURAL_FORWARD).value
        })
        
        logger.info(f"Generated solution in {elapsed*1000:.0f}ms "
                   f"(confidence: {solution.confidence:.0%})")
        
        return solution
        
    async def _get_context(self, request: CodeRequest) -> Dict[str, Any]:
        """Get context from pre-computed indexes."""
        context = request.context.copy()
        
        # Add vector search results
        if self.vector_index:
            similar_code = await self.vector_index.search(
                request.query, k=10
            )
            context['similar_code'] = similar_code
            
        # Add past experiences
        if self.experience_buffer:
            past_experiences = await self.experience_buffer.get_similar_experiences(
                request.query, limit=5
            )
            context['past_experiences'] = [
                {
                    'query': exp.query,
                    'confidence': exp.confidence,
                    'code_preview': exp.generated_code[:200]
                }
                for exp in past_experiences
            ]
        
        # Add system context
        context.update({
            'platform': 'M4 Pro',
            'available_memory_gb': self.memory_manager.get_stats()['system_available_gb'],
            'request_count': self.request_count
        })
        
        return context
        
    async def _get_neural_guidance(self, query: str, 
                                  context: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Get neural network guidance for search."""
        # Create embedding from query
        # For now, use random embedding
        embedding = np.random.randn(1, 768).astype(np.float32)
        
        # Get value and policy predictions
        value_task = self.neural_pool.compute_async('value', embedding)
        policy_task = self.neural_pool.compute_async('policy', embedding)
        
        value, policy = await asyncio.gather(value_task, policy_task)
        
        return {
            'embedding': embedding,
            'value': value,
            'policy': policy
        }
        
    async def _evaluate_and_select(self, search_result: Dict,
                                  guidance: Dict,
                                  context: Dict) -> CodeSolution:
        """Evaluate and select best solution."""
        # For now, return the dummy solution
        return CodeSolution(
            code=search_result['best_code'],
            confidence=float(guidance['value'][0][0]) if guidance['value'].size > 0 else 0.8,
            alternatives=search_result['alternatives'],
            metrics={
                'simulations': self.config.default_simulations,
                'neural_evaluations': 1
            }
        )
        
    async def _record_experience(self, request: CodeRequest, 
                               solution: CodeSolution):
        """Record experience for learning (non-blocking)."""
        try:
            # Save to experience buffer
            exp = GenerationExperience(
                id=f"exp_{uuid.uuid4()}",
                timestamp=time.time(),
                query=request.query,
                generated_code=solution.code,
                confidence=solution.confidence,
                context=request.context,
                alternatives=solution.alternatives,
                metrics=solution.metrics
            )
            
            await self.experience_buffer.add_experience(exp)
            
            # Send to learning worker
            learning_exp = Experience(
                query=request.query,
                code=solution.code,
                context=request.context,
                value=solution.confidence,
                policy_actions=[],  # TODO: Extract from search tree
                timestamp=time.time(),
                metadata=solution.metrics
            )
            
            self.learning_worker.add_experience(learning_exp)
            
            logger.debug(f"Recorded experience for: {request.query[:30]}...")
            
        except Exception as e:
            logger.error(f"Failed to record experience: {e}")
        
    def _generate_dummy_code(self, query: str) -> str:
        """Generate dummy code for testing."""
        query_lower = query.lower()
        
        if "hello" in query_lower:
            return '''def hello_world():
    """Print Hello World."""
    print("Hello, World!")
    
if __name__ == "__main__":
    hello_world()'''
        elif "factorial" in query_lower:
            return '''def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)'''
        else:
            return f'''def solution():
    """Solution for: {query}"""
    # TODO: Implement
    pass'''
            
    async def _run_initialization_benchmarks(self):
        """Run benchmarks to select optimal backends."""
        logger.info("Running initialization benchmarks...")
        
        # Benchmark small matmul for tree operations
        tree_backend = self.device_router.select_optimal_backend(
            OperationType.TREE_SEARCH,
            input_size=(1000, 64)
        )
        logger.info(f"Selected {tree_backend.value} for tree search")
        
        # Benchmark larger matmul for neural ops
        neural_backend = self.device_router.select_optimal_backend(
            OperationType.NEURAL_FORWARD,
            input_size=(256, 768)
        )
        logger.info(f"Selected {neural_backend.value} for neural ops")
        
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down Jarvis2...")
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown workers
        if self.neural_pool:
            self.neural_pool.shutdown()
            
        if self.search_pool:
            self.search_pool.shutdown()
            
        if self.learning_worker:
            self.learning_worker.stop()
            
        if self.experience_buffer:
            await self.experience_buffer.close()
        
        self.memory_manager.cleanup()
        
        logger.info("Jarvis2 shutdown complete")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            'request_count': self.request_count,
            'average_time_ms': (self.total_time / self.request_count * 1000) if self.request_count > 0 else 0,
            'memory_stats': self.memory_manager.get_stats(),
            'backends': {
                op.value: self.device_router.route(op).value
                for op in OperationType
            }
        }
        return stats


# Example usage
async def main():
    """Example usage of Jarvis2."""
    jarvis = Jarvis2Orchestrator()
    await jarvis.initialize()
    
    # Generate code
    request = CodeRequest("Create a function to calculate fibonacci numbers")
    solution = await jarvis.generate_code(request)
    
    print(f"\nGenerated code:")
    print(solution.code)
    print(f"\nConfidence: {solution.confidence:.0%}")
    print(f"Time: {solution.metrics['generation_time_ms']:.0f}ms")
    
    # Get stats
    stats = jarvis.get_stats()
    print(f"\nStats: {stats}")
    
    await jarvis.shutdown()


if __name__ == "__main__":
    asyncio.run(main())