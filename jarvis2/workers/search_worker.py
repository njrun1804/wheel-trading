"""Parallel MCTS search workers for P-cores.

Runs Monte Carlo Tree Search in parallel across P-cores to explore
thousands of code implementations efficiently.
"""
import multiprocessing as mp
import asyncio
import queue
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class SearchRequest:
    """Request for code search."""
    id: str
    query: str
    context: Dict[str, Any]
    guidance: Dict[str, np.ndarray]  # Neural guidance (value, policy)
    simulations: int = 2000
    exploration_constant: float = 1.414


@dataclass
class SearchResult:
    """Result from code search."""
    id: str
    best_code: str
    confidence: float
    alternatives: List[Dict[str, Any]]
    search_tree: Optional['TreeNode'] = None
    stats: Dict[str, Any] = None


class TreeNode:
    """Node in MCTS search tree."""
    
    def __init__(self, state: str, parent: Optional['TreeNode'] = None,
                 action: Optional[str] = None, prior: float = 1.0):
        self.state = state  # Current code state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.prior = prior  # Prior probability from policy network
        
        self.visits = 0
        self.value_sum = 0.0
        self.children: Dict[str, 'TreeNode'] = {}
        self.is_expanded = False
        
    @property
    def value(self) -> float:
        """Average value of this node."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0
        
    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score for selection."""
        if self.visits == 0:
            return float('inf')
        
        exploration = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        return self.value + 1.414 * exploration * self.prior
        
    def select_child(self) -> 'TreeNode':
        """Select best child using UCB."""
        return max(self.children.values(), key=lambda n: n.ucb_score)
        
    def expand(self, actions: List[Tuple[str, float]]) -> 'TreeNode':
        """Expand node with possible actions."""
        self.is_expanded = True
        
        for action, prior in actions:
            if action not in self.children:
                self.children[action] = TreeNode(
                    state=self._apply_action(action),
                    parent=self,
                    action=action,
                    prior=prior
                )
                
        # Return random child for exploration
        if self.children:
            return list(self.children.values())[0]
        return self
        
    def _apply_action(self, action: str) -> str:
        """Apply action to current state to get new state."""
        # Simplified: append action to code
        return f"{self.state}\n{action}"
        
    def backup(self, value: float):
        """Backup value through tree."""
        self.visits += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backup(value)


class CodeActionSpace:
    """Defines possible code transformations."""
    
    # Simplified action types for demo
    ACTION_TYPES = [
        "add_function",
        "add_class", 
        "add_import",
        "modify_function",
        "add_type_hint",
        "add_docstring",
        "refactor_loop",
        "add_error_handling",
        "optimize_algorithm",
        "add_test"
    ]
    
    @staticmethod
    def get_actions(state: str, policy_probs: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """Get possible actions from current state."""
        actions = []
        
        # Use policy network probabilities if available
        if policy_probs is not None and len(policy_probs) >= len(CodeActionSpace.ACTION_TYPES):
            for i, action_type in enumerate(CodeActionSpace.ACTION_TYPES):
                prob = float(policy_probs[i])
                action = CodeActionSpace._generate_action(action_type, state)
                actions.append((action, prob))
        else:
            # Uniform priors
            for action_type in CodeActionSpace.ACTION_TYPES:
                action = CodeActionSpace._generate_action(action_type, state)
                actions.append((action, 1.0 / len(CodeActionSpace.ACTION_TYPES)))
                
        return actions
        
    @staticmethod
    def _generate_action(action_type: str, state: str) -> str:
        """Generate specific action based on type."""
        # Simplified generation
        if action_type == "add_function":
            return "def new_function():\n    pass"
        elif action_type == "add_class":
            return "class NewClass:\n    pass"
        elif action_type == "add_import":
            return "import numpy as np"
        elif action_type == "add_type_hint":
            return "# Add type hints to existing functions"
        elif action_type == "add_docstring":
            return '"""Add docstrings."""'
        elif action_type == "refactor_loop":
            return "# Refactor loops for efficiency"
        elif action_type == "add_error_handling":
            return "try:\n    pass\nexcept Exception:\n    pass"
        elif action_type == "optimize_algorithm":
            return "# Optimize algorithm"
        elif action_type == "add_test":
            return "def test_function():\n    assert True"
        else:
            return f"# {action_type}"


class MCTSSearcher:
    """Monte Carlo Tree Search for code generation."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.action_space = CodeActionSpace()
        
    def search(self, request: SearchRequest) -> SearchResult:
        """Run MCTS to find best code."""
        start_time = time.perf_counter()
        
        # Initialize root with query as initial state
        root = TreeNode(state=f"# Solution for: {request.query}")
        
        # Get policy guidance if available
        policy_probs = request.guidance.get('policy')
        if policy_probs is not None:
            policy_probs = policy_probs.flatten()
            
        # Run simulations
        for sim in range(request.simulations):
            node = root
            
            # Selection: traverse tree using UCB
            path = [node]
            while node.is_expanded and node.children:
                node = node.select_child()
                path.append(node)
                
            # Expansion: add new nodes
            if not node.is_expanded and node.visits > 0:
                actions = self.action_space.get_actions(node.state, policy_probs)
                node = node.expand(actions)
                path.append(node)
                
            # Evaluation: use value network or rollout
            value = self._evaluate(node, request.guidance.get('value'))
            
            # Backup: propagate value up the tree
            node.backup(value)
            
        # Extract best path
        best_code, confidence = self._extract_best_solution(root)
        alternatives = self._extract_alternatives(root, n=3)
        
        elapsed = time.perf_counter() - start_time
        
        return SearchResult(
            id=request.id,
            best_code=best_code,
            confidence=confidence,
            alternatives=alternatives,
            search_tree=root,
            stats={
                'simulations': request.simulations,
                'worker_id': self.worker_id,
                'search_time_ms': elapsed * 1000,
                'nodes_explored': self._count_nodes(root)
            }
        )
        
    def _evaluate(self, node: TreeNode, value_guidance: Optional[np.ndarray]) -> float:
        """Evaluate a node's value."""
        # Use neural value if available
        if value_guidance is not None and value_guidance.size > 0:
            return float(value_guidance[0])
            
        # Simple heuristic based on code length and structure
        code = node.state
        score = 0.5
        
        # Reward complete solutions
        if "def" in code or "class" in code:
            score += 0.1
        if "return" in code:
            score += 0.1
        if "import" in code:
            score += 0.05
        if '"""' in code:
            score += 0.05
            
        # Penalize very long code
        if len(code) > 1000:
            score -= 0.1
            
        return max(0.0, min(1.0, score))
        
    def _extract_best_solution(self, root: TreeNode) -> Tuple[str, float]:
        """Extract best code path from tree."""
        # Follow most visited path
        node = root
        code_parts = [node.state]
        
        while node.children:
            # Choose most visited child
            best_child = max(node.children.values(), key=lambda n: n.visits)
            if best_child.action:
                code_parts.append(best_child.action)
            node = best_child
            
        best_code = "\n".join(code_parts)
        confidence = node.value if node.visits > 0 else 0.5
        
        return best_code, confidence
        
    def _extract_alternatives(self, root: TreeNode, n: int = 3) -> List[Dict[str, Any]]:
        """Extract top N alternative solutions."""
        alternatives = []
        
        # Get top nodes by value
        all_nodes = []
        self._collect_nodes(root, all_nodes)
        
        # Sort by value * visits (quality * confidence)
        sorted_nodes = sorted(all_nodes, 
                            key=lambda n: n.value * np.sqrt(n.visits),
                            reverse=True)
        
        for node in sorted_nodes[1:n+1]:  # Skip root
            if node.visits > 10:  # Minimum visits threshold
                path = self._get_path_to_node(node)
                code = "\n".join(n.state if n == root else n.action 
                               for n in path if n.action or n == root)
                
                alternatives.append({
                    'code': code,
                    'confidence': node.value,
                    'visits': node.visits
                })
                
        return alternatives
        
    def _collect_nodes(self, node: TreeNode, nodes: List[TreeNode]):
        """Recursively collect all nodes."""
        nodes.append(node)
        for child in node.children.values():
            self._collect_nodes(child, nodes)
            
    def _get_path_to_node(self, node: TreeNode) -> List[TreeNode]:
        """Get path from root to node."""
        path = []
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))
        
    def _count_nodes(self, node: TreeNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count


class SearchWorkerProcess:
    """Process running MCTS search."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.process = None
        
        # Communication
        self.request_queue = mp.Queue(maxsize=10)
        self.response_queue = mp.Queue(maxsize=10)
        self.shutdown_event = mp.Event()
        
    def start(self):
        """Start worker process."""
        self.process = mp.Process(
            target=self._run_worker,
            args=(self.worker_id, self.request_queue, self.response_queue,
                  self.shutdown_event),
            daemon=True
        )
        self.process.start()
        logger.info(f"Search worker {self.worker_id} started")
        
    def stop(self):
        """Stop worker process."""
        self.shutdown_event.set()
        if self.process:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
        logger.info(f"Search worker {self.worker_id} stopped")
        
    @staticmethod
    def _run_worker(worker_id: int, request_queue: mp.Queue,
                   response_queue: mp.Queue, shutdown_event: mp.Event):
        """Main worker loop."""
        searcher = MCTSSearcher(worker_id)
        
        while not shutdown_event.is_set():
            try:
                # Get request
                request = request_queue.get(timeout=0.1)
                
                # Run search
                result = searcher.search(request)
                
                # Send result
                response_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Search worker {worker_id} error: {e}")
                if 'request' in locals():
                    response_queue.put(SearchResult(
                        id=request.id,
                        best_code="# Error during search",
                        confidence=0.0,
                        alternatives=[],
                        stats={'error': str(e)}
                    ))


class SearchWorkerPool:
    """Pool of search workers running on P-cores."""
    
    def __init__(self, num_workers: int = 8):
        self.num_workers = num_workers
        self.workers = []
        
        # Start workers
        for i in range(num_workers):
            worker = SearchWorkerProcess(i)
            worker.start()
            self.workers.append(worker)
            
    async def parallel_search(self, query: str, context: Dict[str, Any],
                            guidance: Dict[str, np.ndarray],
                            simulations: int = 2000) -> Dict[str, Any]:
        """Run parallel MCTS search."""
        # Divide simulations across workers
        sims_per_worker = simulations // self.num_workers
        remainder = simulations % self.num_workers
        
        # Create requests
        requests = []
        for i, worker in enumerate(self.workers):
            worker_sims = sims_per_worker + (1 if i < remainder else 0)
            request = SearchRequest(
                id=f"{uuid.uuid4()}-w{i}",
                query=query,
                context=context,
                guidance=guidance,
                simulations=worker_sims
            )
            requests.append((worker, request))
            
        # Submit all requests
        for worker, request in requests:
            worker.request_queue.put(request)
            
        # Gather results with timeout
        results = []
        pending_requests = {req.id: (worker, req) for worker, req in requests}
        start_time = time.time()
        timeout = 30.0  # 30 second timeout
        
        while pending_requests and (time.time() - start_time) < timeout:
            # Check all workers for any results
            for req_id, (worker, request) in list(pending_requests.items()):
                try:
                    result = worker.response_queue.get_nowait()
                    
                    # Check if this result matches any pending request
                    if result.id in pending_requests:
                        results.append(result)
                        del pending_requests[result.id]
                    else:
                        # Not one of our results, put it back
                        worker.response_queue.put(result)
                        
                except queue.Empty:
                    continue
                    
            # Brief sleep to avoid busy waiting
            if pending_requests:
                await asyncio.sleep(0.01)
                
        # Check if we got all results
        if pending_requests:
            logger.warning(f"Timeout waiting for {len(pending_requests)} search results")
                    
        # Combine results
        return self._combine_results(results)
        
    def _combine_results(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Combine results from parallel searches."""
        # Select best overall solution
        best_result = max(results, key=lambda r: r.confidence)
        
        # Aggregate alternatives
        all_alternatives = []
        for result in results:
            all_alternatives.extend(result.alternatives)
            
        # Sort and deduplicate
        unique_alternatives = {}
        for alt in all_alternatives:
            code_hash = hash(alt['code'])
            if code_hash not in unique_alternatives or alt['confidence'] > unique_alternatives[code_hash]['confidence']:
                unique_alternatives[code_hash] = alt
                
        alternatives = sorted(unique_alternatives.values(), 
                            key=lambda a: a['confidence'], 
                            reverse=True)[:5]
        
        # Aggregate stats
        total_nodes = sum(r.stats.get('nodes_explored', 0) for r in results)
        avg_time = sum(r.stats.get('search_time_ms', 0) for r in results) / len(results)
        
        return {
            'best_code': best_result.best_code,
            'confidence': best_result.confidence,
            'alternatives': alternatives,
            'search_tree': best_result.search_tree,
            'stats': {
                'total_simulations': sum(r.stats.get('simulations', 0) for r in results),
                'total_nodes_explored': total_nodes,
                'avg_search_time_ms': avg_time,
                'num_workers': len(results)
            }
        }
        
    def shutdown(self):
        """Shutdown all workers."""
        for worker in self.workers:
            worker.stop()