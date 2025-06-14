"""Lightweight MCTS implementation for Jarvis.

Simplified version that focuses on practical code optimization
rather than complex neural networks.
"""

import math
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
import time


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    state: Dict[str, Any]
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    value: float = 0.0
    action: Optional[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def ucb_score(self) -> float:
        """Calculate UCB1 score for node selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def best_child(self) -> 'MCTSNode':
        """Select best child based on UCB scores."""
        return max(self.children, key=lambda c: c.ucb_score)
    
    def add_child(self, state: Dict[str, Any], action: str) -> 'MCTSNode':
        """Add a child node."""
        child = MCTSNode(state=state, parent=self, action=action)
        self.children.append(child)
        return child


class MCTSLite:
    """Lightweight MCTS for code optimization decisions."""
    
    def __init__(self, max_simulations: int = 1000, max_depth: int = 10):
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.root = None
        
    async def search(self, initial_state: Dict[str, Any], 
                    time_limit_ms: float = 5000) -> Dict[str, Any]:
        """Run MCTS search to find best action."""
        start_time = time.perf_counter()
        self.root = MCTSNode(state=initial_state)
        
        simulations = 0
        
        while simulations < self.max_simulations:
            # Check time limit
            if (time.perf_counter() - start_time) * 1000 > time_limit_ms:
                break
            
            # Run one simulation
            leaf = self._select(self.root)
            value = await self._simulate(leaf)
            self._backpropagate(leaf, value)
            
            simulations += 1
            
            # Yield control periodically
            if simulations % 100 == 0:
                await asyncio.sleep(0)
        
        # Return best action
        best_child = max(self.root.children, key=lambda c: c.visits)
        
        return {
            "best_action": best_child.action,
            "confidence": best_child.visits / simulations,
            "expected_value": best_child.value / best_child.visits if best_child.visits > 0 else 0,
            "simulations_run": simulations,
            "time_ms": (time.perf_counter() - start_time) * 1000
        }
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node for expansion."""
        current = node
        depth = 0
        
        while current.children and depth < self.max_depth:
            current = current.best_child()
            depth += 1
        
        # Expand if not fully expanded
        if not self._is_terminal(current) and len(current.children) < self._get_actions(current.state):
            return self._expand(current)
        
        return current
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by adding a child."""
        available_actions = self._get_actions(node.state)
        tried_actions = {child.action for child in node.children}
        
        # Find untried action
        untried = [a for a in available_actions if a not in tried_actions]
        
        if untried:
            action = random.choice(untried)
            new_state = self._apply_action(node.state, action)
            return node.add_child(new_state, action)
        
        return node
    
    async def _simulate(self, node: MCTSNode) -> float:
        """Simulate from a node to estimate its value."""
        state = node.state.copy()
        depth = 0
        
        # Random playout
        while not self._is_terminal_state(state) and depth < self.max_depth:
            actions = self._get_actions(state)
            if not actions:
                break
                
            action = random.choice(actions)
            state = self._apply_action(state, action)
            depth += 1
        
        # Evaluate terminal state
        return self._evaluate_state(state)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent
    
    def _get_actions(self, state: Dict[str, Any]) -> List[str]:
        """Get available actions for a state."""
        # Code optimization actions
        task_type = state.get("task_type", "general")
        
        if task_type == "optimization":
            return [
                "parallelize_loops",
                "cache_results", 
                "vectorize_operations",
                "remove_redundancy",
                "optimize_imports",
                "use_generators",
                "batch_operations"
            ]
        elif task_type == "refactoring":
            return [
                "extract_method",
                "rename_variable",
                "simplify_condition",
                "remove_duplication",
                "improve_naming",
                "split_class"
            ]
        else:
            return ["modify_code", "add_feature", "fix_issue"]
    
    def _apply_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Apply an action to a state."""
        new_state = state.copy()
        
        # Track actions taken
        if "actions_taken" not in new_state:
            new_state["actions_taken"] = []
        new_state["actions_taken"].append(action)
        
        # Update metrics based on action
        if action == "parallelize_loops":
            new_state["performance_gain"] = new_state.get("performance_gain", 0) + 20
            new_state["complexity_increase"] = new_state.get("complexity_increase", 0) + 5
        elif action == "cache_results":
            new_state["performance_gain"] = new_state.get("performance_gain", 0) + 15
            new_state["memory_usage"] = new_state.get("memory_usage", 0) + 10
        # ... more action effects
        
        return new_state
    
    def _is_terminal(self, node: MCTSNode) -> bool:
        """Check if a node is terminal."""
        return self._is_terminal_state(node.state)
    
    def _is_terminal_state(self, state: Dict[str, Any]) -> bool:
        """Check if a state is terminal."""
        # Terminal if we've taken enough actions or reached goal
        actions_taken = len(state.get("actions_taken", []))
        return actions_taken >= 5  # Max 5 optimizations
    
    def _evaluate_state(self, state: Dict[str, Any]) -> float:
        """Evaluate the quality of a state."""
        # Simple evaluation based on performance vs complexity
        performance = state.get("performance_gain", 0)
        complexity = state.get("complexity_increase", 0)
        memory = state.get("memory_usage", 0)
        
        # Weighted score
        score = performance - (0.5 * complexity) - (0.3 * memory)
        
        # Normalize to 0-1
        return max(0, min(1, score / 100))