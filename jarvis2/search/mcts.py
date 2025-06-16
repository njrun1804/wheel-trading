"""Neural-Guided Monte Carlo Tree Search for code generation.

Implements MCTS with neural value and policy networks to efficiently
explore the vast space of possible code implementations.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.solution import SearchNode
from ..neural.lazy_networks import LazyPolicyNetwork, LazyValueNetwork

logger = logging.getLogger(__name__)


class MCTSConfig:
    """Configuration for MCTS."""

    def __init__(self):
        self.c_puct = 1.414
        self.batch_size = 256
        self.virtual_loss = 3.0
        self.max_depth = 10
        self.temperature = 1.0
        self.epsilon = 0.25
        self.alpha = 0.3


class NeuralGuidedMCTS:
    """MCTS with neural network guidance."""

    def __init__(self, config: MCTSConfig | None = None):
        self.config = config or MCTSConfig()
        self.value_net = LazyValueNetwork(hidden_dim=256, num_layers=2)
        self.policy_net = LazyPolicyNetwork(hidden_dim=256, num_layers=2)
        self.action_space = CodeActionSpace()
        self.total_simulations = 0
        self.cache_hits = 0
        self.neural_evals = 0
        self._model_update_count = 0

    async def initialize(self):
        """Initialize neural networks."""
        logger.info("MCTS initialized with lazy-loading networks")
        self._initialized = True

    async def explore(
        self,
        query: str,
        context: dict[str, Any],
        simulations: int = 2000,
        parallel_batch_size: int = 256,
        hardware_executor: Any | None = None,
    ) -> SearchNode:
        """Run MCTS exploration with neural guidance."""
        root = SearchNode(code="", parent=None)
        await self._add_exploration_noise(root)
        if hardware_executor:
            self.hardware_executor = hardware_executor
        batch_count = simulations // parallel_batch_size
        remainder = simulations % parallel_batch_size
        logger.info(f"Running {simulations} MCTS simulations in {batch_count} batches")
        start_time = time.time()
        for batch_idx in range(batch_count):
            await self._run_batch_simulations(root, query, context, parallel_batch_size)
            if batch_idx % 10 == 0:
                logger.debug(f"Completed batch {batch_idx + 1}/{batch_count}")
        if remainder > 0:
            await self._run_batch_simulations(root, query, context, remainder)
        elapsed = time.time() - start_time
        logger.info(
            f"MCTS complete: {simulations} simulations in {elapsed:.2f}s ({simulations / elapsed:.0f} sims/sec)"
        )
        self.total_simulations += simulations
        return root

    async def fast_search(
        self,
        query: str,
        context: dict[str, Any],
        simulations: int = 100,
        hardware_executor: Any | None = None,
    ) -> SearchNode:
        """Fast search for simple tasks."""
        if hardware_executor:
            self.hardware_executor = hardware_executor
        root = SearchNode(code="", parent=None)
        await self._run_batch_simulations(root, query, context, simulations)
        return root

    async def _run_batch_simulations(
        self, root: SearchNode, query: str, context: dict[str, Any], batch_size: int
    ):
        """Run a batch of simulations in parallel."""
        paths = []
        for _ in range(batch_size):
            leaf, path = self._select_leaf(root)
            paths.append((leaf, path))
            for node in path:
                node.visits += self.config.virtual_loss
        leaves = [leaf for leaf, _ in paths]
        values, action_probs = await self._batch_evaluate(leaves, query, context)
        for i, (leaf, path) in enumerate(paths):
            for node in path:
                node.visits -= self.config.virtual_loss
            if leaf.depth < self.config.max_depth:
                await self._expand_node(leaf, action_probs[i], query, context)
            self._backpropagate(path, values[i])

    def _select_leaf(self, root: SearchNode) -> tuple[SearchNode, list[SearchNode]]:
        """Select a leaf node using PUCT."""
        path = []
        node = root
        while node.children:
            node = self._select_child(node)
            path.append(node)
        return node, [root] + path

    def _select_child(self, node: SearchNode) -> SearchNode:
        """Select best child using PUCT formula."""
        puct_values = []
        sqrt_parent_visits = math.sqrt(node.visits + 1)
        for child in node.children:
            q_value = child.average_value
            u_value = (
                self.config.c_puct
                * child.prior_probability
                * sqrt_parent_visits
                / (1 + child.visits)
            )
            puct_values.append(q_value + u_value)
        best_idx = np.argmax(puct_values)
        return node.children[best_idx]

    async def _batch_evaluate(
        self, nodes: list[SearchNode], query: str, context: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Evaluate multiple nodes in a single GPU batch."""
        await asyncio.sleep(0)
        self.neural_evals += len(nodes)
        batch_features = []
        for node in nodes:
            features = self._extract_features(node, query, context)
            batch_features.append(features)
        batch_tensor = torch.stack(batch_features)
        with torch.no_grad():
            values = self.value_net(batch_tensor)
            action_logits = self.policy_net(batch_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
        values_np = values.cpu().numpy()
        action_probs_np = action_probs.cpu().numpy()
        action_probs_list = [action_probs_np[i] for i in range(len(nodes))]
        return values_np, action_probs_list

    def _extract_features(
        self, node: SearchNode, query: str, context: dict[str, Any]
    ) -> torch.Tensor:
        """Extract features for neural network input."""
        features = []
        query_hash = hash(query) % 1000
        features.extend([query_hash / 1000.0] * 128)
        code_len = len(node.code)
        features.extend(
            [
                code_len / 1000.0,
                node.depth / self.config.max_depth,
                node.visits / 100.0,
                node.average_value,
            ]
        )
        num_files = len(context.get("files", []))
        features.extend(
            [num_files / 100.0, len(context.get("dependencies", [])) / 50.0]
        )
        while len(features) < 768:
            features.append(0.0)
        return torch.tensor(features[:768], dtype=torch.float32)

    def _embed_state(self, state: Any) -> np.ndarray:
        """Create embedding from state for training."""
        if isinstance(state, str):
            features = []
            features.append(len(state) / 1000.0)
            features.append(state.count("\n") / 100.0)
            features.append(state.count("def ") / 10.0)
            features.append(state.count("class ") / 10.0)
            tokens = state.split()
            features.append(len(tokens) / 500.0)
            features.append(len(set(tokens)) / len(tokens) if tokens else 0)
            while len(features) < 768:
                features.append(0.0)
            return np.array(features[:768], dtype=np.float32)
        elif hasattr(state, "__array__"):
            return np.array(state, dtype=np.float32)
        else:
            return np.zeros(768, dtype=np.float32)

    async def _expand_node(
        self,
        node: SearchNode,
        action_probs: np.ndarray,
        query: str,
        context: dict[str, Any],
    ):
        """Expand a node with possible actions."""
        await asyncio.sleep(0)
        valid_actions = self.action_space.get_valid_actions(node, query, context)
        if not valid_actions:
            return
        k = min(5, len(valid_actions))
        action_indices = np.argsort(action_probs)[-k:][::-1]
        for idx in action_indices:
            if idx < len(valid_actions):
                action = valid_actions[idx]
                new_code = self.action_space.apply_action(node.code, action)
                prior = float(action_probs[idx])
                node.add_child(new_code, action["description"], prior)

    def _backpropagate(self, path: list[SearchNode], value: float):
        """Backpropagate value through path."""
        for node in reversed(path):
            node.update(value)

    async def _add_exploration_noise(self, root: SearchNode):
        """Add Dirichlet noise to root node for exploration."""
        if not root.children:
            dummy_probs = np.ones(10) / 10
            await self._expand_node(root, dummy_probs, "", {})
        if root.children:
            noise = np.random.dirichlet([self.config.alpha] * len(root.children))
            for i, child in enumerate(root.children):
                child.prior_probability = (
                    1 - self.config.epsilon
                ) * child.prior_probability + self.config.epsilon * noise[i]

    def _warmup_models(self):
        """Warm up neural networks."""
        try:
            dummy_batch = torch.randn(4, 768)
            with torch.no_grad():
                _ = self.value_net(dummy_batch)
                _ = self.policy_net(dummy_batch)
            logger.info("Neural networks warmed up")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}, continuing anyway")

    async def save_models(self, save_dir: Path):
        """Save neural network models."""
        await asyncio.sleep(0)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.value_net.state_dict(), save_dir / "value_net.pt")
        torch.save(self.policy_net.state_dict(), save_dir / "policy_net.pt")
        metadata = {
            "total_simulations": self.total_simulations,
            "model_updates": self._model_update_count,
            "timestamp": time.time(),
        }
        import json

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        logger.info(f"Models saved to {save_dir}")

    async def load_models(self, load_dir: Path):
        """Load neural network models."""
        await asyncio.sleep(0)
        value_path = load_dir / "value_net.pt"
        policy_path = load_dir / "policy_net.pt"
        if value_path.exists():
            self.value_net.load_state_dict(torch.load(value_path))
            logger.info("Loaded value network")
        if policy_path.exists():
            self.policy_net.load_state_dict(torch.load(policy_path))
            logger.info("Loaded policy network")
        metadata_path = load_dir / "metadata.json"
        if metadata_path.exists():
            import json

            with open(metadata_path) as f:
                metadata = json.load(f)
                self.total_simulations = metadata.get("total_simulations", 0)
                self._model_update_count = metadata.get("model_updates", 0)

    def get_update_count(self) -> int:
        """Get number of model updates."""
        return self._model_update_count

    async def train_batch(
        self, experiences: list[dict[str, Any]]
    ) -> tuple[float, float]:
        """Train networks on a batch of experiences."""
        if not experiences:
            return 0.0, 0.0
        states = []
        values = []
        actions = []
        advantages = []
        for exp in experiences:
            state_embedding = self._embed_state(exp["state"])
            states.append(state_embedding)
            values.append(exp.get("value", 0.5))
            actions.append(exp.get("action_idx", 0))
            advantages.append(exp.get("advantage", 0.0))
        value_data = list(zip(states, values, strict=False))
        value_metrics = await self.value_net.train_batch(value_data)
        policy_data = list(
            zip(states, zip(actions, advantages, strict=False), strict=False)
        )
        policy_metrics = await self.policy_net.train_batch(policy_data)
        self._model_update_count += 1
        return value_metrics.get("loss", 0.0), policy_metrics.get("loss", 0.0)


class CodeActionSpace:
    """Defines the action space for code generation."""

    def __init__(self):
        self.action_templates = [
            {"type": "add_import", "description": "Add import statement"},
            {"type": "add_function", "description": "Add new function"},
            {"type": "add_class", "description": "Add new class"},
            {"type": "modify_function", "description": "Modify existing function"},
            {"type": "add_error_handling", "description": "Add error handling"},
            {"type": "add_validation", "description": "Add input validation"},
            {"type": "optimize_loop", "description": "Optimize loop"},
            {"type": "add_caching", "description": "Add caching"},
            {"type": "refactor_code", "description": "Refactor code structure"},
            {"type": "add_tests", "description": "Add unit tests"},
        ]

    def get_valid_actions(
        self, node: SearchNode, query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get valid actions for current code state."""
        if node.depth >= 8:
            return []
        valid_actions = []
        query_lower = query.lower()
        for action in self.action_templates:
            if (
                "optimize" in query_lower
                and "optimize" in action["type"]
                or "test" in query_lower
                and "test" in action["type"]
                or "error" in query_lower
                and "error" in action["type"]
                or node.depth < 3
            ):
                valid_actions.append(action)
        return valid_actions or self.action_templates[:5]

    def apply_action(self, current_code: str, action: dict[str, Any]) -> str:
        """Apply an action to generate new code."""
        action_type = action["type"]
        if not current_code:
            return self._generate_initial_code(action_type)
        new_lines = []
        if action_type == "add_import":
            new_lines = ["import numpy as np", "import pandas as pd"]
        elif action_type == "add_function":
            new_lines = [
                "",
                "def process_data(data):",
                "    # Process the data",
                "    return data",
            ]
        elif action_type == "add_class":
            new_lines = [
                "",
                "class DataProcessor:",
                "    def __init__(self):",
                "        self.data = None",
                "    ",
                "    def process(self, data):",
                "        return data",
            ]
        elif action_type == "add_error_handling":
            new_lines = [
                "",
                "try:",
                "    # Main logic here",
                "    pass",
                "except Exception as e:",
                "    logger.error(f'Error: {e}')",
                "    raise",
            ]
        elif action_type == "add_validation":
            new_lines = [
                "",
                "def validate_input(data):",
                "    if not data:",
                "        raise ValueError('Data cannot be empty')",
                "    return True",
            ]
        elif action_type == "optimize_loop":
            new_lines = [
                "",
                "# Optimized with vectorization",
                "result = np.vectorize(process_func)(data_array)",
            ]
        elif action_type == "add_caching":
            new_lines = [
                "",
                "from functools import lru_cache",
                "",
                "@lru_cache(maxsize=128)",
                "def cached_computation(x):",
                "    return expensive_operation(x)",
            ]
        else:
            new_lines = [f"\n# Action: {action_type}"]
        return current_code + "\n" + "\n".join(new_lines)

    def _generate_initial_code(self, action_type: str) -> str:
        """Generate initial code based on first action."""
        if "import" in action_type:
            return "import numpy as np\nimport pandas as pd"
        elif "class" in action_type:
            return "class Solution:\n    def __init__(self):\n        pass"
        else:
            return "# Generated code\n"


class MCTSVisualizer:
    """Visualize MCTS tree for debugging."""

    @staticmethod
    def print_tree(root: SearchNode, max_depth: int = 3):
        """Print tree structure."""

        def _print_node(node: SearchNode, depth: int = 0, prefix: str = ""):
            if depth > max_depth:
                return
            indent = "  " * depth
            print(
                f"{indent}{prefix}Visits: {node.visits}, Value: {node.average_value:.3f}, Action: {node.action_taken}"
            )
            for i, child in enumerate(node.children):
                is_last = i == len(node.children) - 1
                _print_node(child, depth + 1, "└─ " if is_last else "├─ ")

        _print_node(root)

    @staticmethod
    def get_best_path(root: SearchNode) -> list[str]:
        """Get best path through tree."""
        path = []
        node = root
        while node.children:
            best_child = max(node.children, key=lambda c: c.visits)
            path.append(best_child.action_taken)
            node = best_child
        return path
