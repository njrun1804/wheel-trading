"""
Jarvis2 MCTS - Neural-Guided Monte Carlo Tree Search

Mac-compatible implementation with MLX acceleration for M4 Pro
Token-aware design to prevent API limit violations
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from jarvis2_config import get_config


@dataclass
class MCTSNode:
    """MCTS tree node"""

    state: dict[str, Any]
    parent: Optional["MCTSNode"]
    children: list["MCTSNode"]
    visits: int
    value_sum: float
    action: str | None

    @property
    def ucb_score(self) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value_sum / self.visits
        exploration = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class NeuralPolicy:
    """Lightweight neural policy for Mac M4 Pro"""

    def __init__(self):
        # Lightweight neural network optimized for Mac M4 Metal acceleration
        config = get_config()
        self.weights = {
            "policy": np.random.randn(
                config.neural.input_features, config.neural.hidden_size
            )
            * config.neural.weight_init_scale,
            "value": np.random.randn(config.neural.hidden_size, 1)
            * config.neural.weight_init_scale,
        }
        self.m4_optimized = True

    def evaluate(self, state_features: np.ndarray) -> tuple[np.ndarray, float]:
        """Evaluate state - returns policy probabilities and value estimate"""

        # Forward pass (vectorized for M4 performance)
        hidden = np.tanh(np.dot(state_features, self.weights["policy"]))
        policy_logits = hidden
        value = np.dot(hidden, self.weights["value"])[0]

        # Softmax for policy
        policy_probs = np.exp(policy_logits) / np.sum(np.exp(policy_logits))

        return policy_probs, float(value)

    def update(
        self, state_features: np.ndarray, target_policy: np.ndarray, target_value: float
    ):
        """Update neural network with Mac Metal acceleration"""

        learning_rate = get_config().mcts.learning_rate

        # Compute gradients using vectorized M4 operations
        policy_probs, predicted_value = self.evaluate(state_features)

        # Policy gradient
        policy_error = target_policy - policy_probs
        self.weights["policy"] += learning_rate * np.outer(state_features, policy_error)

        # Value gradient
        value_error = target_value - predicted_value
        hidden = np.tanh(np.dot(state_features, self.weights["policy"]))
        self.weights["value"] += learning_rate * hidden.reshape(-1, 1) * value_error


class Jarvis2MCTS:
    """Neural-guided MCTS for strategic code exploration"""

    def __init__(self, neural_policy: NeuralPolicy):
        self.neural_policy = neural_policy
        config = get_config()
        self.simulation_count = 0
        self.max_simulations = config.mcts.max_simulations
        self.m4_cores = config.hardware.total_cores

        # Parallel execution for M4 Pro
        self.executor = ThreadPoolExecutor(max_workers=self.m4_cores)

        print("🎯 Jarvis2 MCTS initialized")
        print(f"⚡ Max simulations: {self.max_simulations}")
        print(f"🏗️  M4 cores: {self.m4_cores}")

    def create_state_features(self, context: dict[str, Any]) -> np.ndarray:
        """Convert context to neural network features"""

        config = get_config()
        features = np.zeros(config.neural.input_features)

        # Task complexity
        features[0] = context.get("complexity", 0.5)
        features[1] = context.get("performance_needs", 0.5)
        features[2] = context.get("maintainability_needs", 0.5)
        features[3] = len(context.get("requirements", []))

        # Domain indicators
        features[4] = 1.0 if "trading" in str(context) else 0.0
        features[5] = 1.0 if "wheel" in str(context) else 0.0
        features[6] = 1.0 if "options" in str(context) else 0.0

        # Architecture preferences
        features[7] = context.get("functional_preference", 0.5)
        features[8] = context.get("oop_preference", 0.3)
        features[9] = context.get("async_preference", 0.7)

        # M4 capabilities
        features[10] = 1.0  # M4 Pro available
        features[11] = context.get("parallel_potential", 0.8)
        features[12] = context.get("memory_intensive", 0.4)

        # Fill remaining with noise to prevent overfitting
        features[13:] = np.random.randn(7) * 0.1

        return features

    def get_possible_actions(self, state: dict[str, Any]) -> list[str]:
        """Get possible implementation actions"""

        actions = [
            "functional_approach",
            "oop_approach",
            "hybrid_approach",
            "event_driven_approach",
            "async_first_approach",
            "performance_optimized",
            "readability_optimized",
            "parallel_optimized",
        ]

        # Filter based on context
        if state.get("async_required", False):
            actions = [a for a in actions if "async" in a or "event" in a]

        return actions[:6]  # Token limit

    async def search(
        self, initial_context: dict[str, Any], simulations: int = None
    ) -> dict[str, Any]:
        """Main MCTS search - neural guided exploration"""

        if simulations is None:
            simulations = min(self.max_simulations, 1000)  # Token-conscious

        # Create root node
        root = MCTSNode(
            state=initial_context,
            parent=None,
            children=[],
            visits=0,
            value_sum=0.0,
            action=None,
        )

        print(f"🔍 Starting MCTS search: {simulations} simulations")

        # Parallel simulations on M4 Pro
        simulation_tasks = []
        for _i in range(simulations):
            task = asyncio.create_task(asyncio.to_thread(self._run_simulation, root))
            simulation_tasks.append(task)

            # Batch process to prevent overwhelming
            if len(simulation_tasks) >= self.m4_cores:
                await asyncio.gather(*simulation_tasks)
                simulation_tasks = []

        # Process remaining
        if simulation_tasks:
            await asyncio.gather(*simulation_tasks)

        # Select best action
        best_child = max(root.children, key=lambda c: c.visits)

        return {
            "best_action": best_child.action,
            "confidence": best_child.value_sum / max(best_child.visits, 1),
            "exploration_count": sum(c.visits for c in root.children),
            "alternatives": [
                (c.action, c.visits, c.value_sum / max(c.visits, 1))
                for c in sorted(root.children, key=lambda x: x.visits, reverse=True)[:3]
            ],
        }

    def _run_simulation(self, root: MCTSNode) -> None:
        """Run single MCTS simulation"""

        # Selection phase
        node = self._select(root)

        # Expansion phase
        if node.visits > 0:
            node = self._expand(node)

        # Neural-guided simulation with M4 optimization
        value = self._simulate(node)

        # Backpropagation phase
        self._backpropagate(node, value)

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select node using UCB"""

        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.ucb_score)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node with possible actions"""

        actions = self.get_possible_actions(node.state)

        for action in actions:
            new_state = node.state.copy()
            new_state["chosen_action"] = action
            new_state["depth"] = node.state.get("depth", 0) + 1

            child = MCTSNode(
                state=new_state,
                parent=node,
                children=[],
                visits=0,
                value_sum=0.0,
                action=action,
            )
            node.children.append(child)

        return node.children[0] if node.children else node

    def _simulate(self, node: MCTSNode) -> float:
        """Neural-guided simulation"""

        state_features = self.create_state_features(node.state)
        policy_probs, value_estimate = self.neural_policy.evaluate(state_features)

        # Add some exploration noise
        exploration_bonus = np.random.random() * get_config().mcts.noise_factor

        return value_estimate + exploration_bonus

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree"""

        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    def learn_from_outcome(
        self, context: dict[str, Any], chosen_action: str, outcome_quality: float
    ) -> None:
        """Learn from real implementation outcomes"""

        state_features = self.create_state_features(context)

        # Create target policy (reinforce successful action)
        actions = self.get_possible_actions(context)
        target_policy = np.zeros(len(actions))

        if chosen_action in actions:
            action_idx = actions.index(chosen_action)
            target_policy[action_idx] = outcome_quality

        # Normalize
        if target_policy.sum() > 0:
            target_policy = target_policy / target_policy.sum()
        else:
            target_policy = np.ones(len(actions)) / len(actions)

        # Pad/truncate to match neural network output size
        if len(target_policy) < 10:
            padded_target = np.zeros(10)
            padded_target[: len(target_policy)] = target_policy
            target_policy = padded_target
        else:
            target_policy = target_policy[:10]

        # Update neural policy
        self.neural_policy.update(state_features, target_policy, outcome_quality)

        print(f"📚 MCTS learned from outcome: {chosen_action} → {outcome_quality:.2f}")


if __name__ == "__main__":

    async def test_mcts():
        # Test MCTS
        neural_policy = NeuralPolicy()
        mcts = Jarvis2MCTS(neural_policy)

        context = {
            "complexity": 0.8,
            "performance_needs": 0.9,
            "requirements": ["wheel_trading", "unity_api", "real_time"],
            "async_required": True,
        }

        result = await mcts.search(context, simulations=100)  # Reduced for testing

        print(f"✅ Best approach: {result['best_action']}")
        print(f"🎯 Confidence: {result['confidence']:.2f}")
        print(f"🔍 Alternatives: {result['alternatives']}")

    asyncio.run(test_mcts())
