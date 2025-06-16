"""Base feedback loop infrastructure for adaptive components."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from src.config.loader import get_config
from unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParameterUpdate:
    """Represents a parameter update recommendation."""

    parameter_name: str
    current_value: float
    recommended_value: float
    confidence: float
    reason: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class FeedbackLoop(ABC):
    """
    Base class for implementing feedback loops that learn from outcomes.

    This enables any component to adapt its parameters based on real results.
    """

    def __init__(self, name: str, learning_rate: float = 0.01):
        """
        Initialize feedback loop.

        Args:
            name: Name of the component
            learning_rate: How quickly to adapt (0.01 = 1% change per update)
        """
        self.name = name
        self.learning_rate = learning_rate
        self.config = get_config()

        # Track parameter history
        self.parameter_history: dict[str, list[tuple[datetime, float]]] = {}
        self.outcome_buffer: list[dict[str, Any]] = []
        self.update_count = 0

    @abstractmethod
    def extract_features(self, context: dict[str, Any]) -> np.ndarray:
        """Extract features from the current context."""
        pass

    @abstractmethod
    def calculate_reward(self, outcome: dict[str, Any]) -> float:
        """Calculate reward/penalty from outcome."""
        pass

    @abstractmethod
    def get_current_parameters(self) -> dict[str, float]:
        """Get current parameter values."""
        pass

    @abstractmethod
    def apply_parameter_update(self, updates: list[ParameterUpdate]) -> None:
        """Apply parameter updates to the component."""
        pass

    def record_outcome(self, context: dict[str, Any], outcome: dict[str, Any]) -> None:
        """
        Record an outcome for learning.

        Args:
            context: The context when decision was made
            outcome: The actual outcome
        """
        self.outcome_buffer.append(
            {
                "timestamp": datetime.now(),
                "context": context,
                "outcome": outcome,
                "features": self.extract_features(context),
                "reward": self.calculate_reward(outcome),
            }
        )

        # Trigger learning if buffer is large enough
        if len(self.outcome_buffer) >= self.config.learning.min_samples_for_update:
            self.learn_from_outcomes()

    def learn_from_outcomes(self) -> list[ParameterUpdate]:
        """
        Learn from accumulated outcomes and generate parameter updates.

        Returns:
            List of parameter updates
        """
        if not self.outcome_buffer:
            return []

        logger.info(f"{self.name}: Learning from {len(self.outcome_buffer)} outcomes")

        # Get current parameters
        current_params = self.get_current_parameters()

        # Calculate gradients using outcomes
        gradients = self._calculate_gradients()

        # Generate updates
        updates = []
        for param_name, current_value in current_params.items():
            if param_name in gradients:
                # Calculate new value with learning rate
                gradient = gradients[param_name]
                new_value = current_value + self.learning_rate * gradient

                # Apply safety bounds
                new_value = self._apply_safety_bounds(param_name, new_value)

                # Only update if change is significant
                if abs(new_value - current_value) / current_value > 0.001:
                    updates.append(
                        ParameterUpdate(
                            parameter_name=param_name,
                            current_value=current_value,
                            recommended_value=new_value,
                            confidence=self._calculate_confidence(param_name),
                            reason=f"Gradient: {gradient:.4f}, Outcomes: {len(self.outcome_buffer)}",
                        )
                    )

        # Apply updates if approved
        if updates and self.config.learning.auto_apply_updates:
            self.apply_parameter_update(updates)

        # Clear buffer after learning
        self.outcome_buffer = []
        self.update_count += 1

        return updates

    def _calculate_gradients(self) -> dict[str, float]:
        """Calculate parameter gradients from outcomes."""
        gradients = {}

        # Group outcomes by reward
        positive_outcomes = [o for o in self.outcome_buffer if o["reward"] > 0]
        negative_outcomes = [o for o in self.outcome_buffer if o["reward"] < 0]

        if not positive_outcomes or not negative_outcomes:
            return gradients

        # Calculate feature differences
        positive_features = np.mean([o["features"] for o in positive_outcomes], axis=0)
        negative_features = np.mean([o["features"] for o in negative_outcomes], axis=0)

        feature_diff = positive_features - negative_features

        # Map features to parameters (simplified)
        param_names = list(self.get_current_parameters().keys())
        for i, param_name in enumerate(param_names):
            if i < len(feature_diff):
                gradients[param_name] = feature_diff[i]

        return gradients

    def _apply_safety_bounds(self, param_name: str, value: float) -> float:
        """Apply safety bounds to parameter value."""
        # Get bounds from config if available
        bounds = self.config.learning.parameter_bounds.get(param_name, (0.0, 1.0))

        # Clip to bounds
        return np.clip(value, bounds[0], bounds[1])

    def _calculate_confidence(self, param_name: str) -> float:
        """Calculate confidence in parameter update."""
        # Base confidence on number of samples
        sample_confidence = min(len(self.outcome_buffer) / 100, 1.0)

        # Adjust for consistency of outcomes
        rewards = [o["reward"] for o in self.outcome_buffer]
        if rewards:
            reward_std = np.std(rewards)
            consistency_confidence = 1.0 / (1.0 + reward_std)
        else:
            consistency_confidence = 0.5

        return sample_confidence * consistency_confidence

    def get_parameter_history(self, param_name: str) -> list[tuple[datetime, float]]:
        """Get history of a parameter's values."""
        return self.parameter_history.get(param_name, [])
