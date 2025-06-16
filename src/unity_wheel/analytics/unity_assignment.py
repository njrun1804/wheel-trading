"""Unity assignment model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AssignmentProbability:
    """Assignment probability result."""

    probability: float
    confidence: float = 0.95


class UnityAssignmentModel:
    """Unity-specific assignment model."""

    def __init__(self):
        """Initialize model."""
        pass

    def calculate_assignment_probability(
        self, strike: float, spot: float, dte: int, volatility: float = 0.2
    ) -> AssignmentProbability:
        """Calculate assignment probability."""
        # Simplified calculation
        moneyness = strike / spot
        prob = max(0, min(1, 1 - moneyness))
        return AssignmentProbability(probability=prob)
