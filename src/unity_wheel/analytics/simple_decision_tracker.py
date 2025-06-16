"""Simple decision tracking implementation."""
from __future__ import annotations

from datetime import datetime
from typing import Any


class DecisionTracker:
    """Simple decision tracker implementation."""

    def __init__(self):
        self.decisions = []

    def track_decision(self, decision: dict[str, Any]) -> str:
        """Track a decision."""
        decision_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.decisions.append(
            {
                "id": decision_id,
                "timestamp": datetime.now().isoformat(),
                "decision": decision,
            }
        )
        return decision_id

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get basic performance metrics."""
        return {
            "total_decisions": len(self.decisions),
            "last_decision": self.decisions[-1] if self.decisions else None,
        }
