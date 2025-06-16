"""Mock agent types for testing."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class TaskPriority(Enum):
    """Task priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Task:
    """Mock task definition."""
    id: str
    description: str
    priority: TaskPriority
    data: Dict[str, Any]
    dependencies: List[str]
    estimated_duration: float = 1.0