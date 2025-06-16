"""
Shared types for the Bolt agents system.

Contains common data structures used across orchestrator and task manager.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a task for agent execution."""

    id: str
    description: str
    priority: TaskPriority
    data: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    estimated_duration: float | None = None


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: str
    success: bool
    result: Any = None
    error: str | None = None
    duration: float | None = None
    agent_id: str | None = None


class AgentStatus(Enum):
    """Agent status levels."""

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class Agent:
    """Represents an agent in the system."""

    id: str
    status: AgentStatus = AgentStatus.IDLE
    current_task: str | None = None
    capabilities: list[str] = field(default_factory=list)
