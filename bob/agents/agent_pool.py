"""Mock agent pool for testing Bob functionality."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from ..utils.logging import get_component_logger


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class WorkStealingTask:
    """Mock work stealing task."""
    id: str
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    subdividable: bool = True
    estimated_duration: float = 1.0
    remaining_work: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WorkStealingAgentPool:
    """Mock agent pool for testing."""
    
    def __init__(self, num_agents: int = 8, enable_work_stealing: bool = True):
        self.num_agents = num_agents
        self.enable_work_stealing = enable_work_stealing
        self.logger = get_component_logger("mock_agent_pool")
        self._initialized = False
        self._task_results = {}
    
    async def initialize(self) -> None:
        """Initialize the mock agent pool."""
        self.logger.info(f"Initializing mock agent pool with {self.num_agents} agents")
        await asyncio.sleep(0.1)  # Simulate initialization
        self._initialized = True
    
    async def submit_task(self, task: WorkStealingTask) -> None:
        """Submit a task to the mock pool."""
        if not self._initialized:
            raise RuntimeError("Agent pool not initialized")
        
        self.logger.info(f"Submitting task {task.id} to agent pool")
        
        # Simulate task processing
        processing_time = min(task.estimated_duration + 0.1, 2.0)  # Cap at 2s for testing
        
        # Store result for later retrieval
        self._task_results[task.id] = {
            "success": True,
            "result": f"Mock result for task: {task.description}",
            "duration": processing_time,
            "agent_id": f"agent_{hash(task.id) % self.num_agents}"
        }
    
    async def wait_for_task_completion(self, task_id: str) -> Dict[str, Any]:
        """Wait for task completion and return result."""
        if task_id not in self._task_results:
            raise ValueError(f"Task {task_id} not found")
        
        result = self._task_results[task_id]
        processing_time = result["duration"]
        
        # Simulate processing delay
        await asyncio.sleep(processing_time)
        
        return result
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get mock pool status."""
        return {
            "total_agents": self.num_agents,
            "active_agents": self.num_agents,
            "utilization": 0.7,  # Mock utilization
            "tasks_completed": len(self._task_results)
        }
    
    async def shutdown(self) -> None:
        """Shutdown the mock agent pool."""
        self.logger.info("Shutting down mock agent pool")
        self._initialized = False