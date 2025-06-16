"""Mock orchestrator for testing Bob functionality."""

import asyncio
import time
from typing import Any, Dict, List
from ..utils.logging import get_component_logger


class AgentOrchestrator:
    """Mock agent orchestrator."""
    
    def __init__(self, num_agents: int = 8):
        self.num_agents = num_agents
        self.logger = get_component_logger("mock_orchestrator")
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the mock orchestrator."""
        self.logger.info(f"Initializing mock orchestrator with {self.num_agents} agents")
        await asyncio.sleep(0.1)  # Simulate initialization
        self._initialized = True
    
    async def execute_tasks(self, tasks: List[Any]) -> List[Any]:
        """Execute tasks using mock orchestrator."""
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized")
        
        self.logger.info(f"Executing {len(tasks)} tasks via orchestrator")
        
        results = []
        for task in tasks:
            # Simulate task execution
            await asyncio.sleep(0.2)  # Mock processing time
            
            # Create mock result
            result = type('TaskResult', (), {
                'task_id': task.id,
                'success': True,
                'result': f"Orchestrator result for: {task.description}",
                'error': None,
                'duration': 0.2,
                'agent_id': f"orchestrator_agent_{len(results) % self.num_agents}"
            })()
            
            results.append(result)
        
        return results
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get mock orchestrator statistics."""
        return {
            "total_agents": self.num_agents,
            "tasks_processed": 0,
            "avg_task_duration": 0.2,
            "success_rate": 1.0
        }
    
    async def shutdown(self) -> None:
        """Shutdown the mock orchestrator."""
        self.logger.info("Shutting down mock orchestrator")
        self._initialized = False