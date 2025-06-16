"""Example integration of HardwareState with existing jarvis2 components.

Shows how to replace scattered hardware detection calls with the unified
HardwareState singleton for consistent resource management.
"""

import asyncio
import logging
from typing import Any

# Import the unified hardware state
from bolt.hardware.hardware_state import ResourceBudget, get_hardware_state

# Example existing imports (would be from jarvis2.hardware)
# from jarvis2.hardware.hardware_optimizer import HardwareAwareExecutor
# from jarvis2.hardware.metal_monitor import MetalGPUMonitor

logger = logging.getLogger(__name__)


class UnifiedHardwareAwareExecutor:
    """Example executor using unified HardwareState instead of scattered detection."""

    def __init__(self):
        # Single source of truth for hardware
        self.hw_state = get_hardware_state()
        self.agent_id = f"executor_{id(self)}"
        self.resource_budget: ResourceBudget | None = None

        # Log detected hardware once
        logger.info(f"Executor initialized with: {self.hw_state.get_summary()}")

    async def initialize(self):
        """Initialize with proper resource allocation."""
        # Request resources based on workload
        self.resource_budget = self.hw_state.allocate_resources(
            self.agent_id, requested_memory_mb=4096, task_type="cpu"
        )

        if not self.resource_budget:
            raise RuntimeError("Insufficient resources for executor")

        logger.info(
            f"Allocated resources: {self.resource_budget.cpu_workers} workers, "
            f"{self.resource_budget.memory_pool_mb}MB memory"
        )

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute task with hardware-aware optimization."""
        # Classify task type
        task_type = task.get("type", "general")

        # Get optimal settings for this task type
        if task_type == "neural":
            budget = self.hw_state.get_resource_budget("gpu")
        elif task_type == "search":
            budget = self.hw_state.get_resource_budget("cpu")
        elif task_type == "io":
            budget = self.hw_state.get_resource_budget("io")
        else:
            budget = self.resource_budget or self.hw_state.get_resource_budget(
                "general"
            )

        # Execute with appropriate resources
        logger.info(
            f"Executing {task_type} task with {budget.cpu_workers} workers, "
            f"batch size {budget.batch_size}"
        )

        # Simulate execution
        await asyncio.sleep(0.1)

        return {
            "status": "success",
            "task_type": task_type,
            "resources_used": {
                "cpu_workers": budget.cpu_workers,
                "memory_mb": budget.memory_pool_mb,
                "batch_size": budget.batch_size,
            },
        }

    def shutdown(self):
        """Clean shutdown with resource release."""
        self.hw_state.release_resources(self.agent_id)
        logger.info(f"Executor {self.agent_id} released resources")


class UnifiedMetalMonitor:
    """Example GPU monitor using unified HardwareState."""

    def __init__(self):
        self.hw_state = get_hardware_state()
        self.agent_id = f"gpu_monitor_{id(self)}"

    async def start_monitoring(self):
        """Start GPU monitoring using unified state."""
        # Allocate minimal resources for monitoring
        budget = self.hw_state.allocate_resources(
            self.agent_id, requested_memory_mb=256, task_type="io"
        )

        if not budget:
            logger.warning("Could not allocate resources for GPU monitoring")
            return

        logger.info(f"GPU monitoring started: {self.hw_state.gpu}")

    def get_gpu_utilization(self) -> float:
        """Get GPU utilization from unified state."""
        return self.hw_state.get_utilization()["gpu_percent"]

    def get_gpu_memory_mb(self) -> float:
        """Get GPU memory usage estimate."""
        # For unified memory, this is part of system memory
        mem = self.hw_state.memory
        return mem.used_gb * 1024 * 0.3  # Estimate 30% for GPU

    def get_metrics(self) -> dict[str, Any]:
        """Get all GPU metrics from unified state."""
        utilization = self.hw_state.get_utilization()
        return {
            "utilization_percent": utilization["gpu_percent"],
            "memory_used_mb": self.get_gpu_memory_mb(),
            "gpu_info": self.hw_state.gpu.__dict__,
            "monitoring": True,
        }


class UnifiedResourceManager:
    """Central resource manager using HardwareState for all agents."""

    def __init__(self):
        self.hw_state = get_hardware_state()
        self.agents: dict[str, Any] = {}

    async def create_agent(self, agent_type: str, memory_mb: int = 1024) -> str:
        """Create an agent with proper resource allocation."""
        agent_id = f"{agent_type}_{len(self.agents)}"

        # Map agent type to task type
        task_map = {"search": "cpu", "neural": "gpu", "io": "io", "cache": "memory"}
        task_type = task_map.get(agent_type, "general")

        # Allocate resources
        budget = self.hw_state.allocate_resources(agent_id, memory_mb, task_type)

        if not budget:
            raise RuntimeError(
                f"Cannot create {agent_type} agent: insufficient resources"
            )

        self.agents[agent_id] = {
            "type": agent_type,
            "budget": budget,
            "created_at": budget.timestamp,
        }

        logger.info(
            f"Created {agent_type} agent with {budget.cpu_workers} workers, "
            f"{budget.memory_pool_mb}MB memory"
        )

        return agent_id

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "hardware": self.hw_state.to_dict(),
            "agents": self.agents,
            "utilization": self.hw_state.get_utilization(),
        }

    async def shutdown_agent(self, agent_id: str):
        """Shutdown agent and release resources."""
        if agent_id in self.agents:
            self.hw_state.release_resources(agent_id)
            del self.agents[agent_id]
            logger.info(f"Shutdown agent {agent_id}")


# Example: Migrating from old pattern to new pattern
async def migration_example():
    """Show how to migrate from old hardware detection to unified state."""

    print("=== OLD PATTERN (Multiple detection points) ===")
    # Simulated old pattern - multiple components detecting hardware
    print("HardwareOptimizer: Detecting M4 Pro...")
    print("MetalMonitor: Checking GPU...")
    print("MemoryManager: Calculating memory limits...")
    print("(Multiple syscalls, inconsistent results, slow startup)")

    print("\n=== NEW PATTERN (Unified HardwareState) ===")
    # New pattern - single detection, shared state
    hw = get_hardware_state()
    print(f"Hardware detected once: {hw.get_summary()}")
    print("All components use same hardware info (fast, consistent)")

    # Example usage
    print("\n=== Creating Agents with Proper Resource Management ===")
    manager = UnifiedResourceManager()

    # Create different types of agents
    search_agent = await manager.create_agent("search", memory_mb=2048)
    neural_agent = await manager.create_agent("neural", memory_mb=4096)
    io_agent = await manager.create_agent("io", memory_mb=512)

    # Show system status
    print("\n=== System Status ===")
    status = manager.get_system_status()
    print(f"Total agents: {len(status['agents'])}")
    print(f"CPU utilization: {status['utilization']['cpu_percent']:.1f}%")
    print(f"Memory utilization: {status['utilization']['memory_percent']:.1f}%")
    print(f"Allocated agents: {status['utilization']['allocated_agents']}")

    # Test overcommit protection
    print("\n=== Memory Safety Test ===")
    try:
        # Try to allocate too much memory
        await manager.create_agent("cache", memory_mb=20000)
    except RuntimeError as e:
        print(f"Correctly prevented overcommit: {e}")

    # Cleanup
    await manager.shutdown_agent(search_agent)
    await manager.shutdown_agent(neural_agent)
    await manager.shutdown_agent(io_agent)


# Integration with existing components
async def integrate_with_jarvis2():
    """Show how to integrate with existing jarvis2 components."""

    print("=== Integrating with jarvis2 Components ===")

    # 1. Replace HardwareAwareExecutor
    executor = UnifiedHardwareAwareExecutor()
    await executor.initialize()

    # Execute tasks
    result = await executor.execute_task({"type": "search", "query": "test"})
    print(f"Search task result: {result}")

    result = await executor.execute_task({"type": "neural", "model": "test"})
    print(f"Neural task result: {result}")

    # 2. Replace MetalGPUMonitor
    gpu_monitor = UnifiedMetalMonitor()
    await gpu_monitor.start_monitoring()

    metrics = gpu_monitor.get_metrics()
    print(f"GPU metrics: {metrics}")

    # 3. Show unified resource view
    hw = get_hardware_state()
    print(f"\nUnified system view: {hw.to_dict()}")

    # Cleanup
    executor.shutdown()


if __name__ == "__main__":
    # Run examples
    print("Hardware State Integration Examples\n")

    asyncio.run(migration_example())
    print("\n" + "=" * 60 + "\n")
    asyncio.run(integrate_with_jarvis2())
