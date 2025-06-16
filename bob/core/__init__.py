"""Core BOB coordination components."""

from .ultra_fast_coordination import (
    UltraFastCoordinator,
    CoordinationMode,
    FastTaskRequest,
    FastTaskResult,
    create_ultra_fast_coordinator
)

from .enhanced_12_agent_coordinator import (
    Enhanced12AgentCoordinator,
    Enhanced12AgentTask,
    AgentRole,
    create_enhanced_12_agent_coordinator
)

__all__ = [
    "UltraFastCoordinator",
    "CoordinationMode", 
    "FastTaskRequest",
    "FastTaskResult",
    "create_ultra_fast_coordinator",
    "Enhanced12AgentCoordinator",
    "Enhanced12AgentTask", 
    "AgentRole",
    "create_enhanced_12_agent_coordinator"
]