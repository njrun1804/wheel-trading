"""MCP Orchestrator - Consolidated and optimized for M4 Pro.

New structure:
- orchestrator_consolidated.py: Main orchestrator with strategy pattern
- strategies/: Different execution strategies
- components/: Core components (MCP client, monitors, cache)
- optimization/: Advanced algorithms (MCTS, PBT, learning)
- gpu/: GPU acceleration components
- configs/: Unified configuration system
"""

# Core orchestrator
from .orchestrator_consolidated import (
    ConsolidatedOrchestrator,
    StrategyType,
    orchestrate
)

# Strategy base class
from .strategies.base import ExecutionStrategy

# Configuration
from .configs.unified_config import (
    UnifiedOrchestratorConfig,
    TaskType,
    SearchStrategy,
    MAC_OPTIMIZED_CONFIG,
    detect_task_type,
    get_config,
    get_task_config,
    create_execution_plan,
    optimize_for_command
)

# Core components
from .components.mcp_client import (
    MCPClient,
    MCPConnection,
    MCPServerConfig
)

# Memory monitoring
from .pressure import (
    MemoryPressureMonitor,
    MemorySnapshot
)

# Caching
from .slice_cache import SliceCache

# Legacy support (will be deprecated)
from .orchestrator import MCPOrchestrator, ExecutionPlan, Phase, PhaseResult  # TODO: File doesn't exist

__all__ = [
    # New consolidated orchestrator
    "ConsolidatedOrchestrator",
    "ExecutionStrategy", 
    "StrategyType",
    "orchestrate",
    
    # Configuration
    "UnifiedOrchestratorConfig",
    "MAC_OPTIMIZED_CONFIG",
    "TaskType",
    "SearchStrategy",
    "detect_task_type",
    "get_config",
    "get_task_config",
    "create_execution_plan",
    "optimize_for_command",
    
    # Components
    "MCPClient",
    "MCPConnection", 
    "MCPServerConfig",
    "MemoryPressureMonitor",
    "MemorySnapshot",
    "SliceCache",
    
    # Legacy (deprecated)
    # "MCPOrchestrator",
    # "ExecutionPlan",
    # "Phase", 
    # "PhaseResult"
]

# Version info
__version__ = "2.0.0"  # Major version bump for consolidated structure