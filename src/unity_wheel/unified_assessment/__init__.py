"""
Unified Assessment Engine - Single coherent path for natural language commands.

This module provides a unified interface for processing any natural language command
by combining Einstein semantic search, intelligent intent analysis, sophisticated
action planning, and optimized execution routing.

Key Components:
- Context Gathering Layer: Einstein-powered semantic understanding
- Intent Analysis Layer: Multi-model intent classification with context awareness
- Action Planning Layer: Task decomposition and resource optimization
- Execution Routing Layer: Bolt multi-agent and direct tool coordination

Usage:
    engine = UnifiedAssessmentEngine()
    result = await engine.process_command("fix authentication issue")
"""

from .core.engine import UnifiedAssessmentEngine
from .core.context import (
    ContextGatherer,
    GatheredContext,
    ContextQuery,
    UnifiedContext
)
from .core.intent import (
    IntentAnalyzer,
    IntentAnalysis,
    Intent,
    IntentCategory
)
from .core.planning import (
    ActionPlanner,
    ExecutionPlan,
    ActionTask,
    TaskType
)
from .core.routing import (
    ExecutionRouter,
    ExecutionResult,
    RoutingStrategy
)
from .schemas.command import (
    CommandResult,
    CommandStatus,
    CommandMetrics
)

__all__ = [
    # Main Engine
    "UnifiedAssessmentEngine",
    
    # Context Components
    "ContextGatherer",
    "GatheredContext", 
    "ContextQuery",
    "UnifiedContext",
    
    # Intent Components
    "IntentAnalyzer",
    "IntentAnalysis",
    "Intent", 
    "IntentCategory",
    
    # Planning Components
    "ActionPlanner",
    "ExecutionPlan",
    "ActionTask",
    "TaskType",
    
    # Routing Components
    "ExecutionRouter",
    "ExecutionResult",
    "RoutingStrategy",
    
    # Schema Components
    "CommandResult",
    "CommandStatus", 
    "CommandMetrics"
]

__version__ = "1.0.0"