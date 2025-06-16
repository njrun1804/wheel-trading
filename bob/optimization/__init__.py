"""M4 Pro optimizations for Bob.

This package provides hardware-specific optimizations for Apple M4 Pro,
including HTTP/2 session pooling, P-core utilization, and OpenMP management.
"""

from .m4_claude_optimizer import (
    M4ClaudeOptimizer,
    M4OptimizationConfig,
    create_m4_optimizer,
    get_global_optimizer
)

__all__ = [
    "M4ClaudeOptimizer",
    "M4OptimizationConfig", 
    "create_m4_optimizer",
    "get_global_optimizer"
]