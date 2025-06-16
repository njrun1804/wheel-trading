"""Jarvis - Meta-coder for Claude Code CLI optimized for M4 Pro Mac.

Jarvis helps Claude Code understand and execute complex tasks by:
1. Analyzing requests using simplified phases
2. Leveraging hardware acceleration (12 cores + Metal GPU)
3. Using our fast local tools instead of slow MCP servers
4. Applying MCTS for optimization problems when needed

Usage:
    ./jarvis "optimize all trading functions for performance"
    
    # Or programmatically:
    from jarvis import Jarvis
    j = Jarvis()
    result = await j.assist("find and fix all type errors")
"""

from .core.jarvis import Jarvis, JarvisConfig
from .core.phases import Phase, PhaseResult

__version__ = "1.0.0"
__all__ = ["Jarvis", "JarvisConfig", "Phase", "PhaseResult"]
