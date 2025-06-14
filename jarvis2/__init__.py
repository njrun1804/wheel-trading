"""Jarvis2 - Neural-Guided Meta-Coding System for M4 Pro.

A learning, exploring, evolving AI system that treats code generation 
as an intelligent search problem. Implements MCTS, neural guidance,
diversity generation, and continuous learning.

Usage:
    from jarvis2 import Jarvis2Orchestrator, CodeRequest
    
    jarvis = Jarvis2Orchestrator()
    await jarvis.initialize()
    
    request = CodeRequest("optimize options pricing calculations")
    solution = await jarvis.generate_code(request)
"""

# Check if we have the new orchestrator structure
try:
    from .core.orchestrator import CodeRequest, CodeSolution, Jarvis2Config, Jarvis2Orchestrator
    
    __all__ = [
        "Jarvis2Orchestrator",
        "CodeRequest", 
        "CodeSolution",
        "Jarvis2Config",
        "get_router",
        "Backend",
        "OperationType",
        "get_memory_manager",
    ]
except ImportError:
    # Fallback to old structure if exists
    try:
        from .core.jarvis2 import Jarvis2, Jarvis2Config
        from .core.solution import CodeSolution, SolutionMetrics

        # Provide compatibility layer
        Jarvis2Orchestrator = Jarvis2
        CodeRequest = lambda query, context=None: {"query": query, "context": context or {}}
        
        __all__ = ["Jarvis2", "Jarvis2Config", "CodeSolution", "SolutionMetrics", 
                   "Jarvis2Orchestrator", "CodeRequest"]
    except ImportError:
        raise ImportError("Jarvis2 core modules not found. Please check installation.")

__version__ = "2.0.0"