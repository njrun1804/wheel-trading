"""Jarvis 2.0 - Intelligent Meta-Coder for M4 Pro.

A learning, exploring, evolving AI system that treats code generation 
as an intelligent search problem. Implements MCTS, neural guidance,
diversity generation, and continuous learning.

Usage:
    from jarvis2 import Jarvis2
    
    jarvis = Jarvis2()
    solution = await jarvis.assist("optimize options pricing calculations")
"""

from .core.jarvis2 import Jarvis2, Jarvis2Config
from .core.solution import CodeSolution, SolutionMetrics

__version__ = "2.0.0"
__all__ = ["Jarvis2", "Jarvis2Config", "CodeSolution", "SolutionMetrics"]