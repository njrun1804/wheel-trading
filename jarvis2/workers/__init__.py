"""Jarvis2 worker processes for parallel execution.

IMPORTANT: Set multiprocessing start method before importing workers
to avoid PyTorch MPS deadlocks on macOS.
"""
import multiprocessing as mp
import platform

# Force spawn method on macOS to avoid PyTorch MPS fork issues
if platform.system() == "Darwin":
    mp.set_start_method("spawn", force=True)

from .learning_worker import LearningWorker
from .neural_worker import NeuralWorkerPool, NeuralWorkerProcess
from .search_worker import SearchWorkerPool, SearchWorkerProcess

__all__ = [
    "NeuralWorkerPool",
    "NeuralWorkerProcess",
    "SearchWorkerPool",
    "SearchWorkerProcess",
    "LearningWorker",
]
