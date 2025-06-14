"""Pytest configuration for Jarvis2 tests.

Sets up proper multiprocessing start method to avoid PyTorch MPS deadlocks.
"""
import multiprocessing as mp
import platform
import os

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Force spawn method on macOS BEFORE any test imports
# This prevents PyTorch MPS fork deadlocks
if platform.system() == 'Darwin':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, that's fine
        pass

# Configure pytest
def pytest_configure(config):
    """Configure pytest settings."""
    # Set asyncio mode
    config.option.asyncio_mode = "auto"
    
    # Disable xdist for more predictable testing
    if hasattr(config.option, 'numprocesses'):
        config.option.numprocesses = 0

import pytest
import asyncio

@pytest.fixture(autouse=True)
async def cleanup_tasks():
    """Ensure all tasks are cleaned up after each test."""
    yield
    
    # Get all pending tasks
    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    current = asyncio.current_task()
    
    # Cancel all tasks except the current one
    for task in tasks:
        if task != current:
            task.cancel()
    
    # Wait for all tasks to complete
    if tasks:
        await asyncio.gather(*[t for t in tasks if t != current], return_exceptions=True)