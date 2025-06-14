"""Pytest configuration for Jarvis2 tests.

Sets up proper multiprocessing start method to avoid PyTorch MPS deadlocks.
"""
import multiprocessing as mp
import os
import platform

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
if platform.system() == 'Darwin':
    try:
        mp.set_start_method('spawn', force = True)
    except RuntimeError as e:
        logger.debug(f"Ignored exception in {'conftest.py'}: {e}")


def pytest_configure(config):
    """Configure pytest settings."""
    config.option.asyncio_mode = 'auto'
    if hasattr(config.option, 'numprocesses'):
        config.option.numprocesses = 0


import asyncio
import logging
import shutil
import tempfile

import pytest

logger = logging.getLogger(__name__)
None


@pytest.fixture(autouse = True)
async def cleanup_tasks():
    """Ensure all tasks are cleaned up after each test."""
    yield
    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    current = asyncio.current_task()
    for task in tasks:
        if task != current:
            task.cancel()
    if tasks:
        await asyncio.gather(*[t for t in tasks if t != current],
            return_exceptions = True)


@pytest.fixture
def temp_jarvis_config():
    """Create a temporary directory for Jarvis2 data that's unique per test."""
    temp_dir = tempfile.mkdtemp(prefix='jarvis2_test_')
    from jarvis2.core.orchestrator import Jarvis2Config
    config = Jarvis2Config()
    config.index_path = os.path.join(temp_dir, 'indexes')
    config.experience_path = os.path.join(temp_dir, 'experience.db')
    config.model_path = os.path.join(temp_dir, 'models')
    os.makedirs(config.index_path, exist_ok = True)
    os.makedirs(config.model_path, exist_ok = True)
    yield config
    shutil.rmtree(temp_dir, ignore_errors = True)
