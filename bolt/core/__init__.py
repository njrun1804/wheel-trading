"""
Bolt core module - Core system functionality and integration.

Provides the main integration layer and system coordination:
- BoltIntegration: Main orchestration class
- System information and status
- Configuration management
- Task coordination
"""

from .config import BoltConfig
from .integration import BoltIntegration
from .system_info import get_system_status

__all__ = ["BoltIntegration", "get_system_status", "BoltConfig"]
