"""Lazy-loading neural networks to avoid initialization deadlocks."""
from __future__ import annotations

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class LazyValueNetwork:
    """Lazy-loading wrapper for CodeValueNetwork."""
    
    def __init__(self, **kwargs):
        self._network: Optional[Any] = None
        self._kwargs = kwargs
        self._initialized = False
    
    def _ensure_initialized(self):
        """Initialize network on first use."""
        if not self._initialized:
            logger.info("Lazy-loading value network...")
            from .value_network import CodeValueNetwork
            self._network = CodeValueNetwork(**self._kwargs)
            self._initialized = True
    
    def __call__(self, *args, **kwargs):
        """Forward call to network."""
        self._ensure_initialized()
        return self._network(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to network."""
        if name.startswith('_'):
            raise AttributeError(name)
        self._ensure_initialized()
        return getattr(self._network, name)


class LazyPolicyNetwork:
    """Lazy-loading wrapper for CodePolicyNetwork."""
    
    def __init__(self, **kwargs):
        self._network: Optional[Any] = None
        self._kwargs = kwargs
        self._initialized = False
    
    def _ensure_initialized(self):
        """Initialize network on first use."""
        if not self._initialized:
            logger.info("Lazy-loading policy network...")
            from .policy_network import CodePolicyNetwork
            self._network = CodePolicyNetwork(**self._kwargs)
            self._initialized = True
    
    def __call__(self, *args, **kwargs):
        """Forward call to network."""
        self._ensure_initialized()
        return self._network(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to network."""
        if name.startswith('_'):
            raise AttributeError(name)
        self._ensure_initialized()
        return getattr(self._network, name)