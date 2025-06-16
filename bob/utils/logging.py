"""Logging utilities for Bob."""

import logging
import sys
from typing import Optional


def get_component_logger(component_name: str) -> logging.Logger:
    """Get a logger for a specific component."""
    logger = logging.getLogger(f"bob.{component_name}")
    
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def setup_logging(level: str = "INFO") -> None:
    """Setup logging for Bob."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )