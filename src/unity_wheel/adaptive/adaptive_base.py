"""Adaptive base module."""

from ..utils import get_logger

logger = get_logger(__name__)


class AdaptiveBase:
    """Base class for adaptive components."""

    def __init__(self):
        """Initialize adaptive base."""
        self.logger = logger

    def adapt(self, *args, **kwargs):
        """Adapt method to be overridden."""
        raise NotImplementedError
