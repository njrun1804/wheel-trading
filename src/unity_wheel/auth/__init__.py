"""
Unity Wheel Authentication Module.

Provides autonomous authentication with OAuth2, automatic token refresh,
and zero manual intervention after initial setup.
"""

from ..storage.auth_storage import SecureTokenStorage
from .auth_client import AuthClient
from .exceptions import AuthError, RateLimitError, TokenExpiredError

__all__ = [
    "AuthClient",
    "AuthError",
    "TokenExpiredError",
    "RateLimitError",
    "SecureTokenStorage",
]
