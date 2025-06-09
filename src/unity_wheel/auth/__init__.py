"""
Unity Wheel Authentication Module.

Provides autonomous authentication with OAuth2, automatic token refresh,
and zero manual intervention after initial setup.
"""
from .client import AuthClient
from .exceptions import AuthError, TokenExpiredError, RateLimitError
from .storage import SecureTokenStorage

__all__ = [
    "AuthClient",
    "AuthError", 
    "TokenExpiredError",
    "RateLimitError",
    "SecureTokenStorage",
]