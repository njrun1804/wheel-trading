"""
from __future__ import annotations

Enhanced authentication client that integrates with SecretManager.

This module provides a drop-in replacement for the existing AuthClient
that automatically retrieves credentials from SecretManager.
"""

from typing import Any, Dict, Optional

from ..secrets.integration import SecretInjector
from ..secrets import SecretManager
from ..utils.logging import get_logger
from .auth_client import AuthClient as BaseAuthClient

logger = get_logger(__name__)


class AuthClient(BaseAuthClient):
    """Enhanced AuthClient that integrates with SecretManager."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: str = "https://127.0.0.1:8182/callback",
        storage_path: Optional[str] = None,
        auto_refresh: bool = True,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        rate_limit_rps: float = 10,
        enable_circuit_breaker: bool = True,
        use_secret_manager: bool = True,
    ):
        """Initialize authentication client.

        Args:
            client_id: OAuth client ID (optional if using SecretManager)
            client_secret: OAuth client secret (optional if using SecretManager)
            redirect_uri: OAuth callback URL
            storage_path: Path for token storage
            auto_refresh: Enable automatic token refresh
            enable_cache: Enable response caching
            cache_ttl: Cache TTL in seconds
            rate_limit_rps: Requests per second limit
            enable_circuit_breaker: Enable circuit breaker pattern
            use_secret_manager: Whether to retrieve credentials from SecretManager
        """
        # Get credentials from SecretManager if not provided
        if use_secret_manager and (not client_id or not client_secret):
            logger.info("Retrieving credentials from SecretManager")
            try:
                manager = SecretManager()
                creds = manager.get_credentials("schwab")
                client_id = client_id or creds.get("client_id")
                client_secret = client_secret or creds.get("client_secret")
            except (ValueError, KeyError, AttributeError) as e:
                logger.error(f"Failed to retrieve credentials from SecretManager: {e}")
                if not client_id or not client_secret:
                    raise

        # Initialize base class
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            storage_path=storage_path,
            auto_refresh=auto_refresh,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl,
            rate_limit_rps=rate_limit_rps,
            enable_circuit_breaker=enable_circuit_breaker,
        )

    @classmethod
    async def create_with_env_injection(cls, **kwargs) -> "AuthClient":
        """Create AuthClient with temporary environment variable injection.

        This factory method creates an AuthClient within a context that
        temporarily injects Schwab credentials into environment variables.
        Useful for compatibility with code that expects env vars.

        Args:
            **kwargs: Arguments passed to AuthClient constructor

        Returns:
            Initialized AuthClient instance
        """
        with SecretInjector(service="schwab"):
            # Within this context, WHEEL_AUTH__CLIENT_ID and
            # WHEEL_AUTH__CLIENT_SECRET are available as env vars
            client = cls(**kwargs)
            await client.initialize()
            return client

    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check that includes SecretManager status."""
        health = await super().health_check()

        # Add SecretManager status
        try:
            from ..secrets import SecretManager

            manager = SecretManager()
            configured_services = manager.list_configured_services()

            health["secret_manager"] = {
                "provider": manager.provider.value,
                "schwab_configured": configured_services.get("schwab", False),
                "databento_configured": configured_services.get("databento", False),
                "ofred_configured": configured_services.get("ofred", False),
            }
        except (ValueError, KeyError, AttributeError) as e:
            health["secret_manager"] = {"error": str(e)}

        return health