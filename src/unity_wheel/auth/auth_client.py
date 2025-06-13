"""
from __future__ import annotations

Main authentication client with automatic token management.
"""

import asyncio
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

import aiohttp

from ..storage.auth_storage import SecureTokenStorage

# Cache functionality handled by AuthCache, not general cache
from ..storage.cache.auth_cache import AuthCache
from ..utils.logging import get_logger
from .exceptions import (
    AuthError,
    InvalidCredentialsError,
    NetworkError,
    RateLimitError,
    TokenExpiredError,
)
from .oauth import OAuth2Handler
from .rate_limiter import RateLimiter

logger = get_logger(__name__)


def auth_retry(max_attempts: int = 3, backoff_factor: float = 2.0) -> None:
    """Decorator for automatic retry with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(self, *args, **kwargs)

                except TokenExpiredError:
                    # Refresh token and retry immediately
                    logger.info(f"{func.__name__}", action="refreshing_token")
                    await self.refresh_token()
                    continue

                except RateLimitError as e:
                    # Wait for specified time or use exponential backoff
                    wait_time = e.retry_after or (backoff_factor**attempt)
                    logger.warning(
                        f"{func.__name__}",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                        error="rate_limit",
                    )
                    await asyncio.sleep(wait_time)
                    last_exception = e

                except NetworkError as e:
                    # Network errors get exponential backoff
                    wait_time = backoff_factor**attempt
                    logger.warning(
                        f"{func.__name__}",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                        error="network",
                    )
                    await asyncio.sleep(wait_time)
                    last_exception = e

                except InvalidCredentialsError:
                    # Can't recover from invalid credentials
                    raise

                except (ValueError, KeyError, AttributeError) as e:
                    logger.error(f"{func.__name__}", attempt=attempt + 1, error=str(e))
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(backoff_factor**attempt)

            # All retries exhausted
            raise last_exception or AuthError(f"Failed after {max_attempts} attempts")

        return wrapper

    return decorator


class AuthClient:
    """Main authentication client with autonomous operation."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "https://127.0.0.1:8182/callback",
        storage_path: Optional[str] = None,
        auto_refresh: bool = True,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        rate_limit_rps: float = 10,
        enable_circuit_breaker: bool = True,
    ):
        """Initialize authentication client.

        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret
            redirect_uri: OAuth callback URL
            storage_path: Path for token storage
            auto_refresh: Enable automatic token refresh
            enable_cache: Enable response caching
            cache_ttl: Cache TTL in seconds
            rate_limit_rps: Requests per second limit
            enable_circuit_breaker: Enable circuit breaker pattern
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auto_refresh = auto_refresh

        self.storage = SecureTokenStorage(storage_path)
        self.oauth_handler = OAuth2Handler(
            client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri
        )

        self._access_token: Optional[str] = None
        self._token_lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 100
        self._rate_limit_reset = 0

        # Initialize cache and rate limiter
        self.cache = AuthCache(default_ttl=cache_ttl) if enable_cache else None
        self.rate_limiter = RateLimiter(
            requests_per_second=rate_limit_rps, enable_circuit_breaker=enable_circuit_breaker
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self) -> None:
        """Initialize client and validate stored credentials."""
        logger.info("initialize", action="starting")

        # Create session
        self._session = aiohttp.ClientSession()

        # Try to load existing tokens
        token_data = self.storage.load_tokens()

        if token_data:
            self._access_token = token_data["access_token"]

            # Check if token needs refresh
            if self.storage.is_token_expired():
                logger.info("initialize", action="token_expired_refreshing")
                try:
                    await self.refresh_token()
                except InvalidCredentialsError:
                    logger.warning("initialize", action="refresh_failed_need_reauth")
                    await self.authenticate()
            else:
                # Validate token with a test request
                try:
                    await self.validate_token()
                    logger.info("initialize", status="ready", has_valid_token=True)
                except (TokenExpiredError, InvalidCredentialsError):
                    logger.warning("initialize", action="validation_failed_refreshing")
                    await self.refresh_token()
        else:
            logger.info("initialize", action="no_stored_tokens")
            await self.authenticate()

    async def close(self) -> None:
        """Close client and cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None

    async def authenticate(self) -> None:
        """Perform full OAuth authentication flow."""
        logger.info("authenticate", action="starting_oauth_flow")

        try:
            # Start OAuth flow
            auth_code = await self.oauth_handler.authorize()

            # Exchange code for tokens
            token_response = await self.oauth_handler.exchange_code_for_tokens(auth_code)

            # Store tokens
            self.storage.save_tokens(
                access_token=token_response["access_token"],
                refresh_token=token_response["refresh_token"],
                expires_in=token_response["expires_in"],
                scope=token_response.get("scope", ""),
                token_type=token_response.get("token_type", "Bearer"),
            )

            self._access_token = token_response["access_token"]

            logger.info("authenticate", status="success", expires_in=token_response["expires_in"])

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("authenticate", error=str(e))
            raise InvalidCredentialsError(
                f"Authentication failed: {e}. Please check your credentials and try again."
            )

    async def refresh_token(self) -> None:
        """Refresh access token using stored refresh token."""
        async with self._token_lock:
            logger.info("refresh_token", action="acquiring_lock")

            # Double-check if refresh is still needed
            if not self.storage.is_token_expired():
                logger.info("refresh_token", action="already_refreshed_by_another_task")
                return

            token_data = self.storage.load_tokens()
            if not token_data or not token_data.get("refresh_token"):
                raise InvalidCredentialsError("No refresh token available")

            try:
                # Refresh the token
                token_response = await self.oauth_handler.refresh_access_token(
                    token_data["refresh_token"]
                )

                # Store new tokens
                self.storage.save_tokens(
                    access_token=token_response["access_token"],
                    refresh_token=token_response.get(
                        "refresh_token",
                        token_data["refresh_token"],  # Some providers reuse refresh token
                    ),
                    expires_in=token_response["expires_in"],
                    scope=token_response.get("scope", token_data.get("scope", "")),
                )

                self._access_token = token_response["access_token"]

                logger.info(
                    "refresh_token", status="success", expires_in=token_response["expires_in"]
                )

            except (ValueError, KeyError, AttributeError) as e:
                logger.error("refresh_token", error=str(e))
                raise InvalidCredentialsError(f"Token refresh failed: {e}. Please re-authenticate.")

    async def validate_token(self) -> bool:
        """Validate current token with a test API call."""
        try:
            # Make a lightweight API call to validate token
            headers = self._get_auth_headers()
            async with self._session.get(
                "https://api.schwabapi.com/v1/accounts",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 401:
                    raise TokenExpiredError()
                elif response.status == 200:
                    return True
                else:
                    logger.warning(
                        "validate_token", status_code=response.status, reason=response.reason
                    )
                    return False

        except aiohttp.ClientError as e:
            logger.error("validate_token", error=str(e))
            raise NetworkError(f"Failed to validate token: {e}")

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests."""
        if not self._access_token:
            raise InvalidCredentialsError("No access token available")

        return {"Authorization": f"Bearer {self._access_token}", "Accept": "application/json"}

    def _update_rate_limits(self, headers: Dict[str, str]) -> None:
        """Update rate limit tracking from response headers."""
        if "X-RateLimit-Remaining" in headers:
            self._rate_limit_remaining = int(headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in headers:
            self._rate_limit_reset = int(headers["X-RateLimit-Reset"])

    @auth_retry(max_attempts=3)
    async def make_request(
        self,
        method: str,
        url: str,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make authenticated API request with automatic retry and token refresh.

        Args:
            method: HTTP method
            url: API endpoint URL
            use_cache: Whether to use cache for GET requests
            cache_ttl: Override cache TTL for this request
            **kwargs: Additional request parameters

        Returns:
            Response JSON data

        Raises:
            Various auth exceptions based on response
        """
        # Try cache first for GET requests
        if method.upper() == "GET" and use_cache and self.cache:
            cache_params = kwargs.get("params", {})
            cached_data = self.cache.get(url, cache_params)
            if cached_data is not None:
                logger.debug("make_request", source="cache", url=url)
                return cached_data

        # Apply rate limiting
        await self.rate_limiter.acquire(endpoint=url)

        # Ensure we have valid token
        if self.auto_refresh and self.storage.is_token_expired():
            await self.refresh_token()

        # Add auth headers
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())

        try:
            async with self._session.request(method, url, headers=headers, **kwargs) as response:
                # Update rate limits
                self._update_rate_limits(dict(response.headers))

                # Handle response
                if response.status == 401:
                    self.rate_limiter.report_failure()
                    raise TokenExpiredError()
                elif response.status == 429:
                    self.rate_limiter.report_failure(is_rate_limit=True)
                    retry_after = response.headers.get("Retry-After")
                    raise RateLimitError(retry_after=int(retry_after) if retry_after else None)
                elif response.status >= 500:
                    self.rate_limiter.report_failure()
                    raise NetworkError(f"Server error: {response.status}")
                elif response.status >= 400:
                    self.rate_limiter.report_failure()
                    error_data = await response.json()
                    raise AuthError(f"API error: {error_data}")

                # Success
                response_data = await response.json()
                self.rate_limiter.report_success()

                # Cache successful GET responses
                if method.upper() == "GET" and use_cache and self.cache:
                    cache_params = kwargs.get("params", {})
                    self.cache.set(url, response_data, cache_params, ttl=cache_ttl)

                return response_data

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.rate_limiter.report_failure()

            # Try cache fallback for GET requests
            if method.upper() == "GET" and self.cache:
                cache_params = kwargs.get("params", {})
                fallback_data = self.cache.get_fallback(url, cache_params)
                if fallback_data is not None:
                    logger.warning(
                        "make_request", action="using_stale_cache", url=url, error=str(e)
                    )
                    return fallback_data

            raise NetworkError(f"Request failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check.

        Returns:
            Health status dict
        """
        health = {
            "status": "unknown",
            "has_credentials": bool(self.client_id and self.client_secret),
            "has_stored_token": False,
            "token_valid": False,
            "token_expires_at": None,
            "rate_limit_remaining": self._rate_limit_remaining,
            "rate_limiter": self.rate_limiter.get_status(),
            "cache": self.cache.get_stats() if self.cache else None,
            "errors": [],
        }

        try:
            # Check stored tokens
            token_data = self.storage.load_tokens()
            if token_data:
                health["has_stored_token"] = True
                health["token_expires_at"] = token_data.get("expires_at")

                # Validate token
                try:
                    is_valid = await self.validate_token()
                    health["token_valid"] = is_valid
                except (ValueError, KeyError, AttributeError) as e:
                    health["errors"].append(f"Token validation failed: {e}")

            # Overall status
            if health["token_valid"]:
                health["status"] = "healthy"
            elif health["has_stored_token"]:
                health["status"] = "needs_refresh"
            elif health["has_credentials"]:
                health["status"] = "needs_auth"
            else:
                health["status"] = "unconfigured"

        except (ValueError, KeyError, AttributeError) as e:
            health["status"] = "error"
            health["errors"].append(str(e))

        logger.info("health_check", **health)
        return health