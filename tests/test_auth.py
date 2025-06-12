"""
Comprehensive tests for authentication system.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import web
from unity_wheel.auth import AuthClient, AuthError, RateLimitError, TokenExpiredError
from unity_wheel.auth.rate_limiter import RateLimiter
from unity_wheel.storage.auth_storage import SecureTokenStorage
from unity_wheel.storage.cache.auth_cache import AuthCache


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary storage directory."""
    return tmp_path / "auth_test"


@pytest.fixture
def mock_token_response():
    """Mock successful token response."""
    return {
        "access_token": "test_access_token",
        "refresh_token": "test_refresh_token",
        "expires_in": 3600,
        "scope": "AccountsRead MarketDataRead",
        "token_type": "Bearer",
    }


@pytest.fixture
async def mock_auth_server():
    """Create mock OAuth server for testing."""
    app = web.Application()

    async def authorize(request):
        return web.Response(text="<html>Mock auth page</html>")

    async def token(request):
        data = await request.post()
        grant_type = data.get("grant_type")

        if grant_type == "authorization_code":
            return web.json_response(
                {
                    "access_token": "test_access_token",
                    "refresh_token": "test_refresh_token",
                    "expires_in": 3600,
                    "scope": "AccountsRead MarketDataRead",
                }
            )
        elif grant_type == "refresh_token":
            refresh_token = data.get("refresh_token")
            if refresh_token == "invalid_refresh":
                return web.json_response({"error": "invalid_grant"}, status=401)
            return web.json_response(
                {
                    "access_token": "new_access_token",
                    "refresh_token": "new_refresh_token",
                    "expires_in": 3600,
                }
            )

        return web.json_response({"error": "unsupported_grant_type"}, status=400)

    app.router.add_get("/authorize", authorize)
    app.router.add_post("/token", token)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8080)
    await site.start()

    yield "http://localhost:8080"

    await runner.cleanup()


class TestSecureTokenStorage:
    """Test token storage functionality."""

    def test_save_and_load_tokens(self, temp_storage_dir):
        """Test saving and loading tokens."""
        storage = SecureTokenStorage(temp_storage_dir)

        # Save tokens
        storage.save_tokens(
            access_token="test_access",
            refresh_token="test_refresh",
            expires_in=3600,
            scope="test_scope",
        )

        # Load tokens
        tokens = storage.load_tokens()
        assert tokens["access_token"] == "test_access"
        assert tokens["refresh_token"] == "test_refresh"
        assert tokens["scope"] == "test_scope"
        assert "expires_at" in tokens

    def test_token_expiry_check(self, temp_storage_dir):
        """Test token expiry checking."""
        storage = SecureTokenStorage(temp_storage_dir)

        # Save token that expires in 1 second
        storage.save_tokens(access_token="test_access", refresh_token="test_refresh", expires_in=1)

        # Should not be expired immediately
        assert not storage.is_token_expired(buffer_minutes=0)

        # Should be expired with buffer
        assert storage.is_token_expired(buffer_minutes=1)

    def test_clear_tokens(self, temp_storage_dir):
        """Test clearing tokens."""
        storage = SecureTokenStorage(temp_storage_dir)

        # Save and verify tokens exist
        storage.save_tokens("access", "refresh", 3600)
        assert storage.load_tokens() is not None

        # Clear and verify gone
        storage.clear_tokens()
        assert storage.load_tokens() is None

    def test_corrupted_token_file(self, temp_storage_dir):
        """Test handling corrupted token file."""
        storage = SecureTokenStorage(temp_storage_dir)

        # Create corrupted file
        token_file = storage.token_file
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("corrupted data")

        # Should return None and remove corrupted file
        assert storage.load_tokens() is None
        assert not token_file.exists()


class TestAuthCache:
    """Test response caching functionality."""

    def test_cache_get_set(self, temp_storage_dir):
        """Test basic cache operations."""
        cache = AuthCache(temp_storage_dir)

        # Cache miss
        assert cache.get("/api/test") is None

        # Set cache
        test_data = {"result": "test"}
        cache.set("/api/test", test_data, ttl=60)

        # Cache hit
        cached = cache.get("/api/test")
        assert cached == test_data

    def test_cache_expiry(self, temp_storage_dir):
        """Test cache expiration."""
        cache = AuthCache(temp_storage_dir)

        # Set with 0 TTL
        cache.set("/api/test", {"data": "test"}, ttl=0)

        # Should be expired
        assert cache.get("/api/test") is None

    def test_cache_fallback(self, temp_storage_dir):
        """Test fallback for stale cache."""
        cache = AuthCache(temp_storage_dir)

        # Set cache
        test_data = {"result": "test"}
        cache.set("/api/test", test_data, ttl=0)  # Already expired

        # Normal get returns None
        assert cache.get("/api/test") is None

        # Fallback returns stale data
        assert cache.get_fallback("/api/test") == test_data

    def test_cache_size_limit(self, temp_storage_dir):
        """Test cache size enforcement."""
        cache = AuthCache(temp_storage_dir, max_cache_size_mb=0.001)  # 1KB limit

        # Add large data
        large_data = {"data": "x" * 1000}
        cache.set("/api/1", large_data)
        cache.set("/api/2", large_data)

        # First should be evicted
        cache._enforce_size_limit()

        # Check stats
        stats = cache.get_stats()
        assert stats["disk_entries"] <= 1


class TestRateLimiter:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self):
        """Test basic rate limiting."""
        limiter = RateLimiter(requests_per_second=2, burst_capacity=2)

        # First two should be immediate
        await limiter.acquire()
        await limiter.acquire()

        # Third should wait
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed > 0.4  # Should wait ~0.5s

    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        limiter = RateLimiter(enable_circuit_breaker=True)
        limiter.circuit_breaker.failure_threshold = 2

        # Report failures to open circuit
        limiter.report_failure()
        limiter.report_failure()

        # Circuit should be open
        with pytest.raises(RateLimitError):
            await limiter.acquire()

    def test_endpoint_specific_limits(self):
        """Test endpoint-specific rate limits."""
        limiter = RateLimiter()

        # Add endpoint limit
        limiter.add_endpoint_limit("/api/heavy", 1, 1)

        # Check it was added
        assert "/api/heavy" in limiter.endpoint_buckets


class TestAuthClient:
    """Test main authentication client."""

    @pytest.mark.asyncio
    async def test_initialization_no_tokens(self, temp_storage_dir):
        """Test client initialization without stored tokens."""
        with patch("webbrowser.open", return_value=True):
            with patch.object(AuthClient, "authenticate", new_callable=AsyncMock):
                client = AuthClient(
                    client_id="test_id",
                    client_secret="test_secret",
                    storage_path=str(temp_storage_dir),
                )

                await client.initialize()
                client.authenticate.assert_called_once()

    @pytest.mark.asyncio
    async def test_token_refresh_on_expiry(self, temp_storage_dir, mock_token_response):
        """Test automatic token refresh on expiry."""
        # Setup storage with expired token
        storage = SecureTokenStorage(temp_storage_dir)
        storage.save_tokens(
            access_token="old_token",
            refresh_token="test_refresh",
            expires_in=1,  # Expires in 1 second
        )

        with patch.object(AuthClient, "refresh_token", new_callable=AsyncMock):
            client = AuthClient(
                client_id="test_id", client_secret="test_secret", storage_path=str(temp_storage_dir)
            )

            await client.initialize()
            client.refresh_token.assert_called()

    @pytest.mark.asyncio
    async def test_make_request_with_cache(self, temp_storage_dir):
        """Test making cached requests."""
        client = AuthClient(
            client_id="test_id",
            client_secret="test_secret",
            storage_path=str(temp_storage_dir),
            enable_cache=True,
        )

        # Mock session and token
        client._session = MagicMock()
        client._access_token = "test_token"

        # Pre-populate cache
        test_data = {"result": "cached"}
        client.cache.set("http://api.test/data", test_data)

        # Request should return cached data
        result = await client.make_request("GET", "http://api.test/data")
        assert result == test_data

        # Session should not be called
        client._session.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_make_request_retry_on_token_expired(self, temp_storage_dir):
        """Test retry on token expiry."""
        client = AuthClient(
            client_id="test_id", client_secret="test_secret", storage_path=str(temp_storage_dir)
        )

        # Mock response that returns 401 first, then success
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.json = AsyncMock(return_value={"data": "success"})

        client._session = AsyncMock()
        client._session.request = AsyncMock()

        # First call returns 401, second succeeds
        responses = [
            AsyncMock(status=401),
            AsyncMock(status=200, json=AsyncMock(return_value={"data": "success"})),
        ]
        client._session.request.return_value.__aenter__.side_effect = responses

        # Mock token refresh
        client.refresh_token = AsyncMock()
        client._access_token = "test_token"

        # Should retry and succeed
        result = await client.make_request("GET", "http://api.test/data")
        assert result == {"data": "success"}
        client.refresh_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, temp_storage_dir):
        """Test health check functionality."""
        client = AuthClient(
            client_id="test_id", client_secret="test_secret", storage_path=str(temp_storage_dir)
        )

        # Mock valid token
        client.storage.save_tokens(
            access_token="test_token", refresh_token="test_refresh", expires_in=3600
        )

        with patch.object(client, "validate_token", return_value=True):
            health = await client.health_check()

            assert health["status"] == "healthy"
            assert health["has_credentials"]
            assert health["has_stored_token"]
            assert health["token_valid"]
            assert "rate_limiter" in health
            assert "cache" in health

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, temp_storage_dir):
        """Test fallback to cache on network failure."""
        client = AuthClient(
            client_id="test_id",
            client_secret="test_secret",
            storage_path=str(temp_storage_dir),
            enable_cache=True,
        )

        # Pre-populate cache with stale data
        stale_data = {"result": "stale_but_available"}
        client.cache.set("http://api.test/data", stale_data, ttl=0)

        # Mock network failure
        client._session = AsyncMock()
        client._session.request.side_effect = aiohttp.ClientError("Network error")
        client._access_token = "test_token"

        # Should return stale cache
        result = await client.make_request("GET", "http://api.test/data")
        assert result == stale_data

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, temp_storage_dir):
        """Test rate limit error handling."""
        client = AuthClient(
            client_id="test_id", client_secret="test_secret", storage_path=str(temp_storage_dir)
        )

        # Mock 429 response
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "5"}

        client._session = AsyncMock()
        client._session.request.return_value.__aenter__.return_value = mock_response
        client._access_token = "test_token"

        # Should raise RateLimitError with retry_after
        with pytest.raises(RateLimitError) as exc_info:
            await client.make_request("GET", "http://api.test/data")

        assert exc_info.value.retry_after == 5
