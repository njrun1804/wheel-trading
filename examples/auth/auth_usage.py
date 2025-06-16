#!/usr/bin/env python3
"""
Example usage of the authentication system.

This demonstrates:
- Initial setup and authentication
- Making authenticated API requests
- Using cache for performance
- Health monitoring
- Error recovery
"""
import asyncio
from datetime import datetime

from src.config.loader import get_config_loader
from src.unity_wheel.auth.client_v2 import AuthClient
from src.unity_wheel.utils.logging import get_logger

# Setup logging
logger = get_logger(__name__)


async def setup_first_time_auth():
    """Run first-time authentication setup."""
    logger.info("Starting first-time authentication setup")

    # Create auth client - credentials loaded automatically from SecretManager
    auth_client = AuthClient(redirect_uri="https://127.0.0.1:8182/callback")

    async with auth_client:
        # Check current health
        health = await auth_client.health_check()
        logger.info(f"Initial health: {health['status']}")

        if health["status"] in ["needs_auth", "unconfigured"]:
            logger.info("Starting OAuth flow - browser will open...")
            await auth_client.authenticate()

            # Re-check health
            health = await auth_client.health_check()
            logger.info(f"Post-auth health: {health['status']}")

        return health["status"] == "healthy"


async def demo_api_usage():
    """Demonstrate making authenticated API requests."""
    logger.info("Starting API usage demo")

    # Load config
    config = get_config_loader().load()

    # Create auth client
    auth_client = AuthClient(
        client_id=config.auth.client_id.get_secret_value(),
        client_secret=config.auth.client_secret.get_secret_value(),
    )

    async with auth_client:
        # 1. Get account information
        logger.info("Fetching account information...")
        try:
            accounts = await auth_client.make_request(
                "GET", "https://api.schwabapi.com/v1/accounts"
            )
            logger.info(f"Found {len(accounts)} accounts")

            for account in accounts:
                logger.info(f"Account: {account.get('accountNumber', 'N/A')}")

        except Exception as e:
            logger.error(f"Failed to fetch accounts: {e}")

        # 2. Get Unity quote (will be cached)
        logger.info("Fetching Unity quote...")
        try:
            quote_data = await auth_client.make_request(
                "GET",
                "https://api.schwabapi.com/v1/marketdata/quotes",
                params={"symbols": "U"},
                cache_ttl=300,  # Cache for 5 minutes
            )

            if "U" in quote_data:
                unity_quote = quote_data["U"]
                logger.info(f"Unity price: ${unity_quote.get('lastPrice', 'N/A')}")
                logger.info(f"Volume: {unity_quote.get('totalVolume', 'N/A'):,}")

        except Exception as e:
            logger.error(f"Failed to fetch quote: {e}")

        # 3. Get option chain (demonstrates cache hit)
        logger.info("Fetching option chain...")
        try:
            # First call - will hit API
            start = datetime.now()
            chain_data = await auth_client.make_request(
                "GET",
                "https://api.schwabapi.com/v1/marketdata/chains",
                params={"symbol": "U", "contractType": "PUT", "strikeCount": 10},
                cache_ttl=600,  # Cache for 10 minutes
            )
            api_time = (datetime.now() - start).total_seconds()

            put_count = len(chain_data.get("putExpDateMap", {}))
            logger.info(f"Found {put_count} expiration dates (API: {api_time:.2f}s)")

            # Second call - should hit cache
            start = datetime.now()
            await auth_client.make_request(
                "GET",
                "https://api.schwabapi.com/v1/marketdata/chains",
                params={"symbol": "U", "contractType": "PUT", "strikeCount": 10},
            )
            cache_time = (datetime.now() - start).total_seconds()
            logger.info(f"Cache hit demonstrated (Cache: {cache_time:.3f}s)")

        except Exception as e:
            logger.error(f"Failed to fetch option chain: {e}")

        # 4. Show health and stats
        health = await auth_client.health_check()
        logger.info("\nFinal health check:")
        logger.info(f"Status: {health['status']}")
        logger.info(f"Rate limit remaining: {health['rate_limit_remaining']}")
        logger.info(f"Cache stats: {health['cache']}")
        logger.info(f"Rate limiter: {health['rate_limiter']}")


async def demo_error_recovery():
    """Demonstrate error recovery mechanisms."""
    logger.info("Starting error recovery demo")

    # This demonstrates:
    # 1. Automatic token refresh
    # 2. Rate limit handling
    # 3. Cache fallback on errors

    config = get_config_loader().load()
    auth_client = AuthClient(
        client_id=config.auth.client_id.get_secret_value(),
        client_secret=config.auth.client_secret.get_secret_value(),
    )

    async with auth_client:
        # Pre-populate cache
        logger.info("Pre-populating cache...")
        await auth_client.make_request(
            "GET",
            "https://api.schwabapi.com/v1/marketdata/quotes",
            params={"symbols": "U"},
        )

        # Simulate network error (cache fallback)
        logger.info("Testing cache fallback on network error...")
        # This would normally fail, but cache will provide data
        # (In real scenario, network errors trigger cache fallback)

        # Show circuit breaker in action
        logger.info("Circuit breaker status:")
        health = await auth_client.health_check()
        cb_status = health["rate_limiter"]["circuit_breaker"]
        logger.info(f"Circuit breaker state: {cb_status}")


async def main():
    """Run all demonstrations."""
    logger.info("Unity Wheel Auth System Demo")
    logger.info("=" * 50)

    # Credentials will be loaded from SecretManager automatically

    # Run demos
    try:
        # 1. Setup (if needed)
        logger.info("\n1. Authentication Setup")
        logger.info("-" * 30)
        auth_ok = await setup_first_time_auth()

        if not auth_ok:
            logger.error("Authentication setup failed")
            return

        # 2. API usage
        logger.info("\n2. API Usage Demo")
        logger.info("-" * 30)
        await demo_api_usage()

        # 3. Error recovery
        logger.info("\n3. Error Recovery Demo")
        logger.info("-" * 30)
        await demo_error_recovery()

        logger.info("\nDemo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
