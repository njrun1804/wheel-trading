"""Real-time Databento integration for Unity options data."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime

from src.config import get_settings
from unity_wheel.api.types import MarketSnapshot, OptionData
from unity_wheel.data_providers.databento.client import DatabentoClient
from unity_wheel.data_providers.databento.integration import DatabentoIntegration
from unity_wheel.data_providers.databento.live_client import UnityLiveClient
from unity_wheel.secrets.integration import SecretInjector

logger = logging.getLogger(__name__)


# Removed create_synthetic_unity_options function
# Per user requirement: fail if real data not available, no fallback to synthetic data


async def create_databento_market_snapshot(
    portfolio_value: float,
    ticker: str = "U",
    *,
    risk_free_rate: float = 0.05,
) -> tuple[MarketSnapshot, float]:
    """Create market snapshot from real Databento data.

    Args:
        portfolio_value: Total portfolio value
        ticker: Stock ticker (default: "U" for Unity)
        risk_free_rate: Risk-free rate to use for IV calculations

    Returns:
        MarketSnapshot with real Unity options data

    Raises:
        ValueError: If unable to fetch real Unity options data
    """
    # Load API key via SecretInjector
    with SecretInjector(service="databento"):
        client = DatabentoClient()
        integration = DatabentoIntegration(client)

        try:
            # Get Unity spot price - fail if not available
            try:
                spot_data = await client._get_underlying_price(ticker)
                current_price = float(spot_data.last_price)
                logger.info(f"Unity spot price from Databento: ${current_price:.2f}")
            except (ValueError, KeyError, AttributeError) as e:
                error_msg = f"Failed to get Unity spot price: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Get wheel candidates with adjusted parameters for Unity
            config = get_settings()

            # Adjust DTE range for Unity's monthly options
            # Unity only has monthly options, so we need a wider range
            min_dte = 20  # ~3 weeks
            max_dte = 60  # ~2 months

            try:
                candidates = await integration.get_wheel_candidates(
                    underlying=ticker,
                    target_delta=config.wheel_delta_target,
                    dte_range=(min_dte, max_dte),
                    min_premium_pct=0.5,  # Lower threshold for Unity
                )

                if not candidates:
                    # No Unity options found - fail as requested
                    error_msg = f"No Unity options found in Databento for DTE range {min_dte}-{max_dte} days. Cannot proceed without real market data."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                logger.info(f"Found {len(candidates)} Unity option candidates")

            except (ValueError, KeyError, AttributeError) as e:
                error_msg = f"Failed to get Unity options: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Build option chain format expected by advisor
            option_chain = {}
            for candidate in candidates:
                strike_key = str(candidate["strike"])

                # Ensure timezone-aware datetime
                if hasattr(candidate["expiration"], "replace"):
                    expiration = candidate["expiration"]
                    if expiration.tzinfo is None:
                        expiration = expiration.replace(tzinfo=UTC)
                else:
                    expiration = datetime.fromisoformat(str(candidate["expiration"]))
                    if expiration.tzinfo is None:
                        expiration = expiration.replace(tzinfo=UTC)

                option_chain[strike_key] = OptionData(
                    strike=candidate["strike"],
                    expiration=expiration.isoformat(),
                    bid=candidate["bid"],
                    ask=candidate["ask"],
                    mid=candidate["mid"],
                    volume=candidate.get("volume", 0),
                    open_interest=candidate.get("open_interest", 0),
                    delta=candidate["delta"],
                    gamma=candidate["gamma"],
                    theta=candidate["theta"],
                    vega=candidate["vega"],
                    implied_volatility=candidate["iv"],
                )

            # Create market snapshot with timezone-aware timestamp
            market_snapshot = MarketSnapshot(
                timestamp=datetime.now(UTC),
                ticker=ticker,
                current_price=current_price,
                buying_power=portfolio_value,
                margin_used=0.0,
                positions=[],
                option_chain=option_chain,
                implied_volatility=0.45,  # Unity typical IV
                risk_free_rate=risk_free_rate,
            )
            # Confidence based on number of option candidates
            confidence = 1.0 if len(option_chain) > 10 else 0.8

            return market_snapshot, confidence

        finally:
            await client.close()


def get_market_data_sync(
    portfolio_value: float,
    ticker: str = "U",
    *,
    risk_free_rate: float = 0.05,
) -> tuple[MarketSnapshot, float]:
    """Synchronous wrapper for getting market data.

    Args:
        portfolio_value: Total portfolio value
        ticker: Stock ticker (default: "U" for Unity)

    Returns:
        MarketSnapshot with real Unity options data

    Raises:
        ValueError: If unable to fetch real Unity options data
    """
    # Create new event loop for synchronous execution
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            create_databento_market_snapshot(
                portfolio_value,
                ticker,
                risk_free_rate=risk_free_rate,
            )
        )
    finally:
        loop.close()


async def create_live_market_snapshot(
    portfolio_value: float,
    ticker: str = "U",
    *,
    risk_free_rate: float = 0.05,
) -> tuple[MarketSnapshot, float]:
    """Create market snapshot using Databento Live API for real-time data.

    This is used for ad hoc requests where current data is needed.

    Args:
        portfolio_value: Total portfolio value
        ticker: Stock ticker (default: "U" for Unity)
        risk_free_rate: Risk-free rate to use for IV calculations

    Returns:
        MarketSnapshot with live Unity data and confidence score

    Raises:
        ValueError: If unable to fetch live data
    """
    logger.info("Using Databento Live API for real-time data...")

    # Initialize live client
    live_client = UnityLiveClient()

    try:
        # Get live snapshot
        live_data = await live_client.get_live_snapshot()

        if not live_data["stock_price"]:
            raise ValueError("Could not get live Unity stock price")

        current_price = live_data["stock_price"]
        logger.info(f"Live Unity price: ${current_price:.2f}")

        # Build option chain from live quotes
        option_chain = {}
        for key, quote in live_data["option_quotes"].items():
            # Parse key format: strike_expiry_type
            parts = key.split("_")
            if len(parts) != 3:
                continue

            strike = float(parts[0])
            expiry_str = parts[1]
            opt_type = parts[2]

            # Only include puts for wheel strategy
            if opt_type != "P":
                continue

            # Convert expiry string to date
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()

            option_chain[str(strike)] = OptionData(
                strike=strike,
                expiration=expiry_date.isoformat(),
                bid=quote.bid_price,
                ask=quote.ask_price,
                mid=quote.mid_price,
                volume=0,  # Not available in live quotes
                open_interest=0,  # Not available in live quotes
                delta=0.0,  # Will be calculated by advisor
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                implied_volatility=0.0,  # Will be calculated
            )

        if not option_chain:
            raise ValueError("No live Unity options found")

        logger.info(f"Found {len(option_chain)} live option quotes")

        # Create market snapshot
        snapshot = MarketSnapshot(
            timestamp=datetime.now(UTC),
            ticker=ticker,
            current_price=current_price,
            buying_power=portfolio_value * 0.5,  # Conservative estimate
            margin_used=0.0,
            positions=[],  # Would need broker integration
            option_chain=option_chain,
            implied_volatility=0.65,  # Unity typical
            risk_free_rate=risk_free_rate,
        )

        # High confidence for live data
        confidence = 0.95

        return snapshot, confidence

    except Exception as e:
        logger.error(f"Live data fetch failed: {e}")
        # Fall back to historical data
        logger.info("Falling back to historical data...")
        return await create_databento_market_snapshot(
            portfolio_value, ticker, risk_free_rate=risk_free_rate
        )


def get_market_data_sync(
    portfolio_value: float,
    ticker: str = "U",
    *,
    risk_free_rate: float = 0.05,
    use_live: bool = None,
) -> tuple[MarketSnapshot, float]:
    """Get market data synchronously, using live data for ad hoc requests.

    Args:
        portfolio_value: Total portfolio value
        ticker: Stock ticker
        risk_free_rate: Risk-free rate
        use_live: Force live data (None = auto-detect)

    Returns:
        MarketSnapshot and confidence score
    """
    # Auto-detect: use live data if running interactively (ad hoc)
    if use_live is None:
        # Check if running from cron/automated
        use_live = not os.environ.get("AUTOMATED_RUN", False)

    if use_live:
        logger.info("Ad hoc request detected - using live data")
        # Create new event loop for synchronous execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                create_live_market_snapshot(
                    portfolio_value,
                    ticker,
                    risk_free_rate=risk_free_rate,
                )
            )
        finally:
            loop.close()
    else:
        logger.info("Automated run - using historical data")
        return get_market_data_sync_historical(
            portfolio_value, ticker, risk_free_rate=risk_free_rate
        )
