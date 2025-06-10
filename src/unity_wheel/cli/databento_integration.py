"""Real-time Databento integration for Unity options data."""

import asyncio
import logging
from datetime import datetime, timezone

from src.config import get_settings
from src.unity_wheel.api.types import MarketSnapshot, OptionData
from src.unity_wheel.data_providers.databento.client import DatabentoClient
from src.unity_wheel.data_providers.databento.integration import DatabentoIntegration
from src.unity_wheel.secrets.integration import SecretInjector

logger = logging.getLogger(__name__)


# Removed create_synthetic_unity_options function
# Per user requirement: fail if real data not available, no fallback to synthetic data


async def create_databento_market_snapshot(
    portfolio_value: float,
    ticker: str = "U",
) -> tuple[MarketSnapshot, float]:
    """Create market snapshot from real Databento data.

    Args:
        portfolio_value: Total portfolio value
        ticker: Stock ticker (default: "U" for Unity)

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
            except Exception as e:
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

            except Exception as e:
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
                        expiration = expiration.replace(tzinfo=timezone.utc)
                else:
                    expiration = datetime.fromisoformat(str(candidate["expiration"]))
                    if expiration.tzinfo is None:
                        expiration = expiration.replace(tzinfo=timezone.utc)

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
                timestamp=datetime.now(timezone.utc),
                ticker=ticker,
                current_price=current_price,
                buying_power=portfolio_value,
                margin_used=0.0,
                positions=[],
                option_chain=option_chain,
                implied_volatility=0.45,  # Unity typical IV
                risk_free_rate=0.05,
            )
            # Confidence based on number of option candidates
            confidence = 1.0 if len(option_chain) > 10 else 0.8

            return market_snapshot, confidence

        finally:
            await client.close()


def get_market_data_sync(portfolio_value: float, ticker: str = "U") -> tuple[MarketSnapshot, float]:
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
        return loop.run_until_complete(create_databento_market_snapshot(portfolio_value, ticker))
    finally:
        loop.close()
