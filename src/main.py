"""Main entry point for the wheel trading application."""

from __future__ import annotations

import logging
from datetime import datetime

from .config import get_settings
from .wheel import WheelStrategy

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the wheel trading application."""
    logger.info("Starting Wheel Trading Application")
    logger.info(f"Current time: {datetime.now()}")
    logger.info(f"Trading mode: {settings.trading_mode}")

    if settings.google_cloud_project:
        logger.info(f"Google Cloud Project: {settings.google_cloud_project}")

    # Initialize wheel strategy
    wheel = WheelStrategy()
    logger.info(f"Wheel strategy initialized with delta target: {settings.wheel_delta_target}")

    # Example usage (in production, this would be in a trading loop)
    if settings.trading_mode != "live":
        demo_wheel_strategy(wheel)

    logger.info("Application started successfully")


def demo_wheel_strategy(wheel: WheelStrategy) -> None:
    """Demonstrate wheel strategy functionality."""
    logger.info("Running wheel strategy demo...")

    # Example: Find optimal put strike for Unity
    u_strikes = [30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0]
    current_u = 35.50

    optimal_put = wheel.find_optimal_put_strike(
        current_price=current_u,
        available_strikes=u_strikes,
        volatility=0.65,  # Unity typical IV
        days_to_expiry=45,
    )

    if optimal_put:
        logger.info(f"U optimal put strike: ${optimal_put} (current: ${current_u})")

        # Calculate position size
        portfolio_value = 100000  # Example portfolio
        contracts = wheel.calculate_position_size("U", current_u, portfolio_value)
        logger.info(f"Recommended position size: {contracts} contracts")

        # Track the position
        u_position = wheel.track_position("U")
        logger.info(f"Tracking position for {u_position.symbol}")

    # Example: Find optimal call strike if we own shares
    cost_basis = 33.50
    optimal_call = wheel.find_optimal_call_strike(
        current_price=current_u,
        cost_basis=cost_basis,
        available_strikes=u_strikes,
        volatility=0.65,  # Unity typical IV
        days_to_expiry=45,
    )

    if optimal_call:
        logger.info(f"U optimal call strike: ${optimal_call} (cost basis: ${cost_basis})")


if __name__ == "__main__":
    main()
