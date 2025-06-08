"""Main API entry point for wheel trading advisor."""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Final, Literal, TypedDict

from .config import get_settings
from .config.unity import (
    MIN_BID_ASK_SPREAD,
    MIN_CONFIDENCE_SCORE,
    MIN_OPEN_INTEREST,
    MIN_VOLUME,
    TICKER,
)
from .diagnostics import SelfDiagnostics
from .models import WheelPosition
from .wheel import WheelStrategy

logger = logging.getLogger(__name__)

# Type definitions
Action = Literal["HOLD", "ADJUST"]


class RiskMetrics(TypedDict):
    """Risk metrics for recommendation."""

    max_loss: float
    probability_assignment: float
    expected_return: float
    edge_ratio: float


class Recommendation(TypedDict):
    """Recommendation format as specified."""

    action: Action
    rationale: str
    confidence: float
    risk: RiskMetrics


class MarketSnapshot(TypedDict):
    """Market data snapshot for decision making."""

    timestamp: datetime
    ticker: Literal["U"]
    current_price: float
    buying_power: float
    positions: list[dict[str, Any]]
    option_chain: dict[str, Any]
    iv: float


# Constants
COMMISSION_PER_CONTRACT: Final[float] = 0.65
CONTRACTS_PER_TRADE: Final[int] = 100


def calculate_edge_probability(
    premium: float,
    strike: float,
    current_price: float,
    probability_assign: float,
    contracts: int = 1,
) -> float:
    """
    Calculate edge probability as defined in spec.

    Edge = E[profit] / risk_capital
    E[profit] = premium × (1 − P(assign)) − expected_assignment_loss × P(assign) − 2×$0.65/contract
    """
    # Risk capital for cash-secured put
    risk_capital = strike * CONTRACTS_PER_TRADE * contracts

    # Expected profit calculation
    premium_collected = premium * CONTRACTS_PER_TRADE * contracts
    commission_cost = 2 * COMMISSION_PER_CONTRACT * contracts

    # Expected assignment loss (if assigned, we buy at strike vs market)
    assignment_loss = max(0, strike - current_price) * CONTRACTS_PER_TRADE * contracts
    expected_assignment_loss = assignment_loss * probability_assign

    # Total expected profit
    expected_profit = (
        premium_collected * (1 - probability_assign) - expected_assignment_loss - commission_cost
    )

    # Edge ratio
    edge = expected_profit / risk_capital if risk_capital > 0 else 0.0

    logger.debug(
        "Edge calculation",
        extra={
            "premium": premium,
            "strike": strike,
            "current_price": current_price,
            "probability_assign": probability_assign,
            "expected_profit": expected_profit,
            "risk_capital": risk_capital,
            "edge": edge,
        },
    )

    return edge


def validate_option_liquidity(option_data: dict[str, Any]) -> bool:
    """Validate option meets liquidity requirements."""
    bid = option_data.get("bid", 0)
    ask = option_data.get("ask", float("inf"))
    volume = option_data.get("volume", 0)
    open_interest = option_data.get("open_interest", 0)

    # Check bid-ask spread
    if ask - bid > MIN_BID_ASK_SPREAD:
        logger.warning(f"Wide bid-ask spread: {ask - bid}")
        return False

    # Check volume
    if volume < MIN_VOLUME:
        logger.warning(f"Low volume: {volume}")
        return False

    # Check open interest
    if open_interest < MIN_OPEN_INTEREST:
        logger.warning(f"Low open interest: {open_interest}")
        return False

    return True


def advise_position(market_snapshot: MarketSnapshot) -> Recommendation:
    """
    Main API entry point for position recommendations.

    Parameters
    ----------
    market_snapshot : MarketSnapshot
        Current market data including positions, option chain, and Greeks

    Returns
    -------
    Recommendation
        Action recommendation with rationale, confidence, and risk metrics
    """
    start_time = datetime.now()

    # Run diagnostics first
    diagnostics = SelfDiagnostics()
    if not diagnostics.run_all_checks():
        logger.error("Self-diagnostics failed")
        failed_checks = [r for r in diagnostics.results if r.level in ("ERROR", "CRITICAL")]
        return Recommendation(
            action="HOLD",
            rationale=f"System diagnostics failed: {failed_checks[0].message}",
            confidence=0.0,
            risk=RiskMetrics(
                max_loss=0.0,
                probability_assignment=0.0,
                expected_return=0.0,
                edge_ratio=0.0,
            ),
        )

    # Validate input
    if market_snapshot["ticker"] != TICKER:
        raise ValueError(f"Only Unity (U) supported, got {market_snapshot['ticker']}")

    settings = get_settings()
    wheel = WheelStrategy()

    try:
        current_price = market_snapshot["current_price"]
        buying_power = market_snapshot["buying_power"]
        option_chain = market_snapshot["option_chain"]

        # Extract available strikes from option chain
        strikes = sorted([float(k) for k in option_chain.keys()])

        # Check current positions
        current_positions = market_snapshot.get("positions", [])
        open_puts = sum(1 for p in current_positions if p.get("option_type") == "put")

        # Check position limits
        if open_puts >= 3:
            return Recommendation(
                action="HOLD",
                rationale="Maximum 3 concurrent puts limit reached",
                confidence=1.0,
                risk=RiskMetrics(
                    max_loss=0.0,
                    probability_assignment=0.0,
                    expected_return=0.0,
                    edge_ratio=0.0,
                ),
            )

        # Find optimal put strike
        optimal_put = wheel.find_optimal_put_strike(
            current_price=current_price,
            available_strikes=strikes,
            volatility=market_snapshot["iv"],
            days_to_expiry=settings.days_to_expiry_target,
        )

        if not optimal_put:
            return Recommendation(
                action="HOLD",
                rationale="No suitable put strikes found within delta target",
                confidence=0.0,
                risk=RiskMetrics(
                    max_loss=0.0,
                    probability_assignment=0.0,
                    expected_return=0.0,
                    edge_ratio=0.0,
                ),
            )

        # Get option details
        option_data = option_chain[str(optimal_put)]

        # Validate liquidity
        if not validate_option_liquidity(option_data):
            return Recommendation(
                action="HOLD",
                rationale=f"Insufficient liquidity for ${optimal_put} put",
                confidence=0.0,
                risk=RiskMetrics(
                    max_loss=0.0,
                    probability_assignment=0.0,
                    expected_return=0.0,
                    edge_ratio=0.0,
                ),
            )

        # Calculate position size
        portfolio_value = buying_power  # Simplified for now
        contracts = wheel.calculate_position_size(TICKER, current_price, portfolio_value)

        # Check portfolio allocation limit (12% per leg)
        position_value = optimal_put * CONTRACTS_PER_TRADE * contracts
        if position_value > portfolio_value * 0.12:
            contracts = int((portfolio_value * 0.12) / (optimal_put * CONTRACTS_PER_TRADE))
            if contracts < 1:
                return Recommendation(
                    action="HOLD",
                    rationale="Position size would exceed 12% portfolio limit",
                    confidence=0.0,
                    risk=RiskMetrics(
                        max_loss=0.0,
                        probability_assignment=0.0,
                        expected_return=0.0,
                        edge_ratio=0.0,
                    ),
                )

        # Calculate risk metrics
        from .utils.math import probability_itm

        prob_assign = probability_itm(
            S=current_price,
            K=optimal_put,
            T=settings.days_to_expiry_target / 365.0,
            r=0.05,
            sigma=market_snapshot["iv"],
            option_type="put",
        )

        premium = option_data.get("mid", (option_data["bid"] + option_data["ask"]) / 2)

        edge = calculate_edge_probability(
            premium=premium,
            strike=optimal_put,
            current_price=current_price,
            probability_assign=prob_assign,
            contracts=contracts,
        )

        # Calculate expected return (annualized)
        days_to_expiry = settings.days_to_expiry_target
        premium_return = (premium / optimal_put) * (365 / days_to_expiry)
        expected_return = premium_return * (1 - prob_assign)

        # Check confidence threshold (75% edge probability)
        confidence = min(0.99, edge * 4) if edge > 0 else 0.0  # Scale edge to confidence

        if confidence < 0.75:
            return Recommendation(
                action="HOLD",
                rationale=f"Edge probability {edge:.1%} below 75% threshold",
                confidence=confidence,
                risk=RiskMetrics(
                    max_loss=optimal_put * CONTRACTS_PER_TRADE * contracts,
                    probability_assignment=prob_assign,
                    expected_return=expected_return,
                    edge_ratio=edge,
                ),
            )

        # Decision latency check
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > 0.2:
            logger.warning(f"Decision latency {elapsed:.3f}s exceeds 200ms target")

        # Recommend the trade
        return Recommendation(
            action="ADJUST",
            rationale=f"Sell {contracts} {TICKER} ${optimal_put}P @ ${premium:.2f} ({settings.days_to_expiry_target} DTE)",
            confidence=confidence,
            risk=RiskMetrics(
                max_loss=optimal_put * CONTRACTS_PER_TRADE * contracts,
                probability_assignment=prob_assign,
                expected_return=expected_return,
                edge_ratio=edge,
            ),
        )

    except Exception as e:
        logger.error(f"Failed to generate recommendation: {e}")
        return Recommendation(
            action="HOLD",
            rationale=f"Calculation error: {str(e)}",
            confidence=0.0,
            risk=RiskMetrics(
                max_loss=0.0,
                probability_assignment=0.0,
                expected_return=0.0,
                edge_ratio=0.0,
            ),
        )
