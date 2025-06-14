"""Integration layer between Databento and wheel strategy components.

Bridges:
- Databento data types → Internal Position/Greeks models
- Real-time data → Risk analytics
- Historical data → Backtesting
"""
from __future__ import annotations


import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.loader import get_config
from ...math.options import calculate_all_greeks, implied_volatility_validated
from unity_wheel.models.position import Position
from unity_wheel.utils.logging import StructuredLogger

from .client import DatabentoClient
from .types import InstrumentDefinition, OptionChain, OptionQuote

logger = StructuredLogger(logging.getLogger(__name__))


class DatabentoIntegration:
    """Integrates Databento data with wheel strategy."""

    def __init__(
        self,
        client: DatabentoClient,
        storage_adapter: Optional["DatabentoStorageAdapter"] = None,
        risk_free_rate: float = None,
    ):
        """Initialize integration.

        Args:
            client: Databento client instance
            storage_adapter: Optional storage adapter for caching
            risk_free_rate: Risk-free rate for options calculations
        """
        self.client = client
        self.storage_adapter = storage_adapter

        # Use config value if not provided
        if risk_free_rate is None:
            # Default risk-free rate - could be added to config if needed
            # Using current approximate US Treasury rate
            self.risk_free_rate = 0.05
        else:
            self.risk_free_rate = risk_free_rate

        # Cache for instrument definitions
        self._definition_cache: Dict[int, InstrumentDefinition] = {}

    async def get_wheel_candidates(
        self,
        underlying: str = None,
        target_delta: float = None,
        dte_range: Tuple[int, int] = None,
        min_premium_pct: float = None,
    ) -> List[Dict]:
        """Find suitable options for wheel strategy.

        Args:
            underlying: Underlying symbol
            target_delta: Target delta for short puts
            dte_range: Min/max days to expiration
            min_premium_pct: Minimum premium as % of strike

        Returns:
            List of candidate positions with analytics
        """
        # Use config values as defaults
        config = get_config()
        if underlying is None:
            underlying = config.unity.ticker
        if target_delta is None:
            target_delta = config.strategy.delta_target
        if dte_range is None:
            dte_range = (config.strategy.min_days_to_expiry, config.strategy.days_to_expiry_target)
        if min_premium_pct is None:
            min_premium_pct = config.strategy.min_premium_yield * 100  # Convert to percentage

        logger.info(
            "finding_wheel_candidates",
            extra={
                "underlying": underlying,
                "target_delta": target_delta,
                "dte_range": dte_range,
            },
        )

        candidates = []

        # Get spot price first
        spot_data = await self.client._get_underlying_price(underlying)
        spot_price = float(spot_data.last_price)

        # Find relevant expirations
        today = datetime.now().date()
        min_expiry = today + timedelta(days=dte_range[0])
        max_expiry = today + timedelta(days=dte_range[1])

        # Get monthly expirations in range
        expirations = self._get_monthly_expirations(min_expiry, max_expiry)

        # Use most recent trading day for market data
        today = datetime.now(timezone.utc)

        # Find last trading day (skip weekends)
        if today.weekday() >= 5:  # Saturday or Sunday
            days_back = today.weekday() - 4  # Back to Friday
            last_trading_day = today - timedelta(days=days_back)
        else:
            # Use previous day for weekdays
            last_trading_day = today - timedelta(days=1)

        market_timestamp = last_trading_day.replace(hour=0, minute=0, second=0, microsecond=0)

        semaphore = asyncio.Semaphore(3)

        async def analyze_expiry(expiry: datetime) -> List[Dict]:
            local_candidates: List[Dict] = []
            async with semaphore:
                try:
                    chain = await self.client.get_option_chain(
                        underlying=underlying, expiration=expiry, timestamp=market_timestamp
                    )

                    definitions = await self.client._get_definitions(underlying, expiry)
                    def_map = {d.instrument_id: d for d in definitions}

                    for put_quote in chain.puts:
                        if put_quote.instrument_id not in def_map:
                            continue

                        defn = def_map[put_quote.instrument_id]

                        metrics = await self._calculate_option_metrics(
                            quote=put_quote,
                            definition=defn,
                            spot_price=spot_price,
                            option_type="put",
                        )

                        if (
                            abs(metrics["delta"] - target_delta) < 0.05
                            and metrics["premium_pct"] >= min_premium_pct
                        ):
                            local_candidates.append(
                                {
                                    "instrument_id": put_quote.instrument_id,
                                    "symbol": defn.raw_symbol,
                                    "strike": float(defn.strike_price),
                                    "expiration": defn.expiration,
                                    "dte": defn.days_to_expiry,
                                    "bid": float(put_quote.bid_price),
                                    "ask": float(put_quote.ask_price),
                                    "mid": float(put_quote.mid_price),
                                    "spread_pct": float(put_quote.spread_pct),
                                    **metrics,
                                }
                            )
                except (ValueError, KeyError, AttributeError) as e:  # noqa: BLE001
                    logger.error(
                        "chain_analysis_error",
                        extra={"expiration": expiry.isoformat(), "error": str(e)},
                    )
            return local_candidates

        tasks = [asyncio.create_task(analyze_expiry(exp)) for exp in expirations]
        results = await asyncio.gather(*tasks)

        for cand_list in results:
            candidates.extend(cand_list)

        # Sort by expected return
        candidates.sort(key=lambda x: x["expected_return"], reverse=True)

        logger.info(
            "wheel_candidates_found",
            extra={
                "count": len(candidates),
                "best_return": candidates[0]["expected_return"] if candidates else 0,
            },
        )

        return candidates

    async def _calculate_option_metrics(
        self,
        quote: OptionQuote,
        definition: InstrumentDefinition,
        spot_price: float,
        option_type: str,
    ) -> Dict[str, float]:
        """Calculate comprehensive option metrics."""
        strike = float(definition.strike_price)
        dte_years = definition.days_to_expiry / 365.0
        mid_price = float(quote.mid_price)

        # Calculate IV from mid price
        iv_result = implied_volatility_validated(
            option_price=mid_price,
            spot_price=spot_price,
            strike_price=strike,
            time_to_expiry=dte_years,
            risk_free_rate=self.risk_free_rate,
            option_type=option_type,
        )

        iv = iv_result.value if iv_result.confidence > 0.8 else 0.30

        # Calculate Greeks
        greeks_dict, greeks_confidence = calculate_all_greeks(
            S=spot_price,
            K=strike,
            T=dte_years,
            r=self.risk_free_rate,
            sigma=iv,
            option_type=option_type,
        )

        greeks = greeks_dict

        # Calculate strategy metrics
        premium_pct = (mid_price / strike) * 100
        annualized_return = (premium_pct / definition.days_to_expiry) * 365

        # Probability of profit (simplified)
        if option_type == "put":
            # For short put: profit if spot > strike - premium
            breakeven = strike - mid_price
            prob_profit = 1 - greeks["delta"]  # Rough approximation
        else:
            # For covered call: profit if spot < strike + premium
            breakeven = strike + mid_price
            prob_profit = 1 + greeks["delta"]  # Delta negative for OTM calls

        # Expected return considering assignment probability
        if option_type == "put":
            # Short put: keep premium if not assigned
            expected_return = premium_pct * prob_profit
        else:
            # Covered call: premium + potential upside
            max_gain = ((strike - spot_price) / spot_price + premium_pct / 100) * 100
            expected_return = max_gain * prob_profit

        return {
            "iv": iv,
            "delta": greeks["delta"],
            "gamma": greeks["gamma"],
            "theta": greeks["theta"],
            "vega": greeks["vega"],
            "premium_pct": premium_pct,
            "annualized_return": annualized_return,
            "prob_profit": prob_profit,
            "expected_return": expected_return,
            "breakeven": breakeven,
        }

    def _get_monthly_expirations(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get monthly option expirations (3rd Friday) in date range."""
        expirations = []

        # Convert to datetime if needed
        if isinstance(start_date, datetime):
            current = start_date.replace(day=1)
        else:
            current = datetime.combine(start_date.replace(day=1), datetime.min.time())

        # Ensure timezone info
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)

        while current.date() <= end_date:
            # Find first Friday
            first_day = current.replace(day=1)
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_until_friday)

            # Third Friday
            third_friday = first_friday + timedelta(weeks=2)

            # Handle both date and datetime comparisons
            third_friday_date = third_friday.date()
            start_compare = start_date.date() if isinstance(start_date, datetime) else start_date
            end_compare = end_date.date() if isinstance(end_date, datetime) else end_date

            if start_compare <= third_friday_date <= end_compare:
                # Ensure expiration has timezone info
                if third_friday.tzinfo is None:
                    third_friday = third_friday.replace(tzinfo=timezone.utc)
                expirations.append(third_friday)

            # Next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return expirations

    async def convert_to_position(self, candidate: Dict, quantity: int = 1) -> Position:
        """Convert wheel candidate to Position model.

        Args:
            candidate: Candidate dict from get_wheel_candidates
            quantity: Number of contracts

        Returns:
            Position instance
        """
        # Convert to OCC symbol format
        # Example: U241220P00055000 for U Dec 20 2024 Put $55
        expiry = candidate["expiration"]
        date_str = expiry.strftime("%y%m%d")
        option_type = "P"  # Put for wheel entry
        strike_str = f"{int(candidate['strike'] * 1000):08d}"

        # Parse underlying from raw symbol (e.g., "U 24 06 21 00055 P" -> "U")
        underlying = candidate["symbol"].split()[0]

        occ_symbol = f"{underlying}{date_str}{option_type}{strike_str}"

        # For wheel strategy, we sell puts (negative quantity)
        position = Position(
            symbol=occ_symbol, quantity=-abs(quantity)  # Negative for short position
        )

        return position

    async def get_historical_data_for_backtest(
        self, underlying: str, start_date: datetime, end_date: datetime, frequency: str = "daily"
    ) -> pd.DataFrame:
        """Get historical data formatted for backtesting.

        Args:
            underlying: Underlying symbol
            start_date: Backtest start date
            end_date: Backtest end date
            frequency: Data frequency (daily, hourly)

        Returns:
            DataFrame with columns required for backtesting
        """
        logger.info(
            "fetching_backtest_data",
            extra={
                "underlying": underlying,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "frequency": frequency,
            },
        )

        # This would fetch from storage and format for backtesting
        # Placeholder for now
        import pandas as pd

        # Would implement actual data fetching
        df = pd.DataFrame()

        return df

    async def analyze_positions_on_demand(self, positions: List[Position]) -> Dict[str, Any]:
        """Analyze current positions with latest market data (pull-when-asked).

        Args:
            positions: List of positions to analyze

        Returns:
            Analysis results with current Greeks and recommendations
        """
        results = {}

        for pos in positions:
            # Get latest option data for position
            if pos.is_option():
                # Parse option details from position
                underlying = pos.symbol[:1]  # Simplified parsing

                # Fetch current market data
                chain = await self.client.get_option_chain(
                    underlying=underlying, expiration=pos.expiration, timestamp=None  # Latest data
                )

                # Find specific option in chain
                option_type = "CALL" if "C" in pos.symbol else "PUT"
                options = chain.calls if option_type == "CALL" else chain.puts

                for opt in options:
                    # Match by strike
                    if abs(opt.strike - pos.strike) < 0.01:
                        results[pos.symbol] = {
                            "current_price": opt.mid_price,
                            "bid": opt.bid,
                            "ask": opt.ask,
                            "spread_pct": opt.spread_pct,
                            "volume": opt.volume,
                            "timestamp": chain.timestamp,
                        }
                        break

        logger.info(
            "positions_analyzed",
            extra={"position_count": len(positions), "results_count": len(results)},
        )

        return results