"""Bridge between Databento data and WheelAdvisor market snapshot."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from unity_wheel.models.position import Position
from unity_wheel.storage import Storage
from unity_wheel.utils.logging import StructuredLogger

from ..api.types import MarketSnapshot, OptionData
from ..math.options import implied_volatility_validated
from .types import InstrumentDefinition, OptionChain

logger = StructuredLogger(logging.getLogger(__name__))


class DatentoMarketSnapshotBuilder:
    """Build MarketSnapshot from Databento data."""

    def __init__(self, storage: Storage):
        """Initialize with unified storage."""
        self.storage = storage

    def build_from_chain(
        self,
        chain: OptionChain,
        definitions: Dict[int, InstrumentDefinition],
        buying_power: float,
        margin_used: float = 0.0,
        positions: Optional[list[Position]] = None,
        risk_free_rate: float = 0.05,
    ) -> MarketSnapshot:
        """Build MarketSnapshot from Databento option chain.

        Args:
            chain: Databento option chain
            definitions: Map of instrument_id to definition
            buying_power: Available buying power
            margin_used: Currently used margin
            positions: Existing positions
            risk_free_rate: Risk-free rate

        Returns:
            MarketSnapshot compatible with WheelAdvisor
        """
        # Build option chain dict
        option_chain = {}

        # Process puts for wheel strategy
        for put_quote in chain.puts:
            if put_quote.instrument_id not in definitions:
                continue

            defn = definitions[put_quote.instrument_id]
            strike = float(defn.strike_price)

            # Calculate implied volatility
            mid_price = float(put_quote.mid_price)
            spot_price = float(chain.spot_price)
            dte_years = defn.days_to_expiry / 365.0

            iv_result = implied_volatility_validated(
                option_price=mid_price,
                spot_price=spot_price,
                strike_price=strike,
                time_to_expiry=dte_years,
                risk_free_rate=risk_free_rate,
                option_type="put",
            )

            # Skip if IV calculation failed
            if iv_result.confidence < 0.5:
                logger.warning(
                    "IV calculation failed",
                    extra={"strike": strike, "confidence": iv_result.confidence},
                )
                continue

            # Create OptionData
            option_data = OptionData(
                strike=strike,
                expiration=defn.expiration.strftime("%Y-%m-%d"),
                bid=float(put_quote.bid_price),
                ask=float(put_quote.ask_price),
                mid=mid_price,
                volume=0,  # Would need trade data
                open_interest=0,  # Would need additional data
                delta=0.0,  # Will be calculated by advisor
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                implied_volatility=iv_result.value,
            )

            option_chain[str(strike)] = option_data

        # Calculate overall IV (use ATM or average)
        all_ivs = [
            opt.implied_volatility for opt in option_chain.values() if opt.implied_volatility > 0
        ]
        overall_iv = sum(all_ivs) / len(all_ivs) if all_ivs else 0.65

        return MarketSnapshot(
            timestamp=chain.timestamp,
            ticker=chain.underlying,
            current_price=float(chain.spot_price),
            buying_power=buying_power,
            margin_used=margin_used,
            positions=positions or [],
            option_chain=option_chain,
            implied_volatility=overall_iv,
            risk_free_rate=risk_free_rate,
        )

    async def get_latest_snapshot(
        self,
        ticker: str,
        buying_power: float,
        margin_used: float = 0.0,
        positions: Optional[list[Position]] = None,
    ) -> Optional[MarketSnapshot]:
        """Get latest market snapshot from stored data.

        Args:
            ticker: Underlying ticker
            buying_power: Available buying power
            margin_used: Currently used margin
            positions: Existing positions

        Returns:
            MarketSnapshot or None if no data available
        """
        # Find latest stored chain data
        today = datetime.now().date()

        # Look for recent data (last 7 days)
        for days_back in range(7):
            check_date = today - timedelta(days=days_back)
            date_str = check_date.strftime("%Y%m%d")

            # In pull-when-asked architecture, cache checking is handled by Storage
            # This method would be replaced by storage.get_or_fetch pattern

        logger.warning("No recent data found", extra={"ticker": ticker})
        return None
