"""Simple advisor with Unity fill model and single Schwab account."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from unity_wheel.execution import UnityFillModel
from unity_wheel.portfolio import SingleAccountManager, die
from unity_wheel.risk import RiskLimits
from unity_wheel.strategy import WheelParameters
from unity_wheel.utils import RecoveryStrategy, get_logger, timed_operation, with_recovery
from unity_wheel.utils.data_validator import DataValidator, validate_market_data
from .advisor import WheelAdvisor
from .types import MarketSnapshot, Recommendation

logger = get_logger(__name__)


class SimpleWheelAdvisor(WheelAdvisor):
    """Simplified advisor for single Schwab account with hard failure on missing data."""

    def __init__(
        self,
        wheel_params: Optional[WheelParameters] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        """Initialize simple advisor."""
        super().__init__(wheel_params, risk_limits)

        # Initialize components
        self.fill_model = UnityFillModel()
        self.account_manager = SingleAccountManager()

        logger.info("Simple advisor initialized")

    @timed_operation(threshold_ms=200.0)
    def advise_with_fills(
        self, market_snapshot: MarketSnapshot, account_data: Dict
    ) -> Recommendation:
        """
        Generate recommendation with fill costs and account validation.
        Dies if any required data is missing.

        Parameters
        ----------
        market_snapshot : MarketSnapshot
            Current market data
        account_data : Dict
            Schwab account data

        Returns
        -------
        Recommendation
            Trading recommendation with fill estimates
        """
        # Validate market data first - dies if ANY field missing
        validate_market_data(market_snapshot)

        # Parse account data - dies if any data missing
        account = self.account_manager.parse_account(account_data)

        # Update market snapshot with actual account data
        market_snapshot["buying_power"] = account.buying_power
        market_snapshot["margin_used"] = account.total_value - account.cash_balance

        # Get base recommendation
        base_rec = self.advise_position(market_snapshot)

        # If HOLD, return as-is
        if base_rec["action"] == "HOLD":
            return base_rec

        # Validate buying power - dies if insufficient
        if "margin_required" in base_rec["risk"]:
            self.account_manager.validate_buying_power(base_rec["risk"]["margin_required"], account)

        # Validate position limits - dies if exceeded
        if "strike" in base_rec["details"] and "contracts" in base_rec["details"]:
            strike = base_rec["details"]["strike"]
            contracts = base_rec["details"]["contracts"]
            position_value = strike * 100 * contracts

            self.account_manager.validate_position_limits(position_value, account)

        # Add fill cost estimation
        if "strike" in base_rec["details"] and "contracts" in base_rec["details"]:
            strike = base_rec["details"]["strike"]
            contracts = base_rec["details"]["contracts"]

            # Find option data for the strike
            option_key = self._find_option_key(market_snapshot, strike)
            if not option_key:
                die(f"No option data found for strike {strike}")

            option_data = market_snapshot["option_chain"].get(option_key)
            if not option_data:
                die(f"Missing option data for {option_key}")

            # Get bid/ask - die if missing
            bid = option_data.get("bid")
            ask = option_data.get("ask")

            if bid is None:
                die(f"Missing bid price for {option_key}")
            if ask is None:
                die(f"Missing ask price for {option_key}")

            # Estimate fill price
            fill_estimate, fill_confidence = self.fill_model.estimate_fill_price(
                bid=bid, ask=ask, size=contracts, is_opening=True, urgency=0.5
            )

            # Add fill details to recommendation
            base_rec["details"]["fill_estimate"] = {
                "estimated_fill": fill_estimate.fill_price,
                "commission": fill_estimate.commission,
                "spread_cost": fill_estimate.spread_cost,
                "total_cost": fill_estimate.total_cost,
                "fill_confidence": fill_confidence,
            }

            # Adjust expected return for costs
            if "expected_return" in base_rec["risk"]:
                base_rec["risk"]["expected_return"] -= fill_estimate.total_cost

        # Add account summary
        base_rec["details"]["account_summary"] = {
            "account_id": account.account_id,
            "total_value": account.total_value,
            "buying_power": account.buying_power,
            "unity_shares": account.unity_shares,
            "unity_puts": account.unity_puts,
            "unity_notional": account.unity_notional,
        }

        return base_rec

    def _find_option_key(self, market_snapshot: MarketSnapshot, strike: float) -> Optional[str]:
        """Find option key matching the strike price."""
        for key, option in market_snapshot["option_chain"].items():
            if abs(option["strike"] - strike) < 0.01:
                return key
        return None
