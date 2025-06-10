"""Wheel trading advisor with self-validation and risk management."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..analytics import UnityAssignmentModel
from ..math import probability_itm_validated
from ..metrics import metrics_collector
from ..models import Account, Position
from ..risk import BorrowingCostAnalyzer, RiskAnalyzer, RiskLimits, analyze_borrowing_decision
from ..risk.advanced_financial_modeling import AdvancedFinancialModeling
from ..strategy import WheelParameters, WheelStrategy
from ..strategy.position_evaluator import PositionEvaluator, SwitchAnalysis
from ..utils import (
    DecisionLogger,
    RecoveryStrategy,
    StructuredLogger,
    get_logger,
    is_trading_day,
    timed_operation,
    with_recovery,
)
from .types import Action, MarketSnapshot, OptionData, Recommendation, RiskMetrics

# Lazy imports to avoid circular dependency
_market_validator = None
_anomaly_detector = None


def _get_market_validator():
    """Lazy import market validator."""
    global _market_validator
    if _market_validator is None:
        from ..data_providers.base import get_market_validator

        _market_validator = get_market_validator()
    return _market_validator


def _get_anomaly_detector():
    """Lazy import anomaly detector."""
    global _anomaly_detector
    if _anomaly_detector is None:
        from ..data_providers.base import get_anomaly_detector

        _anomaly_detector = get_anomaly_detector()
    return _anomaly_detector


logger = get_logger(__name__)
structured_logger = StructuredLogger(logger)
decision_logger = DecisionLogger(structured_logger)


class TradingConstraints:
    """Trading constraints and validation rules."""

    def __init__(self):
        """Initialize constraints from config."""
        from src.config.loader import get_config

        config = get_config()

        # Unity-specific constraints
        self.MAX_CONCURRENT_PUTS = config.operations.api.max_concurrent_puts
        self.MAX_POSITION_PCT = config.operations.api.max_position_pct
        self.MIN_CONFIDENCE = config.operations.api.min_confidence
        self.MAX_DECISION_TIME = config.operations.api.max_decision_time

        # Liquidity requirements
        self.MAX_BID_ASK_SPREAD = config.operations.api.max_bid_ask_spread
        self.MIN_VOLUME = config.operations.api.min_volume
        self.MIN_OPEN_INTEREST = config.operations.api.min_open_interest

        # Commission
        self.COMMISSION_PER_CONTRACT = config.trading.execution.commission_per_contract
        self.CONTRACTS_PER_TRADE = config.trading.execution.contracts_per_trade


class WheelAdvisor:
    """Enhanced wheel trading advisor with autonomous operation."""

    def __init__(
        self,
        wheel_params: Optional[WheelParameters] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        """Initialize advisor with strategy and risk components."""
        self.constraints = TradingConstraints()
        self.wheel_params = wheel_params or WheelParameters()
        self.risk_limits = risk_limits or RiskLimits()

        # Initialize components
        self.strategy = WheelStrategy(self.wheel_params)
        self.risk_analyzer = RiskAnalyzer(self.risk_limits)
        self.assignment_model = UnityAssignmentModel()
        self.borrowing_analyzer = BorrowingCostAnalyzer()
        self.financial_modeler = AdvancedFinancialModeling(self.borrowing_analyzer)
        self.position_evaluator = PositionEvaluator()

        logger.info(
            "WheelAdvisor initialized",
            extra={
                "target_delta": self.wheel_params.target_delta,
                "target_dte": self.wheel_params.target_dte,
                "max_concurrent_puts": self.constraints.MAX_CONCURRENT_PUTS,
            },
        )

    # === BEGIN main_recommendation_engine ===
    @timed_operation(threshold_ms=200.0)  # 200ms SLA
    @with_recovery(strategy=RecoveryStrategy.FALLBACK)
    def advise_position(self, market_snapshot: MarketSnapshot) -> Recommendation:
        """
        Generate trading recommendation with full validation.

        Parameters
        ----------
        market_snapshot : MarketSnapshot
            Current market data including positions and option chain

        Returns
        -------
        Recommendation
            Action recommendation with confidence and risk metrics
        """
        start_time = datetime.now(timezone.utc)
        decision_id = f"wheel_{uuid.uuid4().hex[:8]}_{int(start_time.timestamp())}"

        # Log decision start
        decision_logger.log_decision(
            action="START",
            rationale="Starting position analysis",
            confidence=0.0,
            risk_metrics={},
            metadata={
                "decision_id": decision_id,
                "expected_return": 0.0,
                "features_used": [],
            },
        )

        try:
            # Validate market data with comprehensive data quality checks
            validator = _get_market_validator()
            validation_result = validator.validate(market_snapshot)

            if not validation_result.is_valid:
                error_msg = f"Data quality issues: {len(validation_result.issues)} errors found"
                logger.warning(
                    error_msg,
                    extra={
                        "issues": [issue.message for issue in validation_result.issues[:3]],
                        "quality_level": validation_result.quality_level.value,
                    },
                )
                return self._create_hold_recommendation(error_msg)

            # Check for anomalies
            anomaly_detector = _get_anomaly_detector()
            anomalies = anomaly_detector.detect_market_anomalies(market_snapshot)
            if anomalies:
                logger.info(
                    f"Detected {len(anomalies)} market anomalies",
                    extra={"anomalies": anomalies[:2]},
                )

            # Basic validation (legacy)
            basic_validation = self._validate_snapshot(market_snapshot)
            if not basic_validation[0]:
                return self._create_hold_recommendation(basic_validation[1])

            # Check position limits
            current_positions = self._parse_positions(market_snapshot["positions"])
            open_puts = sum(1 for p in current_positions if p.position_type.value == "put")

            if open_puts >= self.constraints.MAX_CONCURRENT_PUTS:
                return self._create_hold_recommendation(
                    f"Maximum {self.constraints.MAX_CONCURRENT_PUTS} concurrent puts limit reached"
                )

            # Extract market data
            current_price = market_snapshot["current_price"]
            volatility = market_snapshot["implied_volatility"]
            option_chain = market_snapshot["option_chain"]

            # Find optimal strikes
            available_strikes = self._extract_liquid_strikes(option_chain)
            if not available_strikes:
                return self._create_hold_recommendation("No liquid strikes available")

            # Get strike recommendation
            strike_rec = self.strategy.find_optimal_put_strike(
                current_price=current_price,
                available_strikes=available_strikes,
                volatility=volatility,
                days_to_expiry=self.wheel_params.target_dte,
                risk_free_rate=market_snapshot.get("risk_free_rate", 0.05),
                portfolio_value=market_snapshot["buying_power"],
            )

            if not strike_rec or strike_rec.confidence < self.constraints.MIN_CONFIDENCE:
                confidence = strike_rec.confidence if strike_rec else 0.0
                return self._create_hold_recommendation(
                    f"Low confidence ({confidence:.0%}) in strike selection"
                )

            # Check if we have existing Unity option positions to compare
            existing_unity_options = [
                p
                for p in current_positions
                if p.underlying == "U" and p.position_type.value in ["put", "call"]
            ]

            # If we have existing options, evaluate switching opportunities
            if existing_unity_options:
                logger.info(
                    f"Found {len(existing_unity_options)} existing Unity option positions",
                    extra={"positions": [str(p) for p in existing_unity_options]},
                )

                # Prepare available strikes for comparison
                available_strikes_list = []
                for option in option_chain:
                    if option["strike"] and option["bid"] and option["ask"]:
                        available_strikes_list.append(
                            (
                                option["strike"],
                                option["days_to_expiry"],
                                option["bid"],
                                option["ask"],
                            )
                        )

                # Analyze each existing position for potential switch
                best_switch = None
                for position in existing_unity_options:
                    # Get current bid/ask for existing position
                    current_option = next(
                        (
                            opt
                            for opt in option_chain
                            if opt["strike"] == position.strike
                            and opt["days_to_expiry"] == self._get_position_dte(position)
                        ),
                        None,
                    )

                    if current_option:
                        switch_analysis = self.position_evaluator.find_best_switch_opportunity(
                            current_position=position,
                            current_bid=current_option["bid"],
                            current_ask=current_option["ask"],
                            current_dte=current_option["days_to_expiry"],
                            available_strikes=available_strikes_list,
                            underlying_price=current_price,
                            volatility=volatility,
                            risk_free_rate=market_snapshot.get("risk_free_rate", 0.05),
                            contracts=abs(position.quantity),
                        )

                        if switch_analysis and (
                            not best_switch
                            or switch_analysis.switch_benefit > best_switch.switch_benefit
                        ):
                            best_switch = switch_analysis

                # If we found a beneficial switch, recommend it
                if best_switch and best_switch.should_switch:
                    logger.info(
                        f"Found beneficial position switch",
                        extra={
                            "current_strike": best_switch.current_position.strike,
                            "new_strike": best_switch.new_position.strike,
                            "benefit": best_switch.switch_benefit,
                            "rationale": best_switch.rationale,
                        },
                    )

                    # Return switch recommendation
                    return self._create_switch_recommendation(
                        best_switch, current_price, market_snapshot["buying_power"]
                    )
                else:
                    # No beneficial switch found, keep existing positions
                    return self._create_hold_recommendation(
                        "Keep existing positions - no better alternatives found"
                    )

            # Calculate position size with risk constraints
            account = Account(
                cash_balance=market_snapshot["buying_power"],
                buying_power=market_snapshot["buying_power"],
                margin_used=market_snapshot.get("margin_used", 0.0),
            )

            # Analyze borrowing costs before position sizing
            available_cash = market_snapshot.get("available_cash", 0)  # Cash without borrowing
            initial_position_value = strike_rec.strike * 100 * 10  # Start with 10 contracts

            # Calculate expected return for borrowing analysis
            expected_return_pct = (strike_rec.premium / strike_rec.strike) * (
                365 / self.wheel_params.target_dte
            )

            borrowing_analysis = self.borrowing_analyzer.analyze_position_allocation(
                position_size=initial_position_value,
                expected_annual_return=expected_return_pct,
                holding_period_days=self.wheel_params.target_dte,
                available_cash=available_cash,
                confidence=strike_rec.confidence,
            )

            # If borrowing not recommended, reduce position size
            if borrowing_analysis.action == "paydown_debt":
                logger.info(
                    "Borrowing not recommended",
                    extra={
                        "reason": borrowing_analysis.reasoning,
                        "hurdle_rate": f"{borrowing_analysis.hurdle_rate:.1%}",
                        "expected_return": f"{borrowing_analysis.expected_return:.1%}",
                    },
                )
                # Limit to available cash only
                max_contracts = int(
                    available_cash / (strike_rec.strike * 100 * 0.2)
                )  # Assume 20% margin
            else:
                max_contracts = None  # No borrowing-based limit

            # CRITICAL: Pass real option premium, no placeholders
            contracts, size_confidence = self.strategy.calculate_position_size(
                strike_price=strike_rec.strike,
                option_price=strike_rec.premium,  # Real market premium from strike selection
                portfolio_value=account.cash_balance,
                current_margin_used=account.margin_used,
                max_contracts=max_contracts,  # Pass borrowing limit if applicable
            )

            # Validate position size
            position_value = strike_rec.strike * self.constraints.CONTRACTS_PER_TRADE * contracts
            if position_value > account.cash_balance * self.constraints.MAX_POSITION_PCT:
                return self._create_hold_recommendation(
                    f"Position size would exceed {self.constraints.MAX_POSITION_PCT:.0%} portfolio limit"
                )

            # Calculate Unity-specific assignment probability
            near_earnings = market_snapshot.get("near_earnings", False)
            now = datetime.now()
            # Check if it's Friday AND a trading day (not a holiday)
            is_trading_friday = now.weekday() == 4 and is_trading_day(now)
            assignment_prob = self.assignment_model.probability_of_assignment(
                spot_price=current_price,
                strike_price=strike_rec.strike,
                days_to_expiry=self.wheel_params.target_dte,
                volatility=volatility,
                near_earnings=near_earnings,
                is_friday=is_trading_friday,
            )

            # Use Unity-specific assignment probability if confidence is good
            if assignment_prob.confidence > 0.7:
                effective_assignment_prob = assignment_prob.probability
                if assignment_prob.warnings:
                    logger.info(f"Assignment warnings: {', '.join(assignment_prob.warnings)}")
            else:
                effective_assignment_prob = strike_rec.probability_itm

            # Calculate comprehensive risk metrics
            risk_metrics = self._calculate_risk_metrics(
                strike=strike_rec.strike,
                premium=strike_rec.premium,
                contracts=contracts,
                current_price=current_price,
                probability_itm=effective_assignment_prob,
                volatility=volatility,
                portfolio_value=account.cash_balance,
            )

            # Calculate edge
            edge = self._calculate_edge(
                premium=strike_rec.premium,
                strike=strike_rec.strike,
                current_price=current_price,
                probability_assign=effective_assignment_prob,
                contracts=contracts,
            )

            risk_metrics["edge_ratio"] = edge

            # Overall confidence
            confidence = min(
                strike_rec.confidence,
                size_confidence,
                1.0 if edge > 0.01 else edge * 100,  # Scale edge to confidence
            )

            # Check assignment probability threshold
            if effective_assignment_prob > 0.50:
                return self._create_hold_recommendation(
                    f"Assignment probability {effective_assignment_prob:.1%} too high (>50%)"
                )

            # Check decision latency
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed > self.constraints.MAX_DECISION_TIME:
                logger.warning(
                    f"Decision latency {elapsed:.3f}s exceeds {self.constraints.MAX_DECISION_TIME}s target"
                )
                confidence *= 0.9

            # Final decision
            if confidence < self.constraints.MIN_CONFIDENCE:
                return Recommendation(
                    action="HOLD",
                    rationale=f"Overall confidence {confidence:.0%} below {self.constraints.MIN_CONFIDENCE:.0%} threshold",
                    confidence=confidence,
                    risk=risk_metrics,
                    details={
                        "strike": strike_rec.strike,
                        "contracts": contracts,
                        "edge": edge,
                        "decision_time": elapsed,
                    },
                )

            # Add borrowing metrics to risk metrics
            risk_metrics["borrowing_analysis"] = {
                "action": borrowing_analysis.action,
                "hurdle_rate": borrowing_analysis.hurdle_rate,
                "expected_return": borrowing_analysis.expected_return,
                "borrowing_cost": borrowing_analysis.borrowing_cost,
                "net_benefit": borrowing_analysis.net_benefit,
                "source": borrowing_analysis.source_to_use,
            }

            # Run Monte Carlo simulation for advanced risk metrics
            mc_result = self.financial_modeler.monte_carlo_simulation(
                expected_return=expected_return_pct,
                volatility=volatility,
                time_horizon=self.wheel_params.target_dte,
                position_size=strike_rec.strike * 100 * contracts,
                borrowed_amount=max(0, strike_rec.strike * 100 * contracts - available_cash),
                n_simulations=1000,  # Quick simulation
            )

            risk_metrics["monte_carlo"] = {
                "mean_return": mc_result.mean_return,
                "probability_profit": mc_result.probability_profit,
                "var_95_mc": mc_result.percentiles[5],  # 5th percentile is 95% VaR
                "expected_shortfall": mc_result.expected_shortfall,
            }

            # Generate recommendation
            expiry_str = f"{self.wheel_params.target_dte}DTE"

            recommendation = Recommendation(
                action="ADJUST",
                rationale=f"Sell {contracts} {market_snapshot['ticker']} ${strike_rec.strike:.0f}P @ ${strike_rec.premium:.2f} ({expiry_str})",
                confidence=confidence,
                risk=risk_metrics,
                details={
                    "strike": strike_rec.strike,
                    "contracts": contracts,
                    "delta": strike_rec.delta,
                    "premium": strike_rec.premium,
                    "edge": edge,
                    "decision_time": elapsed,
                    "strike_reason": strike_rec.reason,
                    "borrowing_recommended": borrowing_analysis.action == "invest",
                    "borrowing_amount": max(
                        0, strike_rec.strike * 100 * contracts - available_cash
                    ),
                },
            )

            # Log successful decision
            decision_logger.log_decision(
                action="ADJUST",
                rationale=f"Sell {contracts} put contracts at {strike_rec.strike} strike",
                confidence=confidence,
                risk_metrics=risk_metrics,
                metadata={
                    "decision_id": decision_id,
                    "expected_return": risk_metrics["expected_return"],
                    "features_used": [
                        "delta",
                        "volatility",
                        "dte",
                        "premium_yield",
                        "edge_ratio",
                        "var_95",
                        "margin_utilization",
                    ],
                    "execution_time_ms": elapsed * 1000,
                },
            )

            # Track in metrics collector
            metrics_collector.record_decision(
                decision_id=decision_id,
                action="ADJUST",
                confidence=confidence,
                expected_return=risk_metrics["expected_return"],
                execution_time_ms=elapsed * 1000,
                features_used=["delta", "volatility", "dte", "edge_ratio"],
            )

            return recommendation

        except Exception as e:
            logger.error(f"Failed to generate recommendation: {e}", exc_info=True)

            # Log failed decision
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            decision_logger.log_decision(
                action="ERROR",
                rationale=f"Error generating recommendation: {str(e)}",
                confidence=0.0,
                risk_metrics={},
                metadata={
                    "decision_id": decision_id,
                    "expected_return": 0.0,
                    "features_used": [],
                    "execution_time_ms": elapsed * 1000,
                    "error": str(e),
                },
            )

            return self._create_hold_recommendation(f"Calculation error: {str(e)}")

    # === END main_recommendation_engine ===

    def _validate_snapshot(self, snapshot: MarketSnapshot) -> Tuple[bool, str]:
        """Validate market snapshot data."""
        # Check required fields
        required = ["ticker", "current_price", "buying_power", "option_chain"]
        for field in required:
            if field not in snapshot:
                return False, f"Missing required field: {field}"

        # Validate values
        if snapshot["current_price"] <= 0:
            return False, "Invalid current price"

        if snapshot["buying_power"] <= 0:
            return False, "Insufficient buying power"

        if not snapshot["option_chain"]:
            return False, "Empty option chain"

        return True, ""

    def _parse_positions(self, positions_data: List[Any]) -> List[Position]:
        """Parse position data into Position objects."""
        positions = []
        for pos_data in positions_data:
            try:
                # Build OCC symbol for options
                if "strike" in pos_data:
                    # Convert position data to OCC format
                    symbol = self._build_occ_symbol(pos_data)
                else:
                    symbol = pos_data["symbol"]

                position = Position(
                    symbol=symbol,
                    quantity=pos_data["quantity"],
                )
                positions.append(position)
            except Exception as e:
                logger.warning(f"Failed to parse position: {e}")

        return positions

    def _build_occ_symbol(self, pos_data: Dict[str, Any]) -> str:
        """Build OCC option symbol from position data."""
        # Format: TICKER + YYMMDD + C/P + 00000000 (strike * 1000)
        ticker = pos_data["symbol"]

        # Parse expiration
        exp_str = pos_data["expiration"]  # Assume YYYY-MM-DD format
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        date_str = exp_date.strftime("%y%m%d")

        # Option type
        opt_type = "C" if pos_data["option_type"] == "call" else "P"

        # Strike
        strike_str = f"{int(pos_data['strike'] * 1000):08d}"

        return f"{ticker}{date_str}{opt_type}{strike_str}"

    def _extract_liquid_strikes(self, option_chain: Dict[str, OptionData]) -> List[float]:
        """Extract strikes that meet liquidity requirements."""
        liquid_strikes = []

        for strike_str, option_data in option_chain.items():
            # Check liquidity
            if self._validate_option_liquidity(option_data):
                liquid_strikes.append(float(strike_str))

        return sorted(liquid_strikes)

    def _validate_option_liquidity(self, option_data: OptionData) -> bool:
        """Check if option meets liquidity requirements."""
        bid = option_data.get("bid", 0)
        ask = option_data.get("ask", float("inf"))
        volume = option_data.get("volume", 0)
        open_interest = option_data.get("open_interest", 0)

        # Check bid-ask spread using configured constraints
        if ask - bid > self.constraints.MAX_BID_ASK_SPREAD:
            return False

        # Check volume
        if volume < self.constraints.MIN_VOLUME:
            return False

        # Check open interest
        if open_interest < self.constraints.MIN_OPEN_INTEREST:
            return False

        return True

    def _calculate_risk_metrics(
        self,
        strike: float,
        premium: float,
        contracts: int,
        current_price: float,
        probability_itm: float,
        volatility: float,
        portfolio_value: float,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        # Max loss (assignment at strike)
        max_loss = strike * self.constraints.CONTRACTS_PER_TRADE * contracts

        # Expected return (annualized)
        days_to_expiry = self.wheel_params.target_dte
        premium_return = (premium / strike) * (365 / days_to_expiry)
        expected_return = premium_return * (1 - probability_itm)

        # VaR calculation (simplified)
        position_var = max_loss * volatility * 1.65 * probability_itm

        # Margin requirement
        margin_required = strike * self.constraints.CONTRACTS_PER_TRADE * contracts * 0.20

        return RiskMetrics(
            max_loss=max_loss,
            probability_assignment=probability_itm,
            expected_return=expected_return,
            edge_ratio=0.0,  # Calculated separately
            var_95=position_var,
            margin_required=margin_required,
        )

    def _calculate_edge(
        self,
        premium: float,
        strike: float,
        current_price: float,
        probability_assign: float,
        contracts: int,
    ) -> float:
        """Calculate edge probability as profit expectation / risk capital."""
        # Risk capital
        risk_capital = strike * self.constraints.CONTRACTS_PER_TRADE * contracts

        # Expected profit
        premium_collected = premium * self.constraints.CONTRACTS_PER_TRADE * contracts
        commission_cost = 2 * self.constraints.COMMISSION_PER_CONTRACT * contracts

        # Assignment loss
        assignment_loss = (
            max(0, strike - current_price) * self.constraints.CONTRACTS_PER_TRADE * contracts
        )
        expected_assignment_loss = assignment_loss * probability_assign

        # Total expected profit
        expected_profit = (
            premium_collected * (1 - probability_assign)
            - expected_assignment_loss
            - commission_cost
        )

        # Edge ratio
        edge = expected_profit / risk_capital if risk_capital > 0 else 0.0

        return edge

    def _create_hold_recommendation(self, reason: str) -> Recommendation:
        """Create a HOLD recommendation with given reason."""
        return Recommendation(
            action="HOLD",
            rationale=reason,
            confidence=1.0,  # Confident in decision to hold
            risk=RiskMetrics(
                max_loss=0.0,
                probability_assignment=0.0,
                expected_return=0.0,
                edge_ratio=0.0,
                var_95=0.0,
                margin_required=0.0,
            ),
            details={},
        )

    def _create_switch_recommendation(
        self, switch_analysis: "SwitchAnalysis", current_price: float, buying_power: float
    ) -> Recommendation:
        """Create a SWITCH recommendation based on analysis."""
        # Create risk metrics from the new position
        new_pos = switch_analysis.new_position

        return Recommendation(
            action="ADJUST",  # Use standard action
            rationale=switch_analysis.rationale,
            confidence=switch_analysis.confidence,
            risk=RiskMetrics(
                max_loss=new_pos.max_risk,
                probability_assignment=new_pos.probability_itm,
                expected_return=new_pos.annualized_return,
                edge_ratio=(
                    new_pos.daily_expected_return / (new_pos.max_risk / buying_power)
                    if new_pos.max_risk > 0
                    else 0
                ),
                var_95=new_pos.max_risk * 0.05,  # Simplified VaR
                margin_required=new_pos.strike * 100 * 0.2,  # Basic margin calc
            ),
            details={
                "action_type": "roll_position",
                "current_strike": switch_analysis.current_position.strike,
                "current_expiry": switch_analysis.current_position.expiry_days,
                "new_strike": new_pos.strike,
                "new_expiry": new_pos.expiry_days,
                "strike": new_pos.strike,  # For compatibility
                "contracts": 1,  # Will be calculated later
                "switch_benefit": switch_analysis.switch_benefit,
                "switch_cost": switch_analysis.total_switch_cost,
                "improvement_pct": switch_analysis.switch_benefit_pct,
            },
        )

    def _get_position_dte(self, position: Position) -> int:
        """Get days to expiry for a position."""
        if not position.expiration:
            return 0

        # Calculate days from now to expiration
        from datetime import datetime

        now = datetime.now().date()
        days_to_expiry = (position.expiration - now).days

        return max(0, days_to_expiry)
