"""
from __future__ import annotations

Integrated decision engine that combines all analytics components.
Provides autonomous wheel strategy recommendations.
"""

from datetime import datetime
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from ..config.loader import get_config
from ..models.position import Position
from ..utils import get_logger, timed_operation, with_recovery
from ..utils.recovery import RecoveryStrategy
from .anomaly_detector import AnomalyDetector, MarketAnomaly
from .dynamic_optimizer import DynamicOptimizer, MarketState, OptimizationResult
from .event_analyzer import EventImpactAnalyzer
from .iv_surface import IVMetrics, IVSurfaceAnalyzer
from .seasonality import SeasonalityDetector

logger = get_logger(__name__)


class WheelRecommendation(NamedTuple):
    """Complete wheel strategy recommendation."""

    action: str  # 'SELL_PUT', 'ROLL', 'CLOSE', 'NO_TRADE'
    symbol: str
    strike: Optional[float]
    expiration: Optional[datetime]
    contracts: int

    # Decision factors
    delta_target: float
    dte_target: int
    position_size: float
    kelly_fraction: float

    # Risk metrics
    expected_return: float
    expected_risk: float
    objective_value: float
    max_loss: float

    # Confidence and warnings
    confidence: float
    warnings: List[str]
    adjustments: Dict[str, str]

    # Supporting data
    market_regime: str
    iv_metrics: Optional[IVMetrics]
    anomalies: List[str]
    active_patterns: List[str]


class IntegratedDecisionEngine:
    """
    Integrates all analytics components for autonomous decision making.
    Central brain of the wheel strategy system.
    """

    def __init__(
        self, symbol: str = None, portfolio_value: float = 100000, config: Optional[Dict] = None
    ):
        if symbol is None:
            app_config = get_config()
            symbol = app_config.unity.ticker
        self.symbol = symbol
        self.portfolio_value = portfolio_value
        self.config = config or {}

        # Initialize all components
        self.optimizer = DynamicOptimizer(symbol)
        self.iv_analyzer = IVSurfaceAnalyzer()
        self.event_analyzer = EventImpactAnalyzer(symbol)
        self.anomaly_detector = AnomalyDetector(symbol)
        self.seasonality_detector = SeasonalityDetector(symbol)

        # Decision history for learning
        self.decision_history: List[Dict] = []

    @timed_operation(threshold_ms=200)
    @with_recovery(strategy=RecoveryStrategy.FALLBACK)
    async def get_recommendation(
        self,
        current_prices: Dict[str, float],
        historical_data: pd.DataFrame,
        option_chain: Optional[Dict] = None,
        current_positions: Optional[List[Position]] = None,
        event_calendar: Optional[List[Dict]] = None,
    ) -> WheelRecommendation:
        """
        Generate comprehensive wheel strategy recommendation.

        This is the main entry point that orchestrates all analytics.

        Args:
            current_prices: Current market data
            historical_data: Historical price/volume data
            option_chain: Current option chain
            current_positions: Existing positions
            event_calendar: Upcoming events

        Returns:
            Complete recommendation with all supporting data
        """
        logger.info(
            "Starting integrated decision process",
            symbol=self.symbol,
            portfolio=self.portfolio_value,
        )

        # 1. Calculate market state
        market_state = self._calculate_market_state(current_prices, historical_data)

        # 2. Run dynamic optimization
        returns = historical_data["returns"].values
        optimization = self.optimizer.optimize_parameters(market_state, returns)

        # 3. Analyze IV surface (if options data available)
        iv_metrics = None
        if option_chain:
            iv_metrics = self.iv_analyzer.analyze_iv_surface(option_chain, self.symbol)
            # Update optimization with IV data
            market_state.iv_rank = iv_metrics.iv_rank

        # 4. Check for events
        event_adjustments = {"delta": 0, "dte": 0, "kelly": 1.0}
        if event_calendar:
            self.event_analyzer.update_event_calendar(event_calendar)
            should_adjust, adjustments = self.event_analyzer.should_adjust_for_event(
                optimization.dte_target, iv_metrics.iv_rank if iv_metrics else 50
            )
            if should_adjust:
                event_adjustments = adjustments

        # 5. Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(
            current_prices, historical_data, option_chain
        )

        # 6. Apply seasonality
        patterns = self.seasonality_detector.analyze_seasonality(historical_data)
        seasonal_params = self.seasonality_detector.apply_seasonal_adjustments(
            {
                "delta": optimization.delta_target,
                "dte": optimization.dte_target,
                "kelly": optimization.kelly_fraction,
            },
            datetime.now(),
        )

        # 7. Integrate all factors
        final_params = self._integrate_all_factors(
            optimization, iv_metrics, event_adjustments, anomalies, seasonal_params
        )

        # 8. Determine action
        action = self._determine_action(final_params, current_positions, anomalies)

        # 9. Find best option (if trading)
        strike, expiration = None, None
        if action in ["SELL_PUT", "ROLL"] and option_chain:
            strike, expiration = self._find_best_option(
                option_chain, final_params["delta"], final_params["dte"]
            )

        # 10. Calculate position size
        contracts = self._calculate_position_size(
            final_params["kelly"], strike or current_prices.get("close", 100)
        )

        # 11. Risk calculations
        risk_metrics = self._calculate_risk_metrics(
            strike or current_prices.get("close", 100), final_params, returns
        )

        # 12. Generate warnings
        warnings = self._generate_warnings(anomalies, final_params, optimization)

        # Create recommendation
        recommendation = WheelRecommendation(
            action=action,
            symbol=self.symbol,
            strike=strike,
            expiration=expiration,
            contracts=contracts,
            # Parameters
            delta_target=final_params["delta"],
            dte_target=final_params["dte"],
            position_size=contracts * 100 * (strike or 100),
            kelly_fraction=final_params["kelly"],
            # Risk
            expected_return=risk_metrics["expected_return"],
            expected_risk=risk_metrics["expected_risk"],
            objective_value=optimization.objective_value,
            max_loss=risk_metrics["max_loss"],
            # Meta
            confidence=final_params["confidence"],
            warnings=warnings,
            adjustments=final_params["adjustments_applied"],
            market_regime=self._determine_regime(market_state, iv_metrics),
            iv_metrics=iv_metrics,
            anomalies=[a.anomaly_type.name for a in anomalies],
            active_patterns=[p.pattern_type for p in patterns if p.strength > 0.5],
        )

        # Log decision
        self._log_decision(recommendation)

        return recommendation

    def _calculate_market_state(
        self, current: Dict[str, float], historical: pd.DataFrame
    ) -> MarketState:
        """Calculate comprehensive market state."""
        # Recent returns
        returns = historical["returns"].values
        recent_returns = returns[-20:] if len(returns) >= 20 else returns

        # Volatility
        realized_vol = np.std(recent_returns) * np.sqrt(252)

        # Volatility percentile
        all_vols = []
        for i in range(20, len(returns)):
            vol = np.std(returns[i - 20 : i]) * np.sqrt(252)
            all_vols.append(vol)

        vol_percentile = 0.5
        if all_vols:
            vol_percentile = sum(v < realized_vol for v in all_vols) / len(all_vols)

        # Momentum
        if len(historical) >= 20:
            momentum = (historical["close"].iloc[-1] - historical["close"].iloc[-20]) / historical[
                "close"
            ].iloc[-20]
        else:
            momentum = 0.0

        # Volume
        if "volume" in current and "volume" in historical.columns:
            avg_volume = historical["volume"].rolling(20).mean().iloc[-1]
            volume_ratio = current["volume"] / (avg_volume + 1)
        else:
            volume_ratio = 1.0

        return MarketState(
            realized_volatility=realized_vol,
            volatility_percentile=vol_percentile,
            price_momentum=float(momentum),
            volume_ratio=float(volume_ratio),
            iv_rank=None,  # Set later
            days_to_earnings=None,  # Set later
        )

    def _integrate_all_factors(
        self,
        optimization: OptimizationResult,
        iv_metrics: Optional[IVMetrics],
        event_adjustments: Dict,
        anomalies: List[MarketAnomaly],
        seasonal_params: Dict,
    ) -> Dict[str, any]:
        """Integrate all analytical factors into final parameters."""

        # Start with optimized parameters
        delta = optimization.delta_target
        dte = optimization.dte_target
        kelly = optimization.kelly_fraction
        confidence = optimization.confidence_score

        adjustments_applied = {}

        # Apply IV adjustments
        if iv_metrics:
            if iv_metrics.iv_rank > 80:
                delta *= 1.1  # Can be more aggressive
                kelly *= 1.2
                adjustments_applied["high_iv_rank"] = f"IV rank {iv_metrics.iv_rank:.0f}"
            elif iv_metrics.iv_rank < 20:
                delta *= 0.8  # More conservative
                kelly *= 0.7
                adjustments_applied["low_iv_rank"] = f"IV rank {iv_metrics.iv_rank:.0f}"

        # Apply event adjustments
        if event_adjustments["kelly"] < 1.0:
            delta += event_adjustments.get("delta_adjustment", 0)
            dte = max(21, dte + int(event_adjustments.get("dte_adjustment", 0)))
            kelly *= event_adjustments["size_adjustment"]
            confidence *= event_adjustments["confidence"]
            adjustments_applied["events"] = "Upcoming events detected"

        # Apply anomaly adjustments
        max_severity = max([a.anomaly_type.severity for a in anomalies], default=0)
        if max_severity > 0.7:
            kelly *= 0.3  # Severe reduction
            confidence *= 0.5
            adjustments_applied["severe_anomaly"] = "Multiple anomalies detected"
        elif max_severity > 0.5:
            kelly *= 0.7
            confidence *= 0.8
            adjustments_applied["anomaly"] = "Market anomalies detected"

        # Apply seasonal adjustments
        if seasonal_params != {"delta": delta, "dte": dte, "kelly": kelly}:
            delta = seasonal_params["delta"]
            kelly = seasonal_params["kelly"]
            adjustments_applied["seasonality"] = "Seasonal patterns applied"

        # Bounds checking
        delta = np.clip(delta, 0.10, 0.40)
        dte = np.clip(dte, 21, 49)
        kelly = np.clip(kelly, 0.0, 0.50)
        confidence = np.clip(confidence, 0.0, 1.0)

        return {
            "delta": delta,
            "dte": dte,
            "kelly": kelly,
            "confidence": confidence,
            "adjustments_applied": adjustments_applied,
        }

    def _determine_action(
        self, params: Dict, positions: Optional[List[Position]], anomalies: List[MarketAnomaly]
    ) -> str:
        """Determine recommended action."""

        # Check for no-trade conditions
        if params["confidence"] < 0.3:
            return "NO_TRADE"

        if params["kelly"] == 0:
            return "NO_TRADE"

        # Severe anomalies
        if any(a.anomaly_type.severity > 0.8 for a in anomalies):
            if positions:
                return "CLOSE"  # Close existing positions
            return "NO_TRADE"

        # Check existing positions
        if positions:
            for pos in positions:
                if pos.symbol == self.symbol:
                    # Check if should roll
                    if self._should_roll_position(pos, params):
                        return "ROLL"
                    elif self._should_close_position(pos, anomalies):
                        return "CLOSE"
                    else:
                        return "NO_TRADE"  # Already have position

        # New position
        return "SELL_PUT"

    def _should_roll_position(self, position: Position, params: Dict) -> bool:
        """Determine if position should be rolled."""
        # DTE check
        if position.days_to_expiry <= params["dte"] / 2:
            return True

        # Profit target
        if position.unrealized_pnl_pct >= 0.50:  # 50% of max profit
            return True

        # Delta breach
        if abs(position.delta) > params["delta"] + 0.20:
            return True

        return False

    def _should_close_position(self, position: Position, anomalies: List[MarketAnomaly]) -> bool:
        """Determine if position should be closed."""
        # Severe anomalies
        if any(a.anomaly_type.severity > 0.8 for a in anomalies):
            return True

        # Large loss
        if position.unrealized_pnl_pct < -1.0:  # -100% of credit
            return True

        return False

    def _find_best_option(
        self, chain: Dict, target_delta: float, target_dte: int
    ) -> Tuple[Optional[float], Optional[datetime]]:
        """Find best option matching target parameters."""
        puts = chain.get("puts", [])
        if not puts:
            return None, None

        # Filter by DTE range
        dte_min = target_dte - 7
        dte_max = target_dte + 14

        candidates = []
        for put in puts:
            dte = put.get("dte", 0)
            if dte_min <= dte <= dte_max:
                delta_diff = abs(abs(put.get("delta", 0)) - target_delta)
                if delta_diff < 0.10:  # Within 0.10 delta
                    candidates.append(
                        {
                            "strike": put["strike"],
                            "expiration": put["expiration"],
                            "delta": put["delta"],
                            "dte": dte,
                            "premium": put.get("bid", 0),
                            "score": delta_diff + abs(dte - target_dte) / 100,
                        }
                    )

        if candidates:
            # Sort by score (lower is better)
            best = min(candidates, key=lambda x: x["score"])
            return best["strike"], best["expiration"]

        return None, None

    def _calculate_position_size(self, kelly: float, strike: float) -> int:
        """Calculate number of contracts based on Kelly sizing."""
        if kelly == 0:
            return 0

        # Position value
        position_value = self.portfolio_value * kelly

        # Contracts (each contract = 100 shares)
        contracts = int(position_value / (strike * 100))

        # Apply limits
        max_contracts = self.config.get("max_contracts", 10)

        return min(contracts, max_contracts)

    def _calculate_risk_metrics(
        self, strike: float, params: Dict, returns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        # Expected return (simplified)
        win_rate = 1 - params["delta"]
        premium_pct = params["delta"] * 0.03  # Rough estimate
        expected_return = win_rate * premium_pct

        # Risk (CVaR)
        if len(returns) > 100:
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean(returns[returns <= var_95])
            expected_risk = abs(cvar_95) * params["kelly"]
        else:
            expected_risk = 0.10 * params["kelly"]

        # Max loss
        max_loss = strike * 100 * self._calculate_position_size(params["kelly"], strike)

        return {
            "expected_return": expected_return,
            "expected_risk": expected_risk,
            "max_loss": max_loss,
        }

    def _generate_warnings(
        self, anomalies: List[MarketAnomaly], params: Dict, optimization: OptimizationResult
    ) -> List[str]:
        """Generate actionable warnings."""
        warnings = []

        # Anomaly warnings
        for anomaly in anomalies[:3]:  # Top 3
            warnings.append(
                f"{anomaly.anomaly_type.name}: {anomaly.anomaly_type.recommended_action}"
            )

        # Low confidence
        if params["confidence"] < 0.5:
            warnings.append(f"Low confidence ({params['confidence']:.1%}) in parameters")

        # Negative objective
        if optimization.objective_value < 0:
            warnings.append("Negative expected value - consider avoiding trade")

        # Many adjustments
        if len(params["adjustments_applied"]) > 2:
            warnings.append("Multiple risk factors present - reduced position size")

        return warnings

    def _determine_regime(self, market_state: MarketState, iv_metrics: Optional[IVMetrics]) -> str:
        """Determine overall market regime."""
        regimes = []

        # Volatility regime
        if market_state.volatility_percentile < 0.33:
            regimes.append("low_vol")
        elif market_state.volatility_percentile > 0.67:
            regimes.append("high_vol")
        else:
            regimes.append("normal_vol")

        # IV regime
        if iv_metrics:
            if iv_metrics.regime == "backwardation":
                regimes.append("event_risk")
            if iv_metrics.iv_rank > 80:
                regimes.append("high_iv")
            elif iv_metrics.iv_rank < 20:
                regimes.append("low_iv")

        # Trend
        if market_state.price_momentum > 0.10:
            regimes.append("uptrend")
        elif market_state.price_momentum < -0.10:
            regimes.append("downtrend")

        return "_".join(regimes) if regimes else "normal"

    def _log_decision(self, recommendation: WheelRecommendation) -> None:
        """Log decision for analysis and learning."""
        decision = {
            "timestamp": datetime.now(),
            "action": recommendation.action,
            "confidence": recommendation.confidence,
            "objective_value": recommendation.objective_value,
            "regime": recommendation.market_regime,
            "anomalies": len(recommendation.anomalies),
            "warnings": len(recommendation.warnings),
        }

        self.decision_history.append(decision)

        logger.info(
            "Decision made",
            extra={
                "action": recommendation.action,
                "confidence": recommendation.confidence,
                "objective": recommendation.objective_value,
                "warnings": recommendation.warnings,
            },
        )

    def generate_decision_report(self, recommendation: WheelRecommendation) -> List[str]:
        """Generate comprehensive decision report."""
        report = [
            "=== WHEEL STRATEGY RECOMMENDATION ===",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Symbol: {recommendation.symbol}",
            "",
        ]

        # Action
        action_emoji = {"SELL_PUT": "🟢", "ROLL": "🔄", "CLOSE": "🔴", "NO_TRADE": "⏸️"}

        report.append(
            f"{action_emoji.get(recommendation.action, '❓')} ACTION: {recommendation.action}"
        )

        if recommendation.action in ["SELL_PUT", "ROLL"] and recommendation.strike is not None:
            report.append(f"   Strike: ${recommendation.strike:.2f}")
            if recommendation.expiration:
                report.append(f"   Expiration: {recommendation.expiration.strftime('%Y-%m-%d')}")
            report.append(f"   Contracts: {recommendation.contracts}")
            report.append(f"   Position Size: ${recommendation.position_size:,.0f}")

        report.append("")

        # Parameters
        report.append("📊 OPTIMIZED PARAMETERS:")
        report.append(f"   Delta Target: {recommendation.delta_target:.3f}")
        report.append(f"   DTE Target: {recommendation.dte_target} days")
        report.append(f"   Kelly Fraction: {recommendation.kelly_fraction:.1%}")
        report.append("")

        # Risk Metrics
        report.append("📈 EXPECTED OUTCOMES:")
        report.append(f"   Expected Return: {recommendation.expected_return:.1%}")
        report.append(f"   Expected Risk (CVaR): {recommendation.expected_risk:.1%}")
        report.append(f"   Objective Value: {recommendation.objective_value:.4f}")
        report.append(f"   Max Loss: ${recommendation.max_loss:,.0f}")
        report.append("")

        # Market Analysis
        report.append("🌍 MARKET ANALYSIS:")
        report.append(f"   Regime: {recommendation.market_regime}")

        if recommendation.iv_metrics:
            report.append(f"   IV Rank: {recommendation.iv_metrics.iv_rank:.0f}")
            report.append(f"   IV Regime: {recommendation.iv_metrics.regime}")

        if recommendation.anomalies:
            report.append(f"   Anomalies: {', '.join(recommendation.anomalies)}")

        if recommendation.active_patterns:
            report.append(f"   Patterns: {', '.join(recommendation.active_patterns)}")

        report.append("")

        # Adjustments
        if recommendation.adjustments:
            report.append("🔧 ADJUSTMENTS APPLIED:")
            for reason, detail in recommendation.adjustments.items():
                report.append(f"   - {reason}: {detail}")
            report.append("")

        # Warnings
        if recommendation.warnings:
            report.append("⚠️  WARNINGS:")
            for warning in recommendation.warnings:
                report.append(f"   - {warning}")
            report.append("")

        # Confidence
        conf_emoji = (
            "🟢"
            if recommendation.confidence > 0.7
            else "🟡" if recommendation.confidence > 0.4 else "🔴"
        )
        report.append(f"{conf_emoji} CONFIDENCE: {recommendation.confidence:.1%}")

        return report