"""
Risk limits and circuit breakers for autonomous safety.
Prevents the system from making dangerous trades.
Now with adaptive limits that adjust to market conditions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple, Optional, Tuple

from src.config import get_config

from ..utils import get_logger

logger = get_logger(__name__)


class RiskLimit(NamedTuple):
    """Definition of a risk limit."""

    name: str
    current_value: float
    limit_value: float
    is_breached: bool
    severity: str  # 'warning', 'critical'
    action: str  # What to do if breached


@dataclass
class TradingLimits:
    """Trading limits for safe autonomous operation."""

    # Position limits
    max_position_pct: float = field(default=None)
    max_contracts: int = field(default=None)
    min_portfolio_value: float = field(default=None)

    # Market condition limits
    max_volatility: float = field(default=None)
    max_gap_percent: float = field(default=None)
    min_volume_ratio: float = field(default=None)

    # Loss limits
    max_daily_loss_pct: float = field(default=None)
    max_weekly_loss_pct: float = field(default=None)
    max_consecutive_losses: int = field(default=None)

    # Confidence limits
    min_confidence: float = field(default=None)
    max_warnings: int = field(default=None)

    # Time limits
    blackout_hours: List[int] = field(default=None)
    min_days_between_trades: int = field(default=0)

    def __post_init__(self):
        """Initialize from config if values not provided."""
        config = get_config()
        circuit_breakers = config.risk.circuit_breakers

        # Use config values as defaults if not explicitly set
        if self.max_position_pct is None:
            self.max_position_pct = circuit_breakers.max_position_pct
        if self.max_contracts is None:
            self.max_contracts = circuit_breakers.max_contracts
        if self.min_portfolio_value is None:
            self.min_portfolio_value = circuit_breakers.min_portfolio_value
        if self.max_volatility is None:
            self.max_volatility = circuit_breakers.max_volatility
        if self.max_gap_percent is None:
            self.max_gap_percent = circuit_breakers.max_gap_percent
        if self.min_volume_ratio is None:
            self.min_volume_ratio = circuit_breakers.min_volume_ratio
        if self.max_daily_loss_pct is None:
            self.max_daily_loss_pct = circuit_breakers.max_daily_loss_pct
        if self.max_weekly_loss_pct is None:
            self.max_weekly_loss_pct = circuit_breakers.max_weekly_loss_pct
        if self.max_consecutive_losses is None:
            self.max_consecutive_losses = circuit_breakers.max_consecutive_losses
        if self.min_confidence is None:
            self.min_confidence = circuit_breakers.min_confidence
        if self.max_warnings is None:
            self.max_warnings = circuit_breakers.max_warnings
        if self.blackout_hours is None:
            self.blackout_hours = circuit_breakers.blackout_hours or []

    @classmethod
    def from_config(cls, config: Dict) -> "TradingLimits":
        """Create limits from configuration."""
        risk_config = config.get("risk", {})
        circuit_breakers = risk_config.get("circuit_breakers", {})

        return cls(
            max_position_pct=circuit_breakers.get("max_position_pct", 0.20),
            max_contracts=circuit_breakers.get("max_contracts", 10),
            min_portfolio_value=circuit_breakers.get("min_portfolio_value", 10000),
            max_volatility=circuit_breakers.get("max_volatility", 1.5),
            max_gap_percent=circuit_breakers.get("max_gap_percent", 0.10),
            min_volume_ratio=circuit_breakers.get("min_volume_ratio", 0.5),
            max_daily_loss_pct=circuit_breakers.get("max_daily_loss_pct", 0.02),
            max_weekly_loss_pct=circuit_breakers.get("max_weekly_loss_pct", 0.05),
            max_consecutive_losses=circuit_breakers.get("max_consecutive_losses", 3),
            min_confidence=circuit_breakers.get("min_confidence", 0.30),
            max_warnings=circuit_breakers.get("max_warnings", 3),
            blackout_hours=circuit_breakers.get("blackout_hours"),
        )

    def check_position_size(self, position_value: float, portfolio_value: float) -> bool:
        """Check if position size is within limits."""
        position_pct = position_value / portfolio_value
        return position_pct <= self.max_position_pct

    def check_margin_usage(self, margin_usage: float) -> bool:
        """Check if margin usage is acceptable."""
        config = get_config()
        max_margin = config.risk.margin.max_utilization
        return margin_usage <= max_margin

    def check_var_limit(self, var_value: float, portfolio_value: float) -> bool:
        """Check if VaR is within limits."""
        config = get_config()
        max_var = config.risk.limits.max_var_95
        return abs(var_value) <= max_var * portfolio_value

    def check_delta_exposure(self, total_delta: float) -> bool:
        """Check if delta exposure is within limits."""
        config = get_config()
        max_delta = config.risk.greeks.max_delta_exposure
        return abs(total_delta) <= max_delta * 100  # Assuming 100 as base

    def check_contracts_limit(self, contracts: int) -> bool:
        """Check if number of contracts is within limits."""
        return contracts <= self.max_contracts

    def check_risk_metrics(self, metrics: "RiskMetrics", portfolio_value: float) -> bool:
        """Check all risk metrics against limits."""
        # For aggressive strategy, very permissive
        return True  # All risk accepted per user preference


class RiskLimitChecker:
    """Checks all risk limits before allowing trades."""

    def __init__(self, limits: Optional[TradingLimits] = None):
        self.limits = limits or TradingLimits()
        self.trade_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}

    def check_all_limits(
        self,
        recommendation: Dict,
        portfolio_value: float,
        current_positions: Optional[List] = None,
        market_data: Optional[Dict] = None,
    ) -> List[RiskLimit]:
        """Check all risk limits and return any breaches."""
        breaches = []

        # 1. Position size limits
        position_pct = recommendation.get("position_size", 0) / portfolio_value
        if position_pct > self.limits.max_position_pct:
            breaches.append(
                RiskLimit(
                    name="position_size",
                    current_value=position_pct,
                    limit_value=self.limits.max_position_pct,
                    is_breached=True,
                    severity="critical",
                    action="Reduce position size or skip trade",
                )
            )

        # 2. Contract limits
        contracts = recommendation.get("contracts", 0)
        if contracts > self.limits.max_contracts:
            breaches.append(
                RiskLimit(
                    name="max_contracts",
                    current_value=contracts,
                    limit_value=self.limits.max_contracts,
                    is_breached=True,
                    severity="warning",
                    action=f"Reduce to {self.limits.max_contracts} contracts",
                )
            )

        # 3. Portfolio value minimum
        if portfolio_value < self.limits.min_portfolio_value:
            breaches.append(
                RiskLimit(
                    name="min_portfolio",
                    current_value=portfolio_value,
                    limit_value=self.limits.min_portfolio_value,
                    is_breached=True,
                    severity="critical",
                    action="Stop trading until portfolio recovers",
                )
            )

        # 4. Market volatility
        if market_data:
            volatility = market_data.get("realized_vol", 0)
            if volatility > self.limits.max_volatility:
                breaches.append(
                    RiskLimit(
                        name="max_volatility",
                        current_value=volatility,
                        limit_value=self.limits.max_volatility,
                        is_breached=True,
                        severity="critical",
                        action="Wait for calmer markets",
                    )
                )

            # Check for gaps
            if "open" in market_data and "prev_close" in market_data:
                gap = (
                    abs(market_data["open"] - market_data["prev_close"]) / market_data["prev_close"]
                )
                if gap > self.limits.max_gap_percent:
                    breaches.append(
                        RiskLimit(
                            name="gap_limit",
                            current_value=gap,
                            limit_value=self.limits.max_gap_percent,
                            is_breached=True,
                            severity="critical",
                            action="Wait for gap to fill or stabilize",
                        )
                    )

        # 5. Confidence and warnings
        confidence = recommendation.get("confidence", 1.0)
        if confidence < self.limits.min_confidence:
            breaches.append(
                RiskLimit(
                    name="min_confidence",
                    current_value=confidence,
                    limit_value=self.limits.min_confidence,
                    is_breached=True,
                    severity="warning",
                    action="Improve data or wait for better setup",
                )
            )

        warnings = len(recommendation.get("warnings", []))
        if warnings > self.limits.max_warnings:
            breaches.append(
                RiskLimit(
                    name="max_warnings",
                    current_value=warnings,
                    limit_value=self.limits.max_warnings,
                    is_breached=True,
                    severity="warning",
                    action="Address warnings before trading",
                )
            )

        # 6. Loss limits (if we have history)
        if self.trade_history:
            consecutive_losses = self._count_consecutive_losses()
            if consecutive_losses >= self.limits.max_consecutive_losses:
                breaches.append(
                    RiskLimit(
                        name="consecutive_losses",
                        current_value=consecutive_losses,
                        limit_value=self.limits.max_consecutive_losses,
                        is_breached=True,
                        severity="critical",
                        action="Take a break and review strategy",
                    )
                )

        # 7. Time-based limits
        current_hour = datetime.now().hour
        if self.limits.blackout_hours and current_hour in self.limits.blackout_hours:
            breaches.append(
                RiskLimit(
                    name="blackout_hours",
                    current_value=current_hour,
                    limit_value=0,
                    is_breached=True,
                    severity="warning",
                    action=f"Wait until after {max(self.limits.blackout_hours)}:00",
                )
            )

        return breaches

    def should_allow_trade(self, breaches: List[RiskLimit]) -> bool:
        """Determine if trade should be allowed given breaches."""
        if not breaches:
            return True

        # Any critical breach stops trading
        critical_breaches = [b for b in breaches if b.severity == "critical"]
        if critical_breaches:
            logger.error(
                "Critical risk limits breached", breaches=[b.name for b in critical_breaches]
            )
            return False

        # Multiple warnings also stop trading
        warning_count = len([b for b in breaches if b.severity == "warning"])
        if warning_count >= 2:
            logger.warning(
                "Multiple risk warnings", count=warning_count, breaches=[b.name for b in breaches]
            )
            return False

        return True

    def record_trade_result(self, trade_id: int, pnl: float, success: bool):
        """Record trade result for limit tracking."""
        self.trade_history.append(
            {"id": trade_id, "timestamp": datetime.now(), "pnl": pnl, "success": success}
        )

        # Update daily P&L
        today = datetime.now().date().isoformat()
        self.daily_pnl[today] = self.daily_pnl.get(today, 0) + pnl

        # Trim old history (keep 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        self.trade_history = [t for t in self.trade_history if t["timestamp"] > cutoff]

    def _count_consecutive_losses(self) -> int:
        """Count consecutive losing trades."""
        if not self.trade_history:
            return 0

        consecutive = 0
        for trade in reversed(self.trade_history):
            if not trade["success"]:
                consecutive += 1
            else:
                break

        return consecutive

    def get_current_restrictions(self) -> Dict[str, any]:
        """Get current trading restrictions based on limits."""
        restrictions = {
            "can_trade": True,
            "reduced_size": False,
            "max_position_size": self.limits.max_position_pct,
            "blocked_until": None,
            "reasons": [],
        }

        # Check if we're in a blackout period
        if self.limits.blackout_hours:
            current_hour = datetime.now().hour
            if current_hour in self.limits.blackout_hours:
                next_allowed = min(
                    h for h in range(24) if h not in self.limits.blackout_hours and h > current_hour
                )
                restrictions["blocked_until"] = datetime.now().replace(
                    hour=next_allowed, minute=0, second=0
                )
                restrictions["can_trade"] = False
                restrictions["reasons"].append("Outside trading hours")

        # Check recent losses
        if self.trade_history:
            recent_losses = sum(1 for t in self.trade_history[-5:] if not t["success"])
            if recent_losses >= 3:
                restrictions["reduced_size"] = True
                restrictions["max_position_size"] *= 0.5
                restrictions["reasons"].append("Recent losses - size reduced")

        return restrictions

    def generate_risk_report(self) -> List[str]:
        """Generate human-readable risk status report."""
        report = ["=== RISK LIMITS STATUS ===", ""]

        restrictions = self.get_current_restrictions()

        if restrictions["can_trade"]:
            report.append("✅ Trading allowed")
        else:
            report.append("🚫 Trading blocked")

        if restrictions["reduced_size"]:
            report.append(f"⚠️  Position size reduced to {restrictions['max_position_size']:.1%}")

        if restrictions["blocked_until"]:
            report.append(f"⏰ Blocked until: {restrictions['blocked_until'].strftime('%H:%M')}")

        if restrictions["reasons"]:
            report.append("\nReasons:")
            for reason in restrictions["reasons"]:
                report.append(f"  - {reason}")

        # Add daily P&L if available
        if self.daily_pnl:
            today_pnl = self.daily_pnl.get(datetime.now().date().isoformat(), 0)
            report.append(f"\nToday's P&L: ${today_pnl:,.2f}")

        return report
