"""
Risk limits and circuit breakers for autonomous safety.
Prevents the system from making dangerous trades.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple, Optional

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
    """Hard limits for safe autonomous operation."""

    # Position limits
    max_position_pct: float = 0.20  # Max 20% of portfolio in one position
    max_contracts: int = 10  # Max contracts per trade
    min_portfolio_value: float = 10000  # Stop trading below this

    # Market condition limits
    max_volatility: float = 1.5  # 150% annual vol
    max_gap_percent: float = 0.10  # 10% gap
    min_volume_ratio: float = 0.5  # Half normal volume

    # Loss limits
    max_daily_loss_pct: float = 0.02  # 2% daily loss
    max_weekly_loss_pct: float = 0.05  # 5% weekly loss
    max_consecutive_losses: int = 3  # Stop after 3 losses

    # Confidence limits
    min_confidence: float = 0.30  # Don't trade below 30% confidence
    max_warnings: int = 3  # Don't trade with >3 warnings

    # Time limits
    blackout_hours: List[int] = None  # Hours to avoid (e.g., [0, 1, 2])
    min_days_between_trades: int = 0  # Cooling off period


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
