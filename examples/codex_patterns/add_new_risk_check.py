"""Example: How to add a new risk check to the system.

This example demonstrates the complete pattern for adding a new risk check,
following all architectural conventions and patterns.

SCENARIO: Adding a check for Unity's earnings announcement risk
"""

# Step 1: Add the check to risk/limits.py
# =====================================

from datetime import datetime, timedelta
from typing import Tuple

from ..utils.logging import get_logger
from ..utils.validate import die

logger = get_logger(__name__)


class RiskLimits:
    """Extended with new earnings risk check."""

    # === BEGIN new_risk_check ===
    def check_earnings_risk(
        self, ticker: str, days_to_expiry: int, earnings_date: Optional[datetime] = None
    ) -> Tuple[bool, str, float]:
        """
        Check if option expiry is too close to earnings.

        CODEX PATTERN:
        1. Validate inputs
        2. Apply business logic
        3. Return (passed, message, confidence)
        4. Log decisions with context

        Parameters
        ----------
        ticker : str
            Stock ticker (typically config.unity.ticker)
        days_to_expiry : int
            Days until option expiration
        earnings_date : datetime, optional
            Known earnings date

        Returns
        -------
        Tuple[bool, str, float]
            (check_passed, reason_message, confidence_score)
        """
        # Validate inputs
        ticker = die(ticker, "Ticker required for earnings check")

        # Default confidence
        confidence = 0.95

        # If no earnings date provided, try to estimate
        if earnings_date is None:
            # Unity typically announces ~45 days after quarter end
            # This is simplified - real implementation would check calendar
            logger.info("No earnings date provided, using estimate", extra={"ticker": ticker})
            confidence *= 0.8  # Lower confidence for estimate

            # Assume we're 30 days from next earnings (example)
            days_to_earnings = 30
        else:
            days_to_earnings = (earnings_date - datetime.now()).days

        # Check if expiry is within danger zone
        EARNINGS_BUFFER_DAYS = 7  # Avoid expiries within 7 days of earnings

        if days_to_expiry > days_to_earnings - EARNINGS_BUFFER_DAYS:
            reason = (
                f"Option expires too close to earnings "
                f"({days_to_earnings} days away, need {EARNINGS_BUFFER_DAYS}+ buffer)"
            )
            logger.warning(
                "Earnings risk detected",
                extra={
                    "ticker": ticker,
                    "days_to_expiry": days_to_expiry,
                    "days_to_earnings": days_to_earnings,
                    "buffer_required": EARNINGS_BUFFER_DAYS,
                },
            )
            return False, reason, confidence

        return True, "", confidence

    # === END new_risk_check ===


# Step 2: Add configuration to schema.py
# =====================================

from pydantic import BaseModel, Field


class EarningsRiskConfig(BaseModel):
    """Configuration for earnings-related risk checks."""

    enabled: bool = Field(True, description="Enable earnings proximity risk check")

    buffer_days: int = Field(
        7, ge=0, le=30, description="Minimum days between option expiry and earnings"
    )

    skip_if_unknown: bool = Field(
        False, description="Skip check if earnings date unknown (vs estimate)"
    )


class RiskConfig(BaseModel):
    """Extended risk configuration."""

    # ... existing fields ...

    earnings_risk: EarningsRiskConfig = Field(
        default_factory=EarningsRiskConfig, description="Earnings announcement risk settings"
    )


# Step 3: Integrate into risk analytics
# ====================================


class RiskAnalytics:
    """Extended to use new risk check."""

    def check_all_limits(
        self,
        position_value: float,
        portfolio_value: float,
        ticker: str,
        days_to_expiry: int,
        **kwargs,
    ) -> Tuple[bool, List[str], float]:
        """
        Run all risk checks including new earnings check.

        CODEX PATTERN:
        1. Run each check independently
        2. Collect all failures
        3. Combine confidence scores
        4. Return aggregate result
        """
        checks = []
        failures = []
        total_confidence = 1.0

        # Existing checks
        passed, msg, conf = self.limits.check_position_size(position_value, portfolio_value)
        checks.append((passed, msg, conf))

        # NEW: Earnings risk check
        if self.config.risk.earnings_risk.enabled:
            earnings_date = kwargs.get("earnings_date")
            passed, msg, conf = self.limits.check_earnings_risk(
                ticker, days_to_expiry, earnings_date
            )
            checks.append((passed, msg, conf))

        # Aggregate results
        for passed, msg, conf in checks:
            total_confidence *= conf
            if not passed:
                failures.append(msg)

        all_passed = len(failures) == 0

        logger.info(
            "Risk checks completed",
            extra={
                "checks_run": len(checks),
                "passed": all_passed,
                "failures": len(failures),
                "confidence": total_confidence,
            },
        )

        return all_passed, failures, total_confidence


# Step 4: Add tests
# ================

from datetime import datetime, timedelta

import pytest


class TestEarningsRisk:
    """Test new earnings risk check."""

    def test_earnings_risk_detection(self):
        """Test that earnings risk is properly detected."""
        limits = RiskLimits()

        # Earnings in 5 days, option expires in 10 days = TOO CLOSE
        earnings_date = datetime.now() + timedelta(days=5)

        passed, msg, conf = limits.check_earnings_risk(
            ticker="U", days_to_expiry=10, earnings_date=earnings_date
        )

        assert not passed  # Should fail
        assert "too close to earnings" in msg
        assert conf > 0.9  # High confidence with known date

    def test_earnings_risk_safe(self):
        """Test safe scenario with enough buffer."""
        limits = RiskLimits()

        # Earnings in 30 days, option expires in 15 days = SAFE
        earnings_date = datetime.now() + timedelta(days=30)

        passed, msg, conf = limits.check_earnings_risk(
            ticker="U", days_to_expiry=15, earnings_date=earnings_date
        )

        assert passed  # Should pass
        assert msg == ""
        assert conf > 0.9

    @pytest.mark.parametrize(
        "days_to_expiry,days_to_earnings,should_pass",
        [
            (45, 50, True),  # Expires before earnings - safe
            (45, 40, False),  # Expires after earnings - risky
            (7, 14, True),  # Short expiry before earnings - safe
            (14, 7, False),  # Expires right after earnings - risky
        ],
    )
    def test_earnings_scenarios(self, days_to_expiry, days_to_earnings, should_pass):
        """Test various earnings/expiry scenarios."""
        limits = RiskLimits()

        earnings_date = datetime.now() + timedelta(days=days_to_earnings)
        passed, _, _ = limits.check_earnings_risk(
            ticker="U", days_to_expiry=days_to_expiry, earnings_date=earnings_date
        )

        assert passed == should_pass


# Step 5: Update configuration
# ===========================

# In config.yaml:
"""
risk:
  earnings_risk:
    enabled: true
    buffer_days: 7
    skip_if_unknown: false
"""


# Step 6: Usage in main flow
# =========================


def advise_position(self, market_snapshot: MarketSnapshot) -> Recommendation:
    """Example of using new risk check in main flow."""
    # ... existing code ...

    # Get earnings date if available
    earnings_date = market_snapshot.get("next_earnings_date")

    # Run risk checks including new one
    checks_passed, failures, confidence = self.risk_analytics.check_all_limits(
        position_value=calculated_position_value,
        portfolio_value=account.cash_balance,
        ticker=market_snapshot["ticker"],
        days_to_expiry=target_dte,
        earnings_date=earnings_date,  # Pass to new check
    )

    if not checks_passed:
        return self._create_hold_recommendation(f"Risk checks failed: {'; '.join(failures)}")

    # ... continue with recommendation ...


# Summary of Changes Required:
# 1. ✅ Add check method to RiskLimits class
# 2. ✅ Add configuration schema
# 3. ✅ Integrate into check_all_limits flow
# 4. ✅ Write comprehensive tests
# 5. ✅ Update config.yaml
# 6. ✅ Use in main recommendation flow
#
# This pattern ensures:
# - Consistent error handling
# - Proper confidence tracking
# - Comprehensive logging
# - Easy configuration
# - Thorough testing
