"""
Seasonality and pattern detection for Unity trading.
Identifies recurring patterns to optimize entry timing.
"""

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config.loader import get_config

from ..utils import get_logger, timed_operation

logger = get_logger(__name__)


class SeasonalPattern(NamedTuple):
    """Identified seasonal pattern."""

    pattern_type: str
    period: str  # 'daily', 'weekly', 'monthly', 'quarterly', 'annual'
    strength: float  # 0-1, pattern reliability
    effect_size: float  # Average impact when pattern occurs
    best_action: str  # Recommended strategy adjustment
    confidence: float


@dataclass
class PatternMetrics:
    """Metrics for a specific pattern."""

    win_rate: float
    avg_return: float
    volatility: float
    sample_size: int
    p_value: float  # Statistical significance


class SeasonalityDetector:
    """Detects and analyzes seasonal patterns in trading data."""

    def __init__(self, symbol: str = None, min_samples: int = 10):
        if symbol is None:
            config = get_config()
            symbol = config.unity.ticker
        self.symbol = symbol
        self.min_samples = min_samples
        self.patterns: Dict[str, SeasonalPattern] = {}

    @timed_operation(threshold_ms=100)
    def analyze_seasonality(
        self, historical_data: pd.DataFrame, min_years: int = 2
    ) -> List[SeasonalPattern]:
        """
        Analyze all types of seasonality patterns.

        Args:
            historical_data: DataFrame with date index and returns
            min_years: Minimum years of data required

        Returns:
            List of detected seasonal patterns
        """
        patterns = []

        # Ensure we have enough data
        years_of_data = (historical_data.index[-1] - historical_data.index[0]).days / 365
        if years_of_data < min_years:
            logger.warning(f"Only {years_of_data:.1f} years of data, need {min_years}")
            return patterns

        # 1. Day of week effects
        dow_pattern = self._analyze_day_of_week(historical_data)
        if dow_pattern:
            patterns.append(dow_pattern)

        # 2. Monthly patterns
        monthly_patterns = self._analyze_monthly_patterns(historical_data)
        patterns.extend(monthly_patterns)

        # 3. Quarterly patterns (earnings cycle)
        quarterly_pattern = self._analyze_quarterly_patterns(historical_data)
        if quarterly_pattern:
            patterns.append(quarterly_pattern)

        # 4. Annual patterns
        annual_pattern = self._analyze_annual_patterns(historical_data)
        if annual_pattern:
            patterns.append(annual_pattern)

        # 5. Options expiration patterns
        opex_pattern = self._analyze_opex_patterns(historical_data)
        if opex_pattern:
            patterns.append(opex_pattern)

        # 6. Gaming industry specific patterns
        gaming_patterns = self._analyze_gaming_seasonality(historical_data)
        patterns.extend(gaming_patterns)

        # Store detected patterns
        for pattern in patterns:
            self.patterns[pattern.pattern_type] = pattern

        # Log summary
        logger.info(
            f"Detected {len(patterns)} seasonal patterns",
            extra={
                "patterns": [p.pattern_type for p in patterns],
                "max_strength": max(p.strength for p in patterns) if patterns else 0,
            },
        )

        return patterns

    def _analyze_day_of_week(self, data: pd.DataFrame) -> Optional[SeasonalPattern]:
        """Analyze day of week effects."""
        if "returns" not in data.columns:
            return None

        # Group by day of week
        data["dow"] = data.index.dayofweek
        dow_stats = data.groupby("dow")["returns"].agg(["mean", "std", "count"])

        # Test for significance
        groups = [data[data["dow"] == i]["returns"].values for i in range(5)]  # Mon-Fri
        f_stat, p_value = stats.f_oneway(*[g for g in groups if len(g) > self.min_samples])

        if p_value < 0.05:  # Significant day of week effect
            # Find best and worst days
            best_day = dow_stats["mean"].idxmax()
            worst_day = dow_stats["mean"].idxmin()

            effect_size = dow_stats["mean"].std() * np.sqrt(252)  # Annualized

            pattern = SeasonalPattern(
                pattern_type="day_of_week",
                period="weekly",
                strength=1 - p_value,
                effect_size=effect_size,
                best_action=f"Prefer {calendar.day_name[best_day]} entries, avoid {calendar.day_name[worst_day]}",
                confidence=min(1.0, dow_stats["count"].min() / 50),
            )

            logger.info(
                "Day of week pattern detected",
                extra={
                    "best_day": calendar.day_name[best_day],
                    "worst_day": calendar.day_name[worst_day],
                    "p_value": p_value,
                },
            )

            return pattern

        return None

    def _analyze_monthly_patterns(self, data: pd.DataFrame) -> List[SeasonalPattern]:
        """Analyze monthly patterns (turn of month, mid-month, etc)."""
        patterns = []

        # Turn of month effect (last 2 days + first 3 days)
        data["day_of_month"] = data.index.day
        data["is_turn_of_month"] = (data["day_of_month"] <= 3) | (
            data["day_of_month"] >= data.index.to_series().dt.days_in_month - 1
        )

        tom_returns = data[data["is_turn_of_month"]]["returns"]
        other_returns = data[~data["is_turn_of_month"]]["returns"]

        if len(tom_returns) > self.min_samples and len(other_returns) > self.min_samples:
            t_stat, p_value = stats.ttest_ind(tom_returns, other_returns)

            if p_value < 0.05:
                effect_size = (tom_returns.mean() - other_returns.mean()) * 252

                pattern = SeasonalPattern(
                    pattern_type="turn_of_month",
                    period="monthly",
                    strength=1 - p_value,
                    effect_size=effect_size,
                    best_action=(
                        "Consider positions around month boundaries"
                        if effect_size > 0
                        else "Avoid month boundaries"
                    ),
                    confidence=min(1.0, len(tom_returns) / 100),
                )
                patterns.append(pattern)

        # Mid-month effect (days 10-20)
        data["is_mid_month"] = (data["day_of_month"] >= 10) & (data["day_of_month"] <= 20)

        mid_returns = data[data["is_mid_month"]]["returns"]
        other_returns = data[~data["is_mid_month"]]["returns"]

        if len(mid_returns) > self.min_samples:
            t_stat, p_value = stats.ttest_ind(mid_returns, other_returns)

            if p_value < 0.05:
                effect_size = (mid_returns.mean() - other_returns.mean()) * 252

                pattern = SeasonalPattern(
                    pattern_type="mid_month",
                    period="monthly",
                    strength=1 - p_value,
                    effect_size=effect_size,
                    best_action=(
                        "Mid-month typically favorable" if effect_size > 0 else "Avoid mid-month"
                    ),
                    confidence=min(1.0, len(mid_returns) / 100),
                )
                patterns.append(pattern)

        return patterns

    def _analyze_quarterly_patterns(self, data: pd.DataFrame) -> Optional[SeasonalPattern]:
        """Analyze quarterly patterns (earnings seasons)."""
        # Unity typically reports in early Feb, May, Aug, Nov
        data["quarter"] = data.index.quarter
        data["month"] = data.index.month

        # Earnings months
        earnings_months = [2, 5, 8, 11]
        data["is_earnings_month"] = data["month"].isin(earnings_months)

        earnings_returns = data[data["is_earnings_month"]]["returns"]
        other_returns = data[~data["is_earnings_month"]]["returns"]

        if len(earnings_returns) > self.min_samples:
            # Volatility comparison
            earnings_vol = earnings_returns.std() * np.sqrt(252)
            other_vol = other_returns.std() * np.sqrt(252)

            # F-test for variance
            f_stat = (earnings_vol**2) / (other_vol**2)
            p_value = stats.f.sf(f_stat, len(earnings_returns) - 1, len(other_returns) - 1) * 2

            if p_value < 0.05 or earnings_vol > other_vol * 1.3:
                pattern = SeasonalPattern(
                    pattern_type="earnings_cycle",
                    period="quarterly",
                    strength=min(1.0, earnings_vol / other_vol - 1),
                    effect_size=earnings_vol - other_vol,
                    best_action="Reduce size or avoid new positions in earnings months",
                    confidence=min(1.0, len(earnings_returns) / 50),
                )

                logger.info(
                    "Earnings cycle pattern detected",
                    extra={
                        "earnings_vol": earnings_vol,
                        "other_vol": other_vol,
                        "vol_ratio": earnings_vol / other_vol,
                    },
                )

                return pattern

        return None

    def _analyze_annual_patterns(self, data: pd.DataFrame) -> Optional[SeasonalPattern]:
        """Analyze annual patterns (tax loss, year-end, etc)."""
        data["month"] = data.index.month

        # Year-end effect (Nov-Dec)
        year_end_returns = data[data["month"].isin([11, 12])]["returns"]
        other_returns = data[~data["month"].isin([11, 12])]["returns"]

        if len(year_end_returns) > self.min_samples * 2:
            t_stat, p_value = stats.ttest_ind(year_end_returns, other_returns)

            if p_value < 0.05:
                effect_size = (year_end_returns.mean() - other_returns.mean()) * 252

                pattern = SeasonalPattern(
                    pattern_type="year_end_effect",
                    period="annual",
                    strength=1 - p_value,
                    effect_size=effect_size,
                    best_action=(
                        "Tax loss selling creates volatility"
                        if effect_size < 0
                        else "Year-end rally opportunity"
                    ),
                    confidence=min(1.0, len(year_end_returns) / 100),
                )

                return pattern

        return None

    def _analyze_opex_patterns(self, data: pd.DataFrame) -> Optional[SeasonalPattern]:
        """Analyze options expiration patterns."""
        # Third Friday of each month
        data["is_opex_week"] = self._mark_opex_weeks(data.index)

        opex_returns = data[data["is_opex_week"]]["returns"]
        other_returns = data[~data["is_opex_week"]]["returns"]

        if len(opex_returns) > self.min_samples:
            # Test for lower volatility (pinning effect)
            opex_vol = opex_returns.std()
            other_vol = other_returns.std()

            f_stat = (other_vol**2) / (opex_vol**2)
            p_value = stats.f.sf(f_stat, len(other_returns) - 1, len(opex_returns) - 1)

            if p_value < 0.05 and opex_vol < other_vol * 0.8:
                pattern = SeasonalPattern(
                    pattern_type="opex_pinning",
                    period="monthly",
                    strength=1 - p_value,
                    effect_size=(other_vol - opex_vol) * np.sqrt(252),
                    best_action="Sell options during OpEx week for pinning effect",
                    confidence=min(1.0, len(opex_returns) / 50),
                )

                return pattern

        return None

    def _analyze_gaming_seasonality(self, data: pd.DataFrame) -> List[SeasonalPattern]:
        """Analyze gaming industry specific patterns."""
        patterns = []
        data["month"] = data.index.month
        data["quarter"] = data.index.quarter

        # Q4 holiday season (Oct-Dec)
        q4_returns = data[data["quarter"] == 4]["returns"]
        other_returns = data[data["quarter"] != 4]["returns"]

        if len(q4_returns) > self.min_samples * 3:
            t_stat, p_value = stats.ttest_ind(q4_returns, other_returns)

            if p_value < 0.10:  # Lower threshold for industry pattern
                effect_size = (q4_returns.mean() - other_returns.mean()) * 252

                pattern = SeasonalPattern(
                    pattern_type="gaming_q4_strength",
                    period="annual",
                    strength=1 - p_value,
                    effect_size=effect_size,
                    best_action=(
                        "Q4 typically strong for gaming stocks"
                        if effect_size > 0
                        else "Q4 shows weakness"
                    ),
                    confidence=min(1.0, len(q4_returns) / 50),
                )
                patterns.append(pattern)

        # Summer slump (Jun-Aug)
        summer_returns = data[data["month"].isin([6, 7, 8])]["returns"]
        other_returns = data[~data["month"].isin([6, 7, 8])]["returns"]

        if len(summer_returns) > self.min_samples * 3:
            t_stat, p_value = stats.ttest_ind(summer_returns, other_returns)

            if p_value < 0.10:
                effect_size = (summer_returns.mean() - other_returns.mean()) * 252

                pattern = SeasonalPattern(
                    pattern_type="gaming_summer_slump",
                    period="annual",
                    strength=1 - p_value,
                    effect_size=effect_size,
                    best_action=(
                        "Summer typically weak for gaming"
                        if effect_size < 0
                        else "Summer strength unusual"
                    ),
                    confidence=min(1.0, len(summer_returns) / 50),
                )
                patterns.append(pattern)

        return patterns

    def _mark_opex_weeks(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Mark options expiration weeks (week containing 3rd Friday)."""
        is_opex = pd.Series(False, index=dates)

        for date in dates:
            # Find third Friday of the month
            first_day = date.replace(day=1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)

            # Check if date is in same week as third Friday
            if abs((date - third_friday).days) <= 3:
                is_opex[date] = True

        return is_opex

    def apply_seasonal_adjustments(
        self, base_params: Dict[str, float], current_date: datetime
    ) -> Dict[str, float]:
        """
        Apply seasonal adjustments to strategy parameters.

        Args:
            base_params: Base strategy parameters
            current_date: Current date for seasonality check

        Returns:
            Adjusted parameters
        """
        adjusted = base_params.copy()
        adjustments_applied = []

        # Check each pattern
        for pattern_type, pattern in self.patterns.items():
            applies = False

            if pattern.period == "weekly":
                if "day_of_week" in pattern_type:
                    current_dow = current_date.weekday()
                    # Apply adjustment based on day
                    applies = True

            elif pattern.period == "monthly":
                if "turn_of_month" in pattern_type:
                    day = current_date.day
                    days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]
                    applies = day <= 3 or day >= days_in_month - 1

                elif "opex" in pattern_type:
                    # Check if opex week
                    applies = self._is_opex_week(current_date)

            elif pattern.period == "quarterly":
                if "earnings" in pattern_type:
                    applies = current_date.month in [2, 5, 8, 11]

            elif pattern.period == "annual":
                if "year_end" in pattern_type:
                    applies = current_date.month in [11, 12]
                elif "q4" in pattern_type:
                    applies = current_date.month in [10, 11, 12]
                elif "summer" in pattern_type:
                    applies = current_date.month in [6, 7, 8]

            if applies and pattern.strength > 0.5:  # Only apply strong patterns
                # Adjust parameters based on pattern
                if pattern.effect_size < 0:  # Negative effect
                    adjusted["kelly"] *= 1 - pattern.strength * 0.3
                    adjusted["delta"] -= pattern.strength * 0.05
                else:  # Positive effect
                    adjusted["kelly"] *= 1 + pattern.strength * 0.2
                    adjusted["delta"] += pattern.strength * 0.03

                adjustments_applied.append(pattern_type)

        if adjustments_applied:
            logger.info(
                "Applied seasonal adjustments",
                extra={"patterns": adjustments_applied, "date": current_date.strftime("%Y-%m-%d")},
            )

        return adjusted

    def _is_opex_week(self, date: datetime) -> bool:
        """Check if date is in options expiration week."""
        first_day = date.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)

        return abs((date - third_friday).days) <= 3

    def generate_seasonality_report(self) -> List[str]:
        """Generate human-readable seasonality report."""
        report = ["=== SEASONALITY ANALYSIS ===", ""]

        if not self.patterns:
            report.append("No significant seasonal patterns detected")
            return report

        # Sort by strength
        sorted_patterns = sorted(self.patterns.values(), key=lambda x: x.strength, reverse=True)

        for pattern in sorted_patterns:
            report.append(f"{pattern.pattern_type.upper()}")
            report.append(f"  Period: {pattern.period}")
            report.append(f"  Strength: {pattern.strength:.1%}")
            report.append(f"  Effect size: {pattern.effect_size:+.1%} annualized")
            report.append(f"  Action: {pattern.best_action}")
            report.append(f"  Confidence: {pattern.confidence:.1%}")
            report.append("")

        # Current applicable patterns
        today = datetime.now()
        current_patterns = []

        for pattern_type, pattern in self.patterns.items():
            test_params = {"kelly": 0.5, "delta": 0.25}
            adjusted = self.apply_seasonal_adjustments(test_params, today)

            if adjusted != test_params:
                current_patterns.append(pattern_type)

        if current_patterns:
            report.append(f"CURRENTLY ACTIVE PATTERNS: {', '.join(current_patterns)}")
        else:
            report.append("No patterns currently active")

        return report
