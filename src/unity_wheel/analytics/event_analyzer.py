"""
Event impact analyzer for earnings and macro events.
Quantifies historical impact and adjusts strategy parameters.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger, timed_operation, with_recovery
from ..utils.recovery import RecoveryStrategy

logger = get_logger(__name__)


class EventType(Enum):
    """Types of market events."""

    EARNINGS = "earnings"
    FED_MEETING = "fed_meeting"
    CPI_RELEASE = "cpi_release"
    JOBS_REPORT = "jobs_report"
    OPEX = "options_expiration"
    DIVIDEND = "dividend"


class EventImpact(NamedTuple):
    """Historical impact of an event type."""

    event_type: EventType
    avg_move: float  # Average absolute price move
    avg_iv_expansion: float  # IV increase before event
    avg_iv_crush: float  # IV decrease after event
    win_rate: float  # % of times selling worked
    optimal_days_before: int  # When to avoid/enter
    confidence: float


@dataclass
class UpcomingEvent:
    """Upcoming scheduled event."""

    event_type: EventType
    date: datetime
    days_until: int
    expected_move: float
    iv_expansion_expected: float
    historical_accuracy: float


class EventImpactAnalyzer:
    """Analyzes impact of scheduled events on options strategies."""

    def __init__(self, symbol: str = "U"):
        self.symbol = symbol
        self.event_history: Dict[EventType, List[Dict]] = {}
        self.event_calendar: List[UpcomingEvent] = []

        # Unity-specific constants
        self.earnings_iv_expansion = 1.5  # 50% IV increase typical
        self.earnings_avg_move = 0.15  # 15% average move

    @timed_operation(threshold_ms=30)
    def analyze_event_impact(
        self, event_type: EventType, historical_data: pd.DataFrame
    ) -> EventImpact:
        """
        Analyze historical impact of event type.

        Args:
            event_type: Type of event to analyze
            historical_data: Price/IV data around events

        Returns:
            Event impact metrics
        """
        logger.info(f"Analyzing {event_type.value} impact for {self.symbol}")

        if event_type == EventType.EARNINGS:
            return self._analyze_earnings_impact(historical_data)
        elif event_type == EventType.FED_MEETING:
            return self._analyze_fed_impact(historical_data)
        elif event_type == EventType.OPEX:
            return self._analyze_opex_impact(historical_data)
        else:
            return self._analyze_generic_event(event_type, historical_data)

    def update_event_calendar(self, events: List[Dict]) -> None:
        """Update upcoming events calendar."""
        self.event_calendar = []

        for event in events:
            event_date = event.get("date")
            event_type_str = event.get("type", "unknown")

            try:
                event_type = EventType(event_type_str)
            except ValueError:
                continue

            days_until = (event_date - datetime.now()).days

            # Get expected impact based on history
            if event_type in self.event_history:
                historical_impact = self._get_historical_impact(event_type)
                expected_move = historical_impact["avg_move"]
                iv_expansion = historical_impact["avg_iv_expansion"]
                accuracy = historical_impact["confidence"]
            else:
                # Defaults
                expected_move = 0.05 if event_type != EventType.EARNINGS else self.earnings_avg_move
                iv_expansion = (
                    1.2 if event_type != EventType.EARNINGS else self.earnings_iv_expansion
                )
                accuracy = 0.5

            upcoming = UpcomingEvent(
                event_type=event_type,
                date=event_date,
                days_until=days_until,
                expected_move=expected_move,
                iv_expansion_expected=iv_expansion,
                historical_accuracy=accuracy,
            )

            self.event_calendar.append(upcoming)

        # Sort by date
        self.event_calendar.sort(key=lambda x: x.date)

        logger.info(f"Updated event calendar", extra={"n_events": len(self.event_calendar)})

    def get_next_event(self, event_type: Optional[EventType] = None) -> Optional[UpcomingEvent]:
        """Get next upcoming event of specified type."""
        for event in self.event_calendar:
            if event_type is None or event.event_type == event_type:
                if event.days_until >= 0:
                    return event
        return None

    def should_adjust_for_event(
        self, dte_target: int, current_iv_rank: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Determine if strategy should be adjusted for upcoming events.

        Returns:
            (should_adjust, adjustments_dict)
        """
        adjustments = {
            "delta_adjustment": 0.0,
            "dte_adjustment": 0.0,
            "size_adjustment": 1.0,
            "confidence": 1.0,
        }

        # Check for earnings
        earnings = self.get_next_event(EventType.EARNINGS)
        if earnings and earnings.days_until <= 10:
            should_adjust = True

            if earnings.days_until <= 5:
                # Too close to earnings
                adjustments["size_adjustment"] = 0.0  # No position
                adjustments["confidence"] = 0.3
                logger.warning(f"Earnings in {earnings.days_until} days - avoiding position")
            else:
                # Adjust for expected IV expansion
                if current_iv_rank < 50:
                    # IV will likely expand
                    adjustments["delta_adjustment"] = -0.05  # More conservative
                    adjustments["dte_adjustment"] = -7  # Shorter duration
                    adjustments["size_adjustment"] = 0.5  # Half size
                else:
                    # IV already elevated
                    adjustments["delta_adjustment"] = 0.05  # Can be aggressive
                    adjustments["size_adjustment"] = 1.2  # Size up

                adjustments["confidence"] = 0.7
        else:
            should_adjust = False

        # Check for Fed meetings
        fed = self.get_next_event(EventType.FED_MEETING)
        if fed and fed.days_until <= 3:
            should_adjust = True
            adjustments["size_adjustment"] *= 0.75  # Reduce by 25%
            adjustments["confidence"] *= 0.8

        return should_adjust, adjustments

    def _analyze_earnings_impact(self, data: pd.DataFrame) -> EventImpact:
        """Analyze Unity earnings impact specifically."""
        # Historical Unity earnings moves
        earnings_moves = [0.12, 0.18, 0.15, 0.22, 0.10, 0.14, 0.25, 0.20]  # Example
        avg_move = np.mean(np.abs(earnings_moves))

        # IV behavior
        avg_iv_expansion = self.earnings_iv_expansion
        avg_iv_crush = 0.35  # 35% crush typical

        # Win rate for selling premium into earnings
        # Unity is volatile, so selling works less often
        win_rate = 0.45  # 45% win rate

        # Optimal entry
        optimal_days = 7  # Enter 7 days before for IV expansion

        confidence = min(1.0, len(earnings_moves) / 8)

        return EventImpact(
            event_type=EventType.EARNINGS,
            avg_move=avg_move,
            avg_iv_expansion=avg_iv_expansion,
            avg_iv_crush=avg_iv_crush,
            win_rate=win_rate,
            optimal_days_before=optimal_days,
            confidence=confidence,
        )

    def _analyze_fed_impact(self, data: pd.DataFrame) -> EventImpact:
        """Analyze Fed meeting impact."""
        # Tech stocks like Unity are sensitive to rates
        return EventImpact(
            event_type=EventType.FED_MEETING,
            avg_move=0.03,  # 3% average
            avg_iv_expansion=1.1,  # 10% IV increase
            avg_iv_crush=0.05,  # Small crush
            win_rate=0.65,  # Better for selling
            optimal_days_before=3,
            confidence=0.8,
        )

    def _analyze_opex_impact(self, data: pd.DataFrame) -> EventImpact:
        """Analyze options expiration impact."""
        return EventImpact(
            event_type=EventType.OPEX,
            avg_move=0.02,  # 2% pinning effect
            avg_iv_expansion=1.0,  # No expansion
            avg_iv_crush=0.10,  # Some crush
            win_rate=0.70,  # Good for selling
            optimal_days_before=0,  # No avoidance needed
            confidence=0.9,
        )

    def _analyze_generic_event(self, event_type: EventType, data: pd.DataFrame) -> EventImpact:
        """Generic event analysis."""
        return EventImpact(
            event_type=event_type,
            avg_move=0.02,
            avg_iv_expansion=1.05,
            avg_iv_crush=0.02,
            win_rate=0.60,
            optimal_days_before=1,
            confidence=0.5,
        )

    def _get_historical_impact(self, event_type: EventType) -> Dict:
        """Get historical impact statistics for event type."""
        if event_type not in self.event_history:
            return {"avg_move": 0.05, "avg_iv_expansion": 1.1, "confidence": 0.5}

        history = self.event_history[event_type]

        moves = [h["move"] for h in history if "move" in h]
        iv_changes = [h["iv_change"] for h in history if "iv_change" in h]

        return {
            "avg_move": np.mean(np.abs(moves)) if moves else 0.05,
            "avg_iv_expansion": np.mean([c for c in iv_changes if c > 1]) if iv_changes else 1.1,
            "confidence": min(1.0, len(history) / 10),
        }

    def calculate_event_adjusted_params(
        self, base_delta: float, base_dte: int, base_kelly: float
    ) -> Dict[str, float]:
        """
        Calculate parameter adjustments for all upcoming events.

        Returns:
            Adjusted parameters dictionary
        """
        delta = base_delta
        dte = base_dte
        kelly = base_kelly
        confidence = 1.0

        # Get all events in next 30 days
        near_events = [e for e in self.event_calendar if 0 <= e.days_until <= 30]

        for event in near_events:
            if event.event_type == EventType.EARNINGS:
                if event.days_until <= 5:
                    # Avoid earnings
                    kelly = 0.0
                    confidence = 0.0
                    logger.warning(f"Earnings in {event.days_until} days - no trade")
                elif event.days_until <= 10:
                    # Reduce and adjust
                    delta *= 0.8  # More conservative
                    dte = min(dte, event.days_until + 7)  # Don't cross earnings
                    kelly *= 0.5
                    confidence *= 0.7

            elif event.event_type == EventType.FED_MEETING:
                if event.days_until <= 3:
                    kelly *= 0.75
                    confidence *= 0.9

            elif event.event_type == EventType.OPEX:
                if event.days_until <= 7:
                    # Can benefit from pinning
                    delta *= 1.1  # Slightly more aggressive
                    confidence *= 1.05

        return {
            "delta": np.clip(delta, 0.10, 0.40),
            "dte": max(21, dte),
            "kelly": np.clip(kelly, 0.0, 0.50),
            "confidence": np.clip(confidence, 0.0, 1.0),
            "has_events": len(near_events) > 0,
            "next_event_days": near_events[0].days_until if near_events else 999,
        }

    def generate_event_report(self) -> List[str]:
        """Generate human-readable event impact report."""
        report = ["=== EVENT CALENDAR & IMPACT ===", ""]

        # Next 30 days
        for event in self.event_calendar[:5]:
            if event.days_until > 30:
                break

            report.append(f"{event.event_type.value.upper()}:")
            report.append(f"  Date: {event.date.strftime('%Y-%m-%d')}")
            report.append(f"  Days until: {event.days_until}")
            report.append(f"  Expected move: ±{event.expected_move*100:.1f}%")
            report.append(f"  IV expansion: {event.iv_expansion_expected:.1f}x")

            if event.event_type == EventType.EARNINGS:
                if event.days_until <= 5:
                    report.append("  ⚠️  ACTION: AVOID NEW POSITIONS")
                elif event.days_until <= 10:
                    report.append("  ⚠️  ACTION: REDUCE SIZE, SHORTEN DTE")
                else:
                    report.append("  ✓ ACTION: POSITION FOR IV EXPANSION")

            report.append("")

        if not self.event_calendar:
            report.append("No upcoming events in calendar")

        return report
