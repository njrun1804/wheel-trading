"""Data validation and sanity checks for Databento feeds.

Ensures:
- No missing trading days
- No dummy/placeholder data
- Consistent pricing relationships
- Proper option arbitrage bounds
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple

import pandas as pd

from unity_wheel.utils.logging import StructuredLogger
from unity_wheel.utils.trading_calendar import SimpleTradingCalendar

from .types import DataQuality, InstrumentDefinition, OptionChain, OptionQuote

logger = StructuredLogger(logging.getLogger(__name__))


class DataValidator:
    """Comprehensive data validation for options data."""

    # Validation thresholds
    MAX_SPREAD_PCT = 10.0  # Maximum bid-ask spread %
    MIN_QUOTE_SIZE = 1  # Minimum bid/ask size
    MAX_PRICE_CHANGE_PCT = 50.0  # Max intraday price change
    MIN_OPTIONS_PER_EXPIRY = 10  # Minimum strikes per expiration

    def __init__(self):
        """Initialize validator."""
        self.validation_results: List[Dict] = []
        self.calendar = SimpleTradingCalendar()

    def validate_historical_completeness(
        self, chains: List[OptionChain], start_date: datetime, end_date: datetime
    ) -> Tuple[bool, List[str]]:
        """Check for missing trading days in historical data.

        Returns:
            (is_complete, list_of_missing_dates)
        """
        logger.info(
            "validating_historical_completeness",
            extra={
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "chain_count": len(chains),
            },
        )

        # Get all dates with data
        data_dates = {chain.timestamp.date() for chain in chains}

        # Generate expected trading days using the trading calendar
        trading_days = self.calendar.get_trading_days_between(start_date, end_date)
        expected_dates = set(trading_days)

        # Find missing dates
        missing_dates = expected_dates - data_dates

        if missing_dates:
            logger.warning(
                "missing_trading_days",
                extra={
                    "count": len(missing_dates),
                    "dates": [d.isoformat() for d in sorted(missing_dates)],
                },
            )

        return len(missing_dates) == 0, [d.isoformat() for d in sorted(missing_dates)]

    def validate_chain_integrity(
        self, chain: OptionChain, definitions: List[InstrumentDefinition]
    ) -> DataQuality:
        """Validate a single option chain for data quality issues."""
        issues = []

        # Create definition lookup
        def_map = {d.instrument_id: d for d in definitions}

        # 1. Check for sufficient strikes
        unique_strikes = set()
        for quote in chain.calls + chain.puts:
            if quote.instrument_id in def_map:
                unique_strikes.add(def_map[quote.instrument_id].strike_price)

        if len(unique_strikes) < self.MIN_OPTIONS_PER_EXPIRY:
            issues.append(f"Insufficient strikes: {len(unique_strikes)}")

        # 2. Check bid-ask spreads
        wide_spreads = []
        for quote in chain.calls + chain.puts:
            if quote.spread_pct > Decimal(str(self.MAX_SPREAD_PCT)):
                wide_spreads.append(quote.instrument_id)

        if wide_spreads:
            issues.append(f"Wide spreads on {len(wide_spreads)} options")

        # 3. Check for zero/negative prices
        invalid_prices = []
        for quote in chain.calls + chain.puts:
            if quote.bid_price <= 0 or quote.ask_price <= 0:
                invalid_prices.append(quote.instrument_id)

        if invalid_prices:
            issues.append(f"Invalid prices on {len(invalid_prices)} options")

        # 4. Check for dummy data patterns
        if self._detect_dummy_data(chain):
            issues.append("Possible dummy data detected")

        # 5. Check arbitrage relationships
        arb_violations = self._check_arbitrage_bounds(chain, definitions)
        if arb_violations:
            issues.append(f"Arbitrage violations: {len(arb_violations)}")

        # Calculate overall quality score
        confidence = 1.0
        if issues:
            confidence = max(0.0, 1.0 - (len(issues) * 0.2))

        quality = DataQuality(
            symbol=chain.underlying,
            timestamp=chain.timestamp,
            bid_ask_spread_ok=len(wide_spreads) == 0,
            sufficient_liquidity=len(unique_strikes) >= self.MIN_OPTIONS_PER_EXPIRY,
            data_staleness_seconds=0,  # Assume fresh for historical
            confidence_score=confidence,
        )

        if issues:
            logger.warning(
                "chain_validation_issues",
                extra={
                    "symbol": chain.underlying,
                    "timestamp": chain.timestamp.isoformat(),
                    "issues": issues,
                    "confidence": confidence,
                },
            )

        return quality

    def _detect_dummy_data(self, chain: OptionChain) -> bool:
        """Detect potential dummy/test data patterns."""
        # Look for suspicious patterns
        all_quotes = chain.calls + chain.puts

        if not all_quotes:
            return False

        # Check if all prices are identical
        unique_bids = {q.bid_price for q in all_quotes}
        unique_asks = {q.ask_price for q in all_quotes}
        if len(unique_bids) == 1 and len(unique_asks) == 1 and len(all_quotes) > 5:
            return True

        # Check if all sizes are identical
        sizes = {(q.bid_size, q.ask_size) for q in all_quotes}
        if len(sizes) == 1 and len(all_quotes) > 10:
            return True

        # Check for perfectly sequential patterns (exact same increment)
        prices = sorted([float(q.mid_price) for q in all_quotes])
        if len(prices) > 5:
            diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
            # Check if all differences are exactly the same (within 0.001)
            if diffs and all(abs(d - diffs[0]) < 0.001 for d in diffs[1:]):
                return True

        return False

    def _check_arbitrage_bounds(
        self, chain: OptionChain, definitions: List[InstrumentDefinition]
    ) -> List[str]:
        """Check for violations of no-arbitrage bounds."""
        violations = []
        def_map = {d.instrument_id: d for d in definitions}

        # Group by strike
        strikes_data: Dict[Decimal, Dict] = {}

        for quote in chain.calls:
            if quote.instrument_id in def_map:
                strike = def_map[quote.instrument_id].strike_price
                if strike not in strikes_data:
                    strikes_data[strike] = {}
                strikes_data[strike]["call"] = quote

        for quote in chain.puts:
            if quote.instrument_id in def_map:
                strike = def_map[quote.instrument_id].strike_price
                if strike not in strikes_data:
                    strikes_data[strike] = {}
                strikes_data[strike]["put"] = quote

        # Check put-call parity
        for strike, data in strikes_data.items():
            if "call" in data and "put" in data:
                call = data["call"]
                put = data["put"]

                # Simplified check: C - P should be approximately S - K
                # (ignoring interest rate and dividends for now)
                synthetic = call.mid_price - put.mid_price
                intrinsic = chain.spot_price - strike

                # Allow 5% deviation
                if abs(float(synthetic - intrinsic)) > float(chain.spot_price) * 0.05:
                    violations.append(
                        f"Put-call parity violation at strike {strike}: "
                        f"synthetic={synthetic:.2f}, intrinsic={intrinsic:.2f}"
                    )

        # Check monotonicity
        call_prices = []
        put_prices = []

        for strike in sorted(strikes_data.keys()):
            if "call" in strikes_data[strike]:
                call_prices.append((strike, strikes_data[strike]["call"].mid_price))
            if "put" in strikes_data[strike]:
                put_prices.append((strike, strikes_data[strike]["put"].mid_price))

        # Calls should decrease with strike
        for i in range(1, len(call_prices)):
            if call_prices[i][1] > call_prices[i - 1][1]:
                violations.append(
                    f"Call price increases with strike: "
                    f"{call_prices[i-1][0]}@{call_prices[i-1][1]} -> "
                    f"{call_prices[i][0]}@{call_prices[i][1]}"
                )

        # Puts should increase with strike
        for i in range(1, len(put_prices)):
            if put_prices[i][1] < put_prices[i - 1][1]:
                violations.append(
                    f"Put price decreases with strike: "
                    f"{put_prices[i-1][0]}@{put_prices[i-1][1]} -> "
                    f"{put_prices[i][0]}@{put_prices[i][1]}"
                )

        return violations

    def validate_quote_sequence(
        self, quotes: List[OptionQuote], max_gap_seconds: int = 60
    ) -> List[str]:
        """Check for gaps or anomalies in quote sequence."""
        if not quotes:
            return ["No quotes provided"]

        issues = []
        quotes_sorted = sorted(quotes, key=lambda q: q.timestamp)

        # Check for time gaps
        for i in range(1, len(quotes_sorted)):
            gap = (quotes_sorted[i].timestamp - quotes_sorted[i - 1].timestamp).total_seconds()
            if gap > max_gap_seconds:
                issues.append(
                    f"Time gap of {gap:.0f}s between "
                    f"{quotes_sorted[i-1].timestamp.isoformat()} and "
                    f"{quotes_sorted[i].timestamp.isoformat()}"
                )

        # Check for price jumps
        for i in range(1, len(quotes_sorted)):
            if quotes_sorted[i].instrument_id == quotes_sorted[i - 1].instrument_id:
                price_change = abs(
                    float(quotes_sorted[i].mid_price - quotes_sorted[i - 1].mid_price)
                )
                pct_change = (price_change / float(quotes_sorted[i - 1].mid_price)) * 100

                if pct_change > self.MAX_PRICE_CHANGE_PCT:
                    issues.append(
                        f"Large price jump of {pct_change:.1f}% on "
                        f"instrument {quotes_sorted[i].instrument_id}"
                    )

        return issues

    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available"

        report = ["Data Validation Report", "=" * 50, ""]

        # Summary statistics
        total_chains = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r["passed"])

        report.append(f"Total chains validated: {total_chains}")
        report.append(f"Passed: {passed} ({passed/total_chains*100:.1f}%)")
        report.append(f"Failed: {total_chains - passed}")
        report.append("")

        # Common issues
        all_issues = []
        for result in self.validation_results:
            all_issues.extend(result.get("issues", []))

        if all_issues:
            report.append("Common Issues:")
            issue_counts = pd.Series(all_issues).value_counts()
            for issue, count in issue_counts.items():
                report.append(f"  - {issue}: {count} occurrences")

        return "\n".join(report)
