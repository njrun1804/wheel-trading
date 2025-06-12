"""Parser for broker account data (Schwab/TD Ameritrade format)."""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from unity_wheel.models.position import Position
from unity_wheel.portfolio.single_account import ManualAccount
from unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)


class BrokerDataParser:
    """Parse broker account data from copy/paste."""

    def parse_broker_data(self, data: str) -> ManualAccount:
        """
        Parse broker account data from copy/paste.

        Handles formats like:
        - Account Summary section with Total Accounts Value
        - Cash & Cash Investments
        - Positions Details with Symbol, Qty, Price, etc.
        """
        lines = [line.strip() for line in data.strip().split("\n") if line.strip()]

        # Initialize values
        account_value = None
        cash_balance = None
        buying_power = None
        positions = []

        # State tracking
        in_positions_section = False

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Look for account summary values
            if "total accounts value" in line_lower:
                account_value = self._extract_next_dollar_amount(lines, i)
            elif (
                "cash & cash investments total" in line_lower
                or "total cash & cash invest" in line_lower
            ):
                cash_balance = self._extract_next_dollar_amount(lines, i)
            elif "cash + borrowing" in line_lower and buying_power is None:
                buying_power = self._extract_next_dollar_amount(lines, i)
            elif "available to day trade" in line_lower and buying_power is None:
                # Alternative buying power location
                buying_power = self._extract_next_dollar_amount(lines, i)

            # Check if we're entering positions section
            if "positions details" in line_lower:
                in_positions_section = True
                continue
            elif "account total" in line_lower or "total" in line_lower:
                # Exit positions section on totals
                in_positions_section = False

            # Parse positions
            if in_positions_section and not any(
                header in line_lower
                for header in ["symbol", "description", "customize", "equities", "total"]
            ):
                position = self._parse_position_row(line, lines, i)
                if position:
                    positions.append(position)

        # Validate we found the required values
        if not account_value:
            raise ValueError("Could not find account value in data")
        if not cash_balance:
            raise ValueError("Could not find cash balance in data")
        if not buying_power:
            buying_power = cash_balance  # Default to cash

        # Create account
        account = ManualAccount(
            account_id="BROKER",
            total_value=account_value,
            cash_balance=cash_balance,
            buying_power=buying_power,
            margin_buying_power=buying_power * 2.0,
            positions=positions,
        )

        # Calculate Unity exposure
        self._calculate_unity_exposure(account)

        logger.info(
            "Parsed broker account data",
            extra={
                "total_value": account.total_value,
                "cash": account.cash_balance,
                "positions": len(positions),
                "unity_shares": account.unity_shares,
                "unity_puts": account.unity_puts,
                "unity_calls": account.unity_calls,
            },
        )

        return account

    def _extract_next_dollar_amount(self, lines: List[str], start_idx: int) -> Optional[float]:
        """Extract dollar amount from current or next few lines."""
        for offset in range(0, min(5, len(lines) - start_idx)):
            line = lines[start_idx + offset]
            amount = self._extract_dollar_amount(line)
            if amount is not None:
                return amount
        return None

    def _extract_dollar_amount(self, text: str) -> Optional[float]:
        """Extract dollar amount from text."""
        # Pattern to match dollar amounts with optional negative sign
        pattern = r"([-]?)\$([0-9,]+\.?[0-9]*)"
        match = re.search(pattern, text)
        if match:
            sign = -1 if match.group(1) == "-" else 1
            amount_str = match.group(2).replace(",", "")
            try:
                return sign * float(amount_str)
            except ValueError:
                pass
        return None

    def _parse_position_row(self, line: str, lines: List[str], line_idx: int) -> Optional[Position]:
        """Parse a position row from broker data."""
        # Skip empty lines or headers
        if not line or line.lower() in ["equities", "options", "cash & money market"]:
            return None

        try:
            # For broker data, we need to look at multiple lines
            # Symbol is on first line, other data may be on same or next lines

            # Check if this is an option symbol like "U 07/18/2025 25.00 C"
            option_match = re.match(
                r"^([A-Z]+)\s+(\d{2})/(\d{2})/(\d{4})\s+([\d.]+)\s+([CP])", line
            )
            if option_match:
                symbol, month, day, year, strike, opt_type = option_match.groups()

                # Convert to OCC format: UYYMMDDP00025000
                yy = year[-2:]
                strike_int = int(float(strike) * 1000)
                occ_symbol = f"{symbol}{yy}{month}{day}{opt_type}{strike_int:08d}"

                # Look for quantity on this or next lines
                quantity = self._find_quantity_in_lines(lines, line_idx)
                if quantity is not None:
                    return Position(symbol=occ_symbol, quantity=quantity)

            # Check if this is a stock symbol
            elif re.match(r"^[A-Z]+$", line.split()[0] if line.split() else ""):
                symbol = line.split()[0]

                # Look for quantity on this or next lines
                quantity = self._find_quantity_in_lines(lines, line_idx)
                if quantity is not None:
                    return Position(symbol=symbol, quantity=quantity)

        except Exception as e:
            logger.debug(f"Could not parse position row: {line}, error: {e}")

        return None

    def _find_quantity_in_lines(self, lines: List[str], start_idx: int) -> Optional[int]:
        """Find quantity in current or next few lines."""
        for offset in range(0, min(5, len(lines) - start_idx)):
            line = lines[start_idx + offset]

            # Look for patterns like "7,500" or "-75" that could be quantities
            # Typically after the symbol and before price
            parts = line.split()
            for part in parts:
                # Remove commas and check if it's a number
                clean_part = part.replace(",", "").replace("-", "")
                if clean_part.isdigit():
                    try:
                        qty = int(part.replace(",", ""))
                        # Reasonable quantity check (not a price or large number)
                        if 1 <= abs(qty) <= 100000:
                            return qty
                    except ValueError:
                        continue
        return None

    def _calculate_unity_exposure(self, account: ManualAccount) -> None:
        """Calculate Unity exposure across all positions."""
        unity_price = 25.0  # Default estimate

        for position in account.positions:
            if position.underlying != "U":
                continue

            if position.position_type.value == "stock":
                account.unity_shares += position.quantity
                # Estimate price from position if possible
                unity_price = 25.0  # Will be overridden by actual data
            elif position.position_type.value == "put":
                account.unity_puts += abs(position.quantity)
            elif position.position_type.value == "call":
                account.unity_calls += abs(position.quantity)

        # Calculate notional
        share_exposure = account.unity_shares * unity_price
        put_exposure = account.unity_puts * 100 * unity_price
        call_exposure = account.unity_calls * 100 * unity_price

        account.unity_notional = share_exposure + put_exposure + call_exposure


def parse_broker_paste(data: str) -> ManualAccount:
    """Convenience function to parse broker data."""
    parser = BrokerDataParser()
    return parser.parse_broker_data(data)
