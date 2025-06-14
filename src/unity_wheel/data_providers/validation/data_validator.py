"""Data validation for API responses before storage."""

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates market data before storage to prevent corruption."""
    
    # Unity Software (U) typically trades in this range
    UNITY_MIN_PRICE = 5.0
    UNITY_MAX_PRICE = 100.0
    UNITY_TYPICAL_RANGE = (10.0, 50.0)
    
    # Option validation
    MIN_OPTION_SPREAD = 0.01  # $0.01 minimum spread
    MAX_OPTION_SPREAD_PCT = 0.50  # 50% max spread percentage
    
    def validate_stock_data(
        self, 
        symbol: str, 
        date: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: Optional[int]
    ) -> Tuple[bool, Optional[str]]:
        """Validate stock price data before storage.
        
        Returns:
            (is_valid, error_message)
        """
        # Check for zero prices
        if close_price == 0:
            return False, f"Zero close price not allowed for {symbol}"
            
        if any(p == 0 for p in [open_price, high_price, low_price] if p is not None):
            # Log warning but don't fail - sometimes only close is available
            logger.warning(f"Zero OHLC prices for {symbol} on {date}: O={open_price} H={high_price} L={low_price}")
        
        # Unity-specific validation
        if symbol == 'U':
            # Price range check
            if not (self.UNITY_MIN_PRICE <= close_price <= self.UNITY_MAX_PRICE):
                return False, f"Unity price ${close_price:.2f} outside valid range ${self.UNITY_MIN_PRICE}-${self.UNITY_MAX_PRICE}"
            
            # Warn on unusual but not impossible prices
            if not (self.UNITY_TYPICAL_RANGE[0] <= close_price <= self.UNITY_TYPICAL_RANGE[1]):
                logger.warning(f"Unity price ${close_price:.2f} outside typical range ${self.UNITY_TYPICAL_RANGE[0]}-${self.UNITY_TYPICAL_RANGE[1]}")
        
        # OHLC consistency
        if all(p is not None and p > 0 for p in [open_price, high_price, low_price, close_price]):
            if high_price < max(open_price, close_price, low_price):
                return False, f"High price ${high_price:.2f} is not the highest"
            if low_price > min(open_price, close_price, high_price):
                return False, f"Low price ${low_price:.2f} is not the lowest"
        
        # Volume validation (only on trading days)
        if self._is_trading_day(date) and volume is None:
            logger.warning(f"Missing volume for {symbol} on trading day {date}")
            # Don't fail - sometimes volume data is delayed
        
        return True, None
    
    def validate_option_quote(
        self,
        symbol: str,
        strike: float,
        expiration: datetime,
        option_type: str,
        bid_price: float,
        ask_price: float,
        bid_size: Optional[int] = None,
        ask_size: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate option quote data.
        
        Returns:
            (is_valid, error_message)
        """
        # Check for zero prices
        if bid_price == 0 and ask_price == 0:
            return False, f"Both bid and ask are zero for {symbol} {strike} {option_type}"
        
        # Bid/ask relationship
        if bid_price > ask_price and ask_price > 0:
            return False, f"Bid ${bid_price:.4f} exceeds ask ${ask_price:.4f}"
        
        # Spread validation
        if bid_price > 0 and ask_price > 0:
            spread = ask_price - bid_price
            spread_pct = spread / ask_price
            
            if spread < self.MIN_OPTION_SPREAD:
                logger.warning(f"Unusually tight spread ${spread:.4f} for {symbol} {strike} {option_type}")
            
            if spread_pct > self.MAX_OPTION_SPREAD_PCT:
                logger.warning(f"Wide spread {spread_pct:.1%} for {symbol} {strike} {option_type}")
        
        # Strike validation for Unity
        if symbol == 'U':
            if not (5.0 <= strike <= 100.0):
                return False, f"Unity strike ${strike} outside reasonable range"
        
        # Expiration validation
        if expiration.date() <= datetime.now(timezone.utc).date():
            return False, f"Option already expired: {expiration.date()}"
        
        return True, None
    
    def validate_fred_data(
        self,
        indicator: str,
        date: datetime,
        value: float
    ) -> Tuple[bool, Optional[str]]:
        """Validate FRED economic data.
        
        Returns:
            (is_valid, error_message)
        """
        # Basic validation
        if value is None:
            return False, f"Null value for {indicator} on {date}"
        
        # Indicator-specific validation
        if indicator == 'VIXCLS':  # VIX
            if not (5.0 <= value <= 100.0):
                return False, f"VIX value {value} outside valid range 5-100"
                
        elif indicator in ['DGS3MO', 'DGS10', 'DFF']:  # Interest rates
            if not (-5.0 <= value <= 25.0):
                return False, f"Interest rate {value} outside valid range -5% to 25%"
                
        elif indicator == 'TEDRATE':  # TED spread
            if not (-2.0 <= value <= 10.0):
                return False, f"TED spread {value} outside valid range -2% to 10%"
        
        return True, None
    
    def validate_volume_profile(
        self,
        symbol: str,
        date: datetime,
        volume: int,
        historical_avg: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate volume looks reasonable.
        
        Returns:
            (is_valid, error_message)
        """
        if volume < 0:
            return False, f"Negative volume {volume}"
        
        # Unity typically trades 1M-10M shares
        if symbol == 'U':
            if volume > 50_000_000:
                logger.warning(f"Unusually high Unity volume: {volume:,}")
            elif volume < 100_000 and self._is_trading_day(date):
                logger.warning(f"Unusually low Unity volume: {volume:,}")
        
        # Compare to historical if available
        if historical_avg and volume > 0:
            ratio = volume / historical_avg
            if ratio > 10:
                logger.warning(f"Volume {ratio:.1f}x higher than average")
            elif ratio < 0.1:
                logger.warning(f"Volume {ratio:.1f}x lower than average")
        
        return True, None
    
    def _is_trading_day(self, date: datetime) -> bool:
        """Check if date is a trading day (basic check)."""
        # Weekend check
        if date.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # TODO: Add holiday calendar check
        return True
    
    def validate_data_consistency(
        self,
        symbol: str,
        current_data: Dict,
        previous_data: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate data consistency with previous values.
        
        Returns:
            (is_valid, error_message)
        """
        if not previous_data:
            return True, None
        
        current_price = current_data.get('close', 0)
        previous_price = previous_data.get('close', 0)
        
        if current_price > 0 and previous_price > 0:
            # Check for extreme moves
            pct_change = abs((current_price - previous_price) / previous_price)
            
            # Unity rarely moves more than 20% in a day
            if symbol == 'U' and pct_change > 0.30:
                return False, f"Extreme price move {pct_change:.1%} from ${previous_price:.2f} to ${current_price:.2f}"
            
            # Warn on large moves
            if pct_change > 0.20:
                logger.warning(f"{symbol} moved {pct_change:.1%} in one day")
        
        return True, None