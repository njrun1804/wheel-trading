"""
Market data fetcher for Unity adaptive system.
Uses premium Databento API for market data.
NO MOCK DATA.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np

from ..data_providers.databento.client import DatabentoClient
from ..data_providers.databento.price_history_loader import PriceHistoryLoader
from ...storage.storage import Storage
from ..secrets.integration import get_databento_api_key
from ...utils import get_logger, with_recovery, RecoveryStrategy

logger = get_logger(__name__)


class MarketDataFetcher:
    """Fetch real market data for Unity wheel trading using Databento."""

    def __init__(
        self, databento_client: Optional[DatabentoClient] = None, storage: Optional[Storage] = None
    ):
        self.logger = get_logger(self.__class__.__name__)

        # Initialize Databento client if not provided
        if databento_client is None:
            api_key = get_databento_api_key()
            self.databento_client = DatabentoClient(api_key=api_key)
        else:
            self.databento_client = databento_client

        # Initialize storage if not provided
        self.storage = storage or Storage()

        # Cache for recent data
        self._unity_data_cache: Optional[pd.DataFrame] = None
        self._qqq_data_cache: Optional[pd.DataFrame] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)

    async def _refresh_cache_if_needed(self) -> bool:
        """Refresh data cache if expired using Databento."""
        if (
            self._cache_timestamp is None
            or datetime.now() - self._cache_timestamp > self._cache_ttl
        ):
            try:
                # Load price history for Unity and QQQ
                loader = PriceHistoryLoader(self.databento_client, self.storage)

                # Load Unity data
                unity_success = await loader.load_price_history("U", days=252)
                qqq_success = await loader.load_price_history("QQQ", days=252)

                if unity_success and qqq_success:
                    # Retrieve from storage by querying DuckDB directly
                    async with self.storage.cache.connection() as conn:
                        # Get Unity data
                        unity_result = await conn.execute(
                            """
                            SELECT date, close, returns
                            FROM price_history
                            WHERE symbol = 'U'
                            ORDER BY date DESC
                            LIMIT 252
                            """
                        )
                        unity_rows = await unity_result.fetchall()

                        if unity_rows:
                            # Convert to DataFrame
                            unity_data = [
                                {
                                    "date": row[0],
                                    "close": float(row[1]),
                                    "returns": float(row[2]) if row[2] else 0,
                                }
                                for row in unity_rows
                            ]
                            unity_df = pd.DataFrame(unity_data)
                            unity_df["date"] = pd.to_datetime(unity_df["date"])
                            unity_df.set_index("date", inplace=True)
                            unity_df = unity_df.sort_index()
                            unity_df["Returns"] = (
                                unity_df["returns"]
                                if unity_df["returns"].sum() != 0
                                else unity_df["close"].pct_change()
                            )
                            unity_df["RealizedVol"] = unity_df["Returns"].rolling(
                                20
                            ).std() * np.sqrt(252)
                            self._unity_data_cache = unity_df

                        # Get QQQ data
                        qqq_result = await conn.execute(
                            """
                            SELECT date, close, returns
                            FROM price_history
                            WHERE symbol = 'QQQ'
                            ORDER BY date DESC
                            LIMIT 252
                            """
                        )
                        qqq_rows = await qqq_result.fetchall()

                        if qqq_rows:
                            # Convert to DataFrame
                            qqq_data = [
                                {
                                    "date": row[0],
                                    "close": float(row[1]),
                                    "returns": float(row[2]) if row[2] else 0,
                                }
                                for row in qqq_rows
                            ]
                            qqq_df = pd.DataFrame(qqq_data)
                            qqq_df["date"] = pd.to_datetime(qqq_df["date"])
                            qqq_df.set_index("date", inplace=True)
                            qqq_df = qqq_df.sort_index()
                            qqq_df["Returns"] = (
                                qqq_df["returns"]
                                if qqq_df["returns"].sum() != 0
                                else qqq_df["close"].pct_change()
                            )
                            self._qqq_data_cache = qqq_df

                    if self._unity_data_cache is not None and self._qqq_data_cache is not None:
                        self._cache_timestamp = datetime.now()
                        return True

            except Exception as e:
                self.logger.error(f"Failed to fetch market data from Databento: {e}")

        return self._unity_data_cache is not None

    def _run_async(self, coro):
        """Helper to run async code in sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context
                task = asyncio.create_task(coro)
                return asyncio.run_coroutine_threadsafe(coro, loop).result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(coro)

    def get_unity_volatility(self, lookback_days: int = 20) -> Optional[float]:
        """
        Calculate Unity's realized volatility from real data.
        Returns None if data unavailable.
        """
        if not self._run_async(self._refresh_cache_if_needed()):
            self.logger.error("Cannot calculate Unity volatility - no data available")
            return None

        if "RealizedVol" not in self._unity_data_cache.columns:
            return None

        # Get most recent realized volatility
        recent_vol = self._unity_data_cache["RealizedVol"].dropna().tail(1)
        if recent_vol.empty:
            return None

        return float(recent_vol.iloc[0])

    def get_unity_price(self) -> Optional[float]:
        """Get current Unity price."""
        if not self._run_async(self._refresh_cache_if_needed()):
            return None

        latest_close = self._unity_data_cache["close"].tail(1)
        if latest_close.empty:
            return None

        # Fix FutureWarning by using iloc[0] on the Series directly
        return float(latest_close.iloc[0])

    def get_unity_iv_rank(self) -> Optional[float]:
        """
        Get Unity's IV rank.

        Since we don't have options data, returns None.
        In production, would need options data provider.
        """
        # Cannot calculate IV rank without options data
        self.logger.warning("IV rank calculation requires options data - not available")
        return None

    def get_unity_qqq_correlation(self, lookback_days: int = 60) -> Optional[float]:
        """
        Calculate actual Unity-QQQ correlation from real data.
        """
        if not self._run_async(self._refresh_cache_if_needed()):
            return None

        # Align dates
        common_dates = self._unity_data_cache.index.intersection(self._qqq_data_cache.index)

        if len(common_dates) < lookback_days:
            self.logger.warning(f"Insufficient data for {lookback_days}-day correlation")
            return None

        # Get returns for both
        unity_returns = self._unity_data_cache.loc[common_dates, "Returns"].dropna()
        qqq_returns = self._qqq_data_cache.loc[common_dates, "Returns"].dropna()

        # Ensure same length
        common_returns_dates = unity_returns.index.intersection(qqq_returns.index)
        if len(common_returns_dates) < lookback_days:
            return None

        # Calculate correlation
        unity_returns = unity_returns.loc[common_returns_dates].tail(lookback_days)
        qqq_returns = qqq_returns.loc[common_returns_dates].tail(lookback_days)

        correlation = unity_returns.corr(qqq_returns)

        return float(correlation) if not np.isnan(correlation) else None

    def calculate_portfolio_drawdown(self, current_value: float, peak_value: float) -> float:
        """Calculate current drawdown from peak."""
        if peak_value <= 0:
            return 0.0
        return (current_value - peak_value) / peak_value

    def get_days_to_earnings(self) -> Optional[int]:
        """
        Get days to next Unity earnings.

        Without real earnings API, returns None.
        In production, would use earnings calendar API.
        """
        # Cannot provide accurate earnings dates without API
        self.logger.warning("Earnings dates require external API - not available")
        return None


class UnityEarningsCalendar:
    """
    Unity earnings calendar.
    In production, would fetch from API.
    """

    @classmethod
    def get_next_earnings_date(cls) -> Optional[datetime]:
        """
        Get next earnings date.
        Returns None without real API.
        """
        # Cannot provide without real API
        return None

    @classmethod
    def days_to_next_earnings(cls) -> Optional[int]:
        """Days to next earnings - requires API."""
        return None


def get_market_regime(volatility: float, correlation: float) -> str:
    """
    Determine market regime from real metrics.
    """
    # High volatility + high correlation = crisis
    if volatility > 0.80 and correlation > 0.85:
        return "CRISIS"

    # High volatility OR high correlation = stressed
    elif volatility > 0.60 or correlation > 0.80:
        return "STRESSED"

    # Moderate levels = volatile
    elif volatility > 0.45:
        return "VOLATILE"

    # Otherwise normal
    else:
        return "NORMAL"
