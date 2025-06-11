"""
Base interfaces for data providers with clear async/sync boundaries.
Ensures consistent API across all data providers.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from ...models import Position


class MarketData(Protocol):
    """Protocol for market data objects."""

    symbol: str
    timestamp: datetime
    price: float
    volume: Optional[int]


class OptionChain(Protocol):
    """Protocol for option chain data."""

    symbol: str
    expiration: datetime
    strikes: List[float]
    options: Dict[float, Dict[str, Any]]


class AsyncDataProvider(ABC):
    """
    Base class for async data providers.
    All I/O operations should be async.
    """

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass

    @abstractmethod
    async def get_option_chain(
        self, symbol: str, expiration: Optional[datetime] = None
    ) -> OptionChain:
        """Get option chain for a symbol."""
        pass

    @abstractmethod
    async def get_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[MarketData]:
        """Get historical price data."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if provider is connected and healthy."""
        pass

    def get_current_price_sync(self, symbol: str) -> float:
        """Synchronous wrapper for get_current_price."""
        return self._run_async(self.get_current_price(symbol))

    def get_option_chain_sync(
        self, symbol: str, expiration: Optional[datetime] = None
    ) -> OptionChain:
        """Synchronous wrapper for get_option_chain."""
        return self._run_async(self.get_option_chain(symbol, expiration))

    def get_historical_data_sync(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[MarketData]:
        """Synchronous wrapper for get_historical_data."""
        return self._run_async(self.get_historical_data(symbol, start_date, end_date))

    def get_positions_sync(self) -> List[Position]:
        """Synchronous wrapper for get_positions."""
        return self._run_async(self.get_positions())

    def is_connected_sync(self) -> bool:
        """Synchronous wrapper for is_connected."""
        return self._run_async(self.is_connected())

    @staticmethod
    def _run_async(coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new thread
                import concurrent.futures
                import threading

                result = None
                exception = None

                def run_in_thread():
                    nonlocal result, exception
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(coro)
                        new_loop.close()
                    except Exception as e:
                        exception = e

                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()

                if exception:
                    raise exception
                return result
            else:
                # No running loop, we can run directly
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()


class SyncDataProvider(ABC):
    """
    Base class for sync data providers.
    Used for local/cached data access.
    """

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass

    @abstractmethod
    def get_option_chain(self, symbol: str, expiration: Optional[datetime] = None) -> OptionChain:
        """Get option chain for a symbol."""
        pass

    @abstractmethod
    def get_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[MarketData]:
        """Get historical price data."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if provider is connected and healthy."""
        pass


class CachedDataProvider(SyncDataProvider):
    """
    Sync data provider that caches results from an async provider.
    Useful for reducing API calls and improving performance.
    """

    def __init__(self, async_provider: AsyncDataProvider, cache_ttl: int = 300):
        """
        Initialize with an async provider and cache TTL.

        Parameters
        ----------
        async_provider : AsyncDataProvider
            The underlying async data provider
        cache_ttl : int
            Cache time-to-live in seconds
        """
        self.async_provider = async_provider
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_times: Dict[str, datetime] = {}

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self._cache_times:
            return False

        age = (datetime.now() - self._cache_times[key]).total_seconds()
        return age < self.cache_ttl

    def _get_from_cache_or_fetch(self, key: str, fetch_func):
        """Get from cache or fetch using sync wrapper."""
        if self._is_cache_valid(key):
            return self._cache[key]

        # Fetch using sync wrapper
        result = fetch_func()

        # Update cache
        self._cache[key] = result
        self._cache_times[key] = datetime.now()

        return result

    def get_current_price(self, symbol: str) -> float:
        """Get current price with caching."""
        key = f"price_{symbol}"
        return self._get_from_cache_or_fetch(
            key, lambda: self.async_provider.get_current_price_sync(symbol)
        )

    def get_option_chain(self, symbol: str, expiration: Optional[datetime] = None) -> OptionChain:
        """Get option chain with caching."""
        key = f"chain_{symbol}_{expiration}"
        return self._get_from_cache_or_fetch(
            key, lambda: self.async_provider.get_option_chain_sync(symbol, expiration)
        )

    def get_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[MarketData]:
        """Get historical data with caching."""
        key = f"hist_{symbol}_{start_date}_{end_date}"
        return self._get_from_cache_or_fetch(
            key, lambda: self.async_provider.get_historical_data_sync(symbol, start_date, end_date)
        )

    def get_positions(self) -> List[Position]:
        """Get positions with caching."""
        key = "positions"
        return self._get_from_cache_or_fetch(key, lambda: self.async_provider.get_positions_sync())

    def is_connected(self) -> bool:
        """Check connection status."""
        # Don't cache connection status
        return self.async_provider.is_connected_sync()

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_times.clear()
