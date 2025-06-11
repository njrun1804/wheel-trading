"""Unified data provider interface for all market data sources.

Consolidates access to Databento, FRED, Schwab, and other data providers
through a single, consistent interface with unified caching and error handling.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd

from src.unity_wheel.utils.logging import StructuredLogger
from src.unity_wheel.utils.memory_optimizer import get_memory_monitor
from src.unity_wheel.utils.performance_cache import cached, risk_cache

logger = StructuredLogger(logging.getLogger(__name__))


class DataSourceType(Enum):
    """Types of data sources available."""

    MARKET_DATA = "market_data"  # Real-time and historical prices
    OPTIONS_DATA = "options_data"  # Option chains and Greeks
    ECONOMIC_DATA = "economic_data"  # FRED economic indicators
    FUNDAMENTAL_DATA = "fundamental_data"  # Company fundamentals
    ALTERNATIVE_DATA = "alternative_data"  # Sentiment, flow, etc.


class DataQuality(Enum):
    """Data quality levels."""

    HIGH = "high"  # Real-time, exchange-quality data
    MEDIUM = "medium"  # Delayed or processed data
    LOW = "low"  # Estimated or derived data
    UNKNOWN = "unknown"


@dataclass
class DataRequest:
    """Standardized data request object."""

    source_type: DataSourceType
    symbol: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class DataResponse:
    """Standardized data response object."""

    data: Union[pd.DataFrame, Dict, Any]
    metadata: Dict[str, Any]
    quality: DataQuality
    source: str
    timestamp: datetime
    cached: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DataProviderProtocol(Protocol):
    """Protocol defining the interface all data providers must implement."""

    async def get_data(self, request: DataRequest) -> DataResponse:
        """Get data for a specific request."""
        ...

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health and connectivity."""
        ...

    def get_supported_sources(self) -> List[DataSourceType]:
        """Get list of supported data source types."""
        ...


class BaseDataProvider(ABC):
    """Base class for all data providers with common functionality."""

    def __init__(self, name: str, priority: int = 100):
        """Initialize provider.

        Args:
            name: Provider name for logging and identification
            priority: Provider priority (lower = higher priority)
        """
        self.name = name
        self.priority = priority
        self._health_status = {}
        self._last_health_check = None

        # Register with memory monitor
        get_memory_monitor().track_object(self)

        logger.info("data_provider_initialized", extra={"provider": name, "priority": priority})

    @abstractmethod
    async def get_data(self, request: DataRequest) -> DataResponse:
        """Get data for a specific request."""
        pass

    @abstractmethod
    def get_supported_sources(self) -> List[DataSourceType]:
        """Get list of supported data source types."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Basic health check implementation."""
        self._last_health_check = datetime.now()
        self._health_status = {
            "provider": self.name,
            "status": "healthy",
            "last_check": self._last_health_check.isoformat(),
            "supported_sources": [s.value for s in self.get_supported_sources()],
        }
        return self._health_status

    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        if not self._health_status:
            return False
        return self._health_status.get("status") == "healthy"


class UnifiedDataProvider:
    """Unified data provider that coordinates access to all data sources."""

    def __init__(self):
        self._providers: Dict[str, BaseDataProvider] = {}
        self._source_mapping: Dict[DataSourceType, List[str]] = {}
        self._fallback_chains: Dict[DataSourceType, List[str]] = {}

        # Performance tracking
        self._request_count = 0
        self._cache_hits = 0

        logger.info("unified_data_provider_initialized")

    def register_provider(self, provider: BaseDataProvider) -> None:
        """Register a data provider."""
        self._providers[provider.name] = provider

        # Update source mapping
        for source_type in provider.get_supported_sources():
            if source_type not in self._source_mapping:
                self._source_mapping[source_type] = []
            self._source_mapping[source_type].append(provider.name)

            # Sort by priority
            self._source_mapping[source_type].sort(key=lambda name: self._providers[name].priority)

        logger.info(
            "provider_registered",
            extra={
                "provider": provider.name,
                "supported_sources": [s.value for s in provider.get_supported_sources()],
            },
        )

    def set_fallback_chain(self, source_type: DataSourceType, provider_names: List[str]) -> None:
        """Set fallback chain for a data source type."""
        # Validate providers exist
        for name in provider_names:
            if name not in self._providers:
                raise ValueError(f"Provider '{name}' not registered")

        self._fallback_chains[source_type] = provider_names
        logger.info(
            "fallback_chain_set", extra={"source_type": source_type.value, "chain": provider_names}
        )

    @cached(
        cache_name="market_data",
        ttl_seconds=300,  # 5 minute cache for market data
        key_func=lambda self, request: f"{request.source_type.value}_{request.symbol}_{request.start_time}_{request.end_time}_{hash(str(request.parameters))}",
    )
    async def get_data(self, request: DataRequest) -> DataResponse:
        """Get data using fallback chain if needed."""
        self._request_count += 1

        # Determine provider chain
        provider_chain = self._get_provider_chain(request.source_type)

        if not provider_chain:
            raise ValueError(f"No providers available for {request.source_type.value}")

        # Try each provider in order
        last_error = None
        for provider_name in provider_chain:
            provider = self._providers[provider_name]

            try:
                # Quick health check
                if not provider.is_healthy():
                    await provider.health_check()

                response = await provider.get_data(request)

                # Log successful retrieval
                logger.debug(
                    "data_retrieved_successfully",
                    extra={
                        "provider": provider_name,
                        "source_type": request.source_type.value,
                        "symbol": request.symbol,
                        "quality": response.quality.value,
                    },
                )

                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    "provider_failed",
                    extra={
                        "provider": provider_name,
                        "error": str(e),
                        "source_type": request.source_type.value,
                        "symbol": request.symbol,
                    },
                )
                continue

        # All providers failed
        raise RuntimeError(
            f"All providers failed for {request.source_type.value} data. Last error: {last_error}"
        )

    def _get_provider_chain(self, source_type: DataSourceType) -> List[str]:
        """Get provider chain for a source type."""
        # Use explicit fallback chain if set
        if source_type in self._fallback_chains:
            return self._fallback_chains[source_type]

        # Use default priority-based chain
        return self._source_mapping.get(source_type, [])

    async def get_market_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        frequency: str = "1D",
    ) -> DataResponse:
        """Convenience method for market data."""
        request = DataRequest(
            source_type=DataSourceType.MARKET_DATA,
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            parameters={"frequency": frequency},
        )
        return await self.get_data(request)

    async def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[datetime] = None,
        strike_filter: Optional[Tuple[float, float]] = None,
    ) -> DataResponse:
        """Convenience method for options data."""
        parameters = {}
        if expiration:
            parameters["expiration"] = expiration
        if strike_filter:
            parameters["min_strike"], parameters["max_strike"] = strike_filter

        request = DataRequest(
            source_type=DataSourceType.OPTIONS_DATA, symbol=symbol, parameters=parameters
        )
        return await self.get_data(request)

    async def get_economic_indicator(
        self,
        indicator: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> DataResponse:
        """Convenience method for economic data."""
        request = DataRequest(
            source_type=DataSourceType.ECONOMIC_DATA,
            symbol=indicator,
            start_time=start_date,
            end_time=end_date,
        )
        return await self.get_data(request)

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Run health checks on all providers."""
        results = {}

        tasks = [provider.health_check() for provider in self._providers.values()]

        health_results = await asyncio.gather(*tasks, return_exceptions=True)

        for provider_name, result in zip(self._providers.keys(), health_results):
            if isinstance(result, Exception):
                results[provider_name] = {"status": "error", "error": str(result)}
            else:
                results[provider_name] = result

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = self._cache_hits / max(1, self._request_count)

        return {
            "total_requests": self._request_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "registered_providers": list(self._providers.keys()),
            "source_mapping": {
                source.value: providers for source, providers in self._source_mapping.items()
            },
        }

    async def invalidate_cache(
        self, source_type: Optional[DataSourceType] = None, symbol: Optional[str] = None
    ) -> None:
        """Invalidate cache entries."""
        # This would integrate with the performance cache system
        # For now, just log the invalidation request
        logger.info(
            "cache_invalidation_requested",
            extra={
                "source_type": source_type.value if source_type else "all",
                "symbol": symbol or "all",
            },
        )


# Specific provider implementations


class DatabentoProvider(BaseDataProvider):
    """Databento data provider implementation."""

    def __init__(self):
        super().__init__(name="databento", priority=10)  # High priority
        self._client = None

    def get_supported_sources(self) -> List[DataSourceType]:
        return [DataSourceType.MARKET_DATA, DataSourceType.OPTIONS_DATA]

    async def get_data(self, request: DataRequest) -> DataResponse:
        """Get data from Databento."""
        # Import here to avoid circular dependency
        from src.unity_wheel.data_providers.databento.client import DatabentoClient
        from src.unity_wheel.data_providers.databento.integration import DatabentoIntegration

        if self._client is None:
            self._client = DatabentoClient()
            self._integration = DatabentoIntegration(self._client)

        if request.source_type == DataSourceType.OPTIONS_DATA:
            candidates = await self._integration.get_wheel_candidates(
                underlying=request.symbol, **request.parameters
            )

            return DataResponse(
                data=candidates,
                metadata={"provider": "databento", "type": "options_candidates"},
                quality=DataQuality.HIGH,
                source="databento",
            )

        elif request.source_type == DataSourceType.MARKET_DATA:
            # Implement market data retrieval
            # For now, return empty response
            return DataResponse(
                data=pd.DataFrame(),
                metadata={"provider": "databento", "type": "market_data"},
                quality=DataQuality.HIGH,
                source="databento",
            )

        else:
            raise ValueError(f"Unsupported source type: {request.source_type}")


class FREDProvider(BaseDataProvider):
    """FRED economic data provider implementation."""

    def __init__(self):
        super().__init__(name="fred", priority=20)
        self._manager = None

    def get_supported_sources(self) -> List[DataSourceType]:
        return [DataSourceType.ECONOMIC_DATA]

    async def get_data(self, request: DataRequest) -> DataResponse:
        """Get data from FRED."""
        # Import here to avoid circular dependency
        from src.unity_wheel.data_providers.base.manager import FREDDataManager

        if self._manager is None:
            # Initialize FRED manager
            from src.unity_wheel.storage.storage import Storage

            storage = Storage()
            await storage.initialize()
            self._manager = FREDDataManager(storage=storage)

        if request.source_type == DataSourceType.ECONOMIC_DATA:
            # Get risk-free rate or other economic indicators
            if request.symbol in ["DGS3", "risk_free_rate"]:
                rate, confidence = await self._manager.get_or_fetch_risk_free_rate()

                data = pd.DataFrame(
                    {"value": [rate], "confidence": [confidence], "date": [datetime.now().date()]}
                )

                return DataResponse(
                    data=data,
                    metadata={"provider": "fred", "series": request.symbol},
                    quality=DataQuality.HIGH,
                    source="fred",
                )
            else:
                raise ValueError(f"Unsupported FRED series: {request.symbol}")

        else:
            raise ValueError(f"Unsupported source type: {request.source_type}")


class MockProvider(BaseDataProvider):
    """Mock data provider for testing and development."""

    def __init__(self):
        super().__init__(name="mock", priority=1000)  # Lowest priority

    def get_supported_sources(self) -> List[DataSourceType]:
        return list(DataSourceType)  # Supports all types

    async def get_data(self, request: DataRequest) -> DataResponse:
        """Generate mock data."""
        if request.source_type == DataSourceType.MARKET_DATA:
            # Generate mock price data
            dates = pd.date_range(
                start=request.start_time or datetime.now() - timedelta(days=30),
                end=request.end_time or datetime.now(),
                freq="D",
            )

            # Simple random walk for mock prices
            np.random.seed(hash(request.symbol) % 2**32)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.cumprod(1 + returns)

            data = pd.DataFrame(
                {
                    "date": dates,
                    "close": prices,
                    "volume": np.random.randint(100000, 1000000, len(dates)),
                }
            )

        elif request.source_type == DataSourceType.OPTIONS_DATA:
            # Generate mock options data
            spot = 35.0
            strikes = np.arange(25, 46, 2.5)

            mock_options = []
            for strike in strikes:
                mock_options.append(
                    {
                        "strike": strike,
                        "expiration": datetime.now() + timedelta(days=45),
                        "bid": max(0.05, spot - strike + np.random.normal(0, 0.1)),
                        "ask": max(0.05, spot - strike + np.random.normal(0.1, 0.1)),
                        "volume": np.random.randint(50, 500),
                        "open_interest": np.random.randint(100, 2000),
                    }
                )

            data = mock_options

        elif request.source_type == DataSourceType.ECONOMIC_DATA:
            # Generate mock economic data
            data = pd.DataFrame(
                {
                    "value": [0.05],  # 5% risk-free rate
                    "confidence": [0.95],
                    "date": [datetime.now().date()],
                }
            )

        else:
            data = {"message": f"Mock data for {request.source_type.value}"}

        return DataResponse(
            data=data,
            metadata={"provider": "mock", "generated": True},
            quality=DataQuality.LOW,
            source="mock",
        )


# Global unified provider instance
_unified_provider = None


def get_unified_provider() -> UnifiedDataProvider:
    """Get the global unified data provider instance."""
    global _unified_provider

    if _unified_provider is None:
        _unified_provider = UnifiedDataProvider()

        # Register default providers
        _unified_provider.register_provider(DatabentoProvider())
        _unified_provider.register_provider(FREDProvider())
        _unified_provider.register_provider(MockProvider())  # Fallback

        # Set up fallback chains
        _unified_provider.set_fallback_chain(DataSourceType.MARKET_DATA, ["databento", "mock"])
        _unified_provider.set_fallback_chain(DataSourceType.OPTIONS_DATA, ["databento", "mock"])
        _unified_provider.set_fallback_chain(DataSourceType.ECONOMIC_DATA, ["fred", "mock"])

        logger.info("global_unified_provider_initialized")

    return _unified_provider


# Convenience functions for common operations
async def get_unity_market_data(days_back: int = 30) -> DataResponse:
    """Get Unity market data."""
    provider = get_unified_provider()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)

    return await provider.get_market_data(symbol="U", start_time=start_time, end_time=end_time)


async def get_unity_options_chain(expiration_days: int = 45) -> DataResponse:
    """Get Unity options chain."""
    provider = get_unified_provider()
    expiration = datetime.now() + timedelta(days=expiration_days)

    return await provider.get_options_chain(symbol="U", expiration=expiration)


async def get_risk_free_rate() -> float:
    """Get current risk-free rate."""
    provider = get_unified_provider()
    response = await provider.get_economic_indicator("risk_free_rate")

    if isinstance(response.data, pd.DataFrame) and not response.data.empty:
        return float(response.data["value"].iloc[0])
    else:
        return 0.05  # Default 5%
