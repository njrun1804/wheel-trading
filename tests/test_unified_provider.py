"""Tests for unified data provider system."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from src.unity_wheel.data_providers.unified_provider import (
    BaseDataProvider,
    DatabentoProvider,
    DataQuality,
    DataRequest,
    DataResponse,
    DataSourceType,
    FREDProvider,
    MockProvider,
    UnifiedDataProvider,
    get_risk_free_rate,
    get_unified_provider,
    get_unity_market_data,
    get_unity_options_chain,
)


class TestDataRequest:
    """Test DataRequest dataclass."""

    def test_data_request_creation(self):
        """Test creating a data request."""
        request = DataRequest(source_type=DataSourceType.MARKET_DATA, symbol="U")

        assert request.source_type == DataSourceType.MARKET_DATA
        assert request.symbol == "U"
        assert request.parameters == {}

    def test_data_request_with_parameters(self):
        """Test creating a data request with parameters."""
        start_time = datetime.now()
        end_time = start_time + timedelta(days=1)

        request = DataRequest(
            source_type=DataSourceType.OPTIONS_DATA,
            symbol="AAPL",
            start_time=start_time,
            end_time=end_time,
            parameters={"strike_filter": (100, 200)},
        )

        assert request.source_type == DataSourceType.OPTIONS_DATA
        assert request.symbol == "AAPL"
        assert request.start_time == start_time
        assert request.end_time == end_time
        assert request.parameters["strike_filter"] == (100, 200)


class TestDataResponse:
    """Test DataResponse dataclass."""

    def test_data_response_creation(self):
        """Test creating a data response."""
        data = pd.DataFrame({"price": [100, 101, 102]})

        response = DataResponse(
            data=data, metadata={"provider": "test"}, quality=DataQuality.HIGH, source="test_source"
        )

        assert isinstance(response.data, pd.DataFrame)
        assert response.metadata["provider"] == "test"
        assert response.quality == DataQuality.HIGH
        assert response.source == "test_source"
        assert isinstance(response.timestamp, datetime)
        assert not response.cached


class TestMockProvider:
    """Test mock data provider."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        return MockProvider()

    def test_mock_provider_initialization(self, mock_provider):
        """Test mock provider initialization."""
        assert mock_provider.name == "mock"
        assert mock_provider.priority == 1000
        assert DataSourceType.MARKET_DATA in mock_provider.get_supported_sources()
        assert DataSourceType.OPTIONS_DATA in mock_provider.get_supported_sources()

    @pytest.mark.asyncio
    async def test_mock_market_data(self, mock_provider):
        """Test mock market data generation."""
        request = DataRequest(
            source_type=DataSourceType.MARKET_DATA,
            symbol="TEST",
            start_time=datetime.now() - timedelta(days=5),
            end_time=datetime.now(),
        )

        response = await mock_provider.get_data(request)

        assert isinstance(response.data, pd.DataFrame)
        assert "close" in response.data.columns
        assert "volume" in response.data.columns
        assert len(response.data) > 0
        assert response.quality == DataQuality.LOW
        assert response.source == "mock"

    @pytest.mark.asyncio
    async def test_mock_options_data(self, mock_provider):
        """Test mock options data generation."""
        request = DataRequest(source_type=DataSourceType.OPTIONS_DATA, symbol="TEST")

        response = await mock_provider.get_data(request)

        assert isinstance(response.data, list)
        assert len(response.data) > 0

        option = response.data[0]
        assert "strike" in option
        assert "expiration" in option
        assert "bid" in option
        assert "ask" in option

    @pytest.mark.asyncio
    async def test_mock_economic_data(self, mock_provider):
        """Test mock economic data generation."""
        request = DataRequest(source_type=DataSourceType.ECONOMIC_DATA, symbol="DGS3")

        response = await mock_provider.get_data(request)

        assert isinstance(response.data, pd.DataFrame)
        assert "value" in response.data.columns
        assert response.data["value"].iloc[0] == 0.05

    @pytest.mark.asyncio
    async def test_health_check(self, mock_provider):
        """Test provider health check."""
        health = await mock_provider.health_check()

        assert health["provider"] == "mock"
        assert health["status"] == "healthy"
        assert "last_check" in health
        assert "supported_sources" in health


class TestUnifiedDataProvider:
    """Test unified data provider."""

    @pytest.fixture
    def provider(self):
        """Create a unified provider for testing."""
        return UnifiedDataProvider()

    @pytest.fixture
    def test_base_provider(self):
        """Create a test base provider."""

        class TestProvider(BaseDataProvider):
            def __init__(self):
                super().__init__(name="test", priority=50)

            def get_supported_sources(self):
                return [DataSourceType.MARKET_DATA]

            async def get_data(self, request):
                return DataResponse(
                    data={"test": "data"},
                    metadata={"provider": "test"},
                    quality=DataQuality.MEDIUM,
                    source="test",
                )

        return TestProvider()

    def test_provider_registration(self, provider, test_base_provider):
        """Test registering a data provider."""
        provider.register_provider(test_base_provider)

        assert "test" in provider._providers
        assert DataSourceType.MARKET_DATA in provider._source_mapping
        assert "test" in provider._source_mapping[DataSourceType.MARKET_DATA]

    def test_fallback_chain_setting(self, provider, test_base_provider):
        """Test setting fallback chains."""
        mock_provider = MockProvider()

        provider.register_provider(test_base_provider)
        provider.register_provider(mock_provider)

        provider.set_fallback_chain(DataSourceType.MARKET_DATA, ["test", "mock"])

        assert provider._fallback_chains[DataSourceType.MARKET_DATA] == ["test", "mock"]

    def test_invalid_fallback_chain(self, provider):
        """Test setting fallback chain with invalid provider."""
        with pytest.raises(ValueError, match="Provider 'nonexistent' not registered"):
            provider.set_fallback_chain(DataSourceType.MARKET_DATA, ["nonexistent"])

    @pytest.mark.asyncio
    async def test_data_retrieval_success(self, provider, test_base_provider):
        """Test successful data retrieval."""
        provider.register_provider(test_base_provider)

        request = DataRequest(source_type=DataSourceType.MARKET_DATA, symbol="TEST")

        response = await provider.get_data(request)

        assert response.data == {"test": "data"}
        assert response.source == "test"
        assert response.quality == DataQuality.MEDIUM

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, provider):
        """Test fallback to another provider on failure."""

        class FailingProvider(BaseDataProvider):
            def __init__(self):
                super().__init__(name="failing", priority=10)

            def get_supported_sources(self):
                return [DataSourceType.MARKET_DATA]

            async def get_data(self, request):
                raise Exception("Provider failed")

        failing_provider = FailingProvider()
        mock_provider = MockProvider()

        provider.register_provider(failing_provider)
        provider.register_provider(mock_provider)

        provider.set_fallback_chain(DataSourceType.MARKET_DATA, ["failing", "mock"])

        request = DataRequest(source_type=DataSourceType.MARKET_DATA, symbol="TEST")

        response = await provider.get_data(request)

        # Should get response from mock provider
        assert response.source == "mock"
        assert isinstance(response.data, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, provider):
        """Test when all providers fail."""

        class FailingProvider(BaseDataProvider):
            def __init__(self, name):
                super().__init__(name=name, priority=10)

            def get_supported_sources(self):
                return [DataSourceType.MARKET_DATA]

            async def get_data(self, request):
                raise Exception(f"{self.name} failed")

        provider.register_provider(FailingProvider("failing1"))
        provider.register_provider(FailingProvider("failing2"))

        request = DataRequest(source_type=DataSourceType.MARKET_DATA, symbol="TEST")

        with pytest.raises(RuntimeError, match="All providers failed"):
            await provider.get_data(request)

    @pytest.mark.asyncio
    async def test_convenience_methods(self, provider):
        """Test convenience methods."""
        mock_provider = MockProvider()
        provider.register_provider(mock_provider)

        # Test market data convenience method
        response = await provider.get_market_data("TEST")
        assert isinstance(response.data, pd.DataFrame)

        # Test options chain convenience method
        response = await provider.get_options_chain("TEST")
        assert isinstance(response.data, list)

        # Test economic indicator convenience method
        response = await provider.get_economic_indicator("DGS3")
        assert isinstance(response.data, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_health_check_all(self, provider, test_base_provider):
        """Test health check on all providers."""
        provider.register_provider(test_base_provider)
        provider.register_provider(MockProvider())

        health_results = await provider.health_check_all()

        assert "test" in health_results
        assert "mock" in health_results
        assert health_results["test"]["status"] == "healthy"
        assert health_results["mock"]["status"] == "healthy"

    def test_performance_stats(self, provider, test_base_provider):
        """Test performance statistics."""
        provider.register_provider(test_base_provider)

        stats = provider.get_performance_stats()

        assert "total_requests" in stats
        assert "cache_hits" in stats
        assert "cache_hit_rate" in stats
        assert "registered_providers" in stats
        assert "test" in stats["registered_providers"]


class TestProviderImplementations:
    """Test specific provider implementations."""

    def test_databento_provider(self):
        """Test Databento provider initialization."""
        provider = DatabentoProvider()

        assert provider.name == "databento"
        assert provider.priority == 10
        assert DataSourceType.MARKET_DATA in provider.get_supported_sources()
        assert DataSourceType.OPTIONS_DATA in provider.get_supported_sources()

    def test_fred_provider(self):
        """Test FRED provider initialization."""
        provider = FREDProvider()

        assert provider.name == "fred"
        assert provider.priority == 20
        assert DataSourceType.ECONOMIC_DATA in provider.get_supported_sources()

    @pytest.mark.asyncio
    async def test_databento_provider_options_data(self):
        """Test Databento provider options data retrieval."""
        provider = DatabentoProvider()

        # Mock the integration
        with (
            patch("src.unity_wheel.data_providers.databento.client.DatabentoClient"),
            patch(
                "src.unity_wheel.data_providers.databento.integration.DatabentoIntegration"
            ) as mock_integration,
        ):

            mock_integration_instance = AsyncMock()
            mock_integration.return_value = mock_integration_instance
            mock_integration_instance.get_wheel_candidates.return_value = [
                {"strike": 35.0, "expiration": datetime.now(), "bid": 1.0}
            ]

            request = DataRequest(source_type=DataSourceType.OPTIONS_DATA, symbol="U")

            response = await provider.get_data(request)

            assert response.quality == DataQuality.HIGH
            assert response.source == "databento"
            assert isinstance(response.data, list)


class TestGlobalProvider:
    """Test global provider functions."""

    @pytest.mark.asyncio
    async def test_get_unified_provider(self):
        """Test getting the global unified provider."""
        provider = get_unified_provider()

        assert isinstance(provider, UnifiedDataProvider)
        assert "databento" in provider._providers
        assert "fred" in provider._providers
        assert "mock" in provider._providers

        # Should return same instance on second call
        provider2 = get_unified_provider()
        assert provider is provider2

    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test global convenience functions."""
        # Mock the provider responses
        with patch(
            "src.unity_wheel.data_providers.unified_provider.get_unified_provider"
        ) as mock_get_provider:
            mock_provider = AsyncMock()
            mock_get_provider.return_value = mock_provider

            # Mock market data response
            mock_provider.get_market_data.return_value = DataResponse(
                data=pd.DataFrame({"close": [35.0]}),
                metadata={},
                quality=DataQuality.HIGH,
                source="test",
            )

            # Mock options chain response
            mock_provider.get_options_chain.return_value = DataResponse(
                data=[{"strike": 35.0}], metadata={}, quality=DataQuality.HIGH, source="test"
            )

            # Mock economic indicator response
            mock_provider.get_economic_indicator.return_value = DataResponse(
                data=pd.DataFrame({"value": [0.05]}),
                metadata={},
                quality=DataQuality.HIGH,
                source="test",
            )

            # Test convenience functions
            market_response = await get_unity_market_data()
            assert isinstance(market_response.data, pd.DataFrame)

            options_response = await get_unity_options_chain()
            assert isinstance(options_response.data, list)

            rate = await get_risk_free_rate()
            assert rate == 0.05


@pytest.mark.integration
class TestProviderIntegration:
    """Integration tests for provider system."""

    @pytest.mark.asyncio
    async def test_end_to_end_market_data(self):
        """Test end-to-end market data retrieval."""
        provider = UnifiedDataProvider()
        provider.register_provider(MockProvider())

        request = DataRequest(
            source_type=DataSourceType.MARKET_DATA,
            symbol="U",
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now(),
        )

        response = await provider.get_data(request)

        assert isinstance(response.data, pd.DataFrame)
        assert not response.data.empty
        assert "close" in response.data.columns
        assert response.quality == DataQuality.LOW  # Mock data

    @pytest.mark.asyncio
    async def test_provider_priority_ordering(self):
        """Test that providers are called in priority order."""
        provider = UnifiedDataProvider()

        call_order = []

        class TestProvider(BaseDataProvider):
            def __init__(self, name, priority):
                super().__init__(name=name, priority=priority)

            def get_supported_sources(self):
                return [DataSourceType.MARKET_DATA]

            async def get_data(self, request):
                call_order.append(self.name)
                raise Exception(f"{self.name} failed")

        # Register providers with different priorities
        provider.register_provider(TestProvider("low_priority", 100))
        provider.register_provider(TestProvider("high_priority", 10))
        provider.register_provider(TestProvider("medium_priority", 50))
        provider.register_provider(MockProvider())  # Success provider

        request = DataRequest(source_type=DataSourceType.MARKET_DATA, symbol="TEST")

        response = await provider.get_data(request)

        # Should have tried providers in priority order
        assert call_order[0] == "high_priority"
        assert call_order[1] == "medium_priority"
        assert call_order[2] == "low_priority"

        # Should eventually succeed with mock provider
        assert response.source == "mock"
