"""Tests for Databento integration."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.unity_wheel.data_providers.databento.client import DatentoClient
from src.unity_wheel.data_providers.databento.types import (
    DataQuality,
    InstrumentDefinition,
    OptionChain,
    OptionQuote,
    OptionType,
    UnderlyingPrice,
)
from src.unity_wheel.data_providers.databento.validation import DataValidator


class TestDatentoTypes:
    """Test Databento data types."""

    def test_instrument_definition_from_databento(self):
        """Test creating instrument definition from raw data."""
        raw_data = {
            "instrument_id": 12345,
            "raw_symbol": "U 24 06 21 00055 C",
            "underlying": "U",
            "instrument_class": "C",
            "strike_price": 55000,  # In 1/1000ths
            "expiration": "2024-06-21T00:00:00",
            "multiplier": 100,
        }

        defn = InstrumentDefinition.from_databento(raw_data)

        assert defn.instrument_id == 12345
        assert defn.raw_symbol == "U 24 06 21 00055 C"
        assert defn.underlying == "U"
        assert defn.option_type == OptionType.CALL
        assert defn.strike_price == Decimal("55")
        assert defn.multiplier == 100

    def test_option_quote_calculations(self):
        """Test option quote derived calculations."""
        quote = OptionQuote(
            instrument_id=12345,
            timestamp=datetime.now(),
            bid_price=Decimal("2.50"),
            ask_price=Decimal("2.60"),
            bid_size=100,
            ask_size=150,
        )

        assert quote.mid_price == Decimal("2.55")
        assert quote.spread == Decimal("0.10")
        assert quote.spread_pct == Decimal("0.10") / Decimal("2.55") * 100

    def test_option_quote_from_databento(self):
        """Test creating quote from raw mbp-1 data."""
        raw_data = {
            "instrument_id": 12345,
            "ts_event": 1718035200000000000,  # Nanoseconds
            "levels": [
                {
                    "bid_px": 2500000000,  # 2.50 in 1e-9
                    "ask_px": 2600000000,  # 2.60 in 1e-9
                    "bid_sz": 100,
                    "ask_sz": 150,
                }
            ],
        }

        quote = OptionQuote.from_databento(raw_data)

        assert quote.instrument_id == 12345
        assert quote.bid_price == Decimal("2.50")
        assert quote.ask_price == Decimal("2.60")
        assert quote.bid_size == 100
        assert quote.ask_size == 150


# DataStorage tests removed - using new pull-when-asked architecture
# Tests for unified storage are in test_storage.py


class TestDataValidator:
    """Test data validation."""

    def test_completeness_check(self):
        """Test checking for missing trading days."""
        validator = DataValidator()

        # Create sample chains for weekdays
        chains = []
        start = datetime(2024, 6, 3)  # Monday

        for i in range(5):  # Monday through Friday
            chains.append(
                OptionChain(
                    underlying="U",
                    expiration=datetime(2024, 6, 21),
                    spot_price=Decimal("50"),
                    timestamp=start + timedelta(days=i),
                    calls=[],
                    puts=[],
                )
            )

        # Check completeness for the week
        is_complete, missing = validator.validate_historical_completeness(
            chains, start_date=start, end_date=start + timedelta(days=4)
        )

        assert is_complete
        assert len(missing) == 0

        # Remove Wednesday
        chains.pop(2)

        is_complete, missing = validator.validate_historical_completeness(
            chains, start_date=start, end_date=start + timedelta(days=4)
        )

        assert not is_complete
        assert len(missing) == 1
        assert "2024-06-05" in missing[0]

    def test_dummy_data_detection(self):
        """Test detection of dummy/test data patterns."""
        validator = DataValidator()

        chain = OptionChain(
            underlying="U",
            expiration=datetime.now() + timedelta(days=45),
            spot_price=Decimal("50"),
            timestamp=datetime.now(),
            calls=[],
            puts=[],
        )

        # Add suspiciously uniform data
        for i in range(10):
            chain.puts.append(
                OptionQuote(
                    instrument_id=i,
                    timestamp=datetime.now(),
                    bid_price=Decimal("1.00"),  # All same price
                    ask_price=Decimal("1.10"),  # All same spread
                    bid_size=100,  # All same size
                    ask_size=100,
                )
            )

        assert validator._detect_dummy_data(chain)

        # Add more realistic data with highly varied prices
        chain.puts = []
        import random

        random.seed(42)  # For reproducibility
        # Use realistic option prices with varying implied volatilities
        base_prices = [1.05, 1.23, 1.32, 1.58, 1.72, 1.95, 2.18, 2.35, 2.67, 2.89]
        for i, base_price in enumerate(base_prices):
            # Add some randomness to base price itself
            price_variation = random.uniform(-0.05, 0.05)
            actual_base = base_price + price_variation
            spread = 0.05 + random.uniform(0, 0.15)  # Wider spread variation
            chain.puts.append(
                OptionQuote(
                    instrument_id=i,
                    timestamp=datetime.now(),
                    bid_price=Decimal(f"{actual_base:.2f}"),
                    ask_price=Decimal(f"{actual_base + spread:.2f}"),
                    bid_size=90 + random.randint(0, 40),  # More varied sizes
                    ask_size=95 + random.randint(0, 50),
                )
            )

        assert not validator._detect_dummy_data(chain)

    def test_arbitrage_validation(self):
        """Test arbitrage bound checking."""
        validator = DataValidator()

        chain = OptionChain(
            underlying="U",
            expiration=datetime.now() + timedelta(days=45),
            spot_price=Decimal("50"),
            timestamp=datetime.now(),
            calls=[],
            puts=[],
        )

        # Create valid put-call pair at 50 strike
        definitions = [
            InstrumentDefinition(
                instrument_id=1,
                raw_symbol="U 24 06 21 00050 C",
                underlying="U",
                option_type=OptionType.CALL,
                strike_price=Decimal("50"),
                expiration=chain.expiration,
            ),
            InstrumentDefinition(
                instrument_id=2,
                raw_symbol="U 24 06 21 00050 P",
                underlying="U",
                option_type=OptionType.PUT,
                strike_price=Decimal("50"),
                expiration=chain.expiration,
            ),
        ]

        # Add quotes that violate put-call parity
        # C - P should be approximately S - K = 0 for ATM
        chain.calls.append(
            OptionQuote(
                instrument_id=1,
                timestamp=datetime.now(),
                bid_price=Decimal("5.00"),
                ask_price=Decimal("5.10"),
                bid_size=100,
                ask_size=100,
            )
        )

        chain.puts.append(
            OptionQuote(
                instrument_id=2,
                timestamp=datetime.now(),
                bid_price=Decimal("2.00"),  # C - P = 3, should be ~0
                ask_price=Decimal("2.10"),
                bid_size=100,
                ask_size=100,
            )
        )

        violations = validator._check_arbitrage_bounds(chain, definitions)
        assert len(violations) > 0
        assert "Put-call parity violation" in violations[0]


@pytest.mark.asyncio
class TestDatabentoClient:
    """Test Databento client functionality."""

    async def test_client_initialization(self):
        """Test client initialization."""
        with patch.dict("os.environ", {"DATABENTO_API_KEY": "test_key"}):
            client = DatabentoClient()
            assert client.api_key == "test_key"
            assert client.use_cache
            await client.close()

    async def test_rate_limiting(self):
        """Test rate limiting enforcement."""
        client = DatabentoClient(api_key="test_key")

        # Mock the underlying client
        client.client = Mock()

        # Track timing
        start = asyncio.get_event_loop().time()

        # Make rapid requests
        async def make_request():
            await client._rate_limit()

        # Should enforce rate limit
        await make_request()
        await make_request()

        elapsed = asyncio.get_event_loop().time() - start

        # Second request should be delayed
        assert elapsed >= client._request_interval

        await client.close()

    async def test_data_quality_validation(self):
        """Test data quality validation."""
        client = DatabentoClient(api_key="test_key")

        # Use fixed timestamps to ensure consistent staleness
        base_time = datetime(2025, 6, 8, 12, 0, 0)  # Fixed time
        chain_time = base_time - timedelta(seconds=30)

        chain = OptionChain(
            underlying="U",
            expiration=base_time + timedelta(days=45),
            spot_price=Decimal("50"),
            timestamp=chain_time,
            calls=[],
            puts=[],
        )

        # Add some quotes with varying quality
        for i in range(5):
            spread_pct = Decimal("2") if i < 3 else Decimal("15")  # Some wide

            chain.puts.append(
                OptionQuote(
                    instrument_id=i,
                    timestamp=base_time,  # Use fixed time
                    bid_price=Decimal("2.00"),
                    ask_price=Decimal("2.00") + (Decimal("2.00") * spread_pct / 100),
                    bid_size=100 if i < 3 else 5,  # Some illiquid
                    ask_size=100 if i < 3 else 5,
                )
            )

        # Mock datetime.now to return our base_time
        from datetime import timezone as tz

        with patch("src.unity_wheel.databento.client.datetime") as mock_datetime:
            # Mock now() to return our base_time with UTC timezone
            mock_datetime.now.return_value = base_time.replace(tzinfo=tz.utc)
            mock_datetime.timedelta = timedelta
            mock_datetime.timezone = tz

            quality = await client.validate_data_quality(chain)

        assert isinstance(quality, DataQuality)
        assert quality.symbol == "U"
        assert not quality.bid_ask_spread_ok  # Has wide spreads
        assert not quality.sufficient_liquidity  # Has low sizes
        assert quality.data_staleness_seconds == pytest.approx(30, abs=5)
        assert quality.confidence_score < 1.0

        await client.close()
