"""
Tests for Schwab data ingestion with comprehensive validation and sanity checks.

DEPRECATED: This test file is for the deprecated data_ingestion module.
New tests should use the pull-when-asked architecture with unified storage.
See test_storage.py for tests of the new storage system.
"""

import asyncio
import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.loader import get_config

import pytest

from src.unity_wheel.data_providers.schwab.ingestion import (
    DataGranularity,
    DataRequirements,
    DataStorage,
    SchwabDataIngestion,
)
from src.unity_wheel.schwab.client import SchwabClient
from src.unity_wheel.schwab.types import PositionType, SchwabAccount, SchwabPosition

TICKER = get_config().unity.ticker


class TestDataStorage:
    """Test data storage functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DataStorage(Path(tmpdir) / "test.db")
            yield storage

    @pytest.mark.asyncio
    async def test_store_and_retrieve_positions(self, temp_storage):
        """Test storing and retrieving positions."""
        # Create test positions
        positions = [
            SchwabPosition(
                symbol=TICKER,
                quantity=Decimal("100"),
                position_type=PositionType.STOCK,
                market_value=Decimal("5000"),
                cost_basis=Decimal("4500"),
                unrealized_pnl=Decimal("500"),
                realized_pnl=Decimal("0"),
            ),
            SchwabPosition(
                symbol="U241220P00045000",
                quantity=Decimal("-1"),
                position_type=PositionType.OPTION,
                market_value=Decimal("-150"),
                cost_basis=Decimal("-200"),
                unrealized_pnl=Decimal("50"),
                realized_pnl=Decimal("0"),
                underlying="U",
                strike=Decimal("45"),
                expiration=datetime(2024, 12, 20),
                option_type="PUT",
            ),
        ]

        # Store positions
        await temp_storage.store_positions(positions, "TEST123")

        # Retrieve positions
        retrieved = await temp_storage.get_latest_positions("TEST123")

        assert len(retrieved) == 2
        assert retrieved[0]["symbol"] == "U"
        assert retrieved[0]["quantity"] == 100
        assert retrieved[1]["symbol"] == "U241220P00045000"
        assert retrieved[1]["strike"] == 45.0

    @pytest.mark.asyncio
    async def test_position_history(self, temp_storage):
        """Test retrieving position history."""
        # Store multiple snapshots
        for i in range(5):
            timestamp = datetime.now(timezone.utc) - timedelta(days=i)
            position = SchwabPosition(
                symbol=TICKER,
                quantity=Decimal("100"),
                position_type=PositionType.STOCK,
                market_value=Decimal(f"{5000 - i * 100}"),
                cost_basis=Decimal("4500"),
                unrealized_pnl=Decimal(f"{500 - i * 100}"),
                realized_pnl=Decimal("0"),
                last_update=timestamp,
            )

            # Manually insert with specific timestamp
            with sqlite3.connect(temp_storage.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO positions
                    (timestamp, account_number, symbol, quantity, position_type,
                     market_value, cost_basis, unrealized_pnl, realized_pnl, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        "TEST123",
                        position.symbol,
                        float(position.quantity),
                        position.position_type.value,
                        float(position.market_value),
                        float(position.cost_basis),
                        float(position.unrealized_pnl),
                        float(position.realized_pnl),
                        "{}",
                    ),
                )

        # Get history
        history = await temp_storage.get_position_history("U", days=30)

        assert len(history) == 5
        # Should be ordered by timestamp DESC
        assert history[0]["market_value"] == 5000
        assert history[4]["market_value"] == 4600

    def test_data_stats(self, temp_storage):
        """Test data statistics calculation."""
        # Insert test data
        with sqlite3.connect(temp_storage.db_path) as conn:
            # Positions
            for i in range(10):
                conn.execute(
                    """
                    INSERT INTO positions
                    (timestamp, account_number, symbol, quantity, position_type,
                     market_value, cost_basis, unrealized_pnl, realized_pnl, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        datetime.now(timezone.utc) - timedelta(days=i),
                        "TEST123",
                        f"SYMBOL{i % 3}",
                        100,
                        "STOCK",
                        5000,
                        4500,
                        500,
                        0,
                        "{}",
                    ),
                )

            # Orders
            for i in range(5):
                conn.execute(
                    """
                    INSERT INTO orders
                    (order_id, account_number, symbol, quantity, order_type,
                     status, price, entered_time, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        f"ORDER{i}",
                        "TEST123",
                        "U",
                        100,
                        "LIMIT",
                        "FILLED" if i < 3 else "CANCELLED",
                        50,
                        datetime.now(timezone.utc) - timedelta(days=i),
                        "{}",
                    ),
                )

        stats = temp_storage.get_data_stats()

        assert stats["positions"]["unique_symbols"] == 3
        assert stats["positions"]["days_with_data"] >= 9
        assert stats["orders"]["total_orders"] == 5
        assert stats["orders"]["filled_orders"] == 3


class TestDataIngestion:
    """Test data ingestion functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Schwab client."""
        client = AsyncMock(spec=SchwabClient)
        return client

    @pytest.fixture
    def temp_ingestion(self, mock_client):
        """Create ingestion instance with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DataStorage(Path(tmpdir) / "test.db")
            ingestion = SchwabDataIngestion(mock_client, storage)
            yield ingestion

    @pytest.mark.asyncio
    async def test_pull_all_data(self, temp_ingestion, mock_client):
        """Test comprehensive data pull."""
        # Mock responses
        mock_positions = [
            SchwabPosition(
                symbol=TICKER,
                quantity=Decimal("100"),
                position_type=PositionType.STOCK,
                market_value=Decimal("5000"),
                cost_basis=Decimal("4500"),
                unrealized_pnl=Decimal("500"),
                realized_pnl=Decimal("0"),
            )
        ]

        mock_account = SchwabAccount(
            account_number="TEST123",
            account_type="MARGIN",
            total_value=Decimal("100000"),
            cash_balance=Decimal("50000"),
            buying_power=Decimal("200000"),
            margin_balance=Decimal("50000"),
        )

        mock_client.get_positions.return_value = mock_positions
        mock_client.get_account.return_value = mock_account
        mock_client._make_request.return_value = {"orders": [], "transactions": []}

        # Pull data
        results = await temp_ingestion.pull_all_data("TEST123")

        assert results["positions"] == 1
        assert results["account_snapshots"] == 1
        assert results["errors"] == []

        # Verify data was stored
        stored_positions = await temp_ingestion.storage.get_latest_positions("TEST123")
        assert len(stored_positions) == 1
        assert stored_positions[0]["symbol"] == "U"

    @pytest.mark.asyncio
    async def test_rate_limiting(self, temp_ingestion, mock_client):
        """Test rate limiting behavior."""
        # Mock responses
        mock_client._make_request.return_value = {"orders": []}

        # Track call times
        call_times = []

        async def track_time(*args, **kwargs):
            call_times.append(datetime.now())
            return {"orders": []}

        mock_client._make_request.side_effect = track_time

        # Make multiple requests
        for _ in range(5):
            await temp_ingestion._pull_orders("TEST123", days=1)

        # Check that calls are rate limited
        # With our rate limit of ~1.67/sec, 5 calls should take ~3 seconds
        if len(call_times) > 1:
            total_time = (call_times[-1] - call_times[0]).total_seconds()
            assert total_time >= 2.0  # Should take at least 2 seconds

    @pytest.mark.asyncio
    async def test_data_validation(self, temp_ingestion):
        """Test data validation and completeness checks."""
        # Insert test data with gaps
        with sqlite3.connect(temp_ingestion.storage.db_path) as conn:
            # Good positions
            for i in [0, 1, 3, 4]:  # Skip day 2
                conn.execute(
                    """
                    INSERT INTO positions
                    (timestamp, account_number, symbol, quantity, position_type,
                     market_value, cost_basis, unrealized_pnl, realized_pnl, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        datetime.now(timezone.utc) - timedelta(days=i),
                        "TEST123",
                        "U",
                        100,
                        "STOCK",
                        5000,
                        4500,
                        500,
                        0,
                        "{}",
                    ),
                )

            # Add expired option
            conn.execute(
                """
                INSERT INTO positions
                (timestamp, account_number, symbol, quantity, position_type,
                 market_value, cost_basis, unrealized_pnl, realized_pnl,
                 expiration, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now(timezone.utc),
                    "TEST123",
                    "U231220P00045000",
                    -1,
                    "OPTION",
                    0,
                    -200,
                    200,
                    0,
                    "2023-12-20",
                    "{}",
                ),
            )

            # Add zero quantity position
            conn.execute(
                """
                INSERT INTO positions
                (timestamp, account_number, symbol, quantity, position_type,
                 market_value, cost_basis, unrealized_pnl, realized_pnl, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (datetime.now(timezone.utc), "TEST123", "AAPL", 0, "STOCK", 0, 0, 0, 0, "{}"),
            )

        # Run validation
        validation = await temp_ingestion._validate_data_completeness("TEST123")

        assert not validation["is_complete"]  # Should fail due to expired option
        assert len(validation["errors"]) >= 1  # Expired option
        assert len(validation["warnings"]) >= 2  # Missing days + zero quantity

        # Check specific warnings
        warning_messages = " ".join(validation["warnings"])
        assert "Zero quantity" in warning_messages
        assert "Missing position data" in warning_messages

        # Check specific errors
        error_messages = " ".join(validation["errors"])
        assert "Expired option" in error_messages


class TestDataSanityChecks:
    """Test data sanity checks and anomaly detection."""

    @pytest.mark.asyncio
    async def test_position_consistency(self):
        """Test position data consistency checks."""
        # Create positions with various issues
        positions = [
            # Normal position
            SchwabPosition(
                symbol=TICKER,
                quantity=Decimal("100"),
                position_type=PositionType.STOCK,
                market_value=Decimal("5000"),
                cost_basis=Decimal("4500"),
                unrealized_pnl=Decimal("500"),
                realized_pnl=Decimal("0"),
            ),
            # Suspicious: negative market value for long position
            SchwabPosition(
                symbol="AAPL",
                quantity=Decimal("50"),
                position_type=PositionType.STOCK,
                market_value=Decimal("-1000"),  # Should be positive
                cost_basis=Decimal("7500"),
                unrealized_pnl=Decimal("-8500"),
                realized_pnl=Decimal("0"),
            ),
            # Option with invalid strike
            SchwabPosition(
                symbol="U241220C00000000",
                quantity=Decimal("-1"),
                position_type=PositionType.OPTION,
                market_value=Decimal("-100"),
                cost_basis=Decimal("-150"),
                unrealized_pnl=Decimal("50"),
                realized_pnl=Decimal("0"),
                underlying="U",
                strike=Decimal("0"),  # Invalid
                expiration=datetime(2024, 12, 20),
                option_type="CALL",
            ),
        ]

        # Validate each position
        issues = []
        for pos in positions:
            if not pos.validate():
                issues.append(f"Invalid position: {pos.symbol}")

            # Additional sanity checks
            if pos.position_type == PositionType.STOCK:
                if pos.quantity > 0 and pos.market_value < 0:
                    issues.append(f"Long stock {pos.symbol} has negative market value")
                elif pos.quantity < 0 and pos.market_value > 0:
                    issues.append(f"Short stock {pos.symbol} has positive market value")

        assert len(issues) == 2
        assert "negative market value" in issues[0]
        assert "Invalid position" in issues[1]

    def test_transaction_reconciliation(self):
        """Test transaction data reconciliation."""
        # Sample transactions
        transactions = [
            {
                "transaction_id": "TX001",
                "transaction_type": "TRADE",
                "symbol": "U",
                "quantity": 100,
                "price": 45.00,
                "amount": -4500.00,
                "fees": -0.65,
            },
            {
                "transaction_id": "TX002",
                "transaction_type": "TRADE",
                "symbol": "U241220P00045000",
                "quantity": -1,
                "price": 2.00,
                "amount": 200.00,
                "fees": -0.65,
            },
            {
                "transaction_id": "TX003",
                "transaction_type": "OPTION_ASSIGNMENT",
                "symbol": "U241220P00045000",
                "quantity": 1,  # Close short put
                "price": 0,
                "amount": 0,
                "fees": 0,
            },
            {
                "transaction_id": "TX004",
                "transaction_type": "OPTION_ASSIGNMENT",
                "symbol": "U",
                "quantity": 100,  # Assigned shares
                "price": 45.00,
                "amount": -4500.00,
                "fees": 0,
            },
        ]

        # Reconcile transactions
        position_changes = {}
        cash_impact = 0

        for tx in transactions:
            symbol = tx["symbol"]

            # Update position quantities
            if symbol not in position_changes:
                position_changes[symbol] = 0
            position_changes[symbol] += tx["quantity"]

            # Update cash
            cash_impact += tx["amount"] + tx["fees"]

        # Verify reconciliation
        assert position_changes["U"] == 200  # Bought 100 + assigned 100
        assert position_changes["U241220P00045000"] == 0  # Sold 1, closed 1
        assert cash_impact == -8801.30  # Total cash outflow

    def test_corporate_action_detection(self):
        """Test detection of corporate actions from position anomalies."""
        positions = [
            # Odd lot - potential stock split
            SchwabPosition(
                symbol=TICKER,
                quantity=Decimal("150"),  # Not a round lot
                position_type=PositionType.STOCK,
                market_value=Decimal("7500"),
                cost_basis=Decimal("4500"),
                unrealized_pnl=Decimal("3000"),
                realized_pnl=Decimal("0"),
            ),
            # Non-standard option strike - potential adjustment
            SchwabPosition(
                symbol="U241220C00047500",
                quantity=Decimal("-1"),
                position_type=PositionType.OPTION,
                market_value=Decimal("-100"),
                cost_basis=Decimal("-150"),
                unrealized_pnl=Decimal("50"),
                realized_pnl=Decimal("0"),
                underlying="U",
                strike=Decimal("47.50"),  # Non-standard for U
                expiration=datetime(2024, 12, 20),
                option_type="CALL",
            ),
        ]

        # Detect anomalies
        anomalies = []

        for pos in positions:
            if pos.position_type == PositionType.STOCK:
                if pos.quantity % 100 != 0 and pos.quantity > 100:
                    anomalies.append(
                        {
                            "type": "ODD_LOT",
                            "symbol": pos.symbol,
                            "quantity": pos.quantity,
                            "message": "Potential stock split or dividend",
                        }
                    )

            elif pos.position_type == PositionType.OPTION:
                # Check for non-standard strikes (not divisible by standard intervals)
                if pos.strike and pos.strike % Decimal("2.5") != 0:
                    anomalies.append(
                        {
                            "type": "NON_STANDARD_STRIKE",
                            "symbol": pos.symbol,
                            "strike": pos.strike,
                            "message": "Potential contract adjustment",
                        }
                    )

        assert len(anomalies) == 2
        assert anomalies[0]["type"] == "ODD_LOT"
        assert anomalies[1]["type"] == "NON_STANDARD_STRIKE"


@pytest.mark.asyncio
async def test_full_data_pipeline():
    """Integration test of full data pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        storage = DataStorage(Path(tmpdir) / "test.db")
        mock_client = AsyncMock(spec=SchwabClient)

        # Configure requirements
        requirements = DataRequirements(
            positions_needed=True,
            position_history_days=30,
            orders_needed=True,
            order_history_days=7,
            transactions_needed=True,
            transaction_history_days=30,
            account_snapshots=True,
            account_snapshot_frequency=timedelta(hours=1),
        )

        ingestion = SchwabDataIngestion(mock_client, storage, requirements)

        # Mock comprehensive data
        mock_client.get_positions.return_value = [
            SchwabPosition(
                symbol=TICKER,
                quantity=Decimal("200"),
                position_type=PositionType.STOCK,
                market_value=Decimal("10000"),
                cost_basis=Decimal("9000"),
                unrealized_pnl=Decimal("1000"),
                realized_pnl=Decimal("500"),
            ),
            SchwabPosition(
                symbol="U241220P00045000",
                quantity=Decimal("-2"),
                position_type=PositionType.OPTION,
                market_value=Decimal("-400"),
                cost_basis=Decimal("-600"),
                unrealized_pnl=Decimal("200"),
                realized_pnl=Decimal("0"),
                underlying="U",
                strike=Decimal("45"),
                expiration=datetime(2024, 12, 20),
                option_type="PUT",
            ),
        ]

        mock_client.get_account.return_value = SchwabAccount(
            account_number="TEST123",
            account_type="MARGIN",
            total_value=Decimal("150000"),
            cash_balance=Decimal("40000"),
            buying_power=Decimal("300000"),
            margin_balance=Decimal("110000"),
            margin_requirement=Decimal("20000"),
        )

        mock_client._make_request.side_effect = [
            {
                "orders": [
                    {
                        "orderId": "ORD001",
                        "symbol": "U241220P00045000",
                        "quantity": 2,
                        "orderType": "LIMIT",
                        "status": "FILLED",
                        "price": 3.00,
                        "filledQuantity": 2,
                        "filledPrice": 3.00,
                        "enteredTime": datetime.now(timezone.utc).isoformat(),
                    }
                ]
            },
            {
                "transactions": [
                    {
                        "transactionId": "TXN001",
                        "transactionType": "TRADE",
                        "symbol": "U",
                        "quantity": 100,
                        "price": 45.00,
                        "amount": -4500.00,
                        "transactionDate": datetime.now(timezone.utc).isoformat(),
                    }
                ]
            },
            {"transactions": []},  # Other transaction types
            {"transactions": []},
            {"transactions": []},
        ]

        # Execute pipeline
        results = await ingestion.pull_all_data("TEST123")

        # Verify results
        assert results["positions"] == 2
        assert results["orders"] == 1
        assert results["transactions"] == 1
        assert results["account_snapshots"] == 1
        assert results["errors"] == []

        # Verify stored data
        stats = storage.get_data_stats()
        assert stats["positions"]["unique_symbols"] == 2

        # Verify data integrity
        positions = await storage.get_latest_positions("TEST123")
        assert len(positions) == 2

        # Check that stock and option positions are properly stored
        stock_pos = next(p for p in positions if p["symbol"] == "U")
        option_pos = next(p for p in positions if p["symbol"] == "U241220P00045000")

        assert stock_pos["quantity"] == 200
        assert stock_pos["unrealized_pnl"] == 1000
        assert option_pos["strike"] == 45.0
        assert option_pos["option_type"] == "PUT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
