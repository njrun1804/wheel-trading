from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.unity_wheel.cli.databento_integration import (
    create_databento_market_snapshot,
    get_market_data_sync,
)


@pytest.mark.asyncio
async def test_create_snapshot_spot_failure():
    """Ensure spot price fetch failures surface as ValueError."""
    mock_client = AsyncMock()
    mock_client._get_underlying_price.side_effect = RuntimeError("boom")
    mock_client.close = AsyncMock()

    with (
        patch("src.unity_wheel.cli.databento_integration.SecretInjector"),
        patch(
            "src.unity_wheel.cli.databento_integration.DatabentoClient",
            return_value=mock_client,
        ),
        patch("src.unity_wheel.cli.databento_integration.DatabentoIntegration"),
    ):
        with pytest.raises(ValueError, match="Failed to get Unity spot price"):
            await create_databento_market_snapshot(100_000.0, "U", risk_free_rate=0.05)
    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_snapshot_no_candidates():
    """Ensure missing option candidates raises ValueError."""
    mock_client = AsyncMock()
    mock_client._get_underlying_price.return_value = Mock(last_price=35.0)
    mock_client.close = AsyncMock()

    mock_integration = AsyncMock()
    mock_integration.get_wheel_candidates.return_value = []

    with (
        patch("src.unity_wheel.cli.databento_integration.SecretInjector"),
        patch(
            "src.unity_wheel.cli.databento_integration.DatabentoClient",
            return_value=mock_client,
        ),
        patch(
            "src.unity_wheel.cli.databento_integration.DatabentoIntegration",
            return_value=mock_integration,
        ),
    ):
        with pytest.raises(ValueError, match="No Unity options found"):
            await create_databento_market_snapshot(100_000.0, "U", risk_free_rate=0.05)
    mock_client.close.assert_awaited_once()


def test_sync_wrapper_propagates_errors():
    """get_market_data_sync should raise same errors as async variant."""
    with patch(
        "src.unity_wheel.cli.databento_integration.create_databento_market_snapshot",
        side_effect=ValueError("bad"),
    ):
        with pytest.raises(ValueError, match="bad"):
            get_market_data_sync(1.0, risk_free_rate=0.05)
