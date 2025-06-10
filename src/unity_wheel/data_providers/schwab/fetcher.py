"""
Schwab data fetching module for pull-when-asked architecture.

This module provides simple functions to fetch data from Schwab API
on demand. All storage is handled by the unified storage layer.
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ...utils.logging import get_logger
from .auth_client import SchwabClient
from .types import SchwabAccount, SchwabPosition

logger = get_logger(__name__)


class SchwabDataFetcher:
    """Fetches data from Schwab API on demand."""

    def __init__(self, client: SchwabClient):
        self.client = client
        logger.info("Initialized Schwab data fetcher")

    async def fetch_all_data(self) -> Dict[str, Any]:
        """Fetch all relevant data from Schwab API.

        Returns:
            Dictionary with positions, account data, and metadata
        """
        logger.info("Fetching all Schwab data")

        # Fetch positions
        positions = await self.client.get_positions()

        # Fetch account data
        account = await self.client.get_account()

        # Check for corporate actions
        corporate_actions = self.client.detect_corporate_actions(positions)

        # Format response
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "positions": [self._position_to_dict(p) for p in positions],
            "account": self._account_to_dict(account),
            "corporate_actions": corporate_actions,
            "metadata": {
                "fetched_at": datetime.utcnow().isoformat(),
                "position_count": len(positions),
                "total_value": float(account.total_value),
            },
        }

    async def fetch_positions(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch current positions from Schwab.

        Returns:
            Dictionary with positions and metadata
        """
        logger.info("Fetching positions from Schwab")

        positions = await self.client.get_positions(account_id)

        return {
            "positions": [self._position_to_dict(p) for p in positions],
            "account_id": account_id or "default",
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(positions),
        }

    async def fetch_account(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch account data from Schwab.

        Returns:
            Dictionary with account data and metadata
        """
        logger.info("Fetching account data from Schwab")

        account = await self.client.get_account(account_id)

        return {
            "account": self._account_to_dict(account),
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_value": float(account.total_value),
                "cash_balance": float(account.cash_balance),
                "buying_power": float(account.buying_power),
            },
        }

    def _position_to_dict(self, position: SchwabPosition) -> Dict[str, Any]:
        """Convert position to dictionary format."""
        return {
            "symbol": position.symbol,
            "quantity": float(position.quantity),
            "position_type": position.position_type.value,
            "market_value": float(position.market_value),
            "cost_basis": float(position.cost_basis),
            "unrealized_pnl": float(position.unrealized_pnl),
            "realized_pnl": float(position.realized_pnl),
            "underlying": position.underlying,
            "strike": float(position.strike) if position.strike else None,
            "expiration": position.expiration.isoformat() if position.expiration else None,
            "option_type": position.option_type,
            "last_update": position.last_update.isoformat(),
        }

    def _account_to_dict(self, account: SchwabAccount) -> Dict[str, Any]:
        """Convert account to dictionary format."""
        return {
            "account_number": account.account_number,
            "account_type": account.account_type,
            "total_value": float(account.total_value),
            "cash_balance": float(account.cash_balance),
            "buying_power": float(account.buying_power),
            "margin_balance": float(account.margin_balance),
            "margin_requirement": float(account.margin_requirement),
            "maintenance_requirement": float(account.maintenance_requirement),
            "daily_pnl": float(account.daily_pnl),
            "total_pnl": float(account.total_pnl),
            "last_update": account.last_update.isoformat(),
        }


# Convenience function for backward compatibility
async def fetch_schwab_data(
    client_id: Optional[str] = None, client_secret: Optional[str] = None
) -> Dict[str, Any]:
    """Fetch all Schwab data using pull-when-asked pattern.

    Args:
        client_id: Optional Schwab client ID
        client_secret: Optional Schwab client secret

    Returns:
        Dictionary with all fetched data
    """
    async with SchwabClient(client_id, client_secret) as client:
        fetcher = SchwabDataFetcher(client)
        return await fetcher.fetch_all_data()
