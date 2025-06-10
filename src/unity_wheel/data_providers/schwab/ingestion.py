"""
Schwab API data ingestion module for wheel trading strategy.

Implements efficient data pulling with rate limiting, caching, and validation
for positions, orders, transactions, and account data needed for wheel strategy decisions.
"""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


from src.unity_wheel.auth.rate_limiter import RateLimiter
from src.unity_wheel.storage.cache.general_cache import IntelligentCache
from src.unity_wheel.utils.data_validator import die
from src.unity_wheel.utils.logging import StructuredLogger, get_logger, timed_operation
from src.unity_wheel.utils.recovery import RecoveryStrategy, with_recovery

from .auth_client import SchwabClient
from .types import SchwabAccount, SchwabPosition

logger = get_logger(__name__)
structured_logger = StructuredLogger(__name__)


class DataGranularity(str, Enum):
    """Granularity levels for data collection."""

    SNAPSHOT = "snapshot"  # Current state only
    DAILY = "daily"  # Daily aggregates
    INTRADAY = "intraday"  # 5-minute intervals
    TICK = "tick"  # Every change (streaming)


@dataclass
class DataRequirements:
    """Data requirements for wheel strategy."""

    # Position data
    positions_needed: bool = True
    position_history_days: int = 90

    # Order data
    orders_needed: bool = True
    order_history_days: int = 30
    filled_orders_only: bool = False

    # Transaction data
    transactions_needed: bool = True
    transaction_history_days: int = 90
    transaction_types: List[str] = None

    # Account data
    account_snapshots: bool = True
    account_snapshot_frequency: timedelta = timedelta(hours=1)

    # Options chain data
    options_chain_needed: bool = True
    option_expiry_range_days: int = 60
    option_strike_range_percent: float = 0.20  # +/- 20% from current price

    def __post_init__(self):
        if self.transaction_types is None:
            self.transaction_types = ["TRADE", "DIVIDEND", "OPTION_ASSIGNMENT", "OPTION_EXERCISE"]


class DataStorage:
    """
    Local SQLite storage for Schwab data.

    Design decisions:
    - SQLite for local single-user operation (no cloud dependency)
    - Separate tables for different data types
    - Indexes on commonly queried fields
    - JSON columns for raw API responses (debugging/audit)
    """

    def __init__(self, db_path: Path = None):
        """Initialize local storage."""
        self.db_path = db_path or Path.home() / ".wheel_trading" / "schwab_data.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Positions table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    account_number TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    position_type TEXT NOT NULL,
                    market_value REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    underlying TEXT,
                    strike REAL,
                    expiration DATE,
                    option_type TEXT,
                    raw_data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, account_number, symbol)
                )
            """
            )

            # Orders table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    account_number TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    price REAL,
                    filled_quantity INTEGER,
                    filled_price REAL,
                    entered_time TIMESTAMP NOT NULL,
                    close_time TIMESTAMP,
                    raw_data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Transactions table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE NOT NULL,
                    account_number TEXT NOT NULL,
                    transaction_date TIMESTAMP NOT NULL,
                    transaction_type TEXT NOT NULL,
                    symbol TEXT,
                    quantity REAL,
                    price REAL,
                    amount REAL NOT NULL,
                    fees REAL,
                    description TEXT,
                    raw_data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Account snapshots table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    account_number TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    buying_power REAL NOT NULL,
                    margin_balance REAL NOT NULL,
                    margin_requirement REAL NOT NULL,
                    maintenance_requirement REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    raw_data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, account_number)
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_positions_timestamp ON positions(timestamp)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_entered_time ON orders(entered_time)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_account_snapshots_timestamp ON account_snapshots(timestamp)"
            )

            conn.commit()

    async def store_positions(self, positions: List[SchwabPosition], account_number: str):
        """Store position snapshot."""
        timestamp = datetime.now(timezone.utc)

        with sqlite3.connect(self.db_path) as conn:
            for pos in positions:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO positions
                    (timestamp, account_number, symbol, quantity, position_type,
                     market_value, cost_basis, unrealized_pnl, realized_pnl,
                     underlying, strike, expiration, option_type, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        account_number,
                        pos.symbol,
                        float(pos.quantity),
                        pos.position_type.value,
                        float(pos.market_value),
                        float(pos.cost_basis),
                        float(pos.unrealized_pnl),
                        float(pos.realized_pnl),
                        pos.underlying,
                        float(pos.strike) if pos.strike else None,
                        pos.expiration.isoformat() if pos.expiration else None,
                        pos.option_type,
                        json.dumps(pos.raw_data),
                    ),
                )

            conn.commit()

        logger.info(f"Stored {len(positions)} positions for account {account_number}")

    async def store_account_snapshot(self, account: SchwabAccount):
        """Store account snapshot."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO account_snapshots
                (timestamp, account_number, account_type, total_value,
                 cash_balance, buying_power, margin_balance, margin_requirement,
                 maintenance_requirement, daily_pnl, total_pnl, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    account.last_update,
                    account.account_number,
                    account.account_type,
                    float(account.total_value),
                    float(account.cash_balance),
                    float(account.buying_power),
                    float(account.margin_balance),
                    float(account.margin_requirement),
                    float(account.maintenance_requirement),
                    float(account.daily_pnl),
                    float(account.total_pnl),
                    json.dumps(account.raw_data),
                ),
            )

            conn.commit()

    async def get_latest_positions(self, account_number: str) -> List[Dict[str, Any]]:
        """Get most recent position snapshot."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM positions
                WHERE account_number = ?
                AND timestamp = (
                    SELECT MAX(timestamp) FROM positions WHERE account_number = ?
                )
                ORDER BY symbol
            """,
                (account_number, account_number),
            )

            return [dict(row) for row in cursor.fetchall()]

    async def get_position_history(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get position history for a symbol."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM positions
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """,
                (symbol, cutoff),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            stats = {}

            # Position stats
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT symbol) as unique_symbols,
                       COUNT(DISTINCT date(timestamp)) as days_with_data,
                       MIN(timestamp) as earliest_data,
                       MAX(timestamp) as latest_data
                FROM positions
            """
            )
            row = cursor.fetchone()
            stats["positions"] = dict(row) if row else {}

            # Order stats
            cursor = conn.execute(
                """
                SELECT COUNT(*) as total_orders,
                       COUNT(DISTINCT symbol) as unique_symbols,
                       SUM(CASE WHEN status = 'FILLED' THEN 1 ELSE 0 END) as filled_orders
                FROM orders
            """
            )
            row = cursor.fetchone()
            stats["orders"] = dict(row) if row else {}

            # Transaction stats
            cursor = conn.execute(
                """
                SELECT COUNT(*) as total_transactions,
                       COUNT(DISTINCT transaction_type) as transaction_types,
                       MIN(transaction_date) as earliest,
                       MAX(transaction_date) as latest
                FROM transactions
            """
            )
            row = cursor.fetchone()
            stats["transactions"] = dict(row) if row else {}

            # Storage size
            stats["storage_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)

            return stats


class SchwabDataIngestion:
    """
    Main class for ingesting Schwab data for wheel trading.

    Features:
    - Intelligent rate limiting (3-4 req/sec sustained)
    - Efficient pagination for large datasets
    - Delta pulls to minimize API calls
    - Local SQLite storage
    - Comprehensive data validation
    """

    def __init__(
        self,
        client: SchwabClient,
        storage: Optional[DataStorage] = None,
        requirements: Optional[DataRequirements] = None,
    ):
        """Initialize data ingestion."""
        self.client = client
        self.storage = storage or DataStorage()
        self.requirements = requirements or DataRequirements()

        # Rate limiter: 100 requests per minute with burst of 10
        self.rate_limiter = RateLimiter(
            requests_per_second=100 / 60,  # ~1.67 per second
            burst_capacity=10,
            enable_circuit_breaker=True,
        )

        # Track last pull times for delta updates
        self.last_pull_times: Dict[str, datetime] = {}

        # Cache for avoiding duplicate API calls
        self.cache = IntelligentCache(max_size_mb=50.0, default_ttl=timedelta(minutes=5))

        logger.info(
            "SchwabDataIngestion initialized",
            extra={
                "storage_path": str(self.storage.db_path),
                "requirements": asdict(self.requirements),
            },
        )

    @timed_operation(threshold_ms=5000.0)
    @with_recovery(strategy=RecoveryStrategy.RETRY)
    async def pull_all_data(self, account_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Pull all required data for wheel strategy.

        Returns summary of data pulled.
        """
        logger.info(
            "Starting comprehensive data pull",
            extra={
                "function": "pull_all_data",
                "account": account_number,
                "requirements": asdict(self.requirements),
            },
        )

        results = {
            "positions": 0,
            "orders": 0,
            "transactions": 0,
            "account_snapshots": 0,
            "errors": [],
            "warnings": [],
        }

        try:
            # Pull positions (never cached, always fresh)
            if self.requirements.positions_needed:
                positions = await self._pull_positions(account_number)
                results["positions"] = len(positions)
                await self.storage.store_positions(positions, account_number)

            # Pull account snapshot
            if self.requirements.account_snapshots:
                account = await self._pull_account(account_number)
                results["account_snapshots"] = 1
                await self.storage.store_account_snapshot(account)

            # Pull orders (with smart pagination)
            if self.requirements.orders_needed:
                orders = await self._pull_orders(
                    account_number, days=self.requirements.order_history_days
                )
                results["orders"] = len(orders)

            # Pull transactions (with delta logic)
            if self.requirements.transactions_needed:
                transactions = await self._pull_transactions(
                    account_number, days=self.requirements.transaction_history_days
                )
                results["transactions"] = len(transactions)

            # Validate data completeness
            validation_results = await self._validate_data_completeness(account_number)
            results["validation"] = validation_results

            if validation_results["warnings"]:
                results["warnings"].extend(validation_results["warnings"])

            logger.info(
                "Data pull completed successfully",
                extra={
                    "function": "pull_all_data",
                    "results": results,
                    "duration_seconds": None,  # Will be filled by timed_operation
                },
            )

        except Exception as e:
            logger.error(f"Error in data pull: {e}")
            results["errors"].append(str(e))
            raise

        return results

    async def _pull_positions(self, account_number: str) -> List[SchwabPosition]:
        """Pull current positions with rate limiting."""
        await self.rate_limiter.acquire()
        return await self.client.get_positions(account_number)

    async def _pull_account(self, account_number: str) -> SchwabAccount:
        """Pull account snapshot."""
        # Check if we need a new snapshot
        cache_key = f"last_account_pull_{account_number}"
        last_pull = self.cache.get(cache_key)

        if last_pull:
            time_since_last = datetime.now(timezone.utc) - last_pull
            if time_since_last < self.requirements.account_snapshot_frequency:
                logger.debug(f"Skipping account pull, last pulled {time_since_last.seconds}s ago")
                # Return cached account data
                return await self.client.get_account(account_number)

        await self.rate_limiter.acquire()
        account = await self.client.get_account(account_number)

        # Update last pull time
        self.cache.set(cache_key, datetime.now(timezone.utc))

        return account

    async def _pull_orders(self, account_number: str, days: int) -> List[Dict[str, Any]]:
        """
        Pull orders with pagination.

        Schwab limits to 200 orders per request.
        """
        all_orders = []
        from_date = datetime.now(timezone.utc) - timedelta(days=days)
        to_date = datetime.now(timezone.utc)

        while True:
            await self.rate_limiter.acquire()

            params = {
                "fromEnteredTime": from_date.isoformat(),
                "toEnteredTime": to_date.isoformat(),
                "maxResults": 200,
            }

            if self.requirements.filled_orders_only:
                params["status"] = "FILLED"

            orders = await self.client._make_request(
                "GET", f"accounts/{account_number}/orders", params=params
            )

            order_list = orders.get("orders", [])
            all_orders.extend(order_list)

            # Check if we got all orders
            if len(order_list) < 200:
                break

            # Update to_date for next page
            if order_list:
                # Validate order has enteredTime field
                if "enteredTime" not in order_list[-1]:
                    die(f"Missing 'enteredTime' in order data for account {account_number}")

                last_order_time = order_list[-1]["enteredTime"]
                to_date = datetime.fromisoformat(last_order_time.replace("Z", "+00:00"))

                # Avoid infinite loop
                if to_date <= from_date:
                    break

        logger.info(f"Pulled {len(all_orders)} orders for account {account_number}")
        return all_orders

    async def _pull_transactions(self, account_number: str, days: int) -> List[Dict[str, Any]]:
        """
        Pull transactions with delta logic.

        Only pulls new transactions since last pull.
        """
        # Check last transaction pull time
        last_pull_key = f"last_transaction_pull_{account_number}"
        last_pull = self.last_pull_times.get(last_pull_key)

        if last_pull:
            # Delta pull - only get new transactions
            from_date = last_pull
            logger.info(f"Delta pull: transactions since {from_date}")
        else:
            # Full pull
            from_date = datetime.now(timezone.utc) - timedelta(days=days)
            logger.info(f"Full pull: transactions for last {days} days")

        to_date = datetime.now(timezone.utc)
        all_transactions = []

        # Pull transactions by type for better organization
        for tx_type in self.requirements.transaction_types:
            await self.rate_limiter.acquire()

            params = {
                "fromDate": from_date.isoformat(),
                "toDate": to_date.isoformat(),
                "type": tx_type,
            }

            transactions = await self.client._make_request(
                "GET", f"accounts/{account_number}/transactions", params=params
            )

            tx_list = transactions.get("transactions", [])
            all_transactions.extend(tx_list)

            logger.debug(f"Pulled {len(tx_list)} {tx_type} transactions")

        # Update last pull time
        self.last_pull_times[last_pull_key] = to_date

        logger.info(f"Pulled {len(all_transactions)} total transactions")
        return all_transactions

    async def _validate_data_completeness(self, account_number: str) -> Dict[str, Any]:
        """
        Validate data completeness and consistency.

        Checks for:
        - Missing days
        - Duplicate entries
        - Data anomalies
        - Stale data
        """
        validation_results = {"is_complete": True, "warnings": [], "errors": [], "stats": {}}

        # Get data statistics
        stats = self.storage.get_data_stats()
        validation_results["stats"] = stats

        # Check for stale data
        if stats["positions"]["latest_data"]:
            latest = datetime.fromisoformat(stats["positions"]["latest_data"])
            age = datetime.now(timezone.utc) - latest

            if age > timedelta(minutes=30):
                validation_results["warnings"].append(
                    f"Position data is {age.total_seconds()/60:.1f} minutes old"
                )

        # Check for data gaps
        with sqlite3.connect(self.storage.db_path) as conn:
            # Check for missing days in position history
            cursor = conn.execute(
                """
                WITH RECURSIVE dates(date) AS (
                    SELECT DATE('now', '-30 days')
                    UNION ALL
                    SELECT DATE(date, '+1 day')
                    FROM dates
                    WHERE date < DATE('now')
                )
                SELECT date FROM dates
                WHERE date NOT IN (
                    SELECT DISTINCT DATE(timestamp) FROM positions
                )
                AND date <= DATE('now', '-1 day')
            """
            )

            missing_days = [row[0] for row in cursor.fetchall()]

            if missing_days:
                validation_results["warnings"].append(
                    f"Missing position data for {len(missing_days)} days: {missing_days[:5]}..."
                )

        # Check for position consistency
        latest_positions = await self.storage.get_latest_positions(account_number)

        # Validate each position
        for pos in latest_positions:
            # Check for zero quantities (should be cleaned up)
            if pos["quantity"] == 0:
                validation_results["warnings"].append(
                    f"Zero quantity position found: {pos['symbol']}"
                )

            # Check for stale option positions
            if pos["expiration"]:
                exp_date = datetime.fromisoformat(pos["expiration"])
                if exp_date.date() < datetime.now(timezone.utc).date():
                    validation_results["errors"].append(
                        f"Expired option position still in account: {pos['symbol']}"
                    )

        # Determine overall completeness
        if validation_results["errors"]:
            validation_results["is_complete"] = False

        return validation_results

    # DEPRECATED: Continuous sync removed in favor of pull-when-asked architecture
    # Use the new on-demand pattern:
    #   from unity_wheel.storage import Storage
    #   storage = Storage()
    #   data = await storage.get_or_fetch_positions(...)


# DEPRECATED: This module is kept for reference but should not be used.
# Use the new pull-when-asked pattern with unified storage:
#
# from unity_wheel.schwab import fetch_schwab_data
# from unity_wheel.storage import Storage
#
# storage = Storage()
# await storage.initialize()
# data = await storage.get_or_fetch_positions(
#     account_id="default",
#     fetch_func=fetch_schwab_data,
#     max_age_minutes=30
# )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEPRECATED: This module is deprecated in favor of pull-when-asked architecture.")
    print("\nUse the new pattern instead:")
    print("\n  from unity_wheel.storage import Storage")
    print("  from unity_wheel.schwab import SchwabDataFetcher")
    print("\n  storage = Storage()")
    print("  await storage.initialize()")
    print("  data = await storage.get_or_fetch_positions(")
    print("      account_id='default',")
    print("      fetch_func=fetcher.fetch_all_data,")
    print("      max_age_minutes=30")
    print("  )")
    print("=" * 80 + "\n")
    raise DeprecationWarning("This module is deprecated. See message above.")
