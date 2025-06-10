import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..secrets.integration import get_schwab_credentials
from ..storage.cache.general_cache import IntelligentCache
from ..utils.logging import get_logger
from .exceptions import (
    SchwabAuthError,
    SchwabDataError,
    SchwabError,
    SchwabNetworkError,
    SchwabRateLimitError,
)
from .types import PositionType, SchwabAccount, SchwabPosition

logger = get_logger(__name__)


class SchwabClient:
    """Reliable Schwab API client with self-validation and error recovery."""

    # OCC option symbol pattern: AAPL  231215C00150000
    OCC_PATTERN = re.compile(
        r"^(?P<underlying>[A-Z]{1,6})\s*"
        r"(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})"
        r"(?P<type>[CP])"
        r"(?P<strike>\d{8})$"
    )

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: str = "https://localhost:8000/callback",
        cache_dir: Optional[Path] = None,
    ):
        # Use provided credentials or fall back to SecretManager
        if not client_id or not client_secret:
            logger.info("No credentials provided, retrieving from SecretManager")
            creds = get_schwab_credentials()
            client_id = client_id or creds["client_id"]
            client_secret = client_secret or creds["client_secret"]

        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

        # Set up caching
        self.cache_dir = cache_dir or Path.home() / ".schwab_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Never cache positions, brief cache for account metadata
        self.account_cache = IntelligentCache(
            max_size_mb=10.0,
            default_ttl=timedelta(seconds=30),
            persistence_path=None,  # Don't persist account data
        )

        # Store last known good state for fallback
        self.last_known_good: Dict[str, Any] = self._load_last_known_good()

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None

        logger.info(
            "SchwabClient initialized",
            extra={"client_id": client_id[:8] + "...", "cache_dir": str(self.cache_dir)},
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self):
        """Initialize session and authenticate."""
        if not self.session:
            from src.config.loader import get_config

            config = get_config()
            timeout = aiohttp.ClientTimeout(
                total=config.data.api_timeouts.total, connect=config.data.api_timeouts.connect
            )
            self.session = aiohttp.ClientSession(timeout=timeout)

        # Authenticate if needed
        if not self._is_authenticated():
            await self._authenticate()

    async def disconnect(self):
        """Clean up session."""
        if self.session:
            await self.session.close()
            self.session = None

    def _is_authenticated(self) -> bool:
        """Check if current auth token is valid."""
        if not self.access_token or not self.token_expiry:
            return False

        # Check with 5 minute buffer
        return datetime.now() < (self.token_expiry - timedelta(minutes=5))

    async def _authenticate(self):
        """Authenticate with Schwab OAuth."""
        # This is a placeholder - actual implementation would handle OAuth flow
        # For now, try to load from cache
        token_file = self.cache_dir / "token.json"

        if token_file.exists():
            try:
                with open(token_file) as f:
                    data = json.load(f)
                    self.access_token = data["access_token"]
                    self.token_expiry = datetime.fromisoformat(data["expires_at"])

                if self._is_authenticated():
                    logger.info("Loaded valid token from cache")
                    return
            except Exception as e:
                logger.warning(f"Failed to load cached token: {e}")

        # In production, this would handle the full OAuth flow
        raise SchwabAuthError("Authentication required - please implement OAuth flow")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((SchwabNetworkError, asyncio.TimeoutError)),
    )
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retries and error handling."""
        if not self.session:
            raise SchwabError("Client not connected")

        if not self._is_authenticated():
            await self._authenticate()

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

        url = f"https://api.schwabapi.com/v1/{endpoint}"

        try:
            async with self.session.request(method, url, headers=headers, **kwargs) as response:
                # Check for rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited, retry after {retry_after}s")
                    raise SchwabRateLimitError(f"Rate limited for {retry_after}s")

                # Check for auth errors
                if response.status == 401:
                    self.access_token = None  # Force re-auth
                    raise SchwabAuthError("Authentication failed")

                # Check for other errors
                if response.status >= 400:
                    text = await response.text()
                    raise SchwabError(f"API error {response.status}: {text}")

                return await response.json()

        except aiohttp.ClientError as e:
            logger.error(f"Network error: {e}")
            raise SchwabNetworkError(f"Network error: {e}")
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            raise SchwabNetworkError("Request timeout")

    async def get_positions(self, account_number: Optional[str] = None) -> List[SchwabPosition]:
        """Get all positions with validation and error handling."""
        logger.info("Fetching positions", extra={"account": account_number})

        try:
            # Get account number if not provided
            if not account_number:
                accounts = await self._get_accounts()
                if not accounts:
                    raise SchwabError("No accounts found")
                account_number = accounts[0]["accountNumber"]

            # Fetch positions
            data = await self._make_request("GET", f"accounts/{account_number}/positions")

            positions = []
            for item in data.get("positions", []):
                try:
                    position = self._parse_position(item)
                    if position.validate():
                        positions.append(position)
                    else:
                        logger.warning(
                            "Invalid position data",
                            extra={"symbol": position.symbol, "data": item},
                        )
                except Exception as e:
                    logger.error(f"Failed to parse position: {e}", extra={"data": item})

            # Validate total quantities
            self._validate_positions(positions)

            # Save as last known good
            self._save_last_known_good("positions", positions)

            logger.info(f"Fetched {len(positions)} positions successfully")
            return positions

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")

            # Try to return last known good
            if "positions" in self.last_known_good:
                logger.warning("Returning last known good positions")
                return self.last_known_good["positions"]

            raise

    def _parse_position(self, data: Dict[str, Any]) -> SchwabPosition:
        """Parse position data from API response."""
        symbol = data["symbol"]
        quantity = Decimal(str(data["quantity"]))

        # Determine position type
        position_type = self._determine_position_type(data)

        # Parse common fields
        position_data = {
            "symbol": symbol,
            "quantity": quantity,
            "position_type": position_type,
            "market_value": Decimal(str(data.get("marketValue", 0))),
            "cost_basis": Decimal(str(data.get("averagePrice", 0))) * abs(quantity),
            "unrealized_pnl": Decimal(str(data.get("unrealizedPnL", 0))),
            "realized_pnl": Decimal(str(data.get("realizedPnL", 0))),
            "raw_data": data,
        }

        # Parse option-specific fields
        if position_type == PositionType.OPTION:
            option_data = self._parse_option_symbol(symbol)
            if option_data:
                position_data.update(option_data)

        return SchwabPosition(**position_data)

    def _determine_position_type(self, data: Dict[str, Any]) -> PositionType:
        """Determine the type of position."""
        asset_type = data.get("assetType", "").upper()

        if asset_type == "EQUITY":
            return PositionType.STOCK
        elif asset_type == "OPTION":
            return PositionType.OPTION
        elif asset_type == "CASH_EQUIVALENT":
            return PositionType.CASH
        else:
            return PositionType.UNKNOWN

    def _parse_option_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Parse OCC option symbol format."""
        match = self.OCC_PATTERN.match(symbol.replace(" ", ""))
        if not match:
            logger.warning(f"Failed to parse option symbol: {symbol}")
            return None

        groups = match.groupdict()

        # Parse expiration date
        year = 2000 + int(groups["year"])
        month = int(groups["month"])
        day = int(groups["day"])
        expiration = datetime(year, month, day)

        # Parse strike price (last 8 digits represent price * 1000)
        strike = Decimal(groups["strike"]) / 1000

        return {
            "underlying": groups["underlying"],
            "strike": strike,
            "expiration": expiration,
            "option_type": "CALL" if groups["type"] == "C" else "PUT",
        }

    def _validate_positions(self, positions: List[SchwabPosition]):
        """Validate position data consistency."""
        # Check for duplicate symbols
        symbols = [p.symbol for p in positions]
        if len(symbols) != len(set(symbols)):
            logger.warning("Duplicate positions detected")

        # Validate option positions have proper data
        for position in positions:
            if position.is_option() and not position.underlying:
                logger.warning(f"Option position missing underlying: {position.symbol}")

        # Check for reasonable values
        total_value = sum(p.market_value for p in positions)
        if total_value < 0:
            logger.warning(f"Negative total portfolio value: {total_value}")

    async def get_account(self, account_number: Optional[str] = None) -> SchwabAccount:
        """Get account balances and buying power."""
        # Check cache first
        cache_key = account_number or "default"
        cached = self.account_cache.get(cache_key)
        if cached:
            logger.debug("Returning cached account data")
            return cached

        logger.info("Fetching account data", extra={"account": account_number})

        try:
            # Get account number if not provided
            if not account_number:
                accounts = await self._get_accounts()
                if not accounts:
                    raise SchwabError("No accounts found")
                account_number = accounts[0]["accountNumber"]

            # Fetch account data
            data = await self._make_request("GET", f"accounts/{account_number}")

            account = self._parse_account(data)

            if not account.validate():
                raise SchwabDataError("Invalid account data")

            # Cache the result
            self.account_cache.set(cache_key, account)

            # Save as last known good
            self._save_last_known_good("account", account)

            logger.info(
                "Fetched account data",
                extra={
                    "total_value": float(account.total_value),
                    "buying_power": float(account.buying_power),
                },
            )

            return account

        except Exception as e:
            logger.error(f"Failed to fetch account: {e}")

            # Try to return last known good
            if "account" in self.last_known_good:
                logger.warning("Returning last known good account data")
                return self.last_known_good["account"]

            raise

    def _parse_account(self, data: Dict[str, Any]) -> SchwabAccount:
        """Parse account data from API response."""
        account_data = data["securitiesAccount"]

        return SchwabAccount(
            account_number=account_data["accountNumber"],
            account_type=account_data.get("type", "UNKNOWN"),
            total_value=Decimal(
                str(account_data.get("currentBalances", {}).get("liquidationValue", 0))
            ),
            cash_balance=Decimal(
                str(account_data.get("currentBalances", {}).get("cashBalance", 0))
            ),
            buying_power=Decimal(
                str(account_data.get("currentBalances", {}).get("buyingPower", 0))
            ),
            margin_balance=Decimal(
                str(account_data.get("currentBalances", {}).get("marginBalance", 0))
            ),
            margin_requirement=Decimal(
                str(account_data.get("currentBalances", {}).get("maintenanceRequirement", 0))
            ),
            maintenance_requirement=Decimal(
                str(account_data.get("currentBalances", {}).get("maintenanceCall", 0))
            ),
            daily_pnl=Decimal(
                str(account_data.get("currentBalances", {}).get("dayTradingBuyingPower", 0))
            ),
            raw_data=data,
        )

    async def _get_accounts(self) -> List[Dict[str, Any]]:
        """Get list of accounts."""
        data = await self._make_request("GET", "accounts")
        return data.get("accounts", [])

    def detect_corporate_actions(self, positions: List[SchwabPosition]) -> List[Dict[str, Any]]:
        """Detect potential corporate actions from position anomalies."""
        actions = []

        # Group positions by underlying
        by_underlying: Dict[str, List[SchwabPosition]] = {}
        for position in positions:
            key = position.underlying or position.symbol
            if key not in by_underlying:
                by_underlying[key] = []
            by_underlying[key].append(position)

        for underlying, group in by_underlying.items():
            # Check for odd lot sizes (potential stock split)
            for position in group:
                if position.position_type == PositionType.STOCK:
                    if position.quantity % 100 != 0 and position.quantity > 100:
                        actions.append(
                            {
                                "type": "POTENTIAL_SPLIT",
                                "symbol": position.symbol,
                                "quantity": position.quantity,
                                "message": f"Odd lot size detected: {position.quantity}",
                            }
                        )

            # Check for mismatched option strikes (potential adjustment)
            option_positions = [p for p in group if p.position_type == PositionType.OPTION]

            if option_positions:
                strikes = {p.strike for p in option_positions if p.strike}
                # Check for non-standard strikes
                for strike in strikes:
                    if strike and strike % Decimal("0.50") != 0:
                        actions.append(
                            {
                                "type": "POTENTIAL_ADJUSTMENT",
                                "symbol": underlying,
                                "strike": strike,
                                "message": f"Non-standard strike detected: {strike}",
                            }
                        )

        return actions

    def _load_last_known_good(self) -> Dict[str, Any]:
        """Load last known good state from disk."""
        state_file = self.cache_dir / "last_known_good.json"

        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)

                # Reconstruct objects
                result = {}

                if "positions" in data:
                    result["positions"] = [
                        SchwabPosition(**self._deserialize_position(p)) for p in data["positions"]
                    ]

                if "account" in data:
                    result["account"] = SchwabAccount(**self._deserialize_account(data["account"]))

                logger.info("Loaded last known good state")
                return result

            except Exception as e:
                logger.warning(f"Failed to load last known good: {e}")

        return {}

    def _save_last_known_good(self, key: str, value: Any):
        """Save last known good state to disk."""
        self.last_known_good[key] = value

        # Serialize to JSON
        data = {}

        if "positions" in self.last_known_good:
            data["positions"] = [
                self._serialize_position(p) for p in self.last_known_good["positions"]
            ]

        if "account" in self.last_known_good:
            data["account"] = self._serialize_account(self.last_known_good["account"])

        # Save to disk
        state_file = self.cache_dir / "last_known_good.json"
        try:
            with open(state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save last known good: {e}")

    def _serialize_position(self, position: SchwabPosition) -> Dict[str, Any]:
        """Serialize position for JSON storage."""
        return {
            "symbol": position.symbol,
            "quantity": str(position.quantity),
            "position_type": position.position_type.value,
            "market_value": str(position.market_value),
            "cost_basis": str(position.cost_basis),
            "unrealized_pnl": str(position.unrealized_pnl),
            "realized_pnl": str(position.realized_pnl),
            "underlying": position.underlying,
            "strike": str(position.strike) if position.strike else None,
            "expiration": position.expiration.isoformat() if position.expiration else None,
            "option_type": position.option_type,
            "last_update": position.last_update.isoformat(),
        }

    def _deserialize_position(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize position from JSON storage."""
        result = {
            "symbol": data["symbol"],
            "quantity": Decimal(data["quantity"]),
            "position_type": PositionType(data["position_type"]),
            "market_value": Decimal(data["market_value"]),
            "cost_basis": Decimal(data["cost_basis"]),
            "unrealized_pnl": Decimal(data["unrealized_pnl"]),
            "realized_pnl": Decimal(data["realized_pnl"]),
            "underlying": data.get("underlying"),
            "strike": Decimal(data["strike"]) if data.get("strike") else None,
            "expiration": (
                datetime.fromisoformat(data["expiration"]) if data.get("expiration") else None
            ),
            "option_type": data.get("option_type"),
            "last_update": datetime.fromisoformat(data["last_update"]),
        }
        return result

    def _serialize_account(self, account: SchwabAccount) -> Dict[str, Any]:
        """Serialize account for JSON storage."""
        return {
            "account_number": account.account_number,
            "account_type": account.account_type,
            "total_value": str(account.total_value),
            "cash_balance": str(account.cash_balance),
            "buying_power": str(account.buying_power),
            "margin_balance": str(account.margin_balance),
            "margin_requirement": str(account.margin_requirement),
            "maintenance_requirement": str(account.maintenance_requirement),
            "daily_pnl": str(account.daily_pnl),
            "total_pnl": str(account.total_pnl),
            "last_update": account.last_update.isoformat(),
        }

    def _deserialize_account(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize account from JSON storage."""
        return {
            "account_number": data["account_number"],
            "account_type": data["account_type"],
            "total_value": Decimal(data["total_value"]),
            "cash_balance": Decimal(data["cash_balance"]),
            "buying_power": Decimal(data["buying_power"]),
            "margin_balance": Decimal(data["margin_balance"]),
            "margin_requirement": Decimal(data["margin_requirement"]),
            "maintenance_requirement": Decimal(data["maintenance_requirement"]),
            "daily_pnl": Decimal(data["daily_pnl"]),
            "total_pnl": Decimal(data["total_pnl"]),
            "last_update": datetime.fromisoformat(data["last_update"]),
        }
