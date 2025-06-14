"""
Databento Live API integration for real-time data
Used for ad hoc trading decisions when current data is needed
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import pandas as pd

import databento as db
from databento import ReconnectPolicy

from unity_wheel.utils import get_logger
from .types import OptionQuote, InstrumentDefinition

logger = get_logger(__name__)


class LiveStreamer:
    """Thin wrapper for Databento Live API with Unity-specific helpers"""
    
    def __init__(
        self,
        dataset: str,
        schema: str,
        symbols="ALL_SYMBOLS",
        *,
        stype_in="raw_symbol",
        start=None,
        snapshot=False,
        reconnect=ReconnectPolicy.RECONNECT,
        hb_interval=30,
        out_path: Optional[Path] = None,
    ):
        self.client = db.Live(
            reconnect_policy=reconnect,
            heartbeat_interval_s=hb_interval,
        )
        self.client.subscribe(
            dataset=dataset,
            schema=schema,
            symbols=symbols,
            stype_in=stype_in,
            start=start,
            snapshot=snapshot,
        )
        
        # Optional: dump raw DBN file for replay/debug
        if out_path:
            self.client.add_stream(out_path.open("wb"))
            
        # Log everything Databento emits
        logging.getLogger("databento").setLevel(logging.INFO)
        
        # Track reconnect gaps
        self.client.add_reconnect_callback(self._on_gap)
        
    def _on_gap(self, last_ts: pd.Timestamp, new_start: pd.Timestamp):
        """Handle data gaps on reconnect"""
        gap = new_start - last_ts
        logger.warning(f"DATA GAP {last_ts} → {new_start} ({gap})")
        
    def run_sync(self, handler: Callable[[db.DBNRecord], None]):
        """Blocking loop; handler(record) is called inline"""
        self.client.add_callback(handler)
        self.client.start()
        self.client.block_for_close()
        
    async def run_async(self, handler: Callable[[db.DBNRecord], Any]):
        """Awaitable loop; ideal inside larger asyncio app"""
        async for rec in self.client:
            await handler(rec)


class UnityLiveClient:
    """Unity-specific live data client for real-time trading decisions"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with optional API key override"""
        if api_key:
            self._api_key = api_key
        else:
            # Load from SecretManager
            from unity_wheel.secrets.manager import SecretManager
            secret_mgr = SecretManager()
            self._api_key = secret_mgr.get_secret("databento_api_key")
            
        if not self._api_key:
            raise ValueError("Databento API key not found")
            
        # Track current state
        self.book: Dict[int, Dict[str, tuple]] = {}  # instrument_id -> {bid/ask: (price, size)}
        self.trades: List[Dict[str, Any]] = []
        self.definitions: Dict[int, InstrumentDefinition] = {}
        
        logger.info("Initialized Unity Live Client")
        
    async def get_live_quotes(self, duration_seconds: int = 5) -> Dict[str, OptionQuote]:
        """Get live Unity option quotes for specified duration"""
        logger.info(f"Collecting live quotes for {duration_seconds} seconds...")
        
        # Clear previous data
        self.book.clear()
        self.trades.clear()
        
        # Create streamer for options
        streamer = LiveStreamer(
            dataset="OPRA.PILLAR",
            schema="mbp-1",  # Market by price
            symbols=["U.OPT"],
            stype_in="parent",
            snapshot=True,  # Get initial book state
        )
        
        # Collect data for specified duration
        start_time = datetime.now()
        
        def handle_quote(rec: db.DBNRecord):
            """Process incoming quote records"""
            if hasattr(rec, 'levels') and rec.levels:
                level = rec.levels[0]
                self.book.setdefault(rec.instrument_id, {})
                self.book[rec.instrument_id]['bid'] = (
                    float(level.bid_px) / 1e9,
                    level.bid_sz
                )
                self.book[rec.instrument_id]['ask'] = (
                    float(level.ask_px) / 1e9,
                    level.ask_sz
                )
                
        # Run collection in background
        collection_task = asyncio.create_task(
            self._collect_with_timeout(streamer, handle_quote, duration_seconds)
        )
        
        # Also get definitions
        await self._get_live_definitions()
        
        # Wait for collection to complete
        await collection_task
        
        # Convert to OptionQuote format
        quotes = {}
        for inst_id, levels in self.book.items():
            if inst_id in self.definitions:
                defn = self.definitions[inst_id]
                bid_price, bid_size = levels.get('bid', (0, 0))
                ask_price, ask_size = levels.get('ask', (0, 0))
                
                if bid_price > 0 and ask_price > 0:
                    quote = OptionQuote(
                        instrument_id=inst_id,
                        symbol=defn.symbol,
                        bid_price=bid_price,
                        ask_price=ask_price,
                        bid_size=bid_size,
                        ask_size=ask_size,
                        mid_price=(bid_price + ask_price) / 2,
                        timestamp=datetime.now()
                    )
                    
                    key = f"{defn.strike_price}_{defn.expiration.strftime('%Y%m%d')}_{defn.option_type}"
                    quotes[key] = quote
                    
        logger.info(f"Collected {len(quotes)} live quotes")
        return quotes
        
    async def get_live_stock_price(self, symbol: str = "U") -> Optional[float]:
        """Get current stock price from live trades"""
        logger.info(f"Getting live price for {symbol}...")
        
        latest_price = None
        
        def handle_trade(rec: db.DBNRecord):
            """Process trade records"""
            nonlocal latest_price
            if rec.schema == "trades":
                latest_price = float(rec.price) / 1e9
                
        streamer = LiveStreamer(
            dataset="EQUS.MINI",
            schema="trades",
            symbols=[symbol],
            stype_in="raw_symbol",
        )
        
        # Collect for 2 seconds
        await self._collect_with_timeout(streamer, handle_trade, 2)
        
        if latest_price:
            logger.info(f"Latest {symbol} price: ${latest_price:.2f}")
        else:
            logger.warning(f"No trades found for {symbol}")
            
        return latest_price
        
    async def _collect_with_timeout(self, streamer: LiveStreamer, handler: Callable, timeout_seconds: int):
        """Run collection with timeout"""
        try:
            # Run collection
            await asyncio.wait_for(
                streamer.run_async(handler),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            # Expected - we want to stop after timeout
            logger.info("Collection timeout reached (expected)")
            
    async def _get_live_definitions(self):
        """Get option definitions"""
        # For live trading, we might need to use Historical API for definitions
        # since they don't change during the day
        pass
        
    async def get_live_snapshot(self) -> Dict[str, Any]:
        """Get complete market snapshot for Unity"""
        logger.info("Getting live market snapshot...")
        
        # Get stock price
        stock_price = await self.get_live_stock_price()
        
        # Get option quotes (5 seconds of data)
        option_quotes = await self.get_live_quotes(duration_seconds=5)
        
        return {
            "timestamp": datetime.now(),
            "stock_price": stock_price,
            "option_quotes": option_quotes,
            "quote_count": len(option_quotes)
        }


class UnityTradingBot:
    """Example production trading bot using live data"""
    
    def __init__(self):
        self.stream = LiveStreamer(
            dataset="OPRA.PILLAR",
            schema="mbp-1",
            symbols=["U.OPT"],
            stype_in="parent",
            snapshot=True,
            reconnect=ReconnectPolicy.RECONNECT,
        )
        self.positions = {}
        self.signals = []
        
    async def on_record(self, rec: db.DBNRecord):
        """Process incoming market data"""
        # 1. Update internal book
        # 2. Check for trading signals
        # 3. Execute if conditions met
        pass
        
    def start(self):
        """Start the trading bot"""
        asyncio.run(self.stream.run_async(self.on_record))