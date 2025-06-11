import asyncio
from datetime import datetime

from src.unity_wheel.storage.duckdb_cache import CacheConfig, DuckDBCache


def test_eviction_by_size(tmp_path):
    """Ensure LRU eviction keeps DB below size limit."""

    async def _run():
        config = CacheConfig(cache_dir=tmp_path, max_size_gb=0.002, vacuum_interval_hours=9999)
        cache = DuckDBCache(config)
        cache._last_vacuum = datetime.utcnow()
        await cache.initialize()

        sample = {"data": "x" * 5000}
        for i in range(50):
            await cache.store_option_chain("TST", datetime.utcnow(), datetime.utcnow(), 1.0, sample)
            await cache.store_positions("ACC", [{"s": i}], {"cash": i})

        size_limit = config.max_size_gb * 1024**3
        assert cache.db_path.stat().st_size <= size_limit

    asyncio.run(_run())
