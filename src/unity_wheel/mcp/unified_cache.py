"""
Unified caching system with L1 (RAM) and L2 (disk) layers.
Includes predictive cache warming and LRU eviction.
"""

import os
import json
import time
import hashlib
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from collections import OrderedDict
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import sqlite3


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    ttl_seconds: int


class TTLCache:
    """In-memory LRU cache with TTL support."""
    
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self.cache:
            self.stats['misses'] += 1
            return None
            
        entry = self.cache[key]
        
        # Check TTL
        if time.time() - entry.created_at > entry.ttl_seconds:
            del self.cache[key]
            self.stats['misses'] += 1
            return None
            
        # Update LRU order
        self.cache.move_to_end(key)
        entry.last_accessed = time.time()
        entry.access_count += 1
        
        self.stats['hits'] += 1
        return entry.value
        
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in cache with TTL."""
        # Calculate size (approximate)
        size_bytes = len(pickle.dumps(value))
        
        # Evict if needed
        while len(self.cache) >= self.maxsize:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats['evictions'] += 1
            
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
            ttl_seconds=ttl_seconds or self.ttl_seconds
        )
        
        self.cache[key] = entry
        
    def clear_expired(self):
        """Remove all expired entries."""
        current_time = time.time()
        expired_keys = [
            k for k, v in self.cache.items()
            if current_time - v.created_at > v.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(e.size_bytes for e in self.cache.values())
        return {
            **self.stats,
            'size': len(self.cache),
            'total_bytes': total_size,
            'hit_rate': self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses'])
        }


class DiskCache:
    """SQLite-based disk cache for L2 storage."""
    
    def __init__(self, cache_dir: str, size_limit_bytes: int = 1_000_000_000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.size_limit = size_limit_bytes
        self.db_path = self.cache_dir / "cache.db"
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    size_bytes INTEGER,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    ttl_seconds INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lru 
                ON cache_entries(last_accessed)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ttl 
                ON cache_entries(created_at, ttl_seconds)
            """)
            
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT value, created_at, ttl_seconds 
                FROM cache_entries 
                WHERE key = ?
            """, (key,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            value_blob, created_at, ttl_seconds = row
            
            # Check TTL
            if time.time() - created_at > ttl_seconds:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                return None
                
            # Update access time and count
            conn.execute("""
                UPDATE cache_entries 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE key = ?
            """, (time.time(), key))
            
            return pickle.loads(value_blob)
            
    def set(self, key: str, value: Any, ttl_seconds: int = 1800):
        """Set value in disk cache."""
        value_blob = pickle.dumps(value)
        size_bytes = len(value_blob)
        
        # Check size limit and evict if needed
        self._ensure_space(size_bytes)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache_entries
                (key, value, size_bytes, created_at, last_accessed, access_count, ttl_seconds)
                VALUES (?, ?, ?, ?, ?, 0, ?)
            """, (key, value_blob, size_bytes, time.time(), time.time(), ttl_seconds))
            
    def _ensure_space(self, needed_bytes: int):
        """Ensure enough space by evicting LRU entries."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Get current size
            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
            current_size = cursor.fetchone()[0] or 0
            
            if current_size + needed_bytes <= self.size_limit:
                return
                
            # Evict LRU entries
            space_to_free = (current_size + needed_bytes) - self.size_limit
            
            cursor = conn.execute("""
                SELECT key, size_bytes 
                FROM cache_entries 
                ORDER BY last_accessed ASC
            """)
            
            freed = 0
            keys_to_delete = []
            
            for key, size in cursor:
                keys_to_delete.append(key)
                freed += size
                if freed >= space_to_free:
                    break
                    
            if keys_to_delete:
                placeholders = ','.join('?' * len(keys_to_delete))
                conn.execute(f"DELETE FROM cache_entries WHERE key IN ({placeholders})", 
                           keys_to_delete)
                           
    def clear_expired(self):
        """Clear expired entries."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                DELETE FROM cache_entries 
                WHERE (? - created_at) > ttl_seconds
            """, (time.time(),))


class UnifiedCache:
    """
    Unified caching system with L1 (RAM) and L2 (disk) layers.
    Features predictive warming and intelligent eviction.
    """
    
    def __init__(self, 
                 l1_size: int = 1000,
                 l1_ttl: int = 300,  # 5 minutes
                 l2_size: int = 1_000_000_000,  # 1GB
                 l2_ttl: int = 1800,  # 30 minutes
                 cache_dir: str = ".cache"):
        
        self.l1_cache = TTLCache(maxsize=l1_size, ttl_seconds=l1_ttl)
        self.l2_cache = DiskCache(cache_dir, size_limit_bytes=l2_size)
        self.l2_ttl = l2_ttl
        
        # Query prediction
        self.query_history = []
        self.prediction_model = QueryPredictor()
        
        # Background tasks
        self._start_maintenance_tasks()
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache, checking L1 then L2.
        Average latency: L1=5ms, L2=50ms
        """
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value
            
        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.set(key, value)
            return value
            
        return None
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in both cache layers."""
        # Always set in L1
        self.l1_cache.set(key, value, ttl)
        
        # Set in L2 with longer TTL
        l2_ttl = (ttl or self.l1_cache.ttl_seconds) * 6  # 6x longer in L2
        self.l2_cache.set(key, value, min(l2_ttl, self.l2_ttl))
        
    def warm_predictive(self, query: str):
        """Warm cache with predicted follow-up queries."""
        # Record query for prediction
        self.query_history.append(query)
        
        # Get predictions
        predictions = self.prediction_model.predict_next(query, self.query_history)
        
        # Schedule warming tasks
        for predicted_query in predictions[:3]:  # Top 3 predictions
            asyncio.create_task(self._warm_query(predicted_query))
            
    async def _warm_query(self, query: str):
        """Warm cache for a specific query (placeholder)."""
        # This would call the actual compute function with minimal config
        pass
        
    def _get_cache_key(self, method: str, params: Dict[str, Any]) -> str:
        """Generate cache key from method and parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return f"{method}:{hashlib.md5(param_str.encode()).hexdigest()}"
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'l1': self.l1_cache.get_stats(),
            'l2': {
                'size_mb': self._get_l2_size() / 1_000_000,
                'entries': self._get_l2_count()
            },
            'predictions': {
                'accuracy': self.prediction_model.get_accuracy(),
                'total_predictions': len(self.query_history)
            }
        }
        
    def _get_l2_size(self) -> int:
        """Get L2 cache size in bytes."""
        with sqlite3.connect(str(self.l2_cache.db_path)) as conn:
            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
            return cursor.fetchone()[0] or 0
            
    def _get_l2_count(self) -> int:
        """Get L2 entry count."""
        with sqlite3.connect(str(self.l2_cache.db_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
            return cursor.fetchone()[0] or 0
            
    def _start_maintenance_tasks(self):
        """Start background maintenance tasks."""
        async def maintenance_loop():
            while True:
                await asyncio.sleep(300)  # Every 5 minutes
                self.l1_cache.clear_expired()
                self.l2_cache.clear_expired()
                
        asyncio.create_task(maintenance_loop())


class QueryPredictor:
    """Simple query prediction based on patterns."""
    
    def __init__(self):
        self.patterns = {
            'find': ['show usage', 'explain', 'where used'],
            'error': ['fix', 'debug', 'trace'],
            'implement': ['test', 'document', 'optimize'],
            'refactor': ['test', 'review', 'document']
        }
        self.accuracy_tracking = {'correct': 0, 'total': 0}
        
    def predict_next(self, current_query: str, history: List[str]) -> List[str]:
        """Predict likely follow-up queries."""
        predictions = []
        query_lower = current_query.lower()
        
        # Pattern-based prediction
        for trigger, follow_ups in self.patterns.items():
            if trigger in query_lower:
                for follow_up in follow_ups:
                    predicted = f"{current_query} {follow_up}"
                    predictions.append(predicted)
                    
        # N-gram based prediction from history
        if len(history) >= 2:
            # Look for repeated patterns
            last_pair = (history[-2], history[-1])
            for i in range(len(history) - 2):
                if (history[i], history[i+1]) == last_pair and i+2 < len(history):
                    predictions.append(history[i+2])
                    
        return predictions[:5]  # Top 5 predictions
        
    def get_accuracy(self) -> float:
        """Get prediction accuracy."""
        if self.accuracy_tracking['total'] == 0:
            return 0.0
        return self.accuracy_tracking['correct'] / self.accuracy_tracking['total']


# Convenience functions
def create_unified_cache() -> UnifiedCache:
    """Create optimized unified cache instance."""
    return UnifiedCache(
        l1_size=1000,
        l1_ttl=300,     # 5 min
        l2_size=1_000_000_000,  # 1GB
        l2_ttl=1800,    # 30 min
        cache_dir=".mcp_cache"
    )