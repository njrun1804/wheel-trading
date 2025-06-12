"""Slice Cache - SHA-1 keyed vector storage for code embeddings."""

import asyncio
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


class SliceCache:
    """High-performance cache for code slice embeddings and metadata."""
    
    def __init__(self, workspace_root: str, cache_size_mb: int = 512):
        self.workspace_root = Path(workspace_root)
        self.cache_dir = self.workspace_root / ".claude" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "slice_cache.db"
        self.cache_size_mb = cache_size_mb
        
        # Thread pool for background operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._shutdown = False
        
        # In-memory LRU cache for hot entries
        self.memory_cache: dict[str, dict[str, Any]] = {}
        self.access_times: dict[str, float] = {}
        self.max_memory_entries = 1000
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    async def initialize(self):
        """Initialize database and indexes."""
        await asyncio.get_event_loop().run_in_executor(
            self.executor, self._init_db
        )
        
    def _init_db(self):
        """Initialize SQLite database with slice cache table."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        
        # Create main table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS slice_cache (
                hash TEXT PRIMARY KEY,
                vector BLOB,
                metadata TEXT,
                file_path TEXT,
                content TEXT,
                size_bytes INTEGER,
                created_at TIMESTAMP,
                accessed_at TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Create indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path ON slice_cache(file_path)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_accessed_at ON slice_cache(accessed_at)
        """)
        
        conn.commit()
        conn.close()
        
    async def store(self, slice_hash: str, data: dict[str, Any], 
                   vector: np.ndarray | None = None) -> bool:
        """Store slice with optional embedding vector."""
        # Add to memory cache first
        self.memory_cache[slice_hash] = data
        self.access_times[slice_hash] = time.time()
        self._evict_if_needed()
        
        # Persist to database
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._store_db, slice_hash, data, vector
        )
        
    def _store_db(self, slice_hash: str, data: dict[str, Any], 
                  vector: np.ndarray | None) -> bool:
        """Store in database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            
            # Serialize vector if provided
            vector_blob = vector.tobytes() if vector is not None else None
            
            # Extract fields
            file_path = data.get("file", "")
            content = data.get("content", "")
            size_bytes = len(content.encode())
            
            conn.execute("""
                INSERT OR REPLACE INTO slice_cache 
                (hash, vector, metadata, file_path, content, size_bytes, 
                 created_at, accessed_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                        COALESCE((SELECT access_count FROM slice_cache WHERE hash = ?), 0) + 1)
            """, (
                slice_hash, vector_blob, json.dumps(data), file_path, content,
                size_bytes, datetime.now(), datetime.now(), slice_hash
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Cache store error: {e}")
            return False
            
    async def retrieve(self, slice_hash: str) -> dict[str, Any] | None:
        """Retrieve slice data by hash."""
        # Check memory cache first
        if slice_hash in self.memory_cache:
            self.hits += 1
            self.access_times[slice_hash] = time.time()
            return self.memory_cache[slice_hash]
            
        # Check database
        self.misses += 1
        data = await asyncio.get_event_loop().run_in_executor(
            self.executor, self._retrieve_db, slice_hash
        )
        
        if data:
            # Add to memory cache
            self.memory_cache[slice_hash] = data
            self.access_times[slice_hash] = time.time()
            self._evict_if_needed()
            
        return data
        
    def _retrieve_db(self, slice_hash: str) -> dict[str, Any] | None:
        """Retrieve from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            
            # Update access time and count
            conn.execute("""
                UPDATE slice_cache 
                SET accessed_at = ?, access_count = access_count + 1
                WHERE hash = ?
            """, (datetime.now(), slice_hash))
            
            # Retrieve data
            cursor = conn.execute("""
                SELECT metadata, content, vector FROM slice_cache WHERE hash = ?
            """, (slice_hash,))
            
            row = cursor.fetchone()
            conn.commit()
            conn.close()
            
            if row:
                data = json.loads(row[0])
                data["content"] = row[1]
                
                # Deserialize vector if present
                if row[2]:
                    data["vector"] = np.frombuffer(row[2], dtype=np.float32)
                    
                return data
                
        except Exception as e:
            print(f"Cache retrieve error: {e}")
            
        return None
        
    async def search_similar(self, query_vector: np.ndarray, 
                           top_k: int = 10) -> list[tuple[str, float]]:
        """Find similar slices using vector similarity."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._search_similar_db, query_vector, top_k
        )
        
    def _search_similar_db(self, query_vector: np.ndarray, 
                          top_k: int) -> list[tuple[str, float]]:
        """Search for similar vectors in database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute("""
                SELECT hash, vector FROM slice_cache 
                WHERE vector IS NOT NULL
                ORDER BY accessed_at DESC
                LIMIT 1000
            """)
            
            # Calculate similarities
            similarities = []
            for row in cursor:
                slice_hash = row[0]
                vector = np.frombuffer(row[1], dtype=np.float32)
                
                # Cosine similarity
                similarity = np.dot(query_vector, vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(vector)
                )
                similarities.append((slice_hash, float(similarity)))
                
            conn.close()
            
            # Sort by similarity and return top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
            
    def _evict_if_needed(self):
        """Evict least recently used entries if cache is full."""
        if len(self.memory_cache) > self.max_memory_entries:
            # Find LRU entry
            lru_hash = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.memory_cache[lru_hash]
            del self.access_times[lru_hash]
            self.evictions += 1
            
    async def cleanup_old_entries(self, days: int = 7):
        """Remove entries older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._cleanup_db, cutoff
        )
        
    def _cleanup_db(self, cutoff: datetime) -> int:
        """Clean up old database entries."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute("""
                DELETE FROM slice_cache 
                WHERE accessed_at < ? AND access_count < 5
            """, (cutoff,))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            return deleted
            
        except Exception as e:
            print(f"Cleanup error: {e}")
            return 0
            
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        
        # Get database stats
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM slice_cache")
            count, total_bytes = cursor.fetchone()
            conn.close()
            
            total_mb = (total_bytes or 0) / (1024 * 1024)
            
        except Exception:
            count, total_mb = 0, 0
            
        return {
            "memory_entries": len(self.memory_cache),
            "database_entries": count,
            "total_size_mb": round(total_mb, 2),
            "hit_rate": round(hit_rate, 3),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions
        }
        
    async def close(self):
        """Shutdown cache and cleanup."""
        self._shutdown = True
        self.executor.shutdown(wait=True)