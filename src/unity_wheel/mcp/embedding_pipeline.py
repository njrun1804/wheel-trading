"""
from __future__ import annotations

Embedding pipeline with SHA-1 slice caching.
Integrates ripgrep, dynamic chunking, and slice cache for efficient code embeddings.
"""

import asyncio
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

from unity_wheel.storage.slice_cache import SliceCache
from .dynamic_chunking import DynamicChunker, ChunkedFileReader
from unity_wheel.utils import get_logger

logger = get_logger(__name__)


class EmbeddingPipeline:
    """High-performance embedding pipeline with slice caching."""
    
    def __init__(self, cache_path: Optional[Path] = None, 
                 embedding_func: Optional[Callable] = None):
        """
        Initialize embedding pipeline.
        
        Args:
            cache_path: Path to slice cache database
            embedding_func: Function that takes text and returns (embedding, token_count)
        """
        self.cache = SliceCache(db_path=cache_path)
        self.cache.initialize()
        
        self.chunker = DynamicChunker()
        self.file_reader = ChunkedFileReader()
        
        # Embedding function (can be replaced with actual API call)
        self.embedding_func = embedding_func or self._mock_embedding
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.stats = {
            "files_processed": 0,
            "slices_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_processed": 0,
            "tokens_saved": 0
        }
        
    def _mock_embedding(self, text: str) -> Tuple[np.ndarray, int]:
        """Mock embedding function for testing."""
        # Simulate embedding dimension of 1536 (like ada-002)
        embedding = np.random.randn(1536).astype(np.float32)
        token_count = len(text.split()) * 1.3  # Rough approximation
        return embedding, int(token_count)
        
    async def embed_file(self, file_path: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Embed a file using cached slices where possible.
        
        Args:
            file_path: Path to file to embed
            force_refresh: Skip cache and re-embed everything
            
        Returns:
            List of slice results with embeddings
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("File not found", path=file_path)
            return []
            
        # Get file chunks
        chunks = self.file_reader.read_file_chunked(str(path), semantic=True)
        if not chunks:
            return []
            
        # Process chunks in parallel
        tasks = []
        for chunk in chunks:
            task = self._process_chunk(chunk, str(path), force_refresh)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        self.stats["files_processed"] += 1
        
        # Log performance
        hit_rate = (self.stats["cache_hits"] / 
                   max(self.stats["cache_hits"] + self.stats["cache_misses"], 1)) * 100
        
        logger.info("File embedded",
                   file=file_path,
                   chunks=len(chunks),
                   hit_rate=f"{hit_rate:.1f}%",
                   tokens_saved=self.stats["tokens_saved"])
                   
        return results
        
    async def _process_chunk(self, chunk: Dict[str, Any], file_path: str, 
                           force_refresh: bool) -> Dict[str, Any]:
        """Process a single chunk with caching."""
        content = chunk["content"]
        start_line = chunk["start_line"]
        end_line = chunk["end_line"]
        
        # Check cache unless forced refresh
        embedding = None
        if not force_refresh:
            embedding = self.cache.get_embedding(
                content, file_path, start_line, end_line
            )
            
        if embedding is not None:
            # Cache hit
            self.stats["cache_hits"] += 1
            self.stats["tokens_saved"] += chunk["tokens"]
            
            return {
                **chunk,
                "embedding": embedding,
                "cached": True
            }
        else:
            # Cache miss - compute embedding
            self.stats["cache_misses"] += 1
            self.stats["slices_processed"] += 1
            
            # Run embedding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding, token_count = await loop.run_in_executor(
                self.executor, self.embedding_func, content
            )
            
            self.stats["tokens_processed"] += token_count
            
            # Store in cache
            self.cache.store_embedding(
                content, file_path, start_line, end_line,
                embedding, token_count
            )
            
            return {
                **chunk,
                "embedding": embedding,
                "cached": False,
                "actual_tokens": token_count
            }
            
    async def embed_search_results(self, pattern: str, directory: str,
                                 context_lines: int = 5) -> List[Dict[str, Any]]:
        """
        Search files with ripgrep and embed matching regions.
        
        Args:
            pattern: Search pattern for ripgrep
            directory: Directory to search
            context_lines: Lines of context around matches
            
        Returns:
            List of embedded search results
        """
        # Find files with matches
        files = await self._ripgrep_search(pattern, directory)
        
        if not files:
            logger.info("No matches found", pattern=pattern, directory=directory)
            return []
            
        # Process each file with matches
        all_results = []
        for file_path in files:
            # Get focused chunks around matches
            chunks = self.file_reader.read_with_ripgrep(
                pattern, file_path, context_lines
            )
            
            # Embed chunks
            for chunk in chunks:
                result = await self._process_chunk(chunk, file_path, force_refresh=False)
                result["pattern"] = pattern
                all_results.append(result)
                
        logger.info("Search results embedded",
                   pattern=pattern,
                   files=len(files),
                   chunks=len(all_results))
                   
        return all_results
        
    async def _ripgrep_search(self, pattern: str, directory: str) -> List[str]:
        """Run ripgrep to find files with matches."""
        try:
            cmd = [
                'rg',
                '--files-with-matches',
                '--type', 'py',  # Python files only
                pattern,
                directory
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                return []
                
            # Parse file paths
            files = stdout.decode().strip().split('\n')
            return [f for f in files if f]
            
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Ripgrep failed", error=str(e))
            return []
            
    async def embed_changed_files(self, changed_files: List[str]) -> Dict[str, Any]:
        """
        Efficiently embed only changed parts of files.
        
        Args:
            changed_files: List of file paths that changed
            
        Returns:
            Summary of embedding operation
        """
        total_chunks = 0
        cached_chunks = 0
        
        for file_path in changed_files:
            results = await self.embed_file(file_path)
            total_chunks += len(results)
            cached_chunks += sum(1 for r in results if r.get("cached", False))
            
        return {
            "files": len(changed_files),
            "total_chunks": total_chunks,
            "cached_chunks": cached_chunks,
            "cache_hit_rate": f"{(cached_chunks / max(total_chunks, 1)) * 100:.1f}%",
            "tokens_saved": self.stats["tokens_saved"]
        }
        
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        cache_stats = self.cache.get_cache_stats()
        
        return {
            "pipeline": {
                "files_processed": self.stats["files_processed"],
                "slices_processed": self.stats["slices_processed"],
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "hit_rate": f"{(self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)) * 100:.1f}%",
                "tokens_processed": self.stats["tokens_processed"],
                "tokens_saved": self.stats["tokens_saved"],
                "estimated_cost_saved": f"${self.stats['tokens_saved'] * 0.0001:.2f}"  # Rough estimate
            },
            "cache": cache_stats
        }
        
    async def warmup_cache(self, directory: str, file_pattern: str = "*.py"):
        """
        Warm up cache by pre-embedding files in a directory.
        
        Args:
            directory: Directory to scan
            file_pattern: File pattern to match
        """
        path = Path(directory)
        files = list(path.rglob(file_pattern))
        
        logger.info("Warming up cache", directory=directory, files=len(files))
        
        # Process in batches to avoid overwhelming the system
        batch_size = 10
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            tasks = [self.embed_file(str(f)) for f in batch]
            await asyncio.gather(*tasks)
            
            # Log progress
            progress = min(i + batch_size, len(files))
            logger.info(f"Progress: {progress}/{len(files)} files")
            
        stats = self.get_pipeline_stats()
        logger.info("Cache warmup complete", stats=stats)
        
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)


# Integration with MCP servers
class MCPEmbeddingHandler:
    """Handler for MCP embedding requests with caching."""
    
    def __init__(self, pipeline: EmbeddingPipeline):
        self.pipeline = pipeline
        
    async def handle_embed_file(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file embedding request."""
        file_path = request.get("file_path")
        force_refresh = request.get("force_refresh", False)
        
        if not file_path:
            return {"error": "file_path required"}
            
        results = await self.pipeline.embed_file(file_path, force_refresh)
        
        return {
            "file_path": file_path,
            "chunks": len(results),
            "results": results,
            "stats": self.pipeline.get_pipeline_stats()
        }
        
    async def handle_embed_search(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search and embed request."""
        pattern = request.get("pattern")
        directory = request.get("directory", ".")
        context_lines = request.get("context_lines", 5)
        
        if not pattern:
            return {"error": "pattern required"}
            
        results = await self.pipeline.embed_search_results(
            pattern, directory, context_lines
        )
        
        return {
            "pattern": pattern,
            "directory": directory,
            "matches": len(results),
            "results": results,
            "stats": self.pipeline.get_pipeline_stats()
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_pipeline():
        """Test the embedding pipeline."""
        # Initialize pipeline
        pipeline = EmbeddingPipeline()
        
        # Test file embedding
        test_file = __file__
        print(f"\nEmbedding file: {test_file}")
        results = await pipeline.embed_file(test_file)
        print(f"Created {len(results)} chunks")
        
        # Test again to see cache hits
        print("\nEmbedding same file again (should hit cache)...")
        results = await pipeline.embed_file(test_file)
        
        # Show stats
        stats = pipeline.get_pipeline_stats()
        print("\nPipeline Stats:")
        print(json.dumps(stats, indent=2))
        
        # Test search embedding
        print("\nTesting search and embed...")
        search_results = await pipeline.embed_search_results(
            "async def", 
            Path(__file__).parent
        )
        print(f"Found and embedded {len(search_results)} matches")
        
        # Cleanup
        pipeline.cleanup()
        
    # Run test
    asyncio.run(test_pipeline())