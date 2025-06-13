"""Direct hardware accelerator for Claude Code - no orchestrator needed."""

import asyncio
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import multiprocessing as mp
import re
import glob as glob_module

# Import only the essentials we need
try:
    import mlx.core as mx
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


class DirectAccelerator:
    """Simple, direct hardware acceleration without orchestrator complexity."""
    
    def __init__(self):
        # Detect hardware once
        self.cpu_count = mp.cpu_count()
        self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.cpu_count * 2)
        
        # Set all performance environment vars
        self._maximize_performance()
        
    def _maximize_performance(self):
        """Set all environment variables for maximum performance."""
        cores = str(self.cpu_count)
        
        # CPU optimization
        for var in ["NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
            os.environ[var] = cores
        
        # GPU optimization
        if GPU_AVAILABLE:
            os.environ["MLX_DEFAULT_STREAM"] = "gpu"
            os.environ["USE_GPU_ACCELERATION"] = "true"
        
        # Python optimization
        os.environ["PYTHONOPTIMIZE"] = "2"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        
    async def parallel_search(self, pattern: str, path: str = ".", 
                            file_pattern: str = "*.py") -> List[Dict[str, Any]]:
        """Hardware-accelerated file search using ripgrep."""
        cmd = ["rg", "--json", "-g", file_pattern, pattern, path]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, _ = await proc.communicate()
        
        results = []
        for line in stdout.decode().splitlines():
            if line.strip():
                import json
                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        results.append({
                            "file": data["data"]["path"]["text"],
                            "line": data["data"]["line_number"],
                            "content": data["data"]["lines"]["text"].strip()
                        })
                except:
                    pass
        
        return results
    
    async def parallel_read_files(self, file_paths: List[str]) -> Dict[str, str]:
        """Read multiple files in parallel."""
        async def read_file(path: str) -> tuple[str, Optional[str]]:
            try:
                async with asyncio.to_thread(open, path, 'r') as f:
                    content = await asyncio.to_thread(f.read)
                return path, content
            except Exception as e:
                return path, f"Error: {e}"
        
        tasks = [read_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    async def parallel_glob(self, patterns: List[str]) -> Dict[str, List[str]]:
        """Find files matching multiple patterns in parallel."""
        async def glob_pattern(pattern: str) -> tuple[str, List[str]]:
            matches = await asyncio.to_thread(glob_module.glob, pattern, recursive=True)
            return pattern, matches
        
        tasks = [glob_pattern(p) for p in patterns]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    async def parallel_execute(self, func: Callable, items: List[Any], 
                             cpu_bound: bool = True) -> List[Any]:
        """Execute function on items in parallel using all cores."""
        if cpu_bound:
            # Use process pool for CPU-bound work
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(self.process_pool, func, item)
                for item in items
            ]
        else:
            # Use thread pool for I/O-bound work
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(self.thread_pool, func, item)
                for item in items
            ]
        
        return await asyncio.gather(*futures)
    
    async def batch_process(self, items: List[Any], batch_size: int = 100,
                          processor: Callable = None) -> List[Any]:
        """Process items in optimized batches."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            if processor:
                batch_results = await self.parallel_execute(processor, batch)
            else:
                batch_results = batch
            results.extend(batch_results)
        
        return results
    
    def cleanup(self):
        """Cleanup pools."""
        self.process_pool.shutdown(wait=False)
        self.thread_pool.shutdown(wait=False)


# Global instance
_accelerator: Optional[DirectAccelerator] = None


def get_accelerator() -> DirectAccelerator:
    """Get or create the global accelerator."""
    global _accelerator
    if _accelerator is None:
        _accelerator = DirectAccelerator()
    return _accelerator


# Simple async wrappers for common operations
async def fast_search(pattern: str, path: str = ".", file_pattern: str = "*.py") -> List[Dict]:
    """Quick parallel search."""
    acc = get_accelerator()
    return await acc.parallel_search(pattern, path, file_pattern)


async def fast_read(file_paths: List[str]) -> Dict[str, str]:
    """Quick parallel file read."""
    acc = get_accelerator()
    return await acc.parallel_read_files(file_paths)


async def fast_glob(patterns: List[str]) -> Dict[str, List[str]]:
    """Quick parallel glob."""
    acc = get_accelerator()
    return await acc.parallel_glob(patterns)


async def fast_process(func: Callable, items: List[Any], cpu_bound: bool = True) -> List[Any]:
    """Quick parallel processing."""
    acc = get_accelerator()
    return await acc.parallel_execute(func, items, cpu_bound)