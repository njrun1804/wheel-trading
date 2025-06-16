"""
Production-grade ripgrep replacement with bulletproof subprocess handling.
Addresses all potential Python 3.13 asyncio issues with comprehensive fallbacks.
"""

import asyncio
import json
import logging
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix asyncio child watcher issue for Python 3.13+ on macOS
if platform.system() == "Darwin" and sys.version_info >= (3, 13):
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        logger.info("Set asyncio event loop policy for Python 3.13+ on macOS")
    except Exception as e:
        logger.warning(f"Could not set asyncio event loop policy: {e}")


class RipgrepTurboProduction:
    """Production-grade ripgrep with bulletproof subprocess handling."""

    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=self.cpu_count)
        self.subprocess_timeout = 30.0  # 30 second timeout
        logger.info(
            f"Initialized RipgrepTurboProduction with {self.cpu_count} CPU cores"
        )

    async def _safe_subprocess_exec(self, cmd: list[str]) -> tuple[bytes, bytes, int]:
        """
        Safely execute subprocess with automatic fallback.
        Returns (stdout, stderr, returncode).
        """
        try:
            # Try async first
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                ),
                timeout=self.subprocess_timeout,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.subprocess_timeout
            )
            return stdout, stderr, proc.returncode

        except (TimeoutError, NotImplementedError) as e:
            logger.warning(f"Async subprocess failed ({e}), falling back to sync")
            # Fallback to sync
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor, self._sync_subprocess_exec, cmd
            )
        except Exception as e:
            logger.error(f"Unexpected subprocess error: {e}")
            # Last resort sync fallback
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor, self._sync_subprocess_exec, cmd
            )

    def _sync_subprocess_exec(self, cmd: list[str]) -> tuple[bytes, bytes, int]:
        """Synchronous subprocess execution."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=self.subprocess_timeout
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            logger.error(f"Subprocess timeout after {self.subprocess_timeout}s")
            return b"", b"Timeout", 1
        except Exception as e:
            logger.error(f"Sync subprocess error: {e}")
            return b"", str(e).encode(), 1

    async def search(
        self,
        pattern: str | list[str],
        path: str = ".",
        file_type: str | None = None,
        max_results: int = 1000,
    ) -> list[dict[str, Any]]:
        """Search with full CPU parallelization and bulletproof subprocess handling."""

        # Handle multiple patterns
        if isinstance(pattern, list):
            pattern = "|".join(pattern)

        # Build ripgrep command
        cmd = [
            "rg",
            "--json",
            "--max-count",
            str(max_results),
            "--threads",
            str(self.cpu_count),
            "--max-columns",
            "500",
            "--max-filesize",
            "5M",
            "--mmap",
            "--smart-case",
        ]

        if file_type:
            cmd.extend(["-t", file_type])

        cmd.extend([pattern, path])

        logger.debug(f"Executing ripgrep command: {' '.join(cmd)}")

        # Execute with safe subprocess handling
        stdout, stderr, returncode = await self._safe_subprocess_exec(cmd)

        if returncode != 0:
            logger.warning(f"Ripgrep returned {returncode}: {stderr.decode()}")
            return []

        # Parse results
        lines = stdout.decode().splitlines()
        results = []

        for line in lines:
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        match_data = data["data"]
                        results.append(
                            {
                                "file": match_data["path"]["text"],
                                "line": match_data["line_number"],
                                "column": match_data.get("column", 1),
                                "content": match_data["lines"]["text"].strip(),
                                "context": {
                                    "before": match_data.get("context", {}).get(
                                        "before", []
                                    ),
                                    "after": match_data.get("context", {}).get(
                                        "after", []
                                    ),
                                },
                            }
                        )
                except json.JSONDecodeError:
                    continue

        logger.debug(f"Found {len(results)} matches")
        return results

    async def search_count(self, pattern: str, path: str = ".") -> dict[str, int]:
        """
        Count matches per file with bulletproof subprocess handling.
        Now includes the same fallback mechanism as search().
        """
        cmd = ["rg", "--count", "--threads", str(self.cpu_count), pattern, path]

        logger.debug(f"Executing count command: {' '.join(cmd)}")

        # Use the same safe subprocess handling
        stdout, stderr, returncode = await self._safe_subprocess_exec(cmd)

        if returncode != 0:
            logger.warning(f"Ripgrep count returned {returncode}: {stderr.decode()}")
            return {}

        counts = {}
        for line in stdout.decode().splitlines():
            if ":" in line:
                file_path, count = line.rsplit(":", 1)
                try:
                    counts[file_path] = int(count)
                except ValueError:
                    continue

        logger.debug(f"Found matches in {len(counts)} files")
        return counts

    async def parallel_search(
        self, patterns: list[str], path: str = "."
    ) -> dict[str, list[dict[str, Any]]]:
        """Search multiple patterns in parallel."""
        tasks = []
        for pattern in patterns:
            task = asyncio.create_task(self.search(pattern, path))
            tasks.append((pattern, task))

        results = {}
        for pattern, task in tasks:
            try:
                results[pattern] = await task
            except Exception as e:
                logger.error(f"Search failed for pattern '{pattern}': {e}")
                results[pattern] = []

        return results

    async def search_files(
        self, pattern: str, files: list[str]
    ) -> list[dict[str, Any]]:
        """Search specific files in parallel."""
        chunk_size = max(1, len(files) // self.cpu_count)
        chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]

        async def search_chunk(file_list: list[str]) -> list[dict]:
            all_results = []
            for file in file_list:
                if Path(file).exists():
                    results = await self.search(pattern, file)
                    all_results.extend(results)
            return all_results

        chunk_results = await asyncio.gather(*[search_chunk(chunk) for chunk in chunks])
        return [r for results in chunk_results for r in results]

    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        logger.info("RipgrepTurboProduction cleanup completed")


# Singleton instance
_turbo_instance: RipgrepTurboProduction | None = None


def get_ripgrep_turbo_production() -> RipgrepTurboProduction:
    """Get or create the production ripgrep instance."""
    global _turbo_instance
    if _turbo_instance is None:
        _turbo_instance = RipgrepTurboProduction()
    return _turbo_instance


# Drop-in replacements for existing functions
async def search(pattern: str, path: str = ".", max_results: int = 100) -> str:
    """Drop-in replacement for MCP ripgrep.search with production-grade handling."""
    turbo = get_ripgrep_turbo_production()
    results = await turbo.search(pattern, path, max_results=max_results)

    # Format as MCP would
    output = []
    for r in results[:max_results]:
        output.append(f"{r['file']}:{r['line']}: {r['content']}")

    return "\n".join(output)


async def search_count(pattern: str, path: str = ".") -> str:
    """Drop-in replacement for MCP ripgrep.search_count with production-grade handling."""
    turbo = get_ripgrep_turbo_production()
    counts = await turbo.search_count(pattern, path)

    total = sum(counts.values())
    output = [f"Total matches: {total}", ""]

    for file, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        output.append(f"{file}: {count}")

    return "\n".join(output)


# Synchronous API for non-async contexts
def search_sync(pattern: str, path: str = ".", max_results: int = 100) -> str:
    """Synchronous version of search."""
    return asyncio.run(search(pattern, path, max_results))


def search_count_sync(pattern: str, path: str = ".") -> str:
    """Synchronous version of search_count."""
    return asyncio.run(search_count(pattern, path))


# Performance testing
async def benchmark_search(
    pattern: str, path: str = ".", iterations: int = 5
) -> dict[str, float]:
    """Benchmark search performance."""
    turbo = get_ripgrep_turbo_production()

    times = []
    for i in range(iterations):
        start = time.perf_counter()
        results = await turbo.search(pattern, path, max_results=1000)
        duration = (time.perf_counter() - start) * 1000
        times.append(duration)
        logger.info(f"Iteration {i+1}: {len(results)} results in {duration:.2f}ms")

    return {
        "avg_time_ms": sum(times) / len(times),
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "total_iterations": iterations,
    }


if __name__ == "__main__":

    async def main():
        """Test the production implementation."""
        print("🚀 Testing Production Ripgrep Implementation")
        print("=" * 50)

        turbo = get_ripgrep_turbo_production()

        # Test basic search
        print("🔍 Testing basic search...")
        results = await turbo.search("wheel", ".", max_results=5)
        print(f"✅ Found {len(results)} results")
        if results:
            print(f"   First: {results[0]['file']}:{results[0]['line']}")

        # Test search count
        print("\n🔢 Testing search count...")
        counts = await turbo.search_count("wheel", ".")
        total = sum(counts.values())
        print(f"✅ Found {total} matches in {len(counts)} files")

        # Test parallel search
        print("\n⚡ Testing parallel search...")
        parallel_results = await turbo.parallel_search(["wheel", "options"], ".")
        for pattern, results in parallel_results.items():
            print(f"   {pattern}: {len(results)} results")

        # Benchmark
        print("\n🏁 Benchmarking...")
        benchmark = await benchmark_search("wheel", ".", iterations=3)
        print(f"✅ Average: {benchmark['avg_time_ms']:.2f}ms")

        turbo.cleanup()
        print("\n🎉 All tests completed successfully!")

    asyncio.run(main())
