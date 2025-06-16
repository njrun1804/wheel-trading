"""Hardware-accelerated ripgrep replacement - Optimized with sync fallback."""

import asyncio
import contextlib
import json
import logging
import multiprocessing as mp
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fix asyncio child watcher issue for Python 3.13+ on macOS
if platform.system() == "Darwin":
    with contextlib.suppress(Exception):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


class RipgrepTurboOptimized:
    """Turbo-charged ripgrep with sync fallback for maximum reliability."""

    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=self.cpu_count)
        self.use_async = self._test_async_subprocess()

        logger.info(
            f"RipgrepTurbo initialized: async={self.use_async}, cores={self.cpu_count}"
        )

    def _test_async_subprocess(self) -> bool:
        """Test if async subprocess works on this system."""
        try:

            async def test():
                proc = await asyncio.create_subprocess_exec(
                    "rg",
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                return True

            # Test in new event loop to avoid conflicts
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(test())
            loop.close()
            return result

        except Exception as e:
            logger.warning(f"Async subprocess test failed: {e}, falling back to sync")
            return False

    def _run_ripgrep_sync(self, cmd: list[str]) -> tuple[bytes, bytes, int]:
        """Run ripgrep synchronously with proper error handling."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,  # 30 second timeout
                check=False,  # Don't raise on non-zero exit
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            logger.error(f"Ripgrep command timed out: {' '.join(cmd[:3])}...")
            return b"", b"timeout", 1
        except Exception as e:
            logger.error(f"Ripgrep command failed: {e}")
            return b"", str(e).encode(), 1

    async def _run_ripgrep_async(self, cmd: list[str]) -> tuple[bytes, bytes, int]:
        """Run ripgrep asynchronously."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            return stdout, stderr, proc.returncode

        except TimeoutError:
            logger.error(f"Async ripgrep command timed out: {' '.join(cmd[:3])}...")
            return b"", b"timeout", 1
        except Exception as e:
            logger.error(f"Async ripgrep command failed: {e}")
            # Fallback to sync on any async error
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor, self._run_ripgrep_sync, cmd
            )

    async def search(
        self,
        pattern: str | list[str],
        path: str = ".",
        file_type: str | None = None,
        max_results: int = 1000,
    ) -> list[dict[str, Any]]:
        """Search with full CPU parallelization and reliable fallback."""

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

        # Execute with appropriate method
        if self.use_async:
            stdout, stderr, returncode = await self._run_ripgrep_async(cmd)
        else:
            loop = asyncio.get_event_loop()
            stdout, stderr, returncode = await loop.run_in_executor(
                self.executor, self._run_ripgrep_sync, cmd
            )

        # Handle errors
        if returncode != 0 and returncode != 1:  # 1 is "no matches", which is OK
            logger.warning(f"Ripgrep returned code {returncode}: {stderr.decode()}")

        # Parse results
        return self._parse_ripgrep_output(stdout.decode())

    def _parse_ripgrep_output(self, output: str) -> list[dict[str, Any]]:
        """Parse ripgrep JSON output into structured results."""
        results = []

        for line in output.splitlines():
            if not line.strip():
                continue

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
                                "after": match_data.get("context", {}).get("after", []),
                            },
                        }
                    )
            except json.JSONDecodeError:
                continue
            except KeyError as e:
                logger.debug(f"Unexpected ripgrep output format: {e}")
                continue

        return results

    async def search_count(self, pattern: str, path: str = ".") -> dict[str, int]:
        """Count matches per file using all cores."""
        cmd = ["rg", "--count", "--threads", str(self.cpu_count), pattern, path]

        if self.use_async:
            stdout, stderr, returncode = await self._run_ripgrep_async(cmd)
        else:
            loop = asyncio.get_event_loop()
            stdout, stderr, returncode = await loop.run_in_executor(
                self.executor, self._run_ripgrep_sync, cmd
            )

        counts = {}
        for line in stdout.decode().splitlines():
            if ":" in line:
                file_path, count = line.rsplit(":", 1)
                try:
                    counts[file_path] = int(count)
                except ValueError:
                    continue

        return counts

    async def parallel_search(
        self, patterns: list[str], path: str = "."
    ) -> dict[str, list[dict[str, Any]]]:
        """Search multiple patterns in parallel using all cores."""
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
        # Split files into chunks for each CPU
        chunk_size = max(1, len(files) // self.cpu_count)
        chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]

        async def search_chunk(file_list: list[str]) -> list[dict]:
            all_results = []
            for file in file_list:
                if Path(file).exists():
                    results = await self.search(pattern, file)
                    all_results.extend(results)
            return all_results

        # Search all chunks in parallel
        chunk_results = await asyncio.gather(*[search_chunk(chunk) for chunk in chunks])

        # Flatten results
        return [r for results in chunk_results for r in results]

    def search_sync(
        self, pattern: str, path: str = ".", max_results: int = 1000
    ) -> list[dict[str, Any]]:
        """Synchronous search method for non-async contexts."""
        cmd = [
            "rg",
            "--json",
            "--max-count",
            str(max_results),
            "--threads",
            str(self.cpu_count),
            "--max-columns",
            "500",
            "--smart-case",
            pattern,
            path,
        ]

        stdout, stderr, returncode = self._run_ripgrep_sync(cmd)
        return self._parse_ripgrep_output(stdout.decode())

    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)


# Singleton instance
_turbo_instance: RipgrepTurboOptimized | None = None


def get_ripgrep_turbo_optimized() -> RipgrepTurboOptimized:
    """Get or create the optimized turbo ripgrep instance."""
    global _turbo_instance
    if _turbo_instance is None:
        _turbo_instance = RipgrepTurboOptimized()
    return _turbo_instance


# Drop-in replacements for MCP functions with enhanced reliability
async def search(pattern: str, path: str = ".", max_results: int = 100) -> str:
    """Drop-in replacement for MCP ripgrep.search with enhanced reliability."""
    turbo = get_ripgrep_turbo_optimized()
    results = await turbo.search(pattern, path, max_results=max_results)

    # Format as MCP would
    output = []
    for r in results[:max_results]:
        output.append(f"{r['file']}:{r['line']}: {r['content']}")

    return "\n".join(output)


async def search_count(pattern: str, path: str = ".") -> str:
    """Drop-in replacement for MCP ripgrep.search_count with enhanced reliability."""
    turbo = get_ripgrep_turbo_optimized()
    counts = await turbo.search_count(pattern, path)

    total = sum(counts.values())
    output = [f"Total matches: {total}", ""]

    for file, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        output.append(f"{file}: {count}")

    return "\n".join(output)


# Synchronous API for non-async contexts
def search_sync(pattern: str, path: str = ".", max_results: int = 100) -> str:
    """Synchronous search for non-async contexts."""
    turbo = get_ripgrep_turbo_optimized()
    results = turbo.search_sync(pattern, path, max_results)

    output = []
    for r in results[:max_results]:
        output.append(f"{r['file']}:{r['line']}: {r['content']}")

    return "\n".join(output)
