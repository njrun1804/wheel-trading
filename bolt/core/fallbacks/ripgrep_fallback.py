"""
Fallback ripgrep implementation using subprocess.

Provides real ripgrep functionality when the accelerated tools are not available.
"""

import asyncio
import json
import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


class RipgrepFallback:
    """Fallback ripgrep implementation using subprocess calls."""

    def __init__(self):
        self.rg_available = self._check_ripgrep_available()
        if not self.rg_available:
            logger.warning("ripgrep not available, using basic grep fallback")

    def _check_ripgrep_available(self) -> bool:
        """Check if ripgrep is available on the system."""
        try:
            result = subprocess.run(
                ["rg", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    async def search(
        self,
        pattern: str,
        path: str = ".",
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
        case_sensitive: bool = False,
        max_results: int = 1000,
    ) -> list[dict[str, Any]]:
        """Search for pattern in files."""
        if self.rg_available:
            return await self._ripgrep_search(
                pattern,
                path,
                include_patterns,
                exclude_patterns,
                case_sensitive,
                max_results,
            )
        else:
            return await self._grep_fallback(
                pattern,
                path,
                include_patterns,
                exclude_patterns,
                case_sensitive,
                max_results,
            )

    async def parallel_search(
        self, patterns: list[str], path: str = "."
    ) -> dict[str, list[dict[str, Any]]]:
        """Search for multiple patterns in parallel."""
        tasks = [self.search(pattern, path) for pattern in patterns]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            patterns[i]: result if not isinstance(result, Exception) else []
            for i, result in enumerate(results)
        }

    async def _ripgrep_search(
        self,
        pattern: str,
        path: str,
        include_patterns: list[str],
        exclude_patterns: list[str],
        case_sensitive: bool,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Search using ripgrep."""
        cmd = ["rg", "--json", "--no-heading"]

        if not case_sensitive:
            cmd.append("--ignore-case")

        if include_patterns:
            for include in include_patterns:
                cmd.extend(["--type-add", f"custom:{include}", "--type", "custom"])

        if exclude_patterns:
            for exclude in exclude_patterns:
                cmd.extend(["--glob", f"!{exclude}"])

        cmd.extend(["--max-count", str(max_results)])
        cmd.append(pattern)
        cmd.append(path)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0 and stderr:
                logger.warning(f"ripgrep warning: {stderr.decode()}")

            results = []
            for line in stdout.decode().strip().split("\n"):
                if line:
                    try:
                        data = json.loads(line)
                        if data.get("type") == "match":
                            match_data = data["data"]
                            results.append(
                                {
                                    "file": match_data["path"]["text"],
                                    "line_number": match_data["line_number"],
                                    "line": match_data["lines"]["text"].strip(),
                                    "column": match_data["submatches"][0]["start"]
                                    if match_data.get("submatches")
                                    else 0,
                                }
                            )
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"Failed to parse ripgrep output: {e}")

            return results[:max_results]

        except Exception as e:
            logger.error(f"ripgrep search failed: {e}")
            return []

    async def _grep_fallback(
        self,
        pattern: str,
        path: str,
        include_patterns: list[str],
        exclude_patterns: list[str],
        case_sensitive: bool,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Fallback search using standard grep."""
        cmd = ["grep", "-rn"]

        if not case_sensitive:
            cmd.append("-i")

        # Add include/exclude patterns if supported
        if include_patterns:
            for include in include_patterns:
                cmd.extend(["--include", include])

        if exclude_patterns:
            for exclude in exclude_patterns:
                cmd.extend(["--exclude", exclude])

        cmd.append(pattern)
        cmd.append(path)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            results = []
            for line in stdout.decode().strip().split("\n")[:max_results]:
                if line and ":" in line:
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        try:
                            line_number = int(parts[1])
                            content = parts[2]
                            results.append(
                                {
                                    "file": file_path,
                                    "line_number": line_number,
                                    "line": content.strip(),
                                    "column": content.find(pattern)
                                    if pattern in content
                                    else 0,
                                }
                            )
                        except ValueError:
                            continue

            return results

        except Exception as e:
            logger.error(f"grep fallback failed: {e}")
            return []

    async def search_files_only(self, pattern: str, path: str = ".") -> list[str]:
        """Search for files matching pattern."""
        if self.rg_available:
            cmd = ["rg", "--files", "--glob", pattern, path]
        else:
            cmd = ["find", path, "-name", pattern, "-type", "f"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                return [
                    line.strip()
                    for line in stdout.decode().strip().split("\n")
                    if line.strip()
                ]
            else:
                logger.warning(f"File search failed: {stderr.decode()}")
                return []

        except Exception as e:
            logger.error(f"File search error: {e}")
            return []

    def search_sync(
        self, pattern: str, path: str = ".", **kwargs
    ) -> list[dict[str, Any]]:
        """Synchronous search wrapper."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.search(pattern, path, **kwargs))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.search(pattern, path, **kwargs))
            finally:
                loop.close()

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            "backend": "ripgrep" if self.rg_available else "grep",
            "parallel_capable": True,
            "json_output": self.rg_available,
            "regex_support": True,
            "glob_support": True,
        }
