#!/usr/bin/env python3
"""
Python MCP Server for Wheel Trading Analysis
Provides real-time analysis and monitoring capabilities
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

try:
    import numpy as np
    import pandas as pd
    import psutil
except ImportError:
    # For testing without full dependencies
    print('{"jsonrpc": "2.0", "method": "initialized", "params": {}}', flush=True)
    sys.exit(0)


class PythonAnalysisMCP:
    """MCP server for Python-specific analysis tools."""

    def __init__(self):
        self.project_root = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"

    async def analyze_position(self, params: dict[str, Any]) -> dict[str, Any]:
        """Analyze a potential position with real-time metrics."""
        try:
            # Import project modules
            sys.path.insert(0, self.project_root)
            from unity_wheel.math.options import calculate_greeks
            from unity_wheel.utils.position_sizing import PositionSizer

            result = {
                "timestamp": datetime.now().isoformat(),
                "greeks": calculate_greeks(
                    params.get("price", 100),
                    params.get("strike", 95),
                    params.get("dte", 30),
                    params.get("iv", 0.25),
                    params.get("rate", 0.05),
                ),
                "position_size": PositionSizer().calculate_position_size(
                    params.get("portfolio_value", 100000), params.get("delta", 0.30)
                ),
                "risk_metrics": self._calculate_risk_metrics(params),
            }
            return result
        except Exception as e:
            return {"error": str(e)}

    def _calculate_risk_metrics(self, params: dict[str, Any]) -> dict[str, Any]:
        """Calculate risk metrics for position."""
        portfolio_value = params.get("portfolio_value", 100000)
        position_size = params.get("position_size", 10000)

        return {
            "position_pct": position_size / portfolio_value,
            "max_loss": position_size * params.get("delta", 0.30),
            "break_even": params.get("strike", 95) - params.get("premium", 2.5),
            "margin_required": position_size * 0.20,  # Approximate
        }

    async def monitor_system(self) -> dict[str, Any]:
        """Real-time system monitoring."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "available_gb": psutil.virtual_memory().available / 1e9,
            },
            "disk": {
                "data_dir_size_mb": self._get_dir_size("data") / 1e6,
                "logs_dir_size_mb": self._get_dir_size("logs") / 1e6,
            },
            "processes": {
                "python": len([p for p in psutil.process_iter() if "python" in p.name()]),
                "mcp": len([p for p in psutil.process_iter() if "mcp" in p.name()]),
            },
        }

    def _get_dir_size(self, dirname: str) -> int:
        """Get directory size in bytes."""
        total = 0
        path = os.path.join(self.project_root, dirname)
        if os.path.exists(path):
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
        return total

    async def data_quality_check(self) -> dict[str, Any]:
        """Check data quality in real-time."""
        try:
            # Check DuckDB data
            import duckdb

            conn = duckdb.connect(
                os.path.join(self.project_root, "data/wheel_trading_master.duckdb")
            )

            checks = {"timestamp": datetime.now().isoformat(), "tables": [], "issues": []}

            # Get table info
            tables = conn.execute("SHOW TABLES").fetchall()
            for table in tables:
                table_name = table[0]
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                latest = conn.execute(f"SELECT MAX(timestamp) FROM {table_name}").fetchone()[0]

                checks["tables"].append(
                    {
                        "name": table_name,
                        "row_count": count,
                        "latest_data": str(latest) if latest else "No data",
                    }
                )

                # Check for data gaps
                if latest:
                    days_old = (datetime.now() - pd.to_datetime(latest)).days
                    if days_old > 1:
                        checks["issues"].append(f"{table_name} data is {days_old} days old")

            conn.close()
            return checks

        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def run_server(self):
        """Run the MCP server on stdio."""
        # Send initialization message
        init_msg = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "analyze_position": {},
                        "monitor_system": {},
                        "data_quality_check": {},
                    }
                },
                "serverInfo": {"name": "python-analysis", "version": "1.0.0"},
            },
        }
        print(json.dumps(init_msg), flush=True)

        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line or line.strip() == "":
                    continue

                request = json.loads(line.strip())
                method = request.get("method")
                params = request.get("params", {})

                if method == "analyze_position":
                    result = await self.analyze_position(params)
                elif method == "monitor_system":
                    result = await self.monitor_system()
                elif method == "data_quality_check":
                    result = await self.data_quality_check()
                elif method == "initialize":
                    result = {"protocolVersion": "2024-11-05", "capabilities": {}}
                else:
                    result = {"error": f"Unknown method: {method}"}

                response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}

                print(json.dumps(response), flush=True)

            except json.JSONDecodeError:
                # Skip malformed JSON
                continue
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id") if "request" in locals() else None,
                    "error": {"code": -1, "message": str(e)},
                }
                print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    server = PythonAnalysisMCP()
    asyncio.run(server.run_server())
