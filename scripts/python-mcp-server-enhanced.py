#!/usr/bin/env python3
"""Enhanced Python analysis MCP server with health checks."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.unity_wheel.mcp.base_server import HealthCheckMCP
import subprocess
import psutil
from datetime import datetime

# Get workspace root from environment or use parent directory
WORKSPACE_ROOT = os.environ.get('WORKSPACE_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mcp = HealthCheckMCP("python-analysis", workspace_root=WORKSPACE_ROOT)

@mcp.tool()
def analyze_position(symbol: str) -> str:
    """Analyze trading position for a given symbol."""
    mcp.track_request()
    try:
        # Import here to avoid startup delays
        from src.unity_wheel.api.advisor import UnityWheelAdvisor
        
        advisor = UnityWheelAdvisor()
        recommendation = advisor.get_recommendation(symbol=symbol, capital=100000)
        
        return f"""Position Analysis for {symbol}:
- Action: {recommendation.get('action', 'N/A')}
- Confidence: {recommendation.get('confidence', 0):.2%}
- Contracts: {recommendation.get('contracts', 0)}
- Strike: ${recommendation.get('strike', 0):.2f}
- Premium: ${recommendation.get('premium', 0):.2f}
"""
    except Exception as e:
        mcp.track_error(str(e))
        return f"Error analyzing position: {str(e)}"

@mcp.tool()
def monitor_system() -> str:
    """Monitor system resource usage and performance."""
    mcp.track_request()
    try:
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        memory_percent = memory.percent
        
        # Disk info
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        disk_percent = disk.percent
        
        # Process info
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024**2)
        process_cpu = process.cpu_percent(interval=0.1)
        
        return f"""System Status:
=== System Resources ===
CPU: {cpu_percent}% ({cpu_count} cores)
Memory: {memory_used_gb:.1f}/{memory_total_gb:.1f} GB ({memory_percent}%)
Disk: {disk_free_gb:.1f} GB free ({100-disk_percent:.1f}% free)

=== Process Info ===
MCP Memory: {process_memory_mb:.1f} MB
MCP CPU: {process_cpu:.1f}%
PID: {process.pid}
Status: {process.status()}
"""
    except Exception as e:
        mcp.track_error(str(e))
        return f"Error monitoring system: {str(e)}"

@mcp.tool()
def data_quality_check() -> str:
    """Check data quality and database health."""
    mcp.track_request()
    try:
        import duckdb
        from pathlib import Path
        
        db_path = Path(WORKSPACE_ROOT) / "data" / "wheel_trading_master.duckdb"
        if not db_path.exists():
            return "Database not found at expected location"
            
        conn = duckdb.connect(str(db_path), read_only=True)
        
        # Get table info
        tables = conn.execute("SHOW TABLES").fetchall()
        
        result = "Database Health Check:\n"
        result += f"Database: {db_path.name}\n"
        result += f"Size: {db_path.stat().st_size / (1024**2):.1f} MB\n\n"
        
        result += "Tables:\n"
        for table in tables:
            table_name = table[0]
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            result += f"- {table_name}: {count:,} rows\n"
            
        # Check for recent data
        try:
            latest = conn.execute("""
                SELECT MAX(timestamp) as latest 
                FROM market_data_unity_1d
            """).fetchone()[0]
            result += f"\nLatest data: {latest}"
        except:
            result += "\nUnable to check latest data timestamp"
            
        conn.close()
        return result
        
    except Exception as e:
        mcp.track_error(str(e))
        return f"Error checking data quality: {str(e)}"

@mcp.tool()
def run_tests(test_type: str = "fast") -> str:
    """Run test suite and return results.
    
    Args:
        test_type: Type of tests to run - 'fast', 'all', or specific test file
    """
    mcp.track_request()
    try:
        if test_type == "fast":
            cmd = ["python", "-m", "pytest", "-v", "-m", "not slow", "--tb=short"]
        elif test_type == "all":
            cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
        else:
            # Run specific test file
            cmd = ["python", "-m", "pytest", "-v", test_type, "--tb=short"]
            
        result = subprocess.run(
            cmd,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout if result.returncode == 0 else result.stderr
        
        # Extract summary
        lines = output.split('\n')
        summary_lines = []
        capture = False
        
        for line in lines:
            if "=" in line and ("passed" in line or "failed" in line):
                capture = True
            if capture:
                summary_lines.append(line)
                
        summary = '\n'.join(summary_lines[-10:]) if summary_lines else "No test summary found"
        
        return f"""Test Results:
Exit Code: {result.returncode}
{'✓ PASSED' if result.returncode == 0 else '✗ FAILED'}

Summary:
{summary}

Full output available in test logs.
"""
    except subprocess.TimeoutExpired:
        mcp.track_error("Test timeout")
        return "Tests timed out after 30 seconds"
    except Exception as e:
        mcp.track_error(str(e))
        return f"Error running tests: {str(e)}"

@mcp.tool()
def check_mcp_servers() -> str:
    """Check status of all MCP servers."""
    mcp.track_request()
    try:
        runtime_dir = Path(WORKSPACE_ROOT) / ".claude" / "runtime"
        if not runtime_dir.exists():
            return "No runtime directory found - no servers running"
            
        results = ["MCP Server Status:\n"]
        
        for health_file in sorted(runtime_dir.glob("*.health")):
            try:
                with open(health_file) as f:
                    data = json.load(f)
                    
                server_name = data.get("server_name", "unknown")
                status = data.get("status", "unknown")
                uptime = data.get("uptime_human", "unknown")
                requests = data.get("request_count", 0)
                errors = data.get("error_count", 0)
                memory_mb = data.get("memory_mb", 0)
                
                # Check if process is alive
                pid = data.get("pid")
                alive = psutil.pid_exists(pid) if pid else False
                
                icon = "✓" if alive and status == "healthy" else "✗"
                results.append(f"{icon} {server_name}:")
                results.append(f"  Status: {status} {'(alive)' if alive else '(dead)'}")
                results.append(f"  Uptime: {uptime}")
                results.append(f"  Requests: {requests} (errors: {errors})")
                results.append(f"  Memory: {memory_mb:.1f} MB")
                results.append("")
                
            except Exception as e:
                results.append(f"✗ Error reading {health_file.name}: {str(e)}\n")
                
        return '\n'.join(results)
        
    except Exception as e:
        mcp.track_error(str(e))
        return f"Error checking MCP servers: {str(e)}"

if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    import json
    
    # Clean up stale health files before starting
    HealthCheckMCP.cleanup_stale_health_files(WORKSPACE_ROOT)
    
    print(f"Starting enhanced Python MCP server...")
    print(f"Workspace: {WORKSPACE_ROOT}")
    print(f"Health checks available at: {WORKSPACE_ROOT}/.claude/runtime/")
    
    # Run server
    mcp.run()