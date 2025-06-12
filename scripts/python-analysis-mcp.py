#!/usr/bin/env python3
"""Python analysis MCP server - fixed version that handles spaces in paths."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from mcp.server import FastMCP

# Get workspace root from environment or use parent directory
WORKSPACE_ROOT = os.environ.get('WORKSPACE_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create MCP server instance
mcp = FastMCP("python-analysis")

# Track metrics
start_time = datetime.now()
request_count = 0
error_count = 0
last_error = None

def track_request():
    """Increment request counter."""
    global request_count
    request_count += 1

def track_error(error: str):
    """Track error for health monitoring."""
    global error_count, last_error
    error_count += 1
    last_error = error

@mcp.tool()
def analyze_position(symbol: str) -> str:
    """Analyze trading position for a given symbol."""
    track_request()
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
        track_error(str(e))
        return f"Error analyzing position: {str(e)}"

@mcp.tool()
def monitor_system() -> str:
    """Monitor system resource usage and performance."""
    track_request()
    try:
        import psutil
        
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
        track_error(str(e))
        return f"Error monitoring system: {str(e)}"

@mcp.tool()
def data_quality_check() -> str:
    """Check data quality and database health."""
    track_request()
    try:
        import duckdb
        
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
        track_error(str(e))
        return f"Error checking data quality: {str(e)}"

@mcp.tool()
def run_tests(test_type: str = "fast") -> str:
    """Run test suite and return results.
    
    Args:
        test_type: Type of tests to run - 'fast', 'all', or specific test file
    """
    track_request()
    try:
        if test_type == "fast":
            cmd = [sys.executable, "-m", "pytest", "-v", "-m", "not slow", "--tb=short"]
        elif test_type == "all":
            cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
        else:
            # Run specific test file
            cmd = [sys.executable, "-m", "pytest", "-v", test_type, "--tb=short"]
            
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
        track_error("Test timeout")
        return "Tests timed out after 30 seconds"
    except Exception as e:
        track_error(str(e))
        return f"Error running tests: {str(e)}"

@mcp.tool()
def healthz() -> Dict[str, Any]:
    """Health check endpoint - returns server health status."""
    uptime_seconds = (datetime.now() - start_time).total_seconds()
    
    # Get process info
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent(interval=0.1)
    except:
        memory_mb = 0
        cpu_percent = 0
    
    health_status = {
        "status": "healthy" if error_count < 10 else "degraded",
        "server_name": "python-analysis",
        "uptime_seconds": int(uptime_seconds),
        "uptime_human": format_uptime(uptime_seconds),
        "request_count": request_count,
        "error_count": error_count,
        "last_error": last_error,
        "memory_mb": round(memory_mb, 2),
        "cpu_percent": round(cpu_percent, 2),
        "pid": os.getpid(),
        "timestamp": datetime.now().isoformat()
    }
    
    return health_status

def format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m {int(seconds % 60)}s"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
    else:
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        return f"{days}d {hours}h"

if __name__ == "__main__":
    print(f"Starting Python Analysis MCP server...")
    print(f"Workspace: {WORKSPACE_ROOT}")
    
    # Run server
    mcp.run()