#!/usr/bin/env python3
"""Simple Python analysis MCP server without complex imports."""

from mcp.server import FastMCP
import subprocess
import os
import json
from datetime import datetime
from pathlib import Path

# Get workspace root
WORKSPACE_ROOT = os.environ.get('WORKSPACE_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mcp = FastMCP("python-analysis")

# Track stats
start_time = datetime.now()
request_count = 0
error_count = 0

@mcp.tool()
def analyze_position(symbol: str, capital: float = 100000) -> str:
    """Analyze trading position for a given symbol."""
    global request_count
    request_count += 1
    
    try:
        # Run the actual advisor script
        cmd = [
            "python3", 
            os.path.join(WORKSPACE_ROOT, "run.py"),
            "-s", symbol,
            "-p", str(capital),
            "--json"
        ]
        
        result = subprocess.run(
            cmd,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                return f"""Position Analysis for {symbol}:
- Action: {data.get('action', 'N/A')}
- Confidence: {data.get('confidence', 0):.2%}
- Contracts: {data.get('contracts', 0)}
- Strike: ${data.get('strike', 0):.2f}
- Premium: ${data.get('premium', 0):.2f}
- Expiry: {data.get('expiry', 'N/A')}
"""
            except json.JSONDecodeError:
                return result.stdout
        else:
            global error_count
            error_count += 1
            return f"Error analyzing position: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        error_count += 1
        return "Analysis timed out after 30 seconds"
    except Exception as e:
        error_count += 1
        return f"Error analyzing position: {str(e)}"

@mcp.tool()
def monitor_system() -> str:
    """Monitor system status and MCP health."""
    global request_count
    request_count += 1
    
    try:
        import psutil
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Process info
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024**2)
        
        # Uptime
        uptime = datetime.now() - start_time
        uptime_str = str(uptime).split('.')[0]
        
        return f"""System Status:
=== System Resources ===
CPU: {cpu_percent}%
Memory: {memory.percent}% used ({memory.used // (1024**3)} GB / {memory.total // (1024**3)} GB)

=== MCP Server Status ===
Server: python-analysis
Uptime: {uptime_str}
Requests: {request_count}
Errors: {error_count}
Memory: {process_memory_mb:.1f} MB
PID: {process.pid}
"""
    except ImportError:
        return "psutil not available - install with: pip install psutil"
    except Exception as e:
        return f"Error monitoring system: {str(e)}"

@mcp.tool()
def data_quality_check() -> str:
    """Check data quality and database health."""
    global request_count
    request_count += 1
    
    try:
        import duckdb
        
        db_path = Path(WORKSPACE_ROOT) / "data" / "wheel_trading_master.duckdb"
        if not db_path.exists():
            # Try cache location
            db_path = Path(WORKSPACE_ROOT) / "data" / "cache" / "wheel_cache.duckdb"
            
        if not db_path.exists():
            return "Database not found at expected locations"
            
        conn = duckdb.connect(str(db_path), read_only=True)
        
        # Get basic info
        result = f"Database: {db_path.name}\n"
        result += f"Size: {db_path.stat().st_size / (1024**2):.1f} MB\n\n"
        
        # Get tables
        tables = conn.execute("SHOW TABLES").fetchall()
        result += f"Tables ({len(tables)}):\n"
        
        for table in tables[:10]:  # Limit to first 10 tables
            table_name = table[0]
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                result += f"- {table_name}: {count:,} rows\n"
            except:
                result += f"- {table_name}: (unable to count)\n"
                
        conn.close()
        return result
        
    except ImportError:
        return "DuckDB not available - install with: pip install duckdb"
    except Exception as e:
        global error_count
        error_count += 1
        return f"Error checking data quality: {str(e)}"

@mcp.tool()
def run_tests(test_type: str = "fast") -> str:
    """Run test suite.
    
    Args:
        test_type: 'fast' (default), 'all', or path to specific test file
    """
    global request_count
    request_count += 1
    
    try:
        if test_type == "fast":
            cmd = ["python3", "-m", "pytest", "-v", "-m", "not slow", "--tb=short"]
        elif test_type == "all":
            cmd = ["python3", "-m", "pytest", "-v", "--tb=short"]
        else:
            cmd = ["python3", "-m", "pytest", "-v", test_type, "--tb=short"]
            
        result = subprocess.run(
            cmd,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Extract summary from output
        output = result.stdout if result.returncode == 0 else result.stderr
        lines = output.split('\n')
        
        # Find test summary
        summary_start = -1
        for i, line in enumerate(lines):
            if "=" in line and ("passed" in line or "failed" in line or "error" in line):
                summary_start = i
                break
                
        if summary_start >= 0:
            summary = '\n'.join(lines[summary_start:])
        else:
            summary = "No test summary found"
            
        return f"""Test Results:
Exit Code: {result.returncode}
Status: {'✓ PASSED' if result.returncode == 0 else '✗ FAILED'}

{summary}
"""
    except subprocess.TimeoutExpired:
        global error_count
        error_count += 1
        return "Tests timed out after 60 seconds"
    except Exception as e:
        error_count += 1
        return f"Error running tests: {str(e)}"

@mcp.tool()
def get_recommendation(symbol: str = "U", capital: float = 100000) -> str:
    """Get trading recommendation - alias for analyze_position."""
    return analyze_position(symbol, capital)

if __name__ == "__main__":
    # Run without stdout output to avoid breaking JSON-RPC
    # Workspace: {WORKSPACE_ROOT}
    mcp.run()