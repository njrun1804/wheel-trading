#!/usr/bin/env python3
"""
Connection pooling wrapper for Python MCP servers.
Reduces startup overhead by keeping Python interpreter warm.
"""

import os
import sys
import importlib
import signal
import time
from pathlib import Path

# Pre-import heavy modules to warm up
PRELOAD_MODULES = [
    'pandas',
    'numpy',
    'sqlalchemy',
    'duckdb',
    'requests',
    'httpx',
    'pydantic',
    'logfire',
    'opentelemetry'
]

def preload_modules():
    """Pre-import heavy modules to reduce cold start time"""
    for module in PRELOAD_MODULES:
        try:
            importlib.import_module(module)
            print(f"✓ Preloaded {module}", file=sys.stderr)
        except ImportError:
            pass  # Module not installed, skip

def warm_cache():
    """Warm up various caches"""
    # Warm Python's import cache
    import_cache_size = len(sys.modules)
    
    # Warm DuckDB if available
    try:
        import duckdb
        conn = duckdb.connect(':memory:')
        conn.execute("SELECT 1").fetchall()
        conn.close()
        print("✓ Warmed DuckDB cache", file=sys.stderr)
    except:
        pass
    
    # Warm sklearn if available
    try:
        import sklearn
        import joblib
        print("✓ Warmed sklearn cache", file=sys.stderr)
    except:
        pass
    
    print(f"✓ Import cache warmed: {len(sys.modules)} modules", file=sys.stderr)

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: mcp-connection-pool.py <mcp_script> [args...]", file=sys.stderr)
        sys.exit(1)
    
    # Preload and warm caches
    print("Warming MCP server...", file=sys.stderr)
    start = time.time()
    preload_modules()
    warm_cache()
    print(f"✓ Ready in {time.time() - start:.2f}s", file=sys.stderr)
    
    # Run the actual MCP server
    mcp_script = sys.argv[1]
    sys.argv = sys.argv[1:]  # Shift arguments
    
    # Execute the MCP server
    exec(open(mcp_script).read(), {'__name__': '__main__'})

if __name__ == '__main__':
    # Handle graceful shutdown
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    
    main()