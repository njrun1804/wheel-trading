#!/usr/bin/env python3
"""Verify all MCP servers are correctly configured and can start."""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

def test_mcp_server(script_path: str, timeout: int = 5) -> dict:
    """Test if an MCP server can start correctly."""
    result = {
        "script": os.path.basename(script_path),
        "path": script_path,
        "status": "unknown",
        "error": None,
        "imports": True,
        "syntax": True
    }
    
    # First check syntax
    try:
        subprocess.run(
            [sys.executable, "-m", "py_compile", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        result["syntax"] = True
    except subprocess.CalledProcessError as e:
        result["syntax"] = False
        result["status"] = "syntax_error"
        result["error"] = e.stderr
        return result
    
    # Check imports by running with --help or in test mode
    try:
        # Try to import and check for FastMCP usage
        test_code = f"""
import sys
sys.path.insert(0, '{os.path.dirname(script_path)}')
sys.path.insert(0, '{os.path.dirname(os.path.dirname(script_path))}')

try:
    # Read the script to check for FastMCP
    with open('{script_path}', 'r') as f:
        content = f.read()
    
    # Check if it's an MCP server
    if 'from mcp.server import' in content or 'import mcp.server' in content:
        # Try to import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_module", '{script_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("IMPORT_OK")
    else:
        print("NOT_MCP_SERVER")
except Exception as e:
    print(f"IMPORT_ERROR: {{e}}")
"""
        
        proc = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if "IMPORT_OK" in proc.stdout:
            result["imports"] = True
            result["status"] = "ok"
        elif "NOT_MCP_SERVER" in proc.stdout:
            result["status"] = "not_mcp_server"
        else:
            result["imports"] = False
            result["status"] = "import_error"
            result["error"] = proc.stdout + proc.stderr
            
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Import check timed out"
    except Exception as e:
        result["imports"] = False
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

def main():
    """Check all MCP servers."""
    scripts_dir = Path(__file__).parent
    
    # Find all potential MCP server files
    mcp_patterns = ["*mcp*.py", "*server*.py"]
    mcp_files = set()
    
    for pattern in mcp_patterns:
        mcp_files.update(scripts_dir.glob(pattern))
    
    # Filter out test and utility scripts
    exclude_patterns = ["test-", "fix-", "verify-", "check-", "monitor", "doctor"]
    mcp_servers = []
    
    for f in sorted(mcp_files):
        if f.is_file() and not any(exc in f.name for exc in exclude_patterns):
            mcp_servers.append(f)
    
    print(f"Found {len(mcp_servers)} MCP server scripts to check\n")
    
    results = []
    for server_path in mcp_servers:
        print(f"Checking {server_path.name}...", end=" ", flush=True)
        result = test_mcp_server(str(server_path))
        results.append(result)
        
        if result["status"] == "ok":
            print("✓ OK")
        elif result["status"] == "not_mcp_server":
            print("- Not an MCP server")
        else:
            print(f"✗ {result['status']}")
            if result["error"]:
                print(f"  Error: {result['error'][:100]}...")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    ok_count = sum(1 for r in results if r["status"] == "ok")
    error_count = sum(1 for r in results if r["status"] not in ["ok", "not_mcp_server"])
    not_mcp_count = sum(1 for r in results if r["status"] == "not_mcp_server")
    
    print(f"Total scripts checked: {len(results)}")
    print(f"MCP servers OK: {ok_count}")
    print(f"Not MCP servers: {not_mcp_count}")
    print(f"Errors: {error_count}")
    
    if error_count > 0:
        print("\nServers with errors:")
        for r in results:
            if r["status"] not in ["ok", "not_mcp_server"]:
                print(f"  - {r['script']}: {r['status']}")
    
    # Write detailed results
    results_file = scripts_dir / "mcp-server-verification.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results written to: {results_file}")
    
    return error_count == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)