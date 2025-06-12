#!/usr/bin/env python3
"""Simple ripgrep MCP server using FastMCP."""

from mcp.server import FastMCP

import subprocess
import os

# Create server
mcp = FastMCP("ripgrep")

@mcp.tool()
def search(pattern: str, path: str = ".", max_results: int = 100) -> str:
    """Search files using ripgrep."""
    try:
        # Ensure path is absolute
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        
        # Build ripgrep command
        cmd = ["rg", "--max-count", str(max_results), "--color", "never", 
               "--line-number", "--with-filename", pattern, path]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            output = result.stdout[:10000]  # Limit output
            lines = output.strip().split('\n') if output.strip() else []
            if len(lines) > max_results:
                output = '\n'.join(lines[:max_results])
                output += f"\n\n... truncated to {max_results} results"
            return output if output else "No matches found"
        elif result.returncode == 1:
            return "No matches found"
        else:
            return f"Error: {result.stderr.strip() if result.stderr else 'Unknown error'}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except FileNotFoundError:
        return "Error: ripgrep (rg) not found. Please install ripgrep."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def search_count(pattern: str, path: str = ".") -> str:
    """Count matches using ripgrep."""
    try:
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        
        cmd = ["rg", "--count", "--color", "never", pattern, path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return result.stdout.strip()
        elif result.returncode == 1:
            return "0 matches"
        else:
            return f"Error: {result.stderr.strip()}"
            
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def list_files(pattern: str, path: str = ".") -> str:
    """List files containing matches."""
    try:
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        
        cmd = ["rg", "--files-with-matches", "--color", "never", pattern, path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return result.stdout.strip()
        elif result.returncode == 1:
            return "No files found"
        else:
            return f"Error: {result.stderr.strip()}"
            
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Run the server
    print("Starting ripgrep MCP server...")
    mcp.run()