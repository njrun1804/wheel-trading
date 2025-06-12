#!/usr/bin/env python3
"""Ripgrep MCP server for fast file searching."""

from mcp.server import FastMCP
import subprocess
import os

mcp = FastMCP("ripgrep")

@mcp.tool()
def search(pattern: str, path: str = ".", max_results: int = 100) -> str:
    """Search files using ripgrep.
    
    Args:
        pattern: Regular expression pattern to search for
        path: Directory to search in (default: current directory)
        max_results: Maximum number of results to return (default: 100)
    """
    try:
        # Ensure path is absolute
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        
        # Build ripgrep command
        cmd = [
            "rg",
            "--max-count", str(max_results),
            "--color", "never",
            "--line-number",
            "--with-filename",
            pattern,
            path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            output = result.stdout[:10000]  # Limit output size
            lines = output.strip().split('\n')
            if len(lines) > max_results:
                output = '\n'.join(lines[:max_results])
                output += f"\n\n... truncated to {max_results} results"
            return output
        elif result.returncode == 1:
            return "No matches found"
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return f"Error: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except FileNotFoundError:
        return "Error: ripgrep (rg) not found. Please install ripgrep first."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def search_with_context(pattern: str, path: str = ".", before: int = 2, after: int = 2) -> str:
    """Search files with context lines.
    
    Args:
        pattern: Regular expression pattern to search for
        path: Directory to search in (default: current directory)
        before: Number of lines to show before match (default: 2)
        after: Number of lines to show after match (default: 2)
    """
    try:
        # Ensure path is absolute
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        
        cmd = [
            "rg",
            "--before-context", str(before),
            "--after-context", str(after),
            "--color", "never",
            "--line-number",
            "--with-filename",
            pattern,
            path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return result.stdout[:10000]  # Limit output size
        elif result.returncode == 1:
            return "No matches found"
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return f"Error: {error_msg}"
            
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def search_file_type(pattern: str, file_type: str, path: str = ".") -> str:
    """Search only in files of a specific type.
    
    Args:
        pattern: Regular expression pattern to search for
        file_type: File extension to search in (e.g., 'py', 'js', 'md')
        path: Directory to search in (default: current directory)
    """
    try:
        # Ensure path is absolute
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        
        cmd = [
            "rg",
            "--type", file_type,
            "--color", "never",
            "--line-number",
            "--with-filename",
            pattern,
            path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return result.stdout[:10000]  # Limit output size
        elif result.returncode == 1:
            return f"No matches found in {file_type} files"
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return f"Error: {error_msg}"
            
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("Starting ripgrep MCP server...")
    mcp.run()
