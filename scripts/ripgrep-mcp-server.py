#!/usr/bin/env python3
"""
Ripgrep MCP Server - A Python wrapper for ripgrep functionality
Provides fast file search capabilities through MCP
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ToolArgument

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RipgrepServer:
    """MCP server that wraps ripgrep functionality"""
    
    def __init__(self):
        self.server = Server("ripgrep-mcp")
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Set up MCP tool handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available ripgrep tools"""
            return [
                Tool(
                    name="search",
                    description="Search for patterns in files using ripgrep",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The pattern to search for (regex supported)"
                            },
                            "path": {
                                "type": "string",
                                "description": "Path to search in (default: current directory)"
                            },
                            "glob": {
                                "type": "string",
                                "description": "File glob pattern (e.g., '*.py')"
                            },
                            "case_sensitive": {
                                "type": "boolean",
                                "description": "Case sensitive search (default: false)"
                            },
                            "max_count": {
                                "type": "integer",
                                "description": "Maximum number of matches per file"
                            },
                            "context_lines": {
                                "type": "integer",
                                "description": "Number of context lines to show"
                            }
                        },
                        "required": ["pattern"]
                    }
                ),
                Tool(
                    name="files_with_matches",
                    description="List files containing matches (no content)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The pattern to search for"
                            },
                            "path": {
                                "type": "string",
                                "description": "Path to search in"
                            },
                            "glob": {
                                "type": "string",
                                "description": "File glob pattern"
                            }
                        },
                        "required": ["pattern"]
                    }
                ),
                Tool(
                    name="count_matches",
                    description="Count occurrences of a pattern",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The pattern to count"
                            },
                            "path": {
                                "type": "string",
                                "description": "Path to search in"
                            },
                            "glob": {
                                "type": "string",
                                "description": "File glob pattern"
                            }
                        },
                        "required": ["pattern"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Execute ripgrep tool"""
            try:
                # Check if ripgrep is available
                if not self._check_ripgrep():
                    return [TextContent(
                        type="text",
                        text="Error: ripgrep (rg) is not installed. Please install it:\n"
                             "  brew install ripgrep    # macOS\n"
                             "  apt install ripgrep     # Ubuntu/Debian"
                    )]
                
                if name == "search":
                    result = await self._search(arguments)
                elif name == "files_with_matches":
                    result = await self._files_with_matches(arguments)
                elif name == "count_matches":
                    result = await self._count_matches(arguments)
                else:
                    result = f"Unknown tool: {name}"
                
                return [TextContent(type="text", text=result)]
                
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _check_ripgrep(self) -> bool:
        """Check if ripgrep is installed"""
        try:
            subprocess.run(["rg", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    async def _search(self, args: Dict[str, Any]) -> str:
        """Perform a ripgrep search"""
        cmd = ["rg"]
        
        # Add case sensitivity
        if not args.get("case_sensitive", False):
            cmd.append("-i")
        
        # Add context lines
        if "context_lines" in args:
            cmd.extend(["-C", str(args["context_lines"])])
        
        # Add max count
        if "max_count" in args:
            cmd.extend(["-m", str(args["max_count"])])
        
        # Add glob pattern
        if "glob" in args:
            cmd.extend(["-g", args["glob"]])
        
        # Add pattern
        cmd.append(args["pattern"])
        
        # Add path
        if "path" in args:
            cmd.append(args["path"])
        
        # Execute ripgrep
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout
            elif result.returncode == 1:
                return "No matches found"
            else:
                return f"Error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "Error: Search timed out after 30 seconds"
        except Exception as e:
            return f"Error executing ripgrep: {str(e)}"
    
    async def _files_with_matches(self, args: Dict[str, Any]) -> str:
        """List files containing matches"""
        cmd = ["rg", "-l"]  # List files only
        
        # Add glob pattern
        if "glob" in args:
            cmd.extend(["-g", args["glob"]])
        
        # Add pattern
        cmd.append(args["pattern"])
        
        # Add path
        if "path" in args:
            cmd.append(args["path"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                return f"Found {len(files)} files with matches:\n" + result.stdout
            elif result.returncode == 1:
                return "No files found with matches"
            else:
                return f"Error: {result.stderr}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _count_matches(self, args: Dict[str, Any]) -> str:
        """Count pattern occurrences"""
        cmd = ["rg", "-c"]  # Count matches
        
        # Add glob pattern
        if "glob" in args:
            cmd.extend(["-g", args["glob"]])
        
        # Add pattern
        cmd.append(args["pattern"])
        
        # Add path
        if "path" in args:
            cmd.append(args["path"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total = sum(int(line.split(':')[-1]) for line in lines if ':' in line)
                return f"Total matches: {total}\n\nPer file:\n{result.stdout}"
            elif result.returncode == 1:
                return "No matches found"
            else:
                return f"Error: {result.stderr}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)

def main():
    """Main entry point"""
    server = RipgrepServer()
    asyncio.run(server.run())

if __name__ == "__main__":
    main()