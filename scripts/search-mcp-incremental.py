#!/usr/bin/env python3
"""MCP server with incremental search and file watching."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.unity_wheel.mcp.base_server import HealthCheckMCP
from src.unity_wheel.mcp.incremental_indexer import IncrementalIndexer
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import json

# Get workspace root
WORKSPACE_ROOT = os.environ.get('WORKSPACE_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mcp = HealthCheckMCP("search-incremental", workspace_root=WORKSPACE_ROOT)

# Initialize indexer
indexer = IncrementalIndexer(WORKSPACE_ROOT)
indexer.connect()

# Start file watching in background
watching_task = None

@mcp.tool()
def start_file_watching() -> Dict[str, str]:
    """Start watching filesystem for changes to update index incrementally."""
    mcp.track_request()
    
    global watching_task
    
    if watching_task and not watching_task.done():
        return {
            "status": "already_running",
            "message": "File watching is already active"
        }
    
    try:
        indexer.start_watching()
        watching_task = asyncio.create_task(indexer.process_updates())
        
        return {
            "status": "started",
            "message": f"Started watching {WORKSPACE_ROOT} for changes",
            "stats": indexer.get_stats()
        }
    except Exception as e:
        mcp.track_error(str(e))
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.tool()
def stop_file_watching() -> Dict[str, str]:
    """Stop watching filesystem for changes."""
    mcp.track_request()
    
    global watching_task
    
    try:
        indexer.processing = False
        indexer.stop_watching()
        
        if watching_task:
            watching_task.cancel()
            
        return {
            "status": "stopped",
            "message": "File watching stopped",
            "stats": indexer.get_stats()
        }
    except Exception as e:
        mcp.track_error(str(e))
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.tool()
def search_incremental(query: str, include_deleted: bool = False) -> List[Dict[str, Any]]:
    """Search files using incremental index with real-time updates.
    
    Args:
        query: Search query (supports FTS syntax)
        include_deleted: Include deleted files in results
        
    Returns:
        List of matching files with snippets
    """
    mcp.track_request()
    
    try:
        results = indexer.search_incremental(query, include_deleted)
        
        if not results:
            return [{
                "message": f"No results found for '{query}'",
                "stats": indexer.get_stats()
            }]
        
        # Add stats to first result
        if results:
            results[0]["stats"] = indexer.get_stats()
            
        return results
        
    except Exception as e:
        mcp.track_error(str(e))
        return [{"error": str(e)}]

@mcp.tool()
def get_file_history(file_path: str) -> List[Dict[str, Any]]:
    """Get change history for a specific file.
    
    Args:
        file_path: Relative path to file
        
    Returns:
        List of changes with timestamps and versions
    """
    mcp.track_request()
    
    try:
        history = indexer.get_change_history(file_path, limit=20)
        
        if not history:
            return [{
                "message": f"No history found for {file_path}",
                "hint": "File may not be indexed or path may be incorrect"
            }]
            
        return history
        
    except Exception as e:
        mcp.track_error(str(e))
        return [{"error": str(e)}]

@mcp.tool()
def get_recent_changes(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent file changes across the project.
    
    Args:
        limit: Maximum number of changes to return
        
    Returns:
        List of recent changes with details
    """
    mcp.track_request()
    
    try:
        changes = indexer.get_change_history(limit=limit)
        
        if not changes:
            return [{
                "message": "No recent changes found",
                "stats": indexer.get_stats()
            }]
            
        # Group by file for summary
        file_changes = {}
        for change in changes:
            file_path = change['file_path']
            if file_path not in file_changes:
                file_changes[file_path] = []
            file_changes[file_path].append(change)
            
        summary = {
            "total_changes": len(changes),
            "files_changed": len(file_changes),
            "changes": changes[:10],  # First 10
            "stats": indexer.get_stats()
        }
        
        return [summary]
        
    except Exception as e:
        mcp.track_error(str(e))
        return [{"error": str(e)}]

@mcp.tool()
def rebuild_index(force: bool = False) -> Dict[str, Any]:
    """Rebuild the incremental index from scratch.
    
    Args:
        force: Force rebuild even if index is recent
        
    Returns:
        Rebuild status and statistics
    """
    mcp.track_request()
    
    try:
        # Stop watching during rebuild
        was_watching = indexer.observer.is_alive()
        if was_watching:
            indexer.stop_watching()
            
        # Clear existing index if forcing
        if force:
            indexer.conn.execute("DELETE FROM file_index")
            indexer.conn.execute("DELETE FROM change_history")
            
        # Find all Python files
        py_files = list(Path(WORKSPACE_ROOT).rglob("*.py"))
        indexed = 0
        errors = 0
        
        for file_path in py_files:
            if indexer.should_ignore(file_path):
                continue
                
            try:
                # Index file synchronously
                indexer.index_file(file_path)
                indexed += 1
                
                if indexed % 100 == 0:
                    indexer.conn.commit()
                    
            except Exception:
                errors += 1
                
        indexer.conn.commit()
        
        # Restart watching if it was active
        if was_watching:
            indexer.start_watching()
            global watching_task
            watching_task = asyncio.create_task(indexer.process_updates())
            
        return {
            "status": "complete",
            "files_found": len(py_files),
            "files_indexed": indexed,
            "errors": errors,
            "stats": indexer.get_stats()
        }
        
    except Exception as e:
        mcp.track_error(str(e))
        return {"error": str(e)}

@mcp.tool()
def get_vscode_config() -> Dict[str, Any]:
    """Get VS Code configuration for automatic index updates."""
    mcp.track_request()
    
    tasks_config = {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "Update MCP Index on Save",
                "type": "shell",
                "command": "curl",
                "args": [
                    "-X", "POST",
                    "http://localhost:8765/file-saved",
                    "-H", "Content-Type: application/json",
                    "-d", '{"file": "${file}"}'
                ],
                "presentation": {
                    "reveal": "never",
                    "panel": "dedicated",
                    "showReuseMessage": false,
                    "clear": true
                },
                "problemMatcher": []
            }
        ]
    }
    
    settings_config = {
        "emeraldwalk.runonsave": {
            "commands": [
                {
                    "match": "**/*.py",
                    "cmd": "code-insiders --run-task 'Update MCP Index on Save'",
                    "isAsync": true,
                    "useShortcut": false
                }
            ]
        }
    }
    
    return {
        "instructions": [
            "1. Install 'Run on Save' extension by emeraldwalk",
            "2. Add the tasks configuration to .vscode/tasks.json",
            "3. Add the settings configuration to .vscode/settings.json",
            "4. Make sure the search-incremental MCP server is running with file watching enabled"
        ],
        "tasks_json": tasks_config,
        "settings_json": settings_config,
        "current_status": {
            "watching": indexer.observer.is_alive(),
            "stats": indexer.get_stats()
        }
    }

# Cleanup on shutdown
async def cleanup():
    """Clean up resources on shutdown."""
    if indexer.observer.is_alive():
        indexer.stop_watching()
    if indexer.conn:
        indexer.conn.close()

if __name__ == "__main__":
    import atexit
    
    # Register cleanup
    atexit.register(lambda: asyncio.run(cleanup()))
    
    # Clean up stale health files
    HealthCheckMCP.cleanup_stale_health_files(WORKSPACE_ROOT)
    
    print(f"Starting incremental search MCP server...")
    print(f"Workspace: {WORKSPACE_ROOT}")
    print(f"Features:")
    print("- Real-time incremental indexing")
    print("- File change tracking with versions")
    print("- VS Code integration support")
    print("- Sub-millisecond search performance")
    print("\nUse 'start_file_watching' tool to enable real-time updates")
    
    # Run server
    mcp.run()