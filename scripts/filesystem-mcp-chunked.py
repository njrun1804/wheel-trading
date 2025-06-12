#!/usr/bin/env python3
"""Enhanced filesystem MCP server with dynamic chunking."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.unity_wheel.mcp.base_server import HealthCheckMCP
from src.unity_wheel.mcp.dynamic_chunking import ChunkedFileReader
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

# Get workspace root
WORKSPACE_ROOT = os.environ.get('WORKSPACE_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mcp = HealthCheckMCP("filesystem-chunked", workspace_root=WORKSPACE_ROOT)
reader = ChunkedFileReader()

@mcp.tool()
def read_file_chunked(file_path: str, chunk_id: Optional[int] = None, 
                     semantic: bool = True) -> Dict[str, Any]:
    """Read file with intelligent chunking for optimal token usage.
    
    Args:
        file_path: Path to file to read
        chunk_id: Specific chunk to read (None for chunk list)
        semantic: Use semantic chunking for code files
        
    Returns:
        If chunk_id is None: List of available chunks with metadata
        If chunk_id specified: The requested chunk content
    """
    mcp.track_request()
    
    try:
        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(WORKSPACE_ROOT) / path
            
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
            
        # Get all chunks
        chunks = reader.read_file_chunked(str(path), semantic=semantic)
        
        if not chunks:
            return {"error": "No chunks created"}
            
        # Return specific chunk or chunk list
        if chunk_id is not None:
            if 1 <= chunk_id <= len(chunks):
                chunk = chunks[chunk_id - 1]
                return {
                    "chunk": chunk,
                    "performance": reader.chunker.get_performance_stats()
                }
            else:
                return {"error": f"Invalid chunk_id: {chunk_id}. Valid range: 1-{len(chunks)}"}
        else:
            # Return chunk summary
            total_tokens = sum(c['tokens'] for c in chunks)
            return {
                "file_path": str(path),
                "total_chunks": len(chunks),
                "total_tokens": total_tokens,
                "chunks": [
                    {
                        "id": c['chunk_id'],
                        "lines": f"{c['start_line']}-{c['end_line']}",
                        "line_count": c['lines'],
                        "tokens": c['tokens']
                    }
                    for c in chunks
                ],
                "performance": reader.chunker.get_performance_stats()
            }
            
    except Exception as e:
        mcp.track_error(str(e))
        return {"error": str(e)}

@mcp.tool()
def search_file_chunked(pattern: str, file_path: str, 
                       context_lines: int = 3) -> List[Dict[str, Any]]:
    """Search file with ripgrep and return contextual chunks.
    
    Args:
        pattern: Search pattern (regex supported)
        file_path: File to search in
        context_lines: Lines of context around matches
        
    Returns:
        List of chunks containing matches with context
    """
    mcp.track_request()
    
    try:
        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(WORKSPACE_ROOT) / path
            
        if not path.exists():
            return [{"error": f"File not found: {file_path}"}]
            
        # Search and chunk
        chunks = reader.read_with_ripgrep(pattern, str(path), context_lines)
        
        if not chunks:
            return [{"message": f"No matches found for '{pattern}'"}]
            
        # Add performance stats to first result
        if chunks:
            chunks[0]["performance"] = reader.chunker.get_performance_stats()
            
        return chunks
        
    except Exception as e:
        mcp.track_error(str(e))
        return [{"error": str(e)}]

@mcp.tool()
def read_large_file(file_path: str, start_line: int = 1, 
                   max_tokens: int = 3000) -> Dict[str, Any]:
    """Read portion of a large file within token budget.
    
    Args:
        file_path: Path to file
        start_line: Line to start reading from
        max_tokens: Maximum tokens to return
        
    Returns:
        File content within token budget with metadata
    """
    mcp.track_request()
    
    try:
        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(WORKSPACE_ROOT) / path
            
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
            
        # Read file
        content = path.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')
        
        if start_line > len(lines):
            return {"error": f"start_line {start_line} exceeds file length {len(lines)}"}
            
        # Build content within token budget
        result_lines = []
        total_tokens = 0
        
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            line_tokens = reader.chunker.count_tokens(line + '\n')
            
            if total_tokens + line_tokens > max_tokens:
                break
                
            result_lines.append(line)
            total_tokens += line_tokens
            
        return {
            "file_path": str(path),
            "start_line": start_line,
            "end_line": start_line + len(result_lines) - 1,
            "lines_read": len(result_lines),
            "total_lines": len(lines),
            "tokens_used": total_tokens,
            "max_tokens": max_tokens,
            "content": '\n'.join(result_lines),
            "more_content": start_line + len(result_lines) - 1 < len(lines)
        }
        
    except Exception as e:
        mcp.track_error(str(e))
        return {"error": str(e)}

@mcp.tool()
def analyze_file_complexity(file_path: str) -> Dict[str, Any]:
    """Analyze file complexity and recommend chunk strategy.
    
    Args:
        file_path: Path to file to analyze
        
    Returns:
        Complexity analysis and chunking recommendations
    """
    mcp.track_request()
    
    try:
        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(WORKSPACE_ROOT) / path
            
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
            
        # Read file
        content = path.read_text(encoding='utf-8', errors='ignore')
        
        # Analyze
        lines = content.split('\n')
        total_tokens = reader.chunker.count_tokens(content)
        complexity = reader.chunker.estimate_complexity(content)
        optimal_chunk_size = reader.chunker.calculate_optimal_chunk_size(str(path), content)
        
        # Calculate metrics
        tokens_per_line = total_tokens / max(len(lines), 1)
        estimated_chunks = max(1, len(lines) // optimal_chunk_size)
        
        # Recommendations
        recommendations = []
        
        if total_tokens > 10000:
            recommendations.append("File is large - use chunked reading")
            
        if complexity > 1.5:
            recommendations.append("High complexity - use semantic chunking")
            
        if tokens_per_line > 50:
            recommendations.append("Dense code - smaller chunks recommended")
            
        if path.suffix == '.py' and 'test' in path.name.lower():
            recommendations.append("Test file - consider reading specific test functions")
            
        return {
            "file_path": str(path),
            "file_size_kb": path.stat().st_size / 1024,
            "total_lines": len(lines),
            "total_tokens": total_tokens,
            "tokens_per_line": round(tokens_per_line, 1),
            "complexity_score": round(complexity, 2),
            "optimal_chunk_size": optimal_chunk_size,
            "estimated_chunks": estimated_chunks,
            "recommendations": recommendations,
            "performance": reader.chunker.get_performance_stats()
        }
        
    except Exception as e:
        mcp.track_error(str(e))
        return {"error": str(e)}

@mcp.tool()
def get_chunking_stats() -> Dict[str, Any]:
    """Get performance statistics for dynamic chunking."""
    mcp.track_request()
    
    stats = reader.chunker.get_performance_stats()
    
    if not stats:
        return {
            "message": "No chunking operations performed yet",
            "hint": "Use read_file_chunked or search_file_chunked to generate statistics"
        }
        
    return {
        "statistics": stats,
        "description": "Dynamic chunking adapts based on file complexity and token density",
        "optimization_tips": [
            "Semantic chunking preserves code structure for Python files",
            "Search operations create focused chunks around matches",
            "Large files are automatically chunked to stay within token limits",
            "Performance improves over time as the system learns optimal chunk sizes"
        ]
    }

if __name__ == "__main__":
    import asyncio
    
    # Clean up stale health files
    HealthCheckMCP.cleanup_stale_health_files(WORKSPACE_ROOT)
    
    print(f"Starting enhanced filesystem MCP server with dynamic chunking...")
    print(f"Workspace: {WORKSPACE_ROOT}")
    print(f"Features:")
    print("- Dynamic chunk sizing based on file complexity")
    print("- Semantic chunking for Python files")
    print("- Ripgrep integration for focused search")
    print("- Token budget management")
    print("- Performance optimization over time")
    
    # Run server
    mcp.run()