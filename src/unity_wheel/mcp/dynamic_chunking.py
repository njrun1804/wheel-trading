#!/usr/bin/env python3
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


Dynamic chunk sizing for optimal token usage in MCP servers.
Adapts chunk size based on file size, complexity, and response time.
"""

import time
import math
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import tiktoken
import re

@dataclass
class ChunkMetrics:
    """Metrics for chunk performance."""
    file_path: str
    chunk_size: int
    token_count: int
    processing_time: float
    success: bool
    complexity_score: float
    
@dataclass
class ChunkStrategy:
    """Dynamic chunking strategy."""
    base_chunk_size: int = 2000  # Base lines per chunk
    min_chunk_size: int = 100
    max_chunk_size: int = 5000
    target_tokens: int = 3000  # Target tokens per chunk
    max_tokens: int = 4000  # Max tokens to avoid limits
    

class DynamicChunker:
    """Intelligent file chunking for optimal token usage."""
    
    def __init__(self, model: str = "gpt-4"):
        self.encoder = tiktoken.encoding_for_model(model)
        self.strategy = ChunkStrategy()
        self.performance_history: List[ChunkMetrics] = []
        self._chunk_size_cache: Dict[str, int] = {}
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))
        
    def estimate_complexity(self, content: str) -> float:
        """Estimate code complexity for chunk sizing."""
        # Simple heuristics for code complexity
        lines = content.split('\n')
        
        # Factors that increase complexity
        complexity = 1.0
        
        # Dense code (short lines with lots of symbols)
        avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
        if avg_line_length < 40:
            complexity *= 1.2
            
        # Many imports (interconnected code)
        import_count = sum(1 for line in lines if line.strip().startswith(('import', 'from')))
        if import_count > 10:
            complexity *= 1.1
            
        # Complex expressions
        complex_patterns = [
            r'\blambda\b',  # Lambda functions
            r'[\[\{].*[\]\}].*[\[\{].*[\]\}]',  # Nested structures
            r'\b(?:if|else|elif)\b.*\b(?:if|else|elif)\b',  # Nested conditionals
            r'(?:def|class)\s+\w+\s*\([^)]{50,}\)',  # Long parameter lists
        ]
        
        for pattern in complex_patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            if matches > 0:
                complexity *= (1 + matches * 0.05)
                
        # Many comments (needs more context)
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        if comment_lines > len(lines) * 0.3:
            complexity *= 1.1
            
        return min(complexity, 2.0)  # Cap at 2x
        
    def calculate_optimal_chunk_size(self, file_path: str, content: str) -> int:
        """Calculate optimal chunk size for a file."""
        # Check cache
        cache_key = f"{file_path}:{len(content)}"
        if cache_key in self._chunk_size_cache:
            return self._chunk_size_cache[cache_key]
            
        # Base calculation
        total_lines = len(content.split('\n'))
        total_tokens = self.count_tokens(content)
        
        if total_tokens == 0:
            return self.strategy.base_chunk_size
            
        # Tokens per line average
        tokens_per_line = total_tokens / max(total_lines, 1)
        
        # Estimate complexity
        complexity = self.estimate_complexity(content)
        
        # Calculate base chunk size to hit target tokens
        base_chunk = int(self.strategy.target_tokens / (tokens_per_line * complexity))
        
        # Adjust based on performance history for similar files
        if self.performance_history:
            similar_metrics = [
                m for m in self.performance_history[-20:]  # Last 20 chunks
                if abs(m.complexity_score - complexity) < 0.2
            ]
            
            if similar_metrics:
                # Find the chunk size that performed best
                best_metric = min(similar_metrics, key=lambda m: m.processing_time / m.chunk_size)
                base_chunk = int(base_chunk * 0.7 + best_metric.chunk_size * 0.3)
                
        # Apply bounds
        chunk_size = max(
            self.strategy.min_chunk_size,
            min(base_chunk, self.strategy.max_chunk_size)
        )
        
        # Cache the result
        self._chunk_size_cache[cache_key] = chunk_size
        
        return chunk_size
        
    def create_chunks(self, file_path: str, content: str) -> List[Tuple[int, int, str]]:
        """Create optimized chunks from file content.
        
        Returns:
            List of (start_line, end_line, chunk_text) tuples
        """
        lines = content.split('\n')
        if not lines:
            return []
            
        chunk_size = self.calculate_optimal_chunk_size(file_path, content)
        chunks = []
        
        # First pass: Create basic chunks
        for i in range(0, len(lines), chunk_size):
            start = i
            end = min(i + chunk_size, len(lines))
            chunk_lines = lines[start:end]
            chunk_text = '\n'.join(chunk_lines)
            
            # Check token count
            tokens = self.count_tokens(chunk_text)
            
            # If too many tokens, split further
            if tokens > self.strategy.max_tokens and len(chunk_lines) > 10:
                # Binary search for optimal split
                left, right = 0, len(chunk_lines)
                while left < right - 1:
                    mid = (left + right) // 2
                    mid_text = '\n'.join(chunk_lines[:mid])
                    if self.count_tokens(mid_text) <= self.strategy.target_tokens:
                        left = mid
                    else:
                        right = mid
                        
                # Create two chunks
                chunk1 = '\n'.join(chunk_lines[:left])
                chunks.append((start + 1, start + left, chunk1))
                
                chunk2 = '\n'.join(chunk_lines[left:])
                chunks.append((start + left + 1, end, chunk2))
            else:
                chunks.append((start + 1, end, chunk_text))
                
        return chunks
        
    def create_semantic_chunks(self, file_path: str, content: str) -> List[Tuple[int, int, str]]:
        """Create chunks that respect code structure (functions, classes)."""
        lines = content.split('\n')
        if not lines:
            return []
            
        chunks = []
        current_chunk = []
        current_start = 1
        current_tokens = 0
        indent_stack = [0]
        
        for i, line in enumerate(lines):
            # Calculate indentation
            indent = len(line) - len(line.lstrip())
            
            # Check for structure boundaries
            is_boundary = (
                line.strip().startswith(('def ', 'class ', 'async def ')) or
                (indent == 0 and line.strip() and current_chunk)  # Top-level code
            )
            
            # Estimate tokens for this line
            line_tokens = self.count_tokens(line)
            
            # Decide whether to start new chunk
            should_split = (
                is_boundary and current_tokens > self.strategy.target_tokens * 0.5 or
                current_tokens + line_tokens > self.strategy.max_tokens or
                len(current_chunk) > self.strategy.max_chunk_size
            )
            
            if should_split and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append((current_start, i, chunk_text))
                
                # Start new chunk
                current_chunk = [line]
                current_start = i + 1
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
                
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append((current_start, len(lines), chunk_text))
            
        return chunks
        
    def record_performance(self, file_path: str, chunk_size: int, 
                         token_count: int, processing_time: float, 
                         success: bool = True):
        """Record chunk processing performance for optimization."""
        # Estimate complexity from the file
        complexity = 1.0  # Default, would be calculated from actual content
        
        metric = ChunkMetrics(
            file_path=file_path,
            chunk_size=chunk_size,
            token_count=token_count,
            processing_time=processing_time,
            success=success,
            complexity_score=complexity
        )
        
        self.performance_history.append(metric)
        
        # Keep history bounded
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_history:
            return {}
            
        successful = [m for m in self.performance_history if m.success]
        if not successful:
            return {"error": "No successful chunks"}
            
        avg_chunk_size = sum(m.chunk_size for m in successful) / len(successful)
        avg_tokens = sum(m.token_count for m in successful) / len(successful)
        avg_time = sum(m.processing_time for m in successful) / len(successful)
        tokens_per_second = sum(m.token_count / m.processing_time for m in successful) / len(successful)
        
        return {
            "total_chunks": len(self.performance_history),
            "successful_chunks": len(successful),
            "average_chunk_size": int(avg_chunk_size),
            "average_tokens": int(avg_tokens),
            "average_processing_time": round(avg_time, 3),
            "tokens_per_second": round(tokens_per_second, 1),
            "cache_size": len(self._chunk_size_cache)
        }


# Integration with MCP servers
class ChunkedFileReader:
    """File reader with dynamic chunking for MCP servers."""
    
    def __init__(self):
        self.chunker = DynamicChunker()
        
    def read_file_chunked(self, file_path: str, semantic: bool = True) -> List[Dict[str, Any]]:
        """Read file in optimized chunks.
        
        Args:
            file_path: Path to file
            semantic: Use semantic chunking (respects code structure)
            
        Returns:
            List of chunk dictionaries with metadata
        """
        path = Path(file_path)
        if not path.exists():
            return []
            
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
        except (ValueError, KeyError, AttributeError) as e:
            return [{"error": str(e)}]
            
        # Choose chunking method
        if semantic and path.suffix == '.py':
            chunks = self.chunker.create_semantic_chunks(str(path), content)
        else:
            chunks = self.chunker.create_chunks(str(path), content)
            
        # Format results
        results = []
        for i, (start, end, text) in enumerate(chunks):
            tokens = self.chunker.count_tokens(text)
            results.append({
                "chunk_id": i + 1,
                "total_chunks": len(chunks),
                "start_line": start,
                "end_line": end,
                "lines": end - start + 1,
                "tokens": tokens,
                "content": text,
                "file_path": str(path)
            })
            
        return results
        
    def read_with_ripgrep(self, pattern: str, file_path: str, 
                         context_lines: int = 3) -> List[Dict[str, Any]]:
        """Use ripgrep for initial search, then chunk results."""
        import subprocess
        
        try:
            # Run ripgrep with context
            cmd = [
                'rg',
                '--json',
                f'--context={context_lines}',
                '--no-heading',
                pattern,
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return []
                
            # Parse ripgrep JSON output
            matches = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    match_data = json.loads(line)
                    if match_data.get('type') == 'match':
                        matches.append(match_data)
                        
            # Group matches and create focused chunks
            if not matches:
                return []
                
            # Read file content
            content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            # Create chunks around matches
            chunks = []
            covered_lines = set()
            
            for match in matches:
                line_num = match['data']['line_number']
                start = max(1, line_num - context_lines * 2)
                end = min(len(lines), line_num + context_lines * 2)
                
                # Skip if already covered
                if any(l in covered_lines for l in range(start, end + 1)):
                    continue
                    
                # Mark lines as covered
                covered_lines.update(range(start, end + 1))
                
                # Create chunk
                chunk_lines = lines[start-1:end]
                chunk_text = '\n'.join(chunk_lines)
                tokens = self.chunker.count_tokens(chunk_text)
                
                chunks.append({
                    "match_line": line_num,
                    "start_line": start,
                    "end_line": end,
                    "lines": end - start + 1,
                    "tokens": tokens,
                    "content": chunk_text,
                    "file_path": file_path,
                    "pattern": pattern
                })
                
            return chunks
            
        except (ValueError, KeyError, AttributeError) as e:
            return [{"error": str(e)}]


# Example usage for MCP integration
if __name__ == "__main__":
    
    # Test chunking
    reader = ChunkedFileReader()
    
    # Test on this file
    chunks = reader.read_file_chunked(__file__, semantic=True)
    
    logger.info("Created {len(chunks)} chunks")
    for chunk in chunks:
        print(f"Chunk {chunk['chunk_id']}/{chunk['total_chunks']}: "
              f"Lines {chunk['start_line']}-{chunk['end_line']} "
              f"({chunk['tokens']} tokens)")
        
    # Show performance stats
    logger.info("\nPerformance Stats:")
    print(json.dumps(reader.chunker.get_performance_stats(), indent=2))