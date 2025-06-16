#!/usr/bin/env python3
"""
Unified Result Format for Einstein Search System

This module provides a standardized result format that ensures consistency
across all Einstein search methods and compatibility with both Einstein 
and unified CLI systems.

Key Features:
- Single, consistent result format for all search types
- Backward compatibility with existing MergedResult and SearchResult
- Automatic format conversion and validation
- Optimized serialization for CLI communication
- Type-safe interfaces with proper error handling
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Protocol
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ResultFormat(Enum):
    """Supported result format types."""
    SEARCH_RESULT = "search_result"  # Basic SearchResult format
    MERGED_RESULT = "merged_result"  # Merged across modalities
    UNIFIED_RESULT = "unified_result"  # New standardized format


@dataclass
class UnifiedSearchResult:
    """
    Standardized result format for all Einstein search operations.
    
    This format combines the best aspects of SearchResult, MergedResult, and
    FastMergedResult while maintaining compatibility with the unified CLI.
    """
    # Core result data (required)
    content: str
    file_path: str 
    line_number: int
    score: float  # Combined/final score for ranking
    
    # Result metadata (required)
    result_type: str  # text, semantic, structural, analytical
    timestamp: float = field(default_factory=time.time)
    
    # Enhanced scoring (optional - for merged results)
    modality_scores: Dict[str, float] = field(default_factory=dict)
    source_modalities: List[str] = field(default_factory=list)
    combined_score: Optional[float] = None  # Alias for score for backward compatibility
    
    # Context and metadata (optional)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Format identification
    format_type: ResultFormat = ResultFormat.UNIFIED_RESULT
    format_version: str = "1.0"

    def __post_init__(self):
        """Post-initialization validation and normalization."""
        # Ensure combined_score matches score for compatibility
        if self.combined_score is None:
            self.combined_score = self.score
        elif abs(self.combined_score - self.score) > 0.001:
            # If they differ, prefer combined_score
            self.score = self.combined_score
            
        # Ensure at least one modality is specified
        if not self.source_modalities and self.result_type:
            self.source_modalities = [self.result_type]
            
        # Ensure modality scores include the primary result type
        if self.result_type and self.result_type not in self.modality_scores:
            self.modality_scores[self.result_type] = self.score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for CLI compatibility."""
        return asdict(self)
    
    def to_legacy_search_result(self) -> Dict[str, Any]:
        """Convert to legacy SearchResult-compatible format."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "score": self.score,
            "result_type": self.result_type,
            "context": self.context,
            "timestamp": self.timestamp
        }
    
    def to_legacy_merged_result(self) -> Dict[str, Any]:
        """Convert to legacy MergedResult-compatible format."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "combined_score": self.combined_score,
            "modality_scores": self.modality_scores,
            "source_modalities": self.source_modalities,
            "context": self.context,
            "timestamp": self.timestamp
        }
    
    @classmethod 
    def from_search_result(cls, result: Any) -> "UnifiedSearchResult":
        """Create UnifiedSearchResult from legacy SearchResult."""
        if hasattr(result, '__dict__'):
            # Handle dataclass or object with attributes
            return cls(
                content=getattr(result, 'content', ''),
                file_path=getattr(result, 'file_path', ''),
                line_number=getattr(result, 'line_number', 0),
                score=getattr(result, 'score', 0.0),
                result_type=getattr(result, 'result_type', 'unknown'),
                context=getattr(result, 'context', {}),
                timestamp=getattr(result, 'timestamp', time.time()),
                format_type=ResultFormat.SEARCH_RESULT
            )
        elif isinstance(result, dict):
            # Handle dictionary format
            return cls(
                content=result.get('content', ''),
                file_path=result.get('file_path', result.get('file', '')),
                line_number=result.get('line_number', result.get('line', 0)),
                score=result.get('score', 0.0),
                result_type=result.get('result_type', 'unknown'),
                context=result.get('context', {}),
                timestamp=result.get('timestamp', time.time()),
                format_type=ResultFormat.SEARCH_RESULT
            )
        else:
            raise ValueError(f"Cannot convert {type(result)} to UnifiedSearchResult")
    
    @classmethod
    def from_merged_result(cls, result: Any) -> "UnifiedSearchResult":
        """Create UnifiedSearchResult from legacy MergedResult."""
        if hasattr(result, '__dict__'):
            # Handle dataclass or object with attributes
            return cls(
                content=getattr(result, 'content', ''),
                file_path=getattr(result, 'file_path', ''),
                line_number=getattr(result, 'line_number', 0),
                score=getattr(result, 'combined_score', getattr(result, 'score', 0.0)),
                result_type=getattr(result, 'source_modalities', ['unknown'])[0] if getattr(result, 'source_modalities', []) else 'unknown',
                modality_scores=getattr(result, 'modality_scores', {}),
                source_modalities=getattr(result, 'source_modalities', []),
                combined_score=getattr(result, 'combined_score', getattr(result, 'score', 0.0)),
                context=getattr(result, 'context', {}),
                timestamp=getattr(result, 'timestamp', time.time()),
                format_type=ResultFormat.MERGED_RESULT
            )
        elif isinstance(result, dict):
            # Handle dictionary format
            combined_score = result.get('combined_score', result.get('score', 0.0))
            source_modalities = result.get('source_modalities', [])
            return cls(
                content=result.get('content', ''),
                file_path=result.get('file_path', result.get('file', '')),
                line_number=result.get('line_number', result.get('line', 0)),
                score=combined_score,
                result_type=source_modalities[0] if source_modalities else 'unknown',
                modality_scores=result.get('modality_scores', {}),
                source_modalities=source_modalities,
                combined_score=combined_score,
                context=result.get('context', {}),
                timestamp=result.get('timestamp', time.time()),
                format_type=ResultFormat.MERGED_RESULT
            )
        else:
            raise ValueError(f"Cannot convert {type(result)} to UnifiedSearchResult")


@dataclass 
class UnifiedSearchResponse:
    """
    Standardized response format for Einstein search operations.
    
    This provides a consistent wrapper around search results that works
    with both Einstein internal operations and CLI communication.
    """
    # Core response data
    query: str
    results: List[UnifiedSearchResult]
    
    # Response metadata
    total_results: int = 0
    search_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    # Search details
    search_modalities: List[str] = field(default_factory=list)
    routing_info: Dict[str, Any] = field(default_factory=dict)
    
    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)
    
    # Format identification
    format_type: str = "unified_search_response"
    format_version: str = "1.0"
    
    def __post_init__(self):
        """Post-initialization validation and summary generation."""
        if self.total_results == 0:
            self.total_results = len(self.results)
            
        # Generate summary if not provided
        if not self.summary and self.results:
            self.summary = self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if not self.results:
            return {
                "unique_files": 0,
                "top_score": 0.0,
                "multi_modality_results": 0,
                "modality_distribution": {}
            }
        
        # Count unique files
        unique_files = len(set(r.file_path for r in self.results))
        
        # Get top score
        top_score = max(r.score for r in self.results) if self.results else 0.0
        
        # Count multi-modality results
        multi_modality_results = sum(
            1 for r in self.results if len(r.source_modalities) > 1
        )
        
        # Modality distribution
        modality_counts = {}
        for result in self.results:
            for modality in result.source_modalities:
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        return {
            "unique_files": unique_files,
            "top_score": top_score,
            "multi_modality_results": multi_modality_results,
            "modality_distribution": modality_counts
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for CLI compatibility."""
        result_dict = asdict(self)
        # Convert results to dict format for better CLI compatibility
        result_dict["results"] = [r.to_dict() for r in self.results]
        return result_dict
    
    def to_cli_format(self) -> Dict[str, Any]:
        """Convert to format expected by unified CLI."""
        return {
            "query": self.query,
            "results": self.results,  # CLI handles both objects and dicts
            "summary": self.summary,
            "search_time_ms": self.search_time_ms,
            "total_results": self.total_results,
            "routing": self.routing_info,
            "system": "einstein",
            "format": "unified"
        }


class ResultConverter:
    """
    Utility class for converting between different result formats.
    
    Provides safe conversion methods with error handling and validation.
    """
    
    @staticmethod
    def to_unified_results(results: List[Any]) -> List[UnifiedSearchResult]:
        """Convert list of mixed result formats to UnifiedSearchResult."""
        unified_results = []
        
        for result in results:
            try:
                if isinstance(result, UnifiedSearchResult):
                    unified_results.append(result)
                elif hasattr(result, 'combined_score') or (
                    isinstance(result, dict) and 'combined_score' in result
                ):
                    # Looks like MergedResult or FastMergedResult
                    unified_results.append(UnifiedSearchResult.from_merged_result(result))
                else:
                    # Assume SearchResult format
                    unified_results.append(UnifiedSearchResult.from_search_result(result))
                    
            except Exception as e:
                logger.warning(f"Failed to convert result to unified format: {e}")
                # Create a fallback result
                unified_results.append(UnifiedSearchResult(
                    content=str(result),
                    file_path="unknown",
                    line_number=0,
                    score=0.0,
                    result_type="conversion_error",
                    context={"conversion_error": str(e), "original_type": str(type(result))}
                ))
        
        return unified_results
    
    @staticmethod
    def create_unified_response(
        query: str,
        results: List[Any],
        search_time_ms: float = 0.0,
        modalities: List[str] = None,
        routing_info: Dict[str, Any] = None
    ) -> UnifiedSearchResponse:
        """Create a unified response from mixed result formats."""
        
        unified_results = ResultConverter.to_unified_results(results)
        
        return UnifiedSearchResponse(
            query=query,
            results=unified_results,
            search_time_ms=search_time_ms,
            search_modalities=modalities or [],
            routing_info=routing_info or {}
        )


# Compatibility aliases for backward compatibility
SearchResult = UnifiedSearchResult  # Alias for legacy code
MergedResult = UnifiedSearchResult  # Alias for legacy code

# Type hints for better IDE support
ResultList = List[UnifiedSearchResult]
SearchResponse = UnifiedSearchResponse