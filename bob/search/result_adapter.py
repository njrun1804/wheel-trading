#!/usr/bin/env python3
"""
Result Standardization Adapter for Einstein Search System

This adapter ensures all Einstein search methods return consistent UnifiedSearchResult
formats while maintaining backward compatibility with existing code.

Key Features:
- Wraps existing search methods to standardize return formats
- Automatic format detection and conversion
- Performance optimization with minimal overhead
- Error handling and recovery for malformed results
- Logging and metrics for format conversion tracking
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from functools import wraps
import inspect

from .unified_result_format import (
    UnifiedSearchResult, 
    UnifiedSearchResponse, 
    ResultConverter,
    ResultFormat
)

logger = logging.getLogger(__name__)


class SearchMethodAdapter:
    """
    Adapter that wraps Einstein search methods to ensure consistent result formats.
    """
    
    def __init__(self):
        self.conversion_stats = {
            "total_conversions": 0,
            "conversion_errors": 0,
            "format_distribution": {},
            "performance_metrics": []
        }
    
    def standardize_search_method(self, method: Callable) -> Callable:
        """
        Decorator that standardizes the return format of search methods.
        
        Works with both sync and async methods, converting results to UnifiedSearchResult.
        """
        
        @wraps(method)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Call original method
                result = method(*args, **kwargs)
                
                # Convert result to unified format
                standardized_result = self._standardize_result(result, method.__name__)
                
                # Record metrics
                conversion_time = (time.time() - start_time) * 1000
                self._record_conversion_metrics(method.__name__, conversion_time, True)
                
                return standardized_result
                
            except Exception as e:
                logger.error(f"Error in standardized search method {method.__name__}: {e}")
                self._record_conversion_metrics(method.__name__, 0, False)
                
                # Return empty result on error
                return []
        
        @wraps(method)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Call original async method
                result = await method(*args, **kwargs)
                
                # Convert result to unified format
                standardized_result = self._standardize_result(result, method.__name__)
                
                # Record metrics
                conversion_time = (time.time() - start_time) * 1000
                self._record_conversion_metrics(method.__name__, conversion_time, True)
                
                return standardized_result
                
            except Exception as e:
                logger.error(f"Error in standardized async search method {method.__name__}: {e}")
                self._record_conversion_metrics(method.__name__, 0, False)
                
                # Return empty result on error
                return []
        
        # Return appropriate wrapper based on whether method is async
        if inspect.iscoroutinefunction(method):
            return async_wrapper
        else:
            return sync_wrapper
    
    def _standardize_result(self, result: Any, method_name: str) -> List[UnifiedSearchResult]:
        """Convert any result format to List[UnifiedSearchResult]."""
        
        self.conversion_stats["total_conversions"] += 1
        
        try:
            # Handle different result types
            if isinstance(result, tuple):
                # Handle (results, metrics) tuples from OptimizedUnifiedSearch
                if len(result) == 2:
                    actual_results, metrics = result
                    return self._convert_results_list(actual_results, method_name)
                else:
                    logger.warning(f"Unexpected tuple format from {method_name}: {len(result)} elements")
                    return []
            
            elif isinstance(result, list):
                # Handle list of results
                return self._convert_results_list(result, method_name)
            
            elif isinstance(result, dict):
                # Handle dictionary response format
                if 'results' in result:
                    return self._convert_results_list(result['results'], method_name)
                else:
                    logger.warning(f"Dictionary result from {method_name} missing 'results' key")
                    return []
            
            elif result is None:
                return []
            
            else:
                # Single result object
                return self._convert_results_list([result], method_name)
                
        except Exception as e:
            logger.error(f"Failed to standardize result from {method_name}: {e}")
            self.conversion_stats["conversion_errors"] += 1
            return []
    
    def _convert_results_list(self, results: List[Any], method_name: str) -> List[UnifiedSearchResult]:
        """Convert a list of results to UnifiedSearchResult format."""
        
        if not results:
            return []
        
        unified_results = []
        
        for i, result in enumerate(results):
            try:
                if isinstance(result, UnifiedSearchResult):
                    unified_results.append(result)
                    self._track_format_type(ResultFormat.UNIFIED_RESULT)
                    
                elif hasattr(result, 'combined_score') or (
                    isinstance(result, dict) and 'combined_score' in result
                ):
                    # MergedResult or FastMergedResult format
                    unified_result = UnifiedSearchResult.from_merged_result(result)
                    unified_results.append(unified_result)
                    self._track_format_type(ResultFormat.MERGED_RESULT)
                    
                elif hasattr(result, 'score') or (
                    isinstance(result, dict) and 'score' in result
                ):
                    # SearchResult format
                    unified_result = UnifiedSearchResult.from_search_result(result)
                    unified_results.append(unified_result)
                    self._track_format_type(ResultFormat.SEARCH_RESULT)
                    
                else:
                    # Unknown format - create best-effort conversion
                    logger.warning(f"Unknown result format in {method_name}[{i}]: {type(result)}")
                    unified_result = self._create_fallback_result(result, method_name)
                    unified_results.append(unified_result)
                    self._track_format_type("unknown")
                    
            except Exception as e:
                logger.error(f"Failed to convert result {i} from {method_name}: {e}")
                # Create error result
                error_result = UnifiedSearchResult(
                    content=f"Conversion error: {str(e)}",
                    file_path="unknown",
                    line_number=0,
                    score=0.0,
                    result_type="conversion_error",
                    context={
                        "conversion_error": str(e),
                        "original_type": str(type(result)),
                        "method_name": method_name,
                        "result_index": i
                    }
                )
                unified_results.append(error_result)
        
        return unified_results
    
    def _create_fallback_result(self, result: Any, method_name: str) -> UnifiedSearchResult:
        """Create a fallback UnifiedSearchResult from unknown format."""
        
        # Try to extract common fields
        content = ""
        file_path = "unknown"
        line_number = 0
        score = 0.0
        result_type = "fallback"
        
        if hasattr(result, '__dict__'):
            # Extract from object attributes
            content = getattr(result, 'content', getattr(result, 'text', str(result)))
            file_path = getattr(result, 'file_path', getattr(result, 'file', getattr(result, 'path', 'unknown')))
            line_number = getattr(result, 'line_number', getattr(result, 'line', 0))
            score = getattr(result, 'score', getattr(result, 'relevance', 0.0))
            result_type = getattr(result, 'result_type', getattr(result, 'type', 'fallback'))
            
        elif isinstance(result, dict):
            # Extract from dictionary
            content = result.get('content', result.get('text', str(result)))
            file_path = result.get('file_path', result.get('file', result.get('path', 'unknown')))
            line_number = result.get('line_number', result.get('line', 0))
            score = result.get('score', result.get('relevance', 0.0))
            result_type = result.get('result_type', result.get('type', 'fallback'))
        
        return UnifiedSearchResult(
            content=content,
            file_path=file_path,
            line_number=line_number,
            score=score,
            result_type=result_type,
            context={
                "fallback_conversion": True,
                "original_type": str(type(result)),
                "method_name": method_name
            }
        )
    
    def _track_format_type(self, format_type: Union[ResultFormat, str]):
        """Track the distribution of result formats for monitoring."""
        format_name = format_type.value if isinstance(format_type, ResultFormat) else str(format_type)
        self.conversion_stats["format_distribution"][format_name] = (
            self.conversion_stats["format_distribution"].get(format_name, 0) + 1
        )
    
    def _record_conversion_metrics(self, method_name: str, conversion_time_ms: float, success: bool):
        """Record performance metrics for conversions."""
        self.conversion_stats["performance_metrics"].append({
            "method": method_name,
            "conversion_time_ms": conversion_time_ms,
            "success": success,
            "timestamp": time.time()
        })
        
        # Keep only last 1000 metrics
        if len(self.conversion_stats["performance_metrics"]) > 1000:
            self.conversion_stats["performance_metrics"] = (
                self.conversion_stats["performance_metrics"][-1000:]
            )
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get statistics about format conversions."""
        return self.conversion_stats.copy()


class UnifiedSearchInterface:
    """
    Unified interface that wraps Einstein search components to provide consistent results.
    
    This class acts as a facade over existing Einstein search methods, ensuring all
    methods return UnifiedSearchResponse format.
    """
    
    def __init__(self, search_hub, result_merger=None, query_router=None):
        self.search_hub = search_hub
        self.result_merger = result_merger
        self.query_router = query_router
        self.adapter = SearchMethodAdapter()
        
        # Wrap existing search methods with standardization
        self._wrap_search_methods()
    
    def _wrap_search_methods(self):
        """Wrap existing search methods to ensure consistent returns."""
        
        if hasattr(self.search_hub, 'search'):
            original_search = self.search_hub.search
            self.search_hub.search = self.adapter.standardize_search_method(original_search)
        
        if hasattr(self.search_hub, '_text_search'):
            original_text_search = self.search_hub._text_search
            self.search_hub._text_search = self.adapter.standardize_search_method(original_text_search)
        
        if hasattr(self.search_hub, '_semantic_search'):
            original_semantic_search = self.search_hub._semantic_search
            self.search_hub._semantic_search = self.adapter.standardize_search_method(original_semantic_search)
        
        if hasattr(self.search_hub, '_structural_search'):
            original_structural_search = self.search_hub._structural_search
            self.search_hub._structural_search = self.adapter.standardize_search_method(original_structural_search)
        
        if hasattr(self.search_hub, '_analytical_search'):
            original_analytical_search = self.search_hub._analytical_search
            self.search_hub._analytical_search = self.adapter.standardize_search_method(original_analytical_search)
    
    async def unified_search(self, query: str, **kwargs) -> UnifiedSearchResponse:
        """
        Perform search using standardized interface.
        
        Returns UnifiedSearchResponse that works with both Einstein and unified CLI.
        """
        start_time = time.time()
        
        try:
            # Determine search modalities
            modalities = kwargs.get('search_modalities', ['text', 'semantic', 'structural'])
            
            # Execute search through hub (now standardized)
            if hasattr(self.search_hub, 'search'):
                # OptimizedUnifiedSearch style
                results = await self.search_hub.search(query, modalities)
            else:
                # Individual search methods
                results = []
                for modality in modalities:
                    method_name = f"_{modality}_search"
                    if hasattr(self.search_hub, method_name):
                        method = getattr(self.search_hub, method_name)
                        modality_results = await method(query)
                        results.extend(modality_results)
            
            # Results are now guaranteed to be List[UnifiedSearchResult]
            search_time_ms = (time.time() - start_time) * 1000
            
            # Create unified response
            response = UnifiedSearchResponse(
                query=query,
                results=results,
                search_time_ms=search_time_ms,
                search_modalities=modalities,
                routing_info=kwargs.get('routing_info', {})
            )
            
            logger.info(f"Unified search completed: {len(results)} results in {search_time_ms:.1f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Unified search failed for query '{query}': {e}")
            # Return empty response on error
            return UnifiedSearchResponse(
                query=query,
                results=[],
                search_time_ms=(time.time() - start_time) * 1000,
                search_modalities=[],
                routing_info={"error": str(e)}
            )
    
    def get_adapter_stats(self) -> Dict[str, Any]:
        """Get statistics about result format adaptations."""
        return self.adapter.get_conversion_stats()


# Global adapter instance for singleton pattern
_global_adapter = None

def get_standardization_adapter() -> SearchMethodAdapter:
    """Get global standardization adapter instance."""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = SearchMethodAdapter()
    return _global_adapter


def standardize_search_results(results: Any, method_name: str = "unknown") -> List[UnifiedSearchResult]:
    """
    Utility function to standardize any search results to unified format.
    
    Can be used standalone for one-off conversions.
    """
    adapter = get_standardization_adapter()
    return adapter._standardize_result(results, method_name)


def ensure_unified_response(response: Any, query: str = "") -> UnifiedSearchResponse:
    """
    Ensure any response is in UnifiedSearchResponse format.
    
    Handles conversion from dictionary responses, legacy formats, etc.
    """
    if isinstance(response, UnifiedSearchResponse):
        return response
    
    elif isinstance(response, dict):
        # Convert dictionary response
        results = response.get('results', [])
        unified_results = standardize_search_results(results, "dict_response")
        
        return UnifiedSearchResponse(
            query=response.get('query', query),
            results=unified_results,
            search_time_ms=response.get('search_time_ms', 0.0),
            search_modalities=response.get('search_modalities', []),
            routing_info=response.get('routing', response.get('routing_info', {})),
            summary=response.get('summary', {})
        )
    
    else:
        # Try to convert as results list
        unified_results = standardize_search_results(response, "unknown_response")
        
        return UnifiedSearchResponse(
            query=query,
            results=unified_results,
            search_time_ms=0.0,
            search_modalities=[],
            routing_info={}
        )