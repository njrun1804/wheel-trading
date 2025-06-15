#!/usr/bin/env python3
"""
Einstein Result Merger

Intelligently merges and ranks results from multiple search modalities:
- Deduplicates across modalities
- Ranks by relevance and confidence
- Provides unified scoring
- Maintains result provenance
"""

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .unified_index import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class MergedResult:
    """Result merged from multiple search modalities."""
    content: str
    file_path: str
    line_number: int
    combined_score: float
    modality_scores: dict[str, float]  # Score from each modality
    source_modalities: list[str]       # Which modalities found this
    context: dict[str, Any]
    timestamp: float
    relevance_rank: int = 0


class ResultMerger:
    """Merges results from multiple search modalities intelligently."""
    
    def __init__(self):
        # Weights for different modalities
        self.modality_weights = {
            'text': 1.0,        # Exact matches are highly relevant
            'semantic': 0.8,    # Semantic similarity is good
            'structural': 0.9,  # Structural matches are precise
            'analytical': 0.7   # Analytics provide context
        }
        
        # Boost for multi-modality results
        self.multi_modality_boost = 0.2
        
    def merge_results(self, results_by_modality: dict[str, list[SearchResult]]) -> list[MergedResult]:
        """Merge results from multiple modalities with intelligent ranking."""
        
        try:
            if not results_by_modality:
                logger.debug("No results to merge")
                return []
            
            total_results = sum(len(results) for results in results_by_modality.values())
            logger.debug(f"Merging {total_results} results from {len(results_by_modality)} modalities")
            
            # Group results by file + line for deduplication
            result_groups = self._group_by_location(results_by_modality)
            
            # Create merged results
            merged_results = []
            for location, modality_results in result_groups.items():
                try:
                    merged = self._merge_location_group(location, modality_results)
                    merged_results.append(merged)
                except Exception as e:
                    logger.warning(f"Failed to merge results for location {location}: {e}",
                                  extra={
                                      'operation': 'merge_location_group',
                                      'error_type': type(e).__name__,
                                      'location': str(location),
                                      'modality_count': len(modality_results),
                                      'modalities': list(modality_results.keys())
                                  })
                    continue
            
            # Rank by combined score
            merged_results.sort(key=lambda r: r.combined_score, reverse=True)
            
            # Assign relevance ranks
            for i, result in enumerate(merged_results):
                result.relevance_rank = i + 1
            
            logger.debug(f"Successfully merged into {len(merged_results)} unique results")
            return merged_results
            
        except Exception as e:
            logger.error(f"Failed to merge results: {e}", exc_info=True,
                        extra={
                            'operation': 'merge_results',
                            'error_type': type(e).__name__,
                            'modality_count': len(results_by_modality) if results_by_modality else 0,
                            'total_input_results': sum(len(results) for results in results_by_modality.values()) if results_by_modality else 0
                        })
            return []
    
    def _group_by_location(self, results_by_modality: dict[str, list[SearchResult]]) -> dict[tuple[str, int], dict[str, SearchResult]]:
        """Group results by file path and line number."""
        
        location_groups = defaultdict(dict)
        
        for modality, results in results_by_modality.items():
            for result in results:
                location_key = (result.file_path, result.line_number)
                location_groups[location_key][modality] = result
        
        return dict(location_groups)
    
    def _merge_location_group(self, location: tuple[str, int], modality_results: dict[str, SearchResult]) -> MergedResult:
        """Merge results from the same location across modalities."""
        
        try:
            file_path, line_number = location
            
            # Combine content (prefer longest/most detailed)
            content = max((r.content for r in modality_results.values()), 
                         key=len, default="")
            
            # Calculate combined score
            modality_scores = {}
            weighted_scores = []
            
            for modality, result in modality_results.items():
                if not isinstance(result.score, (int, float)) or result.score < 0:
                    logger.warning(f"Invalid score {result.score} for modality {modality}, using 0.0")
                    score = 0.0
                else:
                    score = min(result.score, 1.0)  # Cap at 1.0
                
                modality_scores[modality] = score
                weight = self.modality_weights.get(modality, 1.0)
                weighted_scores.append(score * weight)
            
            # Base score is weighted average
            if weighted_scores:
                base_score = sum(weighted_scores) / len(weighted_scores)
            else:
                base_score = 0.0
            
            # Apply multi-modality boost
            if len(modality_results) > 1:
                base_score += self.multi_modality_boost
            
            # Ensure score doesn't exceed 1.0
            combined_score = min(base_score, 1.0)
            
            # Merge contexts
            merged_context = {}
            for result in modality_results.values():
                if hasattr(result, 'context') and isinstance(result.context, dict):
                    merged_context.update(result.context)
            
            # Add merger metadata
            merged_context['merger_info'] = {
                'modality_count': len(modality_results),
                'source_modalities': list(modality_results.keys()),
                'multi_modality_boost': len(modality_results) > 1
            }
            
            # Get timestamp with fallback
            try:
                timestamp = max(r.timestamp for r in modality_results.values() if hasattr(r, 'timestamp'))
            except (ValueError, AttributeError):
                import time
                timestamp = time.time()
            
            return MergedResult(
                content=content,
                file_path=file_path,
                line_number=line_number,
                combined_score=combined_score,
                modality_scores=modality_scores,
                source_modalities=list(modality_results.keys()),
                context=merged_context,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"Failed to merge location group {location}: {e}", exc_info=True,
                        extra={
                            'operation': 'merge_location_group',
                            'error_type': type(e).__name__,
                            'location': str(location),
                            'modality_count': len(modality_results),
                            'modalities': list(modality_results.keys())
                        })
            # Return a fallback result
            import time
            return MergedResult(
                content="",
                file_path=str(location[0]) if len(location) > 0 else "unknown",
                line_number=int(location[1]) if len(location) > 1 else 0,
                combined_score=0.0,
                modality_scores={},
                source_modalities=[],
                context={'error': 'merge_failed'},
                timestamp=time.time()
            )
    
    def deduplicate_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate results based on content similarity."""
        
        seen_hashes = set()
        unique_results = []
        
        for result in results:
            # Create content hash for deduplication
            content_hash = self._content_hash(result)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _content_hash(self, result: SearchResult) -> str:
        """Generate hash for result deduplication."""
        
        # Hash based on file path, line number, and content
        content_str = f"{result.file_path}:{result.line_number}:{result.content}"
        return hashlib.md5(content_str.encode()).hexdigest()[:12]
    
    def boost_relevant_results(self, results: list[MergedResult], query: str) -> list[MergedResult]:
        """Apply additional relevance boosting based on query analysis."""
        
        query_lower = query.lower()
        
        for result in results:
            # Boost for exact query matches in content
            if query_lower in result.content.lower():
                result.combined_score = min(result.combined_score + 0.1, 1.0)
            
            # Boost for query matches in file path
            if query_lower in result.file_path.lower():
                result.combined_score = min(result.combined_score + 0.05, 1.0)
            
            # Boost for recent files (if timestamp is available)
            file_age_days = (result.timestamp - 1640995200) / 86400  # Days since 2022
            if file_age_days < 30:  # Recent modifications
                result.combined_score = min(result.combined_score + 0.03, 1.0)
        
        # Re-sort after boosting
        results.sort(key=lambda r: r.combined_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.relevance_rank = i + 1
        
        return results
    
    def filter_by_confidence(self, results: list[MergedResult], min_confidence: float = 0.3) -> list[MergedResult]:
        """Filter results below minimum confidence threshold."""
        
        return [r for r in results if r.combined_score >= min_confidence]
    
    def get_diversity_subset(self, results: list[MergedResult], max_results: int = 20) -> list[MergedResult]:
        """Get diverse subset of results to avoid redundancy."""
        
        if len(results) <= max_results:
            return results
        
        # Simple diversity: ensure we don't have too many results from same file
        file_counts = defaultdict(int)
        diverse_results = []
        
        for result in results:
            if len(diverse_results) >= max_results:
                break
            
            # Limit results per file to maintain diversity
            file_count = file_counts[result.file_path]
            if file_count < 3:  # Max 3 results per file
                diverse_results.append(result)
                file_counts[result.file_path] += 1
        
        # Fill remaining slots with any results if needed
        if len(diverse_results) < max_results:
            for result in results:
                if len(diverse_results) >= max_results:
                    break
                if result not in diverse_results:
                    diverse_results.append(result)
        
        return diverse_results
    
    def generate_search_summary(self, results: list[MergedResult], query: str) -> dict[str, Any]:
        """Generate summary of search results for reporting."""
        
        if not results:
            return {
                'query': query,
                'total_results': 0,
                'summary': 'No results found'
            }
        
        # Calculate statistics
        modality_coverage = defaultdict(int)
        file_coverage = set()
        
        for result in results:
            for modality in result.source_modalities:
                modality_coverage[modality] += 1
            file_coverage.add(result.file_path)
        
        # Find top scoring result
        top_result = results[0] if results else None
        
        return {
            'query': query,
            'total_results': len(results),
            'unique_files': len(file_coverage),
            'modality_coverage': dict(modality_coverage),
            'top_score': top_result.combined_score if top_result else 0.0,
            'multi_modality_results': len([r for r in results if len(r.source_modalities) > 1]),
            'summary': f"Found {len(results)} results across {len(file_coverage)} files"
        }


if __name__ == "__main__":
    # Test the result merger
    import time

    from unified_index import SearchResult
    
    # Create sample results
    text_results = [
        SearchResult("def calculate_delta", "options.py", 42, 1.0, "text", {}, time.time()),
        SearchResult("delta calculation", "pricing.py", 15, 0.9, "text", {}, time.time())
    ]
    
    structural_results = [
        SearchResult("calculate_delta function", "options.py", 42, 0.95, "structural", {}, time.time()),
        SearchResult("Delta class", "greeks.py", 8, 0.85, "structural", {}, time.time())
    ]
    
    results_by_modality = {
        'text': text_results,
        'structural': structural_results
    }
    
    merger = ResultMerger()
    merged = merger.merge_results(results_by_modality)
    
    print(f"Merged {len(merged)} results:")
    for result in merged:
        print(f"  {result.file_path}:{result.line_number} - Score: {result.combined_score:.2f}")
        print(f"    Modalities: {result.source_modalities}")
        print(f"    Content: {result.content[:50]}...")
        print()
