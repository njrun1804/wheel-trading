"""
Einstein Fallback Manager

Provides fallback mechanisms when primary Einstein tools fail, ensuring
graceful degradation and continued operation with alternative methods.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict, Optional, Union


class FallbackStrategy(Enum):
    """Available fallback strategies."""
    TEXT_SEARCH = "text_search"
    CACHED_RESULTS = "cached_results"
    SIMPLIFIED_INDEX = "simplified_index"
    BASIC_EMBEDDINGS = "basic_embeddings"
    FILE_SCANNING = "file_scanning"
    MANUAL_MODE = "manual_mode"


@dataclass
class FallbackResult:
    """Result from a fallback operation."""
    success: bool
    data: Any
    fallback_type: str
    performance_ms: float
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]


class FallbackChain(ABC):
    """Abstract base class for fallback chains."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.fallbacks: List[callable] = []
        self.stats = {
            'total_attempts': 0,
            'successful_fallbacks': 0,
            'failed_fallbacks': 0,
            'average_time': 0.0
        }
    
    def add_fallback(self, fallback_func: callable, priority: int = 10):
        """Add a fallback function to the chain."""
        self.fallbacks.append((priority, fallback_func))
        self.fallbacks.sort(key=lambda x: x[0])  # Sort by priority
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> FallbackResult:
        """Execute the fallback chain."""
        pass
    
    async def _try_fallback(
        self,
        fallback_func: callable,
        *args,
        **kwargs
    ) -> FallbackResult:
        """Try a specific fallback function."""
        start_time = time.time()
        
        try:
            result = await fallback_func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            
            return FallbackResult(
                success=True,
                data=result,
                fallback_type=fallback_func.__name__,
                performance_ms=duration,
                confidence=0.8,  # Default confidence
                metadata={'method': 'fallback'}
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.logger.warning(f"Fallback {fallback_func.__name__} failed: {e}")
            
            return FallbackResult(
                success=False,
                data=None,
                fallback_type=fallback_func.__name__,
                performance_ms=duration,
                confidence=0.0,
                metadata={'error': str(e)}
            )


class SearchFallbackChain(FallbackChain):
    """Fallback chain for search operations."""
    
    def __init__(self, project_root: Path):
        super().__init__("SearchFallback")
        self.project_root = project_root
        self._setup_fallbacks()
    
    def _setup_fallbacks(self):
        """Setup the search fallback chain."""
        # Priority order: higher number = higher priority
        self.add_fallback(self._ripgrep_search, priority=90)
        self.add_fallback(self._grep_search, priority=80)
        self.add_fallback(self._python_search, priority=70)
        self.add_fallback(self._file_scanning_search, priority=60)
        self.add_fallback(self._cached_search, priority=50)
    
    async def execute(self, query: str, **kwargs) -> FallbackResult:
        """Execute search fallback chain."""
        self.stats['total_attempts'] += 1
        
        for priority, fallback_func in self.fallbacks:
            try:
                result = await self._try_fallback(fallback_func, query, **kwargs)
                
                if result.success:
                    self.stats['successful_fallbacks'] += 1
                    self.logger.info(f"Search fallback successful: {result.fallback_type}")
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Search fallback {fallback_func.__name__} failed: {e}")
                continue
        
        # All fallbacks failed
        self.stats['failed_fallbacks'] += 1
        return FallbackResult(
            success=False,
            data=[],
            fallback_type="all_failed",
            performance_ms=0.0,
            confidence=0.0,
            metadata={'error': 'All search fallbacks failed'}
        )
    
    async def _ripgrep_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Fallback using ripgrep for text search."""
        try:
            # Use accelerated ripgrep if available
            from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
            rg = get_ripgrep_turbo()
            results = await rg.search_content(query, str(self.project_root))
            
            # Convert to consistent format
            formatted_results = []
            for result in results[:50]:  # Limit results
                formatted_results.append({
                    'content': result.get('content', ''),
                    'file_path': result.get('file_path', ''),
                    'line_number': result.get('line_number', 0),
                    'score': 1.0,  # High confidence for exact matches
                    'result_type': 'text',
                    'context': {'method': 'ripgrep_fallback'},
                    'timestamp': time.time()
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Ripgrep fallback failed: {e}")
            raise
    
    async def _grep_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Fallback using system grep."""
        import subprocess
        
        try:
            # Use system grep as fallback
            cmd = ['grep', '-r', '-n', '-i', query, str(self.project_root)]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            results = []
            for line in process.stdout.split('\n')[:50]:  # Limit results
                if ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path, line_num, content = parts
                        results.append({
                            'content': content.strip(),
                            'file_path': file_path,
                            'line_number': int(line_num) if line_num.isdigit() else 0,
                            'score': 0.8,
                            'result_type': 'text',
                            'context': {'method': 'grep_fallback'},
                            'timestamp': time.time()
                        })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Grep fallback failed: {e}")
            raise
    
    async def _python_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Fallback using Python string search."""
        results = []
        
        try:
            # Simple file-by-file search
            for file_path in self.project_root.rglob("*.py"):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                if query.lower() in line.lower():
                                    results.append({
                                        'content': line.strip(),
                                        'file_path': str(file_path),
                                        'line_number': line_num,
                                        'score': 0.6,
                                        'result_type': 'text',
                                        'context': {'method': 'python_search_fallback'},
                                        'timestamp': time.time()
                                    })
                                    
                                    if len(results) >= 50:  # Limit results
                                        return results
                                        
                    except Exception as file_error:
                        self.logger.debug(f"Error reading {file_path}: {file_error}")
                        continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Python search fallback failed: {e}")
            raise
    
    async def _file_scanning_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Fallback using basic file scanning."""
        results = []
        
        try:
            # Scan files with basic pattern matching
            import re
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            
            for file_path in self.project_root.rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.py', '.txt', '.md']:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if pattern.search(content):
                                # Find line number
                                lines = content.split('\n')
                                for line_num, line in enumerate(lines, 1):
                                    if pattern.search(line):
                                        results.append({
                                            'content': line.strip(),
                                            'file_path': str(file_path),
                                            'line_number': line_num,
                                            'score': 0.4,
                                            'result_type': 'text',
                                            'context': {'method': 'file_scanning_fallback'},
                                            'timestamp': time.time()
                                        })
                                        break
                                        
                    except Exception as file_error:
                        continue
                        
                    if len(results) >= 30:  # Limit results
                        break
            
            return results
            
        except Exception as e:
            self.logger.error(f"File scanning fallback failed: {e}")
            raise
    
    async def _cached_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Fallback using cached search results."""
        # Simple cache implementation
        cache_file = self.project_root / '.einstein' / 'search_cache.json'
        
        try:
            if cache_file.exists():
                import json
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Look for similar queries
                for cached_query, cached_results in cache_data.items():
                    if query.lower() in cached_query.lower() or cached_query.lower() in query.lower():
                        # Return cached results with lower confidence
                        for result in cached_results:
                            result['score'] *= 0.3  # Reduce confidence for cached results
                            result['context'] = {'method': 'cached_fallback'}
                            result['timestamp'] = time.time()
                        
                        return cached_results[:20]  # Limit cached results
            
            return []
            
        except Exception as e:
            self.logger.error(f"Cached search fallback failed: {e}")
            return []


class EmbeddingFallbackChain(FallbackChain):
    """Fallback chain for embedding operations."""
    
    def __init__(self):
        super().__init__("EmbeddingFallback")
        self._setup_fallbacks()
    
    def _setup_fallbacks(self):
        """Setup the embedding fallback chain."""
        self.add_fallback(self._cpu_embeddings, priority=90)
        self.add_fallback(self._simple_embeddings, priority=80)
        self.add_fallback(self._keyword_matching, priority=70)
        self.add_fallback(self._text_similarity, priority=60)
    
    async def execute(self, text: str, **kwargs) -> FallbackResult:
        """Execute embedding fallback chain."""
        self.stats['total_attempts'] += 1
        
        for priority, fallback_func in self.fallbacks:
            try:
                result = await self._try_fallback(fallback_func, text, **kwargs)
                
                if result.success:
                    self.stats['successful_fallbacks'] += 1
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Embedding fallback {fallback_func.__name__} failed: {e}")
                continue
        
        # All fallbacks failed
        self.stats['failed_fallbacks'] += 1
        return FallbackResult(
            success=False,
            data=None,
            fallback_type="all_failed",
            performance_ms=0.0,
            confidence=0.0,
            metadata={'error': 'All embedding fallbacks failed'}
        )
    
    async def _cpu_embeddings(self, text: str, **kwargs) -> List[float]:
        """CPU-based embedding fallback."""
        try:
            # Use basic sentence-transformers on CPU
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Force CPU usage
            device = 'cpu'
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            embedding = model.encode(text, convert_to_tensor=False)
            
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"CPU embeddings fallback failed: {e}")
            raise
    
    async def _simple_embeddings(self, text: str, **kwargs) -> List[float]:
        """Simple embedding using TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            # Create simple TF-IDF embedding
            vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            
            # Use the text itself for fitting (simple case)
            embedding = vectorizer.fit_transform([text]).toarray()[0]
            
            # Pad or truncate to 384 dimensions
            if len(embedding) < 384:
                embedding = np.pad(embedding, (0, 384 - len(embedding)))
            else:
                embedding = embedding[:384]
            
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"Simple embeddings fallback failed: {e}")
            raise
    
    async def _keyword_matching(self, text: str, **kwargs) -> List[float]:
        """Keyword-based matching as embedding."""
        try:
            import re
            from collections import Counter
            
            # Extract keywords and create simple vector
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts = Counter(words)
            
            # Create a simple 384-dimensional vector based on word frequencies
            embedding = [0.0] * 384
            
            for i, (word, count) in enumerate(word_counts.most_common(384)):
                if i < 384:
                    embedding[i] = float(count) / len(words)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Keyword matching fallback failed: {e}")
            raise
    
    async def _text_similarity(self, text: str, **kwargs) -> List[float]:
        """Basic text similarity as pseudo-embedding."""
        try:
            import hashlib
            
            # Create deterministic vector from text hash
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            # Convert hash to 384-dimensional vector
            embedding = []
            for i in range(0, len(text_hash), 2):
                if len(embedding) < 384:
                    hex_pair = text_hash[i:i+2]
                    value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
                    embedding.append(value)
            
            # Pad to 384 dimensions if needed
            while len(embedding) < 384:
                embedding.append(0.0)
            
            return embedding[:384]
            
        except Exception as e:
            self.logger.error(f"Text similarity fallback failed: {e}")
            raise


class EinsteinFallbackManager:
    """Manages all fallback mechanisms for Einstein system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(f"{__name__}.EinsteinFallbackManager")
        
        # Initialize fallback chains
        self.search_fallback = SearchFallbackChain(project_root)
        self.embedding_fallback = EmbeddingFallbackChain()
        
        # Statistics
        self.stats = {
            'total_fallbacks': 0,
            'successful_fallbacks': 0,
            'failed_fallbacks': 0,
            'by_type': {
                'search': 0,
                'embedding': 0,
                'index': 0
            }
        }
    
    async def execute_search_fallback(
        self,
        query: str,
        context: Dict[str, Any] | None = None
    ) -> FallbackResult:
        """Execute search fallback chain."""
        self.stats['total_fallbacks'] += 1
        self.stats['by_type']['search'] += 1
        
        result = await self.search_fallback.execute(query, **(context or {}))
        
        if result.success:
            self.stats['successful_fallbacks'] += 1
        else:
            self.stats['failed_fallbacks'] += 1
        
        return result
    
    async def execute_embedding_fallback(
        self,
        text: str,
        context: Dict[str, Any] | None = None
    ) -> FallbackResult:
        """Execute embedding fallback chain."""
        self.stats['total_fallbacks'] += 1
        self.stats['by_type']['embedding'] += 1
        
        result = await self.embedding_fallback.execute(text, **(context or {}))
        
        if result.success:
            self.stats['successful_fallbacks'] += 1
        else:
            self.stats['failed_fallbacks'] += 1
        
        return result
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get comprehensive fallback statistics."""
        return {
            'overall': self.stats.copy(),
            'search_chain': self.search_fallback.stats.copy(),
            'embedding_chain': self.embedding_fallback.stats.copy(),
            'success_rate': (
                self.stats['successful_fallbacks'] / max(1, self.stats['total_fallbacks'])
            )
        }


# Global fallback manager
_fallback_manager: EinsteinFallbackManager | None = None


def get_einstein_fallback_manager(project_root: Path | None = None) -> EinsteinFallbackManager:
    """Get or create the global Einstein fallback manager."""
    global _fallback_manager
    if _fallback_manager is None:
        if project_root is None:
            project_root = Path.cwd()
        _fallback_manager = EinsteinFallbackManager(project_root)
    return _fallback_manager