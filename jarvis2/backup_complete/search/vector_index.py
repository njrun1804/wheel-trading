"""Hybrid vector search index using hnswlib for CPU efficiency.

Provides instant (<5ms) code similarity search using pre-computed embeddings.
"""
import numpy as np
import hnswlib
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import time
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CodeDocument:
    """Document in the code index."""
    id: str
    code: str
    filepath: str
    language: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class EmbeddingCache:
    """Cache for code embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, np.ndarray] = {}
        
    def get_embedding(self, code: str) -> Optional[np.ndarray]:
        """Get cached embedding for code."""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        # Check memory cache
        if code_hash in self.memory_cache:
            return self.memory_cache[code_hash]
            
        # Check disk cache
        cache_file = self.cache_dir / f"{code_hash}.npy"
        if cache_file.exists():
            embedding = np.load(cache_file)
            self.memory_cache[code_hash] = embedding
            return embedding
            
        return None
        
    def cache_embedding(self, code: str, embedding: np.ndarray):
        """Cache embedding for code."""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        # Memory cache
        self.memory_cache[code_hash] = embedding
        
        # Disk cache
        cache_file = self.cache_dir / f"{code_hash}.npy"
        np.save(cache_file, embedding)


class RealEmbedder:
    """Real code embedder using lightweight methods optimized for M4 Pro."""
    
    def __init__(self, dim: int = 768):
        self.dim = dim
        # Import here to avoid circular dependency
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from core.code_embeddings import LightweightCodeEmbedder, MLXCodeEmbedder
        
        # Try MLX first for M4 Pro optimization
        try:
            self.embedder = MLXCodeEmbedder()
            self.use_mlx = True
            logger.info("Using MLX embedder for M4 Pro optimization")
        except:
            # Fallback to lightweight embedder
            self.embedder = LightweightCodeEmbedder(vector_dim=dim)
            self.use_mlx = False
            logger.info("Using lightweight code embedder")
    
    def embed(self, code: str) -> np.ndarray:
        """Create embedding from code."""
        try:
            embedding = self.embedder.embed(code)
            
            # Ensure correct dimension
            if len(embedding) != self.dim:
                # Resize if needed
                if len(embedding) > self.dim:
                    embedding = embedding[:self.dim]
                else:
                    # Pad with zeros
                    padding = np.zeros(self.dim - len(embedding), dtype=np.float32)
                    embedding = np.concatenate([embedding, padding])
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Embedding failed: {e}, using fallback")
            # Fallback to simple features
            return self._fallback_embed(code)
    
    def _fallback_embed(self, code: str) -> np.ndarray:
        """Fallback embedding method."""
        features = []
        
        # Length features
        features.append(len(code) / 1000.0)
        features.append(len(code.split('\n')) / 100.0)
        
        # Keyword features
        keywords = ['def', 'class', 'import', 'if', 'for', 'while', 
                   'return', 'try', 'except', 'async', 'await']
        for kw in keywords:
            features.append(code.count(kw) / 10.0)
            
        # Character distribution
        for char in 'abcdefghijklmnopqrstuvwxyz_()[]{}":':
            features.append(code.count(char) / len(code) if code else 0)
            
        # Pad or truncate to dimension
        if len(features) < self.dim:
            features.extend([0.0] * (self.dim - len(features)))
        else:
            features = features[:self.dim]
        np.random.seed(hash(code) % 2**32)
        noise = np.random.randn(self.dim) * 0.1
        
        embedding = np.array(features, dtype=np.float32) + noise
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding


class HybridVectorIndex:
    """Fast vector search index using hnswlib."""
    
    def __init__(self, index_dir: str, dim: int = 768, max_elements: int = 1000000):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.dim = dim
        self.max_elements = max_elements
        
        # HNSW parameters
        self.M = 16  # Number of bi-directional links
        self.ef_construction = 200  # Size of dynamic list
        self.ef_search = 100  # Size of dynamic list for search
        
        # Components
        self.index = None
        self.documents: Dict[int, CodeDocument] = {}
        self.embedder = RealEmbedder(dim)
        self.cache = EmbeddingCache(self.index_dir / "embeddings")
        
        self.initialized = False
        
    async def initialize(self):
        """Initialize or load existing index."""
        if self.initialized:
            return
            
        start_time = time.perf_counter()
        
        # Try to load existing index
        index_file = self.index_dir / "index.bin"
        docs_file = self.index_dir / "documents.pkl"
        
        if index_file.exists() and docs_file.exists():
            try:
                # Load HNSW index
                self.index = hnswlib.Index(space='cosine', dim=self.dim)
                self.index.load_index(str(index_file))
                
                # Load documents
                with open(docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
                    
                logger.info(f"Loaded index with {len(self.documents)} documents")
                
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
            
        self.initialized = True
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Vector index initialized in {elapsed:.1f}ms")
        
    def _create_new_index(self):
        """Create new empty index."""
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        self.index.init_index(max_elements=self.max_elements, 
                            ef_construction=self.ef_construction, 
                            M=self.M)
        self.index.set_ef(self.ef_search)
        self.documents = {}
        logger.info("Created new vector index")
        
    async def add_document(self, code: str, filepath: str, 
                          language: str = "python",
                          metadata: Optional[Dict] = None):
        """Add code document to index."""
        if not self.initialized:
            await self.initialize()
            
        # Check cache first
        embedding = self.cache.get_embedding(code)
        if embedding is None:
            # Generate embedding
            embedding = self.embedder.embed(code)
            self.cache.cache_embedding(code, embedding)
            
        # Create document
        doc_id = str(len(self.documents))
        doc = CodeDocument(
            id=doc_id,
            code=code,
            filepath=filepath,
            language=language,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        # Add to index
        idx = len(self.documents)
        self.documents[idx] = doc
        self.index.add_items(embedding.reshape(1, -1), np.array([idx]))
        
        logger.debug(f"Added document {doc_id} from {filepath}")
        
    async def search(self, query: str, k: int = 10, 
                    min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar code."""
        if not self.initialized:
            await self.initialize()
            
        if not self.documents:
            return []
            
        start_time = time.perf_counter()
        
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        
        # Search
        try:
            indices, distances = self.index.knn_query(
                query_embedding.reshape(1, -1), 
                k=min(k, len(self.documents))
            )
            
            # Convert cosine distance to similarity
            similarities = 1 - distances[0]
            
            # Build results
            results = []
            for idx, similarity in zip(indices[0], similarities):
                if similarity >= min_similarity:
                    doc = self.documents[idx]
                    results.append({
                        'code': doc.code,
                        'filepath': doc.filepath,
                        'language': doc.language,
                        'similarity': float(similarity),
                        'metadata': doc.metadata
                    })
                    
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Search completed in {elapsed:.1f}ms, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
            
    async def search_by_embedding(self, embedding: np.ndarray, 
                                 k: int = 10) -> List[Dict[str, Any]]:
        """Search using pre-computed embedding."""
        if not self.initialized:
            await self.initialize()
            
        if not self.documents:
            return []
            
        # Search
        indices, distances = self.index.knn_query(embedding.reshape(1, -1), k=k)
        similarities = 1 - distances[0]
        
        # Build results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            doc = self.documents[idx]
            results.append({
                'code': doc.code,
                'filepath': doc.filepath,
                'language': doc.language,
                'similarity': float(similarity),
                'metadata': doc.metadata
            })
            
        return results
        
    async def build_from_directory(self, directory: Path, 
                                  extensions: List[str] = ['.py', '.js', '.ts']):
        """Build index from directory of code files."""
        if not self.initialized:
            await self.initialize()
            
        logger.info(f"Building index from {directory}")
        
        count = 0
        for ext in extensions:
            for filepath in directory.rglob(f"*{ext}"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code = f.read()
                        
                    await self.add_document(
                        code=code,
                        filepath=str(filepath.relative_to(directory)),
                        language=ext[1:],  # Remove dot
                        metadata={
                            'size': len(code),
                            'lines': len(code.splitlines())
                        }
                    )
                    count += 1
                    
                    if count % 100 == 0:
                        logger.info(f"Indexed {count} files...")
                        
                except Exception as e:
                    logger.error(f"Failed to index {filepath}: {e}")
                    
        # Save index
        await self.save()
        logger.info(f"Indexed {count} files total")
        
    async def save(self):
        """Save index to disk."""
        if not self.initialized:
            return
            
        # Save HNSW index
        index_file = self.index_dir / "index.bin"
        self.index.save_index(str(index_file))
        
        # Save documents
        docs_file = self.index_dir / "documents.pkl"
        with open(docs_file, 'wb') as f:
            pickle.dump(self.documents, f)
            
        # Save metadata
        meta_file = self.index_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump({
                'num_documents': len(self.documents),
                'dim': self.dim,
                'max_elements': self.max_elements,
                'M': self.M,
                'ef_construction': self.ef_construction,
                'ef_search': self.ef_search
            }, f, indent=2)
            
        logger.info(f"Saved index with {len(self.documents)} documents")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'initialized': self.initialized,
            'num_documents': len(self.documents),
            'index_size': self.index.get_current_count() if self.index else 0,
            'max_elements': self.max_elements,
            'dim': self.dim,
            'cache_size': len(self.cache.memory_cache)
        }


# Example usage
async def demo():
    """Demo of vector search."""
    index = HybridVectorIndex(".jarvis/indexes")
    await index.initialize()
    
    # Add some code
    await index.add_document(
        code="""def hello_world():
    print("Hello, World!")""",
        filepath="examples/hello.py"
    )
    
    await index.add_document(
        code="""def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",
        filepath="examples/fibonacci.py"
    )
    
    # Search
    results = await index.search("function to print greeting", k=5)
    for result in results:
        print(f"Similarity: {result['similarity']:.2f} - {result['filepath']}")
        print(result['code'][:100] + "...")
        print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())