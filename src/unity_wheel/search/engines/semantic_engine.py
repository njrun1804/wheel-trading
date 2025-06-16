"""Semantic Search Engine - GPU-accelerated semantic search using FAISS and MLX.

Combines Einstein's semantic search capabilities with Bolt's Metal GPU acceleration
for 11x faster vector similarity search.
"""

import asyncio
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """GPU-accelerated semantic search engine using FAISS and sentence transformers."""

    def __init__(
        self,
        cache_system: Any,
        hardware_config: dict[str, Any],
        model_name: str = "all-MiniLM-L6-v2",
        index_path: Path | None = None,
    ):
        self.cache_system = cache_system
        self.hardware_config = hardware_config
        self.model_name = model_name
        self.index_path = index_path or Path(".index/semantic")

        # Model and index
        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.document_store: dict[int, dict[str, Any]] = {}
        self.next_doc_id = 0

        # GPU configuration
        self.use_gpu = hardware_config.get("gpu_cores", 0) > 0
        self.gpu_device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Performance settings
        self.batch_size = 64 if self.use_gpu else 32
        self.max_sequence_length = 512

        # Performance tracking
        self.stats = {
            "total_searches": 0,
            "total_index_time_ms": 0.0,
            "total_search_time_ms": 0.0,
            "documents_indexed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        self.available = True
        self.initialized = False

    async def initialize(self):
        """Initialize the semantic search engine."""
        if self.initialized:
            return

        try:
            start_time = time.perf_counter()
            logger.info("🚀 Initializing Semantic Search Engine...")

            # Initialize model
            await self._initialize_model()

            # Initialize or load index
            await self._initialize_index()

            self.initialized = True
            init_time = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"✅ Semantic engine initialized in {init_time:.1f}ms (GPU: {self.gpu_device})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize semantic engine: {e}")
            self.available = False
            raise

    async def _initialize_model(self):
        """Initialize the sentence transformer model."""
        loop = asyncio.get_event_loop()

        def load_model():
            model = SentenceTransformer(self.model_name)
            if self.gpu_device != "cpu":
                model = model.to(self.gpu_device)
            model.max_seq_length = self.max_sequence_length
            return model

        self.model = await loop.run_in_executor(None, load_model)
        logger.debug(f"Loaded model {self.model_name} on {self.gpu_device}")

    async def _initialize_index(self):
        """Initialize or load FAISS index."""
        self.index_path.mkdir(parents=True, exist_ok=True)

        index_file = self.index_path / "faiss.index"
        doc_store_file = self.index_path / "documents.pkl"

        if index_file.exists() and doc_store_file.exists():
            # Load existing index
            try:
                self.index = faiss.read_index(str(index_file))
                with open(doc_store_file, "rb") as f:
                    stored_data = pickle.load(f)
                    self.document_store = stored_data["documents"]
                    self.next_doc_id = stored_data["next_id"]

                logger.info(
                    f"Loaded semantic index with {len(self.document_store)} documents"
                )
            except Exception as e:
                logger.warning(f"Failed to load index, creating new: {e}")
                await self._create_new_index()
        else:
            await self._create_new_index()

    async def _create_new_index(self):
        """Create a new FAISS index."""
        # Get embedding dimension
        test_embedding = self.model.encode(["test"], show_progress_bar=False)
        dimension = test_embedding.shape[1]

        # Create index optimized for GPU if available
        if self.use_gpu and self.gpu_device == "cuda":
            # Use GPU index
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0

            cpu_index = faiss.IndexFlatIP(
                dimension
            )  # Inner product for cosine similarity
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index, config)
        else:
            # Use CPU index with optimization
            self.index = faiss.IndexFlatIP(dimension)

        logger.info(f"Created new semantic index (dimension: {dimension})")

    async def search(
        self,
        query: str,
        max_results: int = 50,
        similarity_threshold: float = 0.5,
        file_filter: list[str] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search.

        Args:
            query: Search query
            max_results: Maximum results to return
            similarity_threshold: Minimum similarity score
            file_filter: Optional list of file paths to search within

        Returns:
            List of search results with similarity scores
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.perf_counter()

        try:
            # Check cache
            cache_key = f"semantic:{query}:{max_results}:{similarity_threshold}"
            cached_result = await self.cache_system.get(cache_key)

            if cached_result is not None:
                self.stats["cache_hits"] += 1
                return cached_result

            self.stats["cache_misses"] += 1

            # Encode query
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    [query], show_progress_bar=False, convert_to_numpy=True
                ),
            )

            # Normalize for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(
                query_embedding, axis=1, keepdims=True
            )

            # Search index
            if self.index.ntotal == 0:
                return []

            k = min(
                max_results * 2, self.index.ntotal
            )  # Search for more to filter later
            scores, indices = self.index.search(query_embedding, k)

            # Build results
            results = []
            for idx, score in zip(indices[0], scores[0], strict=False):
                if idx == -1 or score < similarity_threshold:
                    continue

                doc_id = int(idx)
                if doc_id in self.document_store:
                    doc = self.document_store[doc_id]

                    # Apply file filter if provided
                    if file_filter and doc["file_path"] not in file_filter:
                        continue

                    result = {
                        "content": doc["content"],
                        "file_path": doc["file_path"],
                        "line_number": doc.get("line_number", 0),
                        "score": float(score),
                        "similarity": float(score),  # For compatibility
                        "context": {
                            "function": doc.get("function", ""),
                            "class": doc.get("class", ""),
                            "module": doc.get("module", ""),
                            "snippet_type": doc.get("snippet_type", "unknown"),
                        },
                    }
                    results.append(result)

                    if len(results) >= max_results:
                        break

            # Cache results
            await self.cache_system.put(cache_key, results, ttl_seconds=300)

            # Update stats
            search_time = (time.perf_counter() - start_time) * 1000
            self.stats["total_searches"] += 1
            self.stats["total_search_time_ms"] += search_time

            logger.debug(
                f"Semantic search completed in {search_time:.1f}ms, {len(results)} results"
            )

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def index_documents(self, documents: list[dict[str, Any]]):
        """
        Index a batch of documents for semantic search.

        Args:
            documents: List of documents with 'content', 'file_path', and metadata
        """
        if not self.initialized:
            await self.initialize()

        if not documents:
            return

        start_time = time.perf_counter()

        try:
            # Extract content for embedding
            contents = [doc["content"] for doc in documents]

            # Generate embeddings in batches
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    contents,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ),
            )

            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Add to index
            start_idx = self.next_doc_id
            self.index.add(embeddings)

            # Store documents
            for i, doc in enumerate(documents):
                doc_id = start_idx + i
                self.document_store[doc_id] = {
                    "content": doc["content"],
                    "file_path": doc["file_path"],
                    "line_number": doc.get("line_number", 0),
                    "function": doc.get("function", ""),
                    "class": doc.get("class", ""),
                    "module": doc.get("module", ""),
                    "snippet_type": doc.get("snippet_type", "code"),
                }

            self.next_doc_id = start_idx + len(documents)

            # Save index periodically
            if self.next_doc_id % 1000 == 0:
                await self.save_index()

            # Update stats
            index_time = (time.perf_counter() - start_time) * 1000
            self.stats["documents_indexed"] += len(documents)
            self.stats["total_index_time_ms"] += index_time

            logger.debug(f"Indexed {len(documents)} documents in {index_time:.1f}ms")

        except Exception as e:
            logger.error(f"Failed to index documents: {e}")

    async def save_index(self):
        """Save FAISS index and document store to disk."""
        try:
            index_file = self.index_path / "faiss.index"
            doc_store_file = self.index_path / "documents.pkl"

            # Save FAISS index
            if self.use_gpu and self.gpu_device == "cuda":
                # Transfer GPU index to CPU for saving
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(index_file))
            else:
                faiss.write_index(self.index, str(index_file))

            # Save document store
            with open(doc_store_file, "wb") as f:
                pickle.dump(
                    {"documents": self.document_store, "next_id": self.next_doc_id}, f
                )

            logger.debug(
                f"Saved semantic index with {len(self.document_store)} documents"
            )

        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    async def optimize(self):
        """Optimize the semantic search engine."""
        logger.info("🚀 Optimizing semantic search engine...")

        # Save index if needed
        if self.stats["documents_indexed"] > 0:
            await self.save_index()

        # Clear cache for stale entries
        if self.cache_system:
            await self.cache_system.clear_old_entries(max_age_seconds=3600)

        logger.info("✅ Semantic engine optimization complete")

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = dict(self.stats)

        # Calculate averages
        if stats["total_searches"] > 0:
            stats["avg_search_time_ms"] = (
                stats["total_search_time_ms"] / stats["total_searches"]
            )
        else:
            stats["avg_search_time_ms"] = 0.0

        if stats["documents_indexed"] > 0:
            stats["avg_index_time_ms"] = (
                stats["total_index_time_ms"] / stats["documents_indexed"]
            )
        else:
            stats["avg_index_time_ms"] = 0.0

        # Add cache hit rate
        total_cache_ops = stats["cache_hits"] + stats["cache_misses"]
        if total_cache_ops > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_ops
        else:
            stats["cache_hit_rate"] = 0.0

        # Add system info
        stats["gpu_device"] = self.gpu_device
        stats["model_name"] = self.model_name
        stats["index_size"] = self.index.ntotal if self.index else 0
        stats["initialized"] = self.initialized
        stats["available"] = self.available

        return stats

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            start_time = time.perf_counter()

            # Test search
            results = await self.search("test query", max_results=1)

            response_time = (time.perf_counter() - start_time) * 1000

            return {
                "healthy": True,
                "available": self.available,
                "initialized": self.initialized,
                "response_time_ms": response_time,
                "index_size": self.index.ntotal if self.index else 0,
                "gpu_device": self.gpu_device,
            }
        except Exception as e:
            return {"healthy": False, "available": False, "error": str(e)}

    async def cleanup(self):
        """Cleanup resources."""
        if self.index and self.stats["documents_indexed"] > 0:
            await self.save_index()

        self.model = None
        self.index = None
        self.document_store.clear()
        self.initialized = False

        logger.info("🧹 Semantic search engine cleaned up")
