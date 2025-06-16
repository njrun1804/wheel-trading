"""
Memory Manager Integration Examples
Shows how to integrate the memory manager with existing components
"""

import asyncio
from contextlib import contextmanager
from typing import Any

from bolt.hardware.memory_manager import (
    DuckDBMemoryGuard,
    EinsteinMemoryGuard,
    JarvisMemoryGuard,
    get_memory_manager,
)

# Initialize memory management
memory_manager = get_memory_manager()


# ==================== DuckDB Integration ====================


class MemoryAwareDuckDBConnection:
    """DuckDB connection with automatic memory management"""

    def __init__(self, db_path: str):
        import duckdb

        self.db_path = db_path
        self.connection = duckdb.connect(db_path)
        self.memory_guard = DuckDBMemoryGuard(self.connection)

    def execute_query(self, query: str, estimated_mb: float = 100) -> Any:
        """Execute query with memory allocation"""
        with self.memory_guard.allocate_for_query(estimated_mb, query[:50]):
            return self.connection.execute(query).fetchall()

    def load_dataframe(self, df, table_name: str):
        """Load DataFrame with memory tracking"""
        # Estimate memory usage
        estimated_mb = (
            df.memory_usage(deep=True).sum() / (1024 * 1024) * 2
        )  # 2x for overhead

        with self.memory_guard.allocate_for_query(estimated_mb, f"Load {table_name}"):
            self.connection.register(table_name, df)

    def close(self):
        """Close connection and release memory"""
        self.connection.close()


# ==================== Jarvis Integration ====================


class MemoryAwareJarvisIndex:
    """Jarvis index with memory management"""

    def __init__(self, index_name: str):
        self.index_name = index_name
        self.memory_guard = JarvisMemoryGuard()
        self.index_allocation = None

    def build_index(self, documents: list, embedding_size: int = 768):
        """Build index with memory allocation"""
        # Estimate memory needed
        estimated_mb = (len(documents) * embedding_size * 4) / (1024 * 1024) * 1.5

        if not self.memory_guard.check_allocation(estimated_mb, "build_index"):
            raise MemoryError(f"Insufficient memory for {self.index_name} index")

        # Allocate memory for the index
        allocation = memory_manager.allocate(
            "jarvis",
            estimated_mb,
            f"Index: {self.index_name}",
            can_evict=False,  # Don't evict active indexes
            priority=8,
        )

        if not allocation:
            raise MemoryError("Failed to allocate memory for index")

        self.index_allocation = allocation

        # Build index here...
        print(f"Building index with {estimated_mb:.1f}MB allocation")

    def search(self, query: str, k: int = 10):
        """Search with temporary memory allocation"""
        # Small allocation for search operation
        with memory_manager.allocate_context(
            "jarvis", 10, f"Search in {self.index_name}"
        ):
            # Perform search...
            return [f"Result {i}" for i in range(k)]

    def close(self):
        """Release index memory"""
        if self.index_allocation:
            memory_manager.deallocate(self.index_allocation)
            self.index_allocation = None


# ==================== Einstein Integration ====================


class MemoryAwareEinsteinModel:
    """Einstein model with memory management"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.memory_guard = EinsteinMemoryGuard()
        self.model_allocation = None

    def load_model(self, model_size_mb: float):
        """Load model with memory allocation"""
        # Allocate memory for model weights
        self.model_allocation = memory_manager.allocate(
            "einstein",
            model_size_mb,
            f"Model: {self.model_name}",
            can_evict=False,  # Models should not be evicted
            priority=9,  # High priority
        )

        if not self.model_allocation:
            raise MemoryError(f"Cannot load {self.model_name}, insufficient memory")

        print(f"Model loaded with {model_size_mb:.1f}MB allocation")

    def generate_embeddings(self, texts: list, batch_size: int = 32):
        """Generate embeddings with batch memory management"""
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Allocate memory for this batch
            with self.memory_guard.allocate_for_embeddings(len(batch), 768):
                # Generate embeddings...
                embeddings = [[0.1] * 768 for _ in batch]  # Dummy embeddings
                results.extend(embeddings)

        return results

    def unload_model(self):
        """Unload model and free memory"""
        if self.model_allocation:
            memory_manager.deallocate(self.model_allocation)
            self.model_allocation = None


# ==================== Meta System Integration ====================


class MemoryAwareMetaOperation:
    """Meta system operations with memory tracking"""

    @staticmethod
    @contextmanager
    def allocate_for_analysis(file_count: int):
        """Allocate memory for file analysis"""
        # Estimate: 1MB per file for AST and analysis
        estimated_mb = file_count * 1.0

        with memory_manager.allocate_context(
            "meta_system", estimated_mb, f"Analyzing {file_count} files", priority=6
        ):
            yield

    @staticmethod
    @contextmanager
    def allocate_for_generation(template_size_kb: int, output_count: int):
        """Allocate memory for code generation"""
        estimated_mb = (template_size_kb * output_count) / 1024 * 2  # 2x for overhead

        with memory_manager.allocate_context(
            "meta_system", estimated_mb, f"Generating {output_count} files", priority=7
        ):
            yield


# ==================== Cache Management ====================


class MemoryAwareCache:
    """Cache with automatic memory management and eviction"""

    def __init__(self, cache_name: str, max_size_mb: float = 100):
        self.cache_name = cache_name
        self.max_size_mb = max_size_mb
        self.items = {}
        self.allocation = None

        # Pre-allocate cache memory
        self.allocation = memory_manager.allocate(
            "cache",
            max_size_mb,
            f"Cache: {cache_name}",
            can_evict=True,  # Caches can be evicted
            priority=3,  # Low priority
        )

    def put(self, key: str, value: Any, size_mb: float):
        """Add item to cache with size tracking"""
        if not self.allocation:
            return  # Cache disabled due to memory pressure

        # Simple size management
        self.items[key] = (value, size_mb)

        # Check total size
        total_size = sum(item[1] for item in self.items.values())
        if total_size > self.max_size_mb:
            # Evict oldest items
            self._evict_lru()

    def get(self, key: str) -> Any | None:
        """Get item from cache"""
        if key in self.items:
            return self.items[key][0]
        return None

    def _evict_lru(self):
        """Evict least recently used items"""
        # Simple implementation - evict first 20%
        items_to_evict = int(len(self.items) * 0.2)
        for key in list(self.items.keys())[:items_to_evict]:
            del self.items[key]

    def clear(self):
        """Clear cache (useful during memory pressure)"""
        self.items.clear()

    def __del__(self):
        """Release cache memory allocation"""
        if self.allocation:
            memory_manager.deallocate(self.allocation)


# ==================== Async Integration Example ====================


async def memory_aware_parallel_processing(tasks: list):
    """Process tasks in parallel with memory constraints"""

    # Check available memory
    stats = memory_manager.get_component_stats("other")
    available_mb = stats["max_mb"] - stats["allocated_mb"]

    # Determine batch size based on available memory
    memory_per_task = 10  # MB
    max_parallel = min(len(tasks), int(available_mb / memory_per_task))

    print(f"Processing {len(tasks)} tasks with max parallelism: {max_parallel}")

    results = []
    for i in range(0, len(tasks), max_parallel):
        batch = tasks[i : i + max_parallel]

        # Allocate memory for this batch
        with memory_manager.allocate_context(
            "other",
            len(batch) * memory_per_task,
            f"Parallel batch {i//max_parallel + 1}",
        ):
            # Process batch
            batch_results = await asyncio.gather(
                *[process_task(task) for task in batch]
            )
            results.extend(batch_results)

    return results


async def process_task(task):
    """Dummy task processor"""
    await asyncio.sleep(0.1)
    return f"Processed: {task}"


# ==================== Example Usage ====================


def demo_memory_management():
    """Demonstrate integrated memory management"""

    print("=== Memory Management Demo ===\n")

    # 1. DuckDB Example
    print("1. DuckDB Operations:")
    db = MemoryAwareDuckDBConnection(":memory:")
    try:
        results = db.execute_query("SELECT 1 as test", estimated_mb=10)
        print(f"   Query result: {results}")
    finally:
        db.close()

    # 2. Jarvis Example
    print("\n2. Jarvis Index Operations:")
    index = MemoryAwareJarvisIndex("test_index")
    try:
        index.build_index(["doc1", "doc2", "doc3"], embedding_size=768)
        results = index.search("test query", k=5)
        print(f"   Search results: {results[:2]}...")
    finally:
        index.close()

    # 3. Einstein Example
    print("\n3. Einstein Model Operations:")
    model = MemoryAwareEinsteinModel("test_model")
    try:
        model.load_model(model_size_mb=500)  # 500MB model
        embeddings = model.generate_embeddings(["text1", "text2"], batch_size=1)
        print(f"   Generated {len(embeddings)} embeddings")
    finally:
        model.unload_model()

    # 4. Cache Example
    print("\n4. Cache Operations:")
    cache = MemoryAwareCache("test_cache", max_size_mb=50)
    cache.put("key1", "value1", size_mb=10)
    print(f"   Cached value: {cache.get('key1')}")

    # 5. Status Report
    print("\n5. Memory Status:")
    report = memory_manager.get_status_report()
    print(f"   System usage: {report['system']['system_usage_percent']:.1f}%")
    print(f"   Total allocated: {report['system']['total_allocated_mb']:.1f}MB")

    for component, stats in report["components"].items():
        if stats["allocated_mb"] > 0:
            print(
                f"   {component}: {stats['allocated_mb']:.1f}MB / {stats['max_mb']:.1f}MB ({stats['usage_percent']:.1f}%)"
            )


if __name__ == "__main__":
    demo_memory_management()

    # Run async example
    print("\n6. Async Parallel Processing:")
    tasks = [f"task_{i}" for i in range(20)]
    results = asyncio.run(memory_aware_parallel_processing(tasks))
    print(f"   Processed {len(results)} tasks")
