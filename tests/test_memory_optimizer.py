"""Tests for memory optimization utilities."""

import time

import numpy as np
import pytest

from src.unity_wheel.utils.memory_optimizer import (
    MemoryEfficientArray,
    MemoryMonitor,
    MemoryPool,
    StreamingDataProcessor,
    create_memory_efficient_dict,
    get_memory_monitor,
    memory_profiler,
    optimize_numpy_memory,
)


class TestMemoryMonitor:
    """Test memory monitoring functionality."""

    def test_memory_stats(self):
        """Test getting memory statistics."""
        monitor = MemoryMonitor()
        stats = monitor.get_memory_stats()

        assert stats.process_mb > 0
        assert stats.total_mb > 0
        assert 0 <= stats.process_percent <= 100
        assert stats.process_mb <= stats.total_mb

    def test_object_tracking(self):
        """Test object tracking functionality."""
        monitor = MemoryMonitor()

        # Track some objects
        test_list = [1, 2, 3, 4, 5]
        test_array = np.array([1.0, 2.0, 3.0])
        test_dict = {"a": 1, "b": 2}

        monitor.track_object(test_list)
        monitor.track_object(test_array)
        monitor.track_object(test_dict, size_hint=0.1)  # 100KB hint

        # Get object stats
        object_stats = monitor.get_object_stats()

        assert "list" in object_stats
        assert "ndarray" in object_stats
        assert "dict" in object_stats

        assert object_stats["list"].count == 1
        assert object_stats["ndarray"].count == 1
        assert object_stats["dict"].count == 1

        # Check that dict uses size hint
        assert object_stats["dict"].total_size_mb == 0.1

    def test_memory_thresholds(self):
        """Test memory threshold checking."""
        monitor = MemoryMonitor(
            warning_threshold_mb=1.0, critical_threshold_mb=2.0  # Very low threshold for testing
        )

        # Should not trigger cleanup initially
        triggered, reason = monitor.check_memory_usage()
        # Can't assert False here as it may trigger based on actual memory usage

        # Force cleanup should always trigger
        triggered, reason = monitor.check_memory_usage(force_cleanup=True)
        assert triggered
        assert "Forced cleanup" in reason

    def test_cleanup_interval(self):
        """Test automatic cleanup based on time interval."""
        monitor = MemoryMonitor(cleanup_interval_seconds=0.1)  # 100ms interval

        # Wait for interval to pass
        time.sleep(0.2)

        triggered, reason = monitor.check_memory_usage()
        assert triggered
        assert "Scheduled cleanup" in reason

    def test_memory_summary(self):
        """Test comprehensive memory summary."""
        monitor = MemoryMonitor()

        # Track some objects
        monitor.track_object([1, 2, 3])
        monitor.track_object(np.array([1.0, 2.0]))

        summary = monitor.get_summary()

        required_keys = [
            "current_stats",
            "peak_memory_mb",
            "peak_objects",
            "tracked_objects",
            "object_stats",
            "thresholds",
            "cleanup",
        ]

        for key in required_keys:
            assert key in summary

        assert summary["tracked_objects"] > 0
        assert len(summary["object_stats"]) > 0


class TestMemoryEfficientArray:
    """Test memory-efficient array implementation."""

    def test_basic_operations(self):
        """Test basic array operations."""
        array = MemoryEfficientArray(initial_capacity=10, max_size=50)

        # Test append
        array.append(1.0)
        array.append(2.0)
        array.append(3.0)

        assert len(array) == 3
        assert array[0] == 1.0
        assert array[1] == 2.0
        assert array[2] == 3.0

        # Test get_data
        data = array.get_data()
        np.testing.assert_array_equal(data, [1.0, 2.0, 3.0])

    def test_array_append(self):
        """Test appending numpy arrays."""
        array = MemoryEfficientArray(initial_capacity=10)

        # Append numpy array
        array.append(np.array([1.0, 2.0, 3.0]))
        array.append(np.array([4.0, 5.0]))

        assert len(array) == 5
        data = array.get_data()
        np.testing.assert_array_equal(data, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_resize_behavior(self):
        """Test array resizing when capacity is exceeded."""
        array = MemoryEfficientArray(initial_capacity=3, max_size=20)

        # Fill beyond initial capacity
        for i in range(10):
            array.append(float(i))

        assert len(array) == 10
        assert array._capacity > 3  # Should have resized

        # Check data integrity
        data = array.get_data()
        np.testing.assert_array_equal(data, np.arange(10, dtype=float))

    def test_recycling(self):
        """Test array recycling when max size is exceeded."""
        array = MemoryEfficientArray(initial_capacity=5, max_size=10)

        # Fill beyond max size
        for i in range(15):
            array.append(float(i))

        # Should have recycled and kept recent data
        assert len(array) <= 10

        # Should contain the most recent values
        data = array.get_data()
        assert data[-1] == 14.0  # Last value should be preserved

    def test_clear(self):
        """Test clearing array."""
        array = MemoryEfficientArray()
        array.append(1.0)
        array.append(2.0)

        assert len(array) == 2

        array.clear()
        assert len(array) == 0

    def test_different_dtypes(self):
        """Test array with different data types."""
        int_array = MemoryEfficientArray(dtype=np.int32)
        int_array.append(1)
        int_array.append(2)

        assert int_array.get_data().dtype == np.int32
        np.testing.assert_array_equal(int_array.get_data(), [1, 2])


class TestStreamingDataProcessor:
    """Test streaming data processing."""

    def test_chunk_processing(self):
        """Test processing data in chunks."""
        processor = StreamingDataProcessor(chunk_size=3)

        # Create data source
        data = range(10)  # 0, 1, 2, ..., 9

        # Process in chunks, sum each chunk
        results = list(
            processor.process_chunks(
                data_source=iter(data), processor_func=lambda chunk: sum(chunk)
            )
        )

        # Should have chunks: [0,1,2], [3,4,5], [6,7,8], [9]
        expected = [3, 12, 21, 9]  # Sums of each chunk
        assert results == expected

    def test_accumulator_processing(self):
        """Test processing with accumulator function."""
        processor = StreamingDataProcessor(chunk_size=2)

        data = [1, 2, 3, 4, 5]

        # Process chunks and accumulate results
        results = list(
            processor.process_chunks(
                data_source=iter(data),
                processor_func=lambda chunk: sum(chunk),  # Sum each chunk
                accumulator_func=lambda acc, result: (acc or 0) + result,  # Sum all chunk sums
            )
        )

        # Should return single accumulated result
        assert len(results) == 1
        assert results[0] == 15  # Sum of all numbers 1+2+3+4+5

    def test_batch_processing(self):
        """Test batch processing."""
        processor = StreamingDataProcessor(chunk_size=3)

        items = list(range(10))

        # Process batches
        results = processor.batch_process(
            items=items, batch_func=lambda batch: len(batch)  # Return batch size
        )

        # Should have batches of sizes [3, 3, 3, 1]
        assert results == [3, 3, 3, 1]

    def test_batch_processing_with_combine(self):
        """Test batch processing with combine function."""
        processor = StreamingDataProcessor(chunk_size=2)

        items = [1, 2, 3, 4, 5, 6]

        # Process batches and combine
        result = processor.batch_process(
            items=items,
            batch_func=lambda batch: sum(batch),  # Sum each batch
            combine_func=lambda results: sum(results),  # Sum all batch sums
        )

        assert result == 21  # Sum of 1+2+3+4+5+6


class TestMemoryPool:
    """Test memory pool for object reuse."""

    def test_basic_pool_operations(self):
        """Test basic pool get/put operations."""

        def create_list():
            return []

        def reset_list(lst):
            lst.clear()

        pool = MemoryPool(factory_func=create_list, max_size=5, reset_func=reset_list)

        # Get objects from pool
        obj1 = pool.get()
        obj2 = pool.get()

        assert isinstance(obj1, list)
        assert isinstance(obj2, list)
        assert obj1 is not obj2

        # Modify objects
        obj1.append(1)
        obj2.append(2)

        # Return to pool
        pool.put(obj1)
        pool.put(obj2)

        # Get objects again (should be reused and reset)
        obj3 = pool.get()
        obj4 = pool.get()

        assert obj3 is obj1  # Should be the same object
        assert obj4 is obj2
        assert len(obj3) == 0  # Should be reset
        assert len(obj4) == 0

    def test_pool_size_limit(self):
        """Test that pool respects size limits."""
        pool = MemoryPool(factory_func=lambda: [], max_size=2)

        # Create objects
        obj1 = pool.get()
        obj2 = pool.get()
        obj3 = pool.get()

        # Return all to pool
        pool.put(obj1)
        pool.put(obj2)
        pool.put(obj3)  # This should be ignored due to size limit

        stats = pool.get_stats()
        assert stats["pool_size"] == 2  # Only first 2 should be in pool
        assert stats["max_size"] == 2

    def test_pool_stats(self):
        """Test pool statistics."""
        pool = MemoryPool(factory_func=lambda: {})

        # Get some objects
        obj1 = pool.get()
        obj2 = pool.get()

        # Return one
        pool.put(obj1)

        # Get another (should reuse)
        obj3 = pool.get()

        stats = pool.get_stats()
        assert stats["created_count"] == 2  # Created 2 objects
        assert stats["reused_count"] == 1  # Reused 1 object
        assert stats["reuse_rate"] > 0  # Should have some reuse

    def test_pool_clear(self):
        """Test clearing the pool."""
        pool = MemoryPool(factory_func=lambda: [])

        obj = pool.get()
        pool.put(obj)

        assert pool.get_stats()["pool_size"] == 1

        pool.clear()
        assert pool.get_stats()["pool_size"] == 0


class TestMemoryUtilities:
    """Test utility functions."""

    def test_memory_profiler(self):
        """Test memory profiling context manager."""
        # This is a basic test since actual memory changes are hard to predict
        with memory_profiler("test_operation"):
            # Do some memory allocation
            data = [i for i in range(1000)]
            data = None  # Free memory

        # Test should complete without errors
        assert True

    def test_numpy_memory_optimization(self):
        """Test numpy array memory optimization."""
        # Test float64 to float32 optimization
        array_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        optimized = optimize_numpy_memory(array_f64)

        # Should be optimized to float32 if values allow it
        assert optimized.dtype in [np.float32, np.float64]
        np.testing.assert_allclose(array_f64, optimized)

        # Test int64 to int32 optimization
        array_i64 = np.array([1, 2, 3], dtype=np.int64)
        optimized_int = optimize_numpy_memory(array_i64)

        assert optimized_int.dtype in [np.int32, np.int64]
        np.testing.assert_array_equal(array_i64, optimized_int)

    def test_memory_efficient_dict(self):
        """Test memory-efficient dictionary."""
        max_size = 3
        d = create_memory_efficient_dict(max_size=max_size)

        # Add items up to limit
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3

        assert len(d) == 3
        assert "a" in d
        assert "b" in d
        assert "c" in d

        # Add one more item (should evict least recently used)
        d["d"] = 4

        assert len(d) == 3
        assert "a" not in d  # Should be evicted
        assert "d" in d

        # Access "b" to make it recently used
        _ = d["b"]

        # Add another item
        d["e"] = 5

        # "c" should be evicted (least recently used), "b" should remain
        assert "b" in d
        assert "c" not in d
        assert "e" in d


class TestGlobalMemoryMonitor:
    """Test global memory monitor functionality."""

    def test_get_memory_monitor(self):
        """Test getting global memory monitor."""
        monitor = get_memory_monitor()
        assert isinstance(monitor, MemoryMonitor)

        # Should be the same instance on multiple calls
        monitor2 = get_memory_monitor()
        assert monitor is monitor2

    def test_memory_functions(self):
        """Test global memory functions."""
        from src.unity_wheel.utils.memory_optimizer import (
            force_memory_cleanup,
            get_memory_summary,
            set_memory_thresholds,
        )

        # Test getting summary
        summary = get_memory_summary()
        assert isinstance(summary, dict)
        assert "current_stats" in summary

        # Test setting thresholds
        set_memory_thresholds(100.0, 200.0)
        monitor = get_memory_monitor()
        assert monitor.warning_threshold_mb == 100.0
        assert monitor.critical_threshold_mb == 200.0

        # Test force cleanup (should not raise errors)
        force_memory_cleanup()


@pytest.mark.performance
class TestMemoryPerformance:
    """Performance tests for memory optimization."""

    def test_memory_efficient_array_performance(self):
        """Test performance of memory-efficient array."""
        array = MemoryEfficientArray(initial_capacity=1000, max_size=10000)

        # Time large number of appends
        start_time = time.time()
        for i in range(5000):
            array.append(float(i))
        append_time = time.time() - start_time

        assert len(array) <= 10000  # Should respect max size
        assert append_time < 1.0  # Should be fast

        print(f"Appended 5000 items in {append_time:.3f}s")

    def test_streaming_processor_performance(self):
        """Test performance of streaming processor."""
        processor = StreamingDataProcessor(chunk_size=1000)

        # Large dataset
        large_data = range(50000)

        start_time = time.time()
        results = list(
            processor.process_chunks(
                data_source=iter(large_data),
                processor_func=lambda chunk: len(chunk),  # Simple processing
            )
        )
        process_time = time.time() - start_time

        assert len(results) == 50  # 50 chunks of 1000 items
        assert sum(results) == 50000  # Total items processed
        assert process_time < 2.0  # Should be reasonably fast

        print(f"Processed 50,000 items in {process_time:.3f}s")

    def test_memory_pool_performance(self):
        """Test performance benefits of memory pool."""

        def create_large_dict():
            return {i: f"value_{i}" for i in range(100)}

        pool = MemoryPool(
            factory_func=create_large_dict, max_size=10, reset_func=lambda d: d.clear()
        )

        # Time object creation without pool
        start_time = time.time()
        objects_no_pool = [create_large_dict() for _ in range(100)]
        no_pool_time = time.time() - start_time

        # Time object creation with pool
        start_time = time.time()
        objects_with_pool = []
        for _ in range(100):
            obj = pool.get()
            objects_with_pool.append(obj)
            pool.put(obj)
        pool_time = time.time() - start_time

        stats = pool.get_stats()

        print(f"Without pool: {no_pool_time:.3f}s")
        print(f"With pool: {pool_time:.3f}s")
        print(f"Pool reuse rate: {stats['reuse_rate']:.1%}")

        # Pool should provide some performance benefit through reuse
        assert stats["reuse_rate"] > 0.8  # Should reuse most objects
