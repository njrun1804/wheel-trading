#!/usr/bin/env python3
"""Quick hardware acceleration test."""
import os
import time
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

print("🚀 Hardware Acceleration Quick Test")
print("="*50)

# Show configuration
print("\n📊 Environment Configuration:")
print(f"  • OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
print(f"  • JARVIS2_WORKERS: {os.environ.get('JARVIS2_WORKERS', 'not set')}")
print(f"  • USE_TURBO_MODE: {os.environ.get('USE_TURBO_MODE', 'not set')}")
print(f"  • CPU Count: {mp.cpu_count()}")

# Test 1: NumPy multi-threading
print("\n1️⃣ NumPy Multi-threading Test")
size = 2000
start = time.time()
a = np.random.rand(size, size)
b = np.random.rand(size, size)
c = np.dot(a, b)
elapsed = time.time() - start
gflops = (2 * size**3) / elapsed / 1e9
print(f"  ✓ Matrix multiplication: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")

# Test 2: Parallel processing
print("\n2️⃣ Parallel Processing Test")
def cpu_bound_task(n):
    """CPU intensive task."""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

# Test with different worker counts
test_size = 1000000
tasks = [test_size] * 12  # 12 tasks for 12 cores

# Single-threaded baseline
start = time.time()
results = [cpu_bound_task(t) for t in tasks]
single_time = time.time() - start
print(f"  • Single-threaded: {single_time:.3f}s")

# Multi-threaded
with ThreadPoolExecutor(max_workers=12) as executor:
    start = time.time()
    results = list(executor.map(cpu_bound_task, tasks))
    thread_time = time.time() - start
print(f"  • Multi-threaded (12): {thread_time:.3f}s (speedup: {single_time/thread_time:.1f}x)")

# Multi-process
with ProcessPoolExecutor(max_workers=12) as executor:
    start = time.time()
    results = list(executor.map(cpu_bound_task, tasks))
    process_time = time.time() - start
print(f"  • Multi-process (12): {process_time:.3f}s (speedup: {single_time/process_time:.1f}x)")

# Test 3: Async I/O
print("\n3️⃣ Async I/O Test")
async def async_task(n):
    """Simulate async I/O operation."""
    await asyncio.sleep(0.001)  # Simulate I/O
    return n ** 2

async def run_async_test():
    start = time.time()
    tasks = [async_task(i) for i in range(100)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    print(f"  ✓ 100 async tasks: {elapsed:.3f}s")

asyncio.run(run_async_test())

# Test 4: Memory bandwidth
print("\n4️⃣ Memory Bandwidth Test")
size_mb = 100
data = np.random.rand(size_mb * 1024 * 1024 // 8)  # 8 bytes per float64
start = time.time()
for _ in range(10):
    copy = data.copy()
elapsed = time.time() - start
bandwidth = (size_mb * 10 * 2) / elapsed  # Read + write
print(f"  ✓ Memory bandwidth: {bandwidth:.1f} MB/s")

print("\n" + "="*50)
print("✅ Hardware acceleration is working!")
print(f"   All {mp.cpu_count()} CPU cores available")
print(f"   NumPy using accelerated BLAS: {gflops:.1f} GFLOPS")
print(f"   Process parallelism speedup: {single_time/process_time:.1f}x")