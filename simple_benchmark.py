#!/usr/bin/env python3
"""Simple hardware verification."""
import os
import time
import numpy as np

print("✅ Hardware Acceleration Status")
print("="*40)

# Environment check
print("\n📊 Configuration:")
print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
print(f"  JARVIS2_WORKERS: {os.environ.get('JARVIS2_WORKERS', 'not set')}")
print(f"  USE_TURBO_MODE: {os.environ.get('USE_TURBO_MODE', 'not set')}")

# NumPy test - this shows BLAS acceleration
print("\n⚡ Performance Test:")
size = 2000
start = time.time()
a = np.random.rand(size, size)
b = np.random.rand(size, size)
c = np.dot(a, b)
elapsed = time.time() - start
gflops = (2 * size**3) / elapsed / 1e9

print(f"  Matrix multiply: {elapsed:.3f}s = {gflops:.1f} GFLOPS")
print(f"  Status: {'🚀 ACCELERATED' if gflops > 50 else '🐌 SLOW'}")

# Expected on M4 Pro: >100 GFLOPS with acceleration
print(f"\n{'✅ Hardware acceleration ACTIVE!' if gflops > 50 else '❌ Acceleration may not be working'}")