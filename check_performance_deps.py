#!/usr/bin/env python3
"""Check available performance dependencies."""

import subprocess
import sys
import importlib

print("System Information:")
print("=" * 50)
print(f"Python: {sys.version}")

# Check installed packages
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)

# Performance packages to check
perf_packages = {
    'Core Computation': ['numpy', 'scipy', 'numba', 'cython'],
    'ML Frameworks': ['mlx', 'torch', 'tensorflow', 'jax'],
    'Parallel Processing': ['ray', 'dask', 'joblib', 'multiprocess'],
    'Async/Network': ['uvloop', 'httpx', 'aiohttp', 'grpcio'],
    'Serialization': ['orjson', 'msgpack', 'pyarrow', 'lz4'],
    'GPU/Acceleration': ['cupy', 'pycuda', 'accelerate', 'onnx'],
    'Memory/Caching': ['blosc', 'redis', 'diskcache', 'lmdb']
}

installed = {}
for line in result.stdout.split('\n'):
    parts = line.split()
    if parts:
        installed[parts[0].lower()] = parts[1] if len(parts) > 1 else ''

print("\nInstalled Performance Packages:")
print("=" * 50)

for category, packages in perf_packages.items():
    print(f"\n{category}:")
    found = False
    for pkg in packages:
        if pkg in installed:
            print(f"  ✓ {pkg} {installed[pkg]}")
            found = True
    if not found:
        print("  ✗ None found")

# Check what we can actually import
print("\n\nImportable Modules:")
print("=" * 50)

test_imports = [
    'numpy', 'mlx', 'torch', 'numba', 'joblib', 'multiprocessing',
    'asyncio', 'concurrent.futures', 'cython', 'psutil'
]

for module in test_imports:
    try:
        importlib.import_module(module)
        print(f"✓ {module}")
    except ImportError:
        print(f"✗ {module}")

# Check Metal/GPU availability
print("\n\nGPU/Metal Support:")
print("=" * 50)

try:
    import mlx.core as mx
    print(f"✓ MLX available")
    if hasattr(mx, 'metal') and mx.metal.is_available():
        print(f"✓ Metal GPU available")
        device_info = mx.default_device()
        print(f"  Default device: {device_info}")
except:
    print("✗ MLX not available")

# Check system limits
print("\n\nSystem Limits:")
print("=" * 50)
import resource

limits = {
    'RLIMIT_NOFILE': resource.RLIMIT_NOFILE,
    'RLIMIT_NPROC': resource.RLIMIT_NPROC,
    'RLIMIT_AS': resource.RLIMIT_AS,
}

for name, limit in limits.items():
    soft, hard = resource.getrlimit(limit)
    print(f"{name}: soft={soft}, hard={hard}")

# CPU features
print("\n\nCPU Features:")
print("=" * 50)
result = subprocess.run(['sysctl', '-a'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if any(x in line for x in ['hw.perflevel', 'hw.ncpu', 'hw.memsize', 'hw.cpufrequency']):
        print(line)

# Recommendations
print("\n\nPerformance Optimization Opportunities:")
print("=" * 50)