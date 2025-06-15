#!/usr/bin/env python3
"""Benchmark sequential thinking implementations."""

import asyncio
import time
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.unity_wheel.accelerated_tools.sequential_thinking_turbo import get_sequential_thinking
from src.unity_wheel.accelerated_tools.sequential_thinking_turbo_v2 import get_sequential_thinking_v2


async def benchmark_implementation(name: str, thinking_engine, test_cases: list):
    """Benchmark a thinking implementation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    results = []
    
    for test_name, goal, constraints, strategy in test_cases:
        print(f"\nTest: {test_name}")
        
        times = []
        steps_counts = []
        
        # Run 3 iterations
        for i in range(3):
            start = time.perf_counter()
            
            steps = await thinking_engine.think(
                goal=goal,
                constraints=constraints,
                strategy=strategy,
                max_steps=20,
                timeout=30.0
            )
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            steps_counts.append(len(steps))
            
            print(f"  Run {i+1}: {elapsed:.3f}s, {len(steps)} steps")
        
        avg_time = statistics.mean(times)
        avg_steps = statistics.mean(steps_counts)
        
        print(f"  Average: {avg_time:.3f}s, {avg_steps:.1f} steps")
        print(f"  Steps/second: {avg_steps/avg_time:.1f}")
        
        results.append({
            'test': test_name,
            'avg_time': avg_time,
            'avg_steps': avg_steps,
            'steps_per_sec': avg_steps/avg_time
        })
    
    # Get final stats
    stats = thinking_engine.get_stats()
    
    return results, stats


async def main():
    """Run comprehensive benchmark."""
    
    # Test cases
    test_cases = [
        ("Simple Task", 
         "Fix a typo in the documentation",
         ["Must be grammatically correct", "Preserve formatting"],
         "parallel_explore"),
         
        ("Medium Complexity",
         "Refactor a function for better performance",
         ["Maintain functionality", "Improve readability", "Use type hints"],
         "parallel_explore"),
         
        ("Complex Task",
         "Design a distributed caching system",
         ["Must scale to 1M requests/sec", "Fault tolerant", "Low latency", "Cost effective"],
         "parallel_explore"),
         
        ("GPU-Heavy Task",
         "Implement parallel matrix operations",
         ["Use all GPU cores", "Minimize memory transfers", "Support batching"],
         "gpu_beam_search" if hasattr(get_sequential_thinking_v2(), 'strategies') else "parallel_explore"),
         
        ("Creative Task",
         "Invent a new algorithm for option pricing",
         ["Must be faster than Black-Scholes", "Handle edge cases", "Mathematically sound"],
         "quantum_inspired" if hasattr(get_sequential_thinking_v2(), 'strategies') else "parallel_explore")
    ]
    
    # Benchmark V1
    v1_engine = get_sequential_thinking()
    v1_results, v1_stats = await benchmark_implementation(
        "Sequential Thinking V1 (Original)", 
        v1_engine, 
        test_cases[:3]  # V1 doesn't have all strategies
    )
    
    # Benchmark V2
    v2_engine = get_sequential_thinking_v2()
    v2_results, v2_stats = await benchmark_implementation(
        "Sequential Thinking V2 (Ultra-Optimized)",
        v2_engine,
        test_cases
    )
    
    # Comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    # Compare same tests
    print("\nSpeed Improvements (V2 vs V1):")
    for v1_result in v1_results:
        test_name = v1_result['test']
        v2_result = next((r for r in v2_results if r['test'] == test_name), None)
        
        if v2_result:
            speedup = v2_result['steps_per_sec'] / v1_result['steps_per_sec']
            time_reduction = (v1_result['avg_time'] - v2_result['avg_time']) / v1_result['avg_time'] * 100
            
            print(f"\n{test_name}:")
            print(f"  V1: {v1_result['avg_time']:.3f}s ({v1_result['steps_per_sec']:.1f} steps/s)")
            print(f"  V2: {v2_result['avg_time']:.3f}s ({v2_result['steps_per_sec']:.1f} steps/s)")
            print(f"  Speedup: {speedup:.2f}x faster")
            print(f"  Time reduction: {time_reduction:.1f}%")
    
    # Hardware utilization
    print(f"\n{'='*60}")
    print("HARDWARE UTILIZATION")
    print(f"{'='*60}")
    
    print("\nV1 Stats:")
    for key, value in v1_stats.items():
        print(f"  {key}: {value}")
    
    print("\nV2 Stats:")
    for key, value in v2_stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # V2-only features
    print(f"\n{'='*60}")
    print("V2 EXCLUSIVE FEATURES")
    print(f"{'='*60}")
    
    print("\n✓ Numba JIT compilation for numerical operations")
    print("✓ PyTorch MPS (Metal Performance Shaders) support")
    print("✓ LMDB caching for repeated queries")
    print("✓ Differential evolution for path optimization")
    print("✓ Quantum-inspired strategy superposition")
    print("✓ Dedicated P-core/E-core task distribution")
    print("✓ Advanced serialization (msgpack/orjson)")
    
    # Memory efficiency
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"\nCurrent memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    
    # Cleanup
    v1_engine.close()
    v2_engine.close()


if __name__ == "__main__":
    # Optimal performance settings
    import os
    os.environ['MKL_NUM_THREADS'] = '8'
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['NUMBA_NUM_THREADS'] = '8'
    
    asyncio.run(main())