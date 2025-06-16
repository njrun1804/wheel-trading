#!/usr/bin/env python3
"""
Quick validation of parallel processing optimizations.
"""

import asyncio
import json
import logging
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SimpleParallelValidator:
    """Simple validator for parallel processing improvements."""
    
    def __init__(self):
        self.results = {}
    
    def cpu_task(self, n: int) -> float:
        """CPU-intensive task."""
        result = 0.0
        for i in range(n):
            result += math.sin(i) * math.cos(i) + math.sqrt(abs(i))
        return result
    
    def memory_task(self, size_kb: int) -> int:
        """Memory-intensive task."""
        data = bytearray(size_kb * 1024)
        for i in range(0, len(data), 100):
            data[i:i+10] = bytes(range(10))
        return sum(data[::100])
    
    async def test_thread_parallelism(self) -> dict:
        """Test thread-based parallelism."""
        logger.info("Testing thread parallelism...")
        
        tasks = [10000 + random.randint(0, 5000) for _ in range(16)]
        
        # Serial execution
        start = time.perf_counter()
        serial_results = [self.cpu_task(n) for n in tasks]
        serial_time = time.perf_counter() - start
        
        # Parallel execution with ThreadPoolExecutor
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as executor:
            parallel_results = list(executor.map(self.cpu_task, tasks))
        parallel_time = time.perf_counter() - start
        
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        
        return {
            'test': 'thread_parallelism',
            'tasks': len(tasks),
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'target_met': speedup >= 3.0
        }
    
    async def test_process_parallelism(self) -> dict:
        """Test process-based parallelism."""
        logger.info("Testing process parallelism...")
        
        tasks = [5000 + random.randint(0, 2000) for _ in range(12)]
        
        # Serial execution
        start = time.perf_counter()
        serial_results = [self.cpu_task(n) for n in tasks]
        serial_time = time.perf_counter() - start
        
        # Parallel execution with ProcessPoolExecutor
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=8) as executor:
            parallel_results = list(executor.map(self.cpu_task, tasks))
        parallel_time = time.perf_counter() - start
        
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        
        return {
            'test': 'process_parallelism',
            'tasks': len(tasks),
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'target_met': speedup >= 4.0
        }
    
    async def test_memory_parallel(self) -> dict:
        """Test memory-intensive parallel processing."""
        logger.info("Testing memory parallelism...")
        
        tasks = [100 + random.randint(0, 50) for _ in range(10)]  # KB sizes
        
        # Serial execution
        start = time.perf_counter()
        serial_results = [self.memory_task(size) for size in tasks]
        serial_time = time.perf_counter() - start
        
        # Parallel execution
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=6) as executor:
            parallel_results = list(executor.map(self.memory_task, tasks))
        parallel_time = time.perf_counter() - start
        
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        
        return {
            'test': 'memory_parallelism',
            'tasks': len(tasks),
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'target_met': speedup >= 2.0
        }
    
    async def run_validation(self) -> dict:
        """Run all validation tests."""
        logger.info("🚀 Starting parallel processing validation...")
        
        tests = [
            self.test_thread_parallelism,
            self.test_process_parallelism,
            self.test_memory_parallel
        ]
        
        results = []
        for test in tests:
            try:
                result = await test()
                results.append(result)
                logger.info(f"   {result['test']}: {result['speedup']:.2f}x speedup "
                           f"({'✅' if result['target_met'] else '❌'})")
            except Exception as e:
                logger.error(f"Test {test.__name__} failed: {e}")
                results.append({'test': test.__name__, 'error': str(e)})
        
        # Calculate summary
        speedups = [r['speedup'] for r in results if 'speedup' in r]
        targets_met = sum(1 for r in results if r.get('target_met', False))
        
        summary = {
            'validation_results': results,
            'summary': {
                'total_tests': len(results),
                'targets_met': targets_met,
                'success_rate': (targets_met / len(results)) * 100 if results else 0,
                'avg_speedup': statistics.mean(speedups) if speedups else 0,
                'max_speedup': max(speedups) if speedups else 0,
                'overall_success': statistics.mean(speedups) >= 3.0 if speedups else False
            }
        }
        
        return summary

async def main():
    """Main validation function."""
    validator = SimpleParallelValidator()
    results = await validator.run_validation()
    
    # Save results
    with open('parallel_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    summary = results['summary']
    logger.info("=" * 50)
    logger.info("🏁 VALIDATION SUMMARY:")
    logger.info(f"   Tests: {summary['total_tests']}")
    logger.info(f"   Targets Met: {summary['targets_met']}")
    logger.info(f"   Success Rate: {summary['success_rate']:.1f}%")
    logger.info(f"   Average Speedup: {summary['avg_speedup']:.2f}x")
    logger.info(f"   Maximum Speedup: {summary['max_speedup']:.2f}x")
    logger.info(f"   Overall Success: {'✅' if summary['overall_success'] else '❌'}")
    
    return summary['overall_success']

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)