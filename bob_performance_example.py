#!/usr/bin/env python3
"""Example of BOB using performance optimization for fast command execution.

Shows how BOB can leverage M4 Pro hardware for:
- <500ms simple commands
- <2s complex workflows  
- Parallel execution
- Memory-efficient caching
"""

import asyncio
import time
from typing import List, Dict, Any

from bob.performance import PerformanceOptimizer


class BobWithPerformance:
    """Example BOB implementation with performance optimization."""
    
    def __init__(self):
        """Initialize BOB with performance optimization."""
        # Use realtime profile for fast response
        self.optimizer = PerformanceOptimizer(profile='realtime')
        
        # Create caches for common operations
        self.search_cache = self.optimizer.memory.create_cache(
            'search_results', 
            max_size=1000,
            max_memory_mb=100
        )
        
        self.file_cache = self.optimizer.memory.create_cache(
            'file_contents',
            max_size=500,
            max_memory_mb=200
        )
        
    @property
    def metrics(self):
        """Access performance metrics."""
        return self.optimizer.metrics
        
    async def process_simple_command(self, command: str) -> str:
        """Process a simple command with <500ms target."""
        with self.metrics.measure('simple_command'):
            # Check cache first
            cached = self.search_cache.get(command)
            if cached:
                return cached
                
            # Simulate command processing
            await asyncio.sleep(0.1)  # 100ms processing
            
            result = f"Processed: {command}"
            
            # Cache result
            self.search_cache.put(command, result, size_bytes=len(result))
            
            return result
            
    async def process_complex_workflow(self, tasks: List[str]) -> List[str]:
        """Process complex workflow with <2s target."""
        with self.metrics.measure('complex_workflow'):
            # Optimize for the workflow
            self.optimizer.optimize_for_operation('analysis')
            
            # Process tasks in parallel
            results = await self.optimizer.execute_async(
                self.process_simple_command,
                tasks
            )
            
            return results
            
    def search_codebase(self, query: str, files: List[str]) -> List[Dict[str, Any]]:
        """Search across codebase with parallel execution."""
        with self.metrics.measure('search_operation'):
            # Check cache
            cache_key = f"{query}:{len(files)}"
            cached = self.search_cache.get(cache_key)
            if cached:
                return cached
                
            # Search function
            def search_file(filepath: str) -> Dict[str, Any]:
                # Simulate file search
                time.sleep(0.01)  # 10ms per file
                return {
                    'file': filepath,
                    'matches': [f"Line containing {query}"],
                    'score': 0.95
                }
                
            # Execute in parallel
            results = self.optimizer.execute_parallel(
                search_file,
                files,
                use_processes=False  # I/O bound
            )
            
            # Filter None results
            results = [r for r in results if r is not None]
            
            # Cache results
            self.search_cache.put(cache_key, results, size_bytes=len(str(results)))
            
            return results
            
    def analyze_code(self, files: List[str]) -> Dict[str, Any]:
        """Analyze code files with GPU acceleration if available."""
        with self.metrics.measure('code_analysis'):
            # Switch to throughput profile for analysis
            self.optimizer.switch_profile('throughput')
            
            def analyze_file(filepath: str) -> Dict[str, Any]:
                # Simulate code analysis
                time.sleep(0.05)  # 50ms per file
                return {
                    'file': filepath,
                    'complexity': 42,
                    'issues': [],
                    'suggestions': ['Consider refactoring']
                }
                
            # Use processes for CPU-intensive analysis
            results = self.optimizer.execute_parallel(
                analyze_file,
                files,
                use_processes=True
            )
            
            # Aggregate results
            analysis = {
                'total_files': len(files),
                'analyzed': len([r for r in results if r]),
                'total_complexity': sum(r['complexity'] for r in results if r),
                'all_issues': []
            }
            
            # Switch back to realtime
            self.optimizer.switch_profile('realtime')
            
            return analysis
            
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        summary = self.metrics.get_summary()
        status = self.optimizer.get_status()
        
        return {
            'performance': summary,
            'hardware': status['hardware'],
            'memory': status['memory'],
            'caches': {
                'search_cache': self.search_cache.get_stats(),
                'file_cache': self.file_cache.get_stats()
            }
        }


async def demo_bob_performance():
    """Demonstrate BOB with performance optimization."""
    print("BOB Performance Optimization Demo")
    print("=================================\n")
    
    bob = BobWithPerformance()
    
    # Test simple commands
    print("Testing simple commands (<500ms target)...")
    start = time.time()
    
    for i in range(10):
        result = await bob.process_simple_command(f"command_{i}")
        
    simple_time = time.time() - start
    print(f"10 simple commands: {simple_time:.2f}s ({simple_time/10*1000:.0f}ms average)")
    
    # Test complex workflow
    print("\nTesting complex workflow (<2s target)...")
    tasks = [f"task_{i}" for i in range(20)]
    
    start = time.time()
    results = await bob.process_complex_workflow(tasks)
    workflow_time = time.time() - start
    
    print(f"Complex workflow (20 tasks): {workflow_time:.2f}s")
    
    # Test codebase search
    print("\nTesting codebase search...")
    files = [f"src/file_{i}.py" for i in range(100)]
    
    start = time.time()
    search_results = bob.search_codebase("TODO", files)
    search_time = time.time() - start
    
    print(f"Search across 100 files: {search_time:.2f}s ({search_time*10:.0f}ms per 10 files)")
    
    # Test code analysis
    print("\nTesting code analysis...")
    analysis_files = files[:50]  # Analyze 50 files
    
    start = time.time()
    analysis = bob.analyze_code(analysis_files)
    analysis_time = time.time() - start
    
    print(f"Analyzed {analysis['analyzed']} files: {analysis_time:.2f}s")
    print(f"Total complexity: {analysis['total_complexity']}")
    
    # Show performance report
    print("\n=== Performance Report ===")
    report = bob.get_performance_report()
    
    # Performance metrics
    perf_status = report['performance']['performance_status']
    print(f"\nPerformance Status:")
    print(f"  ✓ Good: {perf_status['good']}")
    print(f"  ⚠ Warning: {perf_status['warning']}")  
    print(f"  ✗ Critical: {perf_status['critical']}")
    
    # Cache effectiveness
    search_cache = report['caches']['search_cache']
    print(f"\nCache Performance:")
    print(f"  Search cache: {search_cache['hit_rate']:.1%} hit rate, "
          f"{search_cache['size']} items")
    
    # Hardware utilization
    hw = report['hardware']
    print(f"\nHardware Utilization:")
    print(f"  CPU: {hw['cpu']['usage_percent']:.1f}%")
    print(f"  Memory: {hw['memory']['percent']:.1f}%")
    
    # Check if targets are met
    print("\n=== Target Achievement ===")
    
    # Check simple command performance
    simple_stats = bob.metrics.collector.get_stats('simple_command')
    if simple_stats:
        print(f"Simple commands: {simple_stats['mean_ms']:.0f}ms average "
              f"({'✓ PASS' if simple_stats['mean_ms'] < 500 else '✗ FAIL'})")
        
    # Check complex workflow performance  
    workflow_stats = bob.metrics.collector.get_stats('complex_workflow')
    if workflow_stats:
        print(f"Complex workflows: {workflow_stats['mean_ms']:.0f}ms average "
              f"({'✓ PASS' if workflow_stats['mean_ms'] < 2000 else '✗ FAIL'})")
        
    # Cleanup
    bob.optimizer.shutdown()


if __name__ == '__main__':
    asyncio.run(demo_bob_performance())