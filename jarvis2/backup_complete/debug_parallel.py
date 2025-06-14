"""Debug parallel request issue."""
import asyncio
import os
import time
import logging

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from jarvis2.core.orchestrator import Jarvis2Orchestrator, CodeRequest


async def test_minimal():
    """Minimal test to isolate the issue."""
    print("=== Minimal Parallel Test ===\n")
    
    jarvis = Jarvis2Orchestrator()
    
    # Reduce workers to minimum
    jarvis.config.num_neural_workers = 1
    jarvis.config.num_search_workers = 2
    jarvis.config.default_simulations = 10  # Very few simulations
    
    try:
        print("1. Initializing with minimal config...")
        await jarvis.initialize()
        print("   ✅ Initialized\n")
        
        print("2. Testing single request...")
        start = time.time()
        req1 = CodeRequest("Test 1")
        sol1 = await jarvis.generate_code(req1)
        t1 = time.time() - start
        print(f"   ✅ Single request: {t1*1000:.0f}ms\n")
        
        print("3. Testing 2 parallel requests...")
        start = time.time()
        
        req2 = CodeRequest("Test 2")
        req3 = CodeRequest("Test 3")
        
        # Create tasks
        task2 = asyncio.create_task(jarvis.generate_code(req2))
        task3 = asyncio.create_task(jarvis.generate_code(req3))
        
        print("   Tasks created, waiting...")
        
        # Wait with timeout
        try:
            sol2, sol3 = await asyncio.wait_for(
                asyncio.gather(task2, task3),
                timeout=30.0
            )
            t2 = time.time() - start
            print(f"   ✅ Parallel requests: {t2*1000:.0f}ms")
            print(f"   Speedup: {(t1*2)/t2:.1f}x\n")
            
        except asyncio.TimeoutError:
            print("   ❌ TIMEOUT after 30s")
            
            # Check task status
            print(f"   Task 2 done: {task2.done()}")
            print(f"   Task 3 done: {task3.done()}")
            
            # Cancel tasks
            task2.cancel()
            task3.cancel()
            
            # Try to get any partial results
            try:
                await asyncio.gather(task2, task3, return_exceptions=True)
            except:
                pass
                
        print("\n4. Checking worker status...")
        print(f"   Neural workers alive: {all(w.process.is_alive() for w in jarvis.neural_pool.workers)}")
        print(f"   Search workers alive: {all(w.process.is_alive() for w in jarvis.search_pool.workers)}")
        
    finally:
        print("\n5. Shutting down...")
        await jarvis.shutdown()
        print("   ✅ Complete")


async def test_components():
    """Test individual components."""
    print("\n=== Component Test ===\n")
    
    # Test neural worker alone
    from jarvis2.workers.neural_worker import NeuralWorkerPool
    import numpy as np
    
    print("1. Testing neural workers alone...")
    pool = NeuralWorkerPool(num_workers=2)
    
    try:
        # Single request
        data = np.random.randn(100, 768).astype(np.float32)
        result = await pool.compute_async('value', data)
        print(f"   ✅ Single neural compute: {result.shape}")
        
        # Parallel
        tasks = [
            pool.compute_async('value', data),
            pool.compute_async('policy', data)
        ]
        results = await asyncio.gather(*tasks)
        print(f"   ✅ Parallel neural compute: {len(results)} results")
        
    finally:
        pool.shutdown()
        
    # Test search worker alone
    print("\n2. Testing search workers alone...")
    from jarvis2.workers.search_worker import SearchWorkerPool
    
    pool = SearchWorkerPool(num_workers=2)
    try:
        result = await pool.parallel_search(
            "test query",
            {},
            {'value': np.array([[0.5]]), 'policy': np.ones(50)/50},
            simulations=20
        )
        print(f"   ✅ Search complete: {result['stats']['total_simulations']} simulations")
    finally:
        pool.shutdown()


async def main():
    """Run debug tests."""
    print("Debugging Jarvis2 Parallel Request Issue")
    print("="*50 + "\n")
    
    # Test minimal case
    await test_minimal()
    
    # Test components separately
    await test_components()
    
    print("\n" + "="*50)
    print("Debug complete")


if __name__ == "__main__":
    asyncio.run(main())