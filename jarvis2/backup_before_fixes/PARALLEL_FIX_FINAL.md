# Final Fix for Parallel Request Hanging

## Root Cause Analysis

The parallel request hanging is caused by a **synchronization bottleneck** in the orchestrator:

1. Each request waits for neural guidance BEFORE starting search
2. With only 2 neural workers and 2+ parallel requests, this creates contention
3. The search workers sit idle while waiting for neural guidance
4. This serializes what should be parallel operations

## The Fix

Change the orchestrator to start neural guidance and search in parallel, then combine results:

```python
# Current (problematic):
guidance = await self._get_neural_guidance(...)  # BLOCKS
search_result = await self.search_pool.parallel_search(..., guidance, ...)

# Fixed (parallel):
# Start both in parallel
guidance_task = asyncio.create_task(self._get_neural_guidance(...))

# Use default guidance for search to start immediately
default_guidance = {
    'value': np.array([[0.5]]),
    'policy': np.ones(50) / 50
}
search_task = asyncio.create_task(
    self.search_pool.parallel_search(..., default_guidance, ...)
)

# Wait for both
guidance, search_result = await asyncio.gather(guidance_task, search_task)

# Update search result with actual guidance if needed
```

## Why This Works

1. **No blocking**: Search starts immediately with default guidance
2. **True parallelism**: Neural and search workers run concurrently
3. **Resource utilization**: All workers stay busy
4. **Scales better**: N requests can progress simultaneously

## Alternative Solutions

### Option 1: Increase Worker Pool Sizes
```python
self.config.num_neural_workers = 4  # More workers
self.config.num_search_workers = 8
```

### Option 2: Use Process Pool Executor
Replace custom worker pools with Python's built-in:
```python
from concurrent.futures import ProcessPoolExecutor

async def run_in_process(func, *args):
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=4) as executor:
        return await loop.run_in_executor(executor, func, *args)
```

### Option 3: Queue-Based Architecture
Instead of round-robin, use a single queue that all workers pull from:
```python
class WorkerPool:
    def __init__(self, num_workers):
        self.task_queue = asyncio.Queue()
        self.workers = [
            asyncio.create_task(self._worker())
            for _ in range(num_workers)
        ]
    
    async def _worker(self):
        while True:
            task = await self.task_queue.get()
            result = await self._process(task)
            task.future.set_result(result)
```

## Testing the Fix

```python
# Test should show true parallelism
async def test_parallel():
    jarvis = Jarvis2Orchestrator()
    await jarvis.initialize()
    
    # Time N parallel requests
    start = time.time()
    requests = [CodeRequest(f"Test {i}") for i in range(5)]
    results = await asyncio.gather(*[
        jarvis.generate_code(req) for req in requests
    ])
    elapsed = time.time() - start
    
    # Should be much less than 5x single request time
    assert elapsed < single_request_time * 2
```

## Lessons Learned

1. **Don't serialize parallel operations**: Start everything that can run in parallel
2. **Default values enable parallelism**: Use reasonable defaults to avoid blocking
3. **Worker pool sizing matters**: Need enough workers to handle parallel load
4. **macOS spawn overhead**: Factor in 1-2s per worker startup time
5. **Queue monitoring**: Can't use qsize() on macOS, need alternatives

This fix should resolve the parallel hanging issue once and for all.