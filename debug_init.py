#!/usr/bin/env python3
"""Debug initialization issues."""
import asyncio
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

async def test():
    print("1. Testing hardware init...")
    from jarvis2.hardware.hardware_optimizer import HardwareAwareExecutor
    hw = HardwareAwareExecutor()
    await hw.initialize()
    print("✅ Hardware initialized")
    
    print("\n2. Testing process pool...")
    # The issue might be with ProcessPoolExecutor
    from concurrent.futures import ProcessPoolExecutor
    executor = ProcessPoolExecutor(max_workers=2)
    
    # Test simple task
    def simple_task(x):
        return x * 2
    
    future = executor.submit(simple_task, 5)
    result = future.result(timeout=5)
    print(f"✅ Process pool works: {result}")
    
    executor.shutdown()
    
    print("\n3. All tests passed!")

if __name__ == "__main__":
    asyncio.run(test())