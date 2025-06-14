#!/usr/bin/env python3
"""Debug the initialization to find the exact issue."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import asyncio
import logging
import sys
import traceback

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def debug_init():
    """Debug initialization step by step."""
    
    print("Starting debug initialization...\n")
    
    # Test 1: Hardware executor with monitoring
    print("1. Testing HardwareAwareExecutor...")
    try:
        from jarvis2.hardware.hardware_optimizer import HardwareAwareExecutor
        hw = HardwareAwareExecutor()
        
        # The issue might be in _monitor_resources
        print("   Creating monitoring task...")
        
        # Manually create and test the monitoring task
        async def test_monitor():
            """Test monitoring."""
            import psutil
            for i in range(3):
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                print(f"   Monitor {i}: CPU={cpu}%, Memory={mem.percent}%")
                await asyncio.sleep(0.5)
        
        # Run with timeout
        await asyncio.wait_for(test_monitor(), timeout=3.0)
        print("   ✅ Monitoring works\n")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return
    
    # Test 2: MCTS initialization
    print("2. Testing MCTS initialization...")
    try:
        from jarvis2.search.mcts import NeuralGuidedMCTS, MCTSConfig
        
        config = MCTSConfig()
        mcts = NeuralGuidedMCTS(config)
        
        # Test model initialization
        print("   Testing neural network creation...")
        print(f"   Value net: {mcts.value_net}")
        print(f"   Policy net: {mcts.policy_net}")
        
        # Test warmup
        print("   Testing model warmup...")
        await mcts._warmup_models()
        print("   ✅ MCTS initialized\n")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return
    
    # Test 3: Full Jarvis initialization
    print("3. Testing full Jarvis initialization...")
    try:
        from jarvis2 import Jarvis2, Jarvis2Config
        
        config = Jarvis2Config(
            max_parallel_simulations=10,
            gpu_batch_size=4,
            num_diverse_solutions=2
        )
        
        jarvis = Jarvis2(config)
        
        # Test the actual initialization
        print("   Calling jarvis.initialize()...")
        await jarvis.initialize()
        
        print("   ✅ Jarvis initialized successfully!")
        
        # Cleanup
        await jarvis.shutdown()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(debug_init())