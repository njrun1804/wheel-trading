#!/usr/bin/env python3
"""Debug MCTS initialization specifically."""
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

async def debug_mcts_init():
    """Debug MCTS initialization step by step."""
    
    print("Starting MCTS debug...\n")
    
    # Test 1: Import modules
    print("1. Testing imports...")
    try:
        from jarvis2.search.mcts import NeuralGuidedMCTS, MCTSConfig
        print("   ✅ Import successful\n")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        traceback.print_exc()
        return
    
    # Test 2: Create config
    print("2. Creating config...")
    try:
        config = MCTSConfig()
        print(f"   ✅ Config created: batch_size={config.batch_size}\n")
    except Exception as e:
        print(f"   ❌ Config creation failed: {e}")
        traceback.print_exc()
        return
    
    # Test 3: Create MCTS instance
    print("3. Creating MCTS instance...")
    try:
        mcts = NeuralGuidedMCTS(config)
        print("   ✅ MCTS instance created")
        print(f"   - Value net: {type(mcts.value_net).__name__}")
        print(f"   - Policy net: {type(mcts.policy_net).__name__}\n")
    except Exception as e:
        print(f"   ❌ MCTS creation failed: {e}")
        traceback.print_exc()
        return
    
    # Test 4: Test neural network creation separately
    print("4. Testing neural network creation...")
    try:
        from jarvis2.neural.value_network import CodeValueNetwork
        from jarvis2.neural.policy_network import CodePolicyNetwork
        
        print("   Creating value network...")
        value_net = CodeValueNetwork(hidden_dim=256, num_layers=2)
        print("   ✅ Value network created")
        
        print("   Creating policy network...")
        policy_net = CodePolicyNetwork(hidden_dim=256, num_layers=2)
        print("   ✅ Policy network created\n")
    except Exception as e:
        print(f"   ❌ Neural network creation failed: {e}")
        traceback.print_exc()
        return
    
    # Test 5: Test warmup directly
    print("5. Testing model warmup...")
    try:
        import torch
        
        # Test value network forward pass
        print("   Testing value network forward pass...")
        dummy_input = torch.randn(4, 768)
        with torch.no_grad():
            value_output = mcts.value_net(dummy_input)
        print(f"   ✅ Value output shape: {value_output.shape}")
        
        # Test policy network forward pass  
        print("   Testing policy network forward pass...")
        with torch.no_grad():
            policy_output = mcts.policy_net(dummy_input)
        print(f"   ✅ Policy output shape: {policy_output.shape}\n")
    except Exception as e:
        print(f"   ❌ Model warmup failed: {e}")
        traceback.print_exc()
        return
    
    # Test 6: Call actual warmup method
    print("6. Calling _warmup_models...")
    try:
        await mcts._warmup_models()
        print("   ✅ Warmup completed\n")
    except Exception as e:
        print(f"   ❌ Warmup failed: {e}")
        traceback.print_exc()
        return
    
    # Test 7: Full initialization
    print("7. Testing full MCTS initialization...")
    try:
        # Create fresh instance
        mcts2 = NeuralGuidedMCTS(config)
        
        # Time the initialization
        import time
        start = time.time()
        await asyncio.wait_for(mcts2.initialize(), timeout=5.0)
        elapsed = time.time() - start
        
        print(f"   ✅ MCTS initialized in {elapsed:.2f}s\n")
    except asyncio.TimeoutError:
        print("   ❌ MCTS initialization timed out!")
    except Exception as e:
        print(f"   ❌ MCTS initialization failed: {e}")
        traceback.print_exc()
        return
    
    print("✅ All MCTS tests passed!")

if __name__ == "__main__":
    asyncio.run(debug_mcts_init())