#!/usr/bin/env python3
"""Debug Jarvis initialization step by step."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONWARNINGS'] = 'ignore'

import asyncio
import logging
import time

logging.basicConfig(level=logging.DEBUG)

async def debug_jarvis_init():
    """Debug Jarvis initialization."""
    print("Debugging Jarvis initialization...")
    
    from jarvis2 import Jarvis2, Jarvis2Config
    
    # Minimal config
    config = Jarvis2Config(
        max_parallel_simulations=10,
        gpu_batch_size=4,
        num_diverse_solutions=2,
        index_update_interval=10000
    )
    
    print("\n1. Creating Jarvis instance...")
    start = time.time()
    
    # Manually create components to debug
    from jarvis2.hardware.hardware_optimizer import HardwareAwareExecutor
    from jarvis2.index.index_manager import DeepIndexManager
    from jarvis2.core.jarvis2 import ComplexityEstimator
    from jarvis2.search.mcts import NeuralGuidedMCTS
    from jarvis2.diversity.diversity_engine import DiversityEngine
    from jarvis2.evaluation.evaluator import MultiObjectiveEvaluator
    from jarvis2.experience.experience_buffer import ExperienceReplaySystem
    
    print("   Creating hardware executor...")
    hw = HardwareAwareExecutor()
    print("   ✅ Hardware executor created")
    
    print("   Creating index manager...")
    index = DeepIndexManager(config.index_path)
    print("   ✅ Index manager created")
    
    print("   Creating complexity estimator...")
    complexity = ComplexityEstimator()
    print("   ✅ Complexity estimator created")
    
    print("   Creating MCTS...")
    mcts = NeuralGuidedMCTS(config)
    print("   ✅ MCTS created")
    
    print("   Creating diversity engine...")
    diversity = DiversityEngine()
    print("   ✅ Diversity engine created")
    
    print("   Creating evaluator...")
    evaluator = MultiObjectiveEvaluator()
    print("   ✅ Evaluator created")
    
    print("   Creating experience buffer...")
    exp_buffer = ExperienceReplaySystem(config.experience_path, config.experience_buffer_size)
    print("   ✅ Experience buffer created")
    
    elapsed = time.time() - start
    print(f"\n   All components created in {elapsed:.2f}s")
    
    # Now test actual Jarvis creation
    print("\n2. Creating Jarvis with all components...")
    jarvis = Jarvis2(config)
    print("   ✅ Jarvis instance created")
    
    # Test initialization
    print("\n3. Testing initialization...")
    print("   Creating directories...")
    config.index_path.mkdir(parents=True, exist_ok=True)
    config.model_path.mkdir(parents=True, exist_ok=True)
    config.experience_path.mkdir(parents=True, exist_ok=True)
    print("   ✅ Directories created")
    
    print("   Initializing hardware...")
    await hw.initialize()
    print("   ✅ Hardware initialized")
    
    print("   Initializing index manager...")
    await asyncio.wait_for(index.initialize(), timeout=5.0)
    print("   ✅ Index manager initialized")
    
    print("   Initializing MCTS...")
    await asyncio.wait_for(mcts.initialize(), timeout=5.0)
    print("   ✅ MCTS initialized")
    
    print("   Initializing experience buffer...")
    await asyncio.wait_for(exp_buffer.initialize(), timeout=5.0)
    print("   ✅ Experience buffer initialized")
    
    print("\n✅ All initialization steps completed successfully!")

if __name__ == "__main__":
    asyncio.run(debug_jarvis_init())