#!/usr/bin/env python3
"""
Example: Using ANE-accelerated Einstein pipeline.

This example shows how to use the ANE-accelerated Einstein embedding pipeline
for maximum performance on M4 Pro systems.
"""

import asyncio

from src.unity_wheel.accelerated_tools.einstein_neural_integration import (
    EinsteinEmbeddingConfig,
    get_einstein_ane_pipeline,
)


async def main():
    """Example usage of ANE-accelerated Einstein pipeline."""

    # Configure ANE acceleration
    config = EinsteinEmbeddingConfig(
        use_ane=True,  # Enable ANE acceleration
        fallback_on_error=True,  # Graceful fallback to CPU
        max_batch_size=256,  # Optimal for ANE
        cache_embeddings=True,  # Cache for better performance
        performance_logging=True,  # Monitor performance
        warmup_on_startup=True,  # Warm up ANE on startup
    )

    # Get ANE-accelerated pipeline
    pipeline = get_einstein_ane_pipeline(config=config)

    # Example: Embed code files
    code_files = [
        "src/unity_wheel/strategy/wheel.py",
        "src/unity_wheel/analytics/decision_engine.py",
        "src/unity_wheel/math/options.py",
    ]

    print("🚀 Embedding code files with ANE acceleration...")
    await pipeline.embed_file_batch(code_files)

    print(f"✅ Embedded {len(code_files)} files")

    # Get enhanced statistics
    stats = pipeline.get_enhanced_stats()
    print("\n📊 Performance Statistics:")
    print(f"Files processed: {stats['pipeline_stats']['files_processed']}")
    print(f"ANE accelerated: {stats['pipeline_stats']['ane_accelerated']}")
    print(f"Cache hits: {stats['pipeline_stats']['cache_hits']}")

    if stats["performance_comparison"]["ane_calls"] > 0:
        perf = stats["performance_comparison"]
        print(f"ANE usage: {perf['ane_usage_percent']:.1f}%")
        print(f"Speedup: {perf['speedup_factor']:.1f}x")
        print(f"ANE tokens/sec: {perf['ane_tokens_per_sec']:.0f}")


if __name__ == "__main__":
    asyncio.run(main())
