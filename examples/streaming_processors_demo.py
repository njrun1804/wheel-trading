#!/usr/bin/env python3
"""
Comprehensive demo of streaming data processors for preventing Claude Code overflow.

This demo shows how to use the streaming processors with realistic trading data
to prevent memory issues and string overflow errors.
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for demo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unity_wheel.utils import (
    # Safe output handling
    safe_output,
    safe_json_output,
    safe_dataframe_output,
    OutputConfig,
    SafeOutputHandler,
    
    # Streaming processors  
    stream_large_json,
    stream_large_text,
    stream_large_data,
    StreamConfig,
    
    # Memory-aware chunking
    chunk_large_data,
    chunk_and_process,
    ChunkingConfig,
    AdaptiveChunker,
    get_optimal_chunk_config,
)


def generate_sample_trading_data(num_records: int = 10000) -> list[dict]:
    """Generate sample trading data for demo."""
    print(f"Generating {num_records:,} sample trading records...")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
    
    data = []
    base_date = datetime.now() - timedelta(days=365)
    
    for i in range(num_records):
        record = {
            'id': i,
            'timestamp': (base_date + timedelta(minutes=i)).isoformat(),
            'symbol': random.choice(symbols),
            'price': round(random.uniform(50, 500), 2),
            'volume': random.randint(100, 10000),
            'bid': round(random.uniform(50, 500) * 0.995, 2),
            'ask': round(random.uniform(50, 500) * 1.005, 2),
            'options_chain': [
                {
                    'strike': round(random.uniform(40, 600), 2),
                    'call_price': round(random.uniform(1, 50), 2),
                    'put_price': round(random.uniform(1, 50), 2),
                    'iv': round(random.uniform(0.1, 1.0), 3),
                    'delta': round(random.uniform(-1, 1), 3),
                    'gamma': round(random.uniform(0, 0.1), 4),
                    'theta': round(random.uniform(-0.1, 0), 4),
                }
                for _ in range(random.randint(5, 20))
            ],
            'metadata': {
                'source': 'demo_generator',
                'quality_score': round(random.uniform(0.8, 1.0), 2),
                'processing_time_ms': round(random.uniform(1, 100), 2),
            }
        }
        data.append(record)
    
    return data


async def demo_safe_output_handling():
    """Demonstrate safe output handling with large datasets."""
    print("\n" + "="*60)
    print("DEMO 1: Safe Output Handling")
    print("="*60)
    
    # Generate large trading dataset
    trading_data = generate_sample_trading_data(5000)
    
    # Test 1: Default safe output
    print("\n1. Testing default safe output with large dataset...")
    result = safe_output(trading_data, output_type="trading_data")
    print(f"   • Original size: {result.original_size:,} bytes")
    print(f"   • Truncated: {result.is_truncated}")
    print(f"   • File created: {result.file_path is not None}")
    if result.file_path:
        print(f"   • File size: {result.compressed_size:,} bytes")
        print(f"   • Compression ratio: {result.metadata.get('compression_ratio', 1.0):.2f}")
    
    # Test 2: Custom configuration for very large data
    print("\n2. Testing with custom configuration...")
    config = OutputConfig(
        max_string_length=50000,  # 50KB limit
        max_memory_mb=5,          # 5MB memory limit
        use_temp_files=True,
        compress_files=True,
        preview_lines=20,
    )
    
    with SafeOutputHandler(config) as handler:
        result = handler.handle_output(trading_data, "large_trading_dataset")
        print(f"   • Handled {len(trading_data):,} records")
        print(f"   • Output truncated: {result.is_truncated}")
        print(f"   • File path: {result.file_path}")
        
        if result.file_path and result.file_path.exists():
            # Read a sample from the file
            with open(result.file_path, 'r') as f:
                sample = f.read(500)  # First 500 chars
                print(f"   • File sample: {sample[:100]}...")
    
    # Test 3: JSON-specific formatting
    print("\n3. Testing JSON-specific safe output...")
    sample_data = {
        "summary": {
            "total_records": len(trading_data),
            "symbols": list(set(record['symbol'] for record in trading_data[:100])),
            "date_range": {
                "start": trading_data[0]['timestamp'],
                "end": trading_data[-1]['timestamp'],
            }
        },
        "sample_records": trading_data[:10],  # Just first 10 for demo
        "statistics": {
            "avg_price": sum(r['price'] for r in trading_data[:100]) / 100,
            "total_volume": sum(r['volume'] for r in trading_data[:100]),
        }
    }
    
    result = safe_json_output(sample_data, pretty=True)
    print(f"   • JSON formatted: {len(result.content.splitlines())} lines")
    print(f"   • Preview:\n{result.content[:300]}...")


async def demo_streaming_processors():
    """Demonstrate streaming processors with large data."""
    print("\n" + "="*60)
    print("DEMO 2: Streaming Data Processors")  
    print("="*60)
    
    # Generate large dataset
    trading_data = generate_sample_trading_data(25000)
    
    # Test 1: Stream large JSON data
    print("\n1. Streaming large JSON dataset...")
    stream_config = StreamConfig(
        max_memory_mb=20,
        default_chunk_size=64 * 1024,  # 64KB chunks
        auto_cleanup=True,
    )
    
    processed_count = 0
    symbol_counts = {}
    
    async for record in stream_large_json(trading_data, config=stream_config):
        processed_count += 1
        symbol = record.get('symbol', 'UNKNOWN')
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Show progress every 5000 records
        if processed_count % 5000 == 0:
            print(f"   • Processed {processed_count:,} records...")
    
    print(f"   • Total records processed: {processed_count:,}")
    print(f"   • Symbol distribution: {dict(sorted(symbol_counts.items()))}")
    
    # Test 2: Stream text data (like logs)
    print("\n2. Streaming text data (simulated logs)...")
    log_lines = []
    for i in range(10000):
        log_lines.append(
            f"{datetime.now().isoformat()} INFO [Trading] "
            f"Processing order {i:06d} for {random.choice(['AAPL', 'GOOGL', 'MSFT'])} "
            f"- Price: ${random.uniform(100, 500):.2f}, Volume: {random.randint(100, 1000)}"
        )
    
    log_text = "\n".join(log_lines)
    print(f"   • Generated log text: {len(log_text):,} characters")
    
    line_count = 0
    error_count = 0
    
    async for line in stream_large_text(log_text, line_based=True, config=stream_config):
        line_count += 1
        if "ERROR" in line:
            error_count += 1
        
        # Show first few lines
        if line_count <= 3:
            print(f"   • Line {line_count}: {line[:80]}...")
    
    print(f"   • Total lines processed: {line_count:,}")
    print(f"   • Error lines found: {error_count}")
    
    # Test 3: Stream binary data (simulated market data feed)
    print("\n3. Streaming binary data...")
    # Simulate binary market data
    binary_data = b""
    for i in range(50000):
        # Simple binary record: timestamp(8) + symbol(4) + price(8) + volume(4)
        timestamp = i * 1000  # milliseconds
        symbol = f"SYM{i % 100:02d}".encode('ascii')[:4].ljust(4, b'\x00')
        price = int(random.uniform(50, 500) * 100)  # price in cents
        volume = random.randint(100, 10000)
        
        record = (
            timestamp.to_bytes(8, 'big') +
            symbol +
            price.to_bytes(8, 'big') +
            volume.to_bytes(4, 'big')
        )
        binary_data += record
    
    print(f"   • Generated binary data: {len(binary_data):,} bytes")
    
    chunk_count = 0
    total_records = 0
    
    async for chunk in stream_large_data(binary_data, config=stream_config):
        chunk_count += 1
        # Each record is 24 bytes
        records_in_chunk = len(chunk) // 24
        total_records += records_in_chunk
        
        if chunk_count <= 3:
            print(f"   • Chunk {chunk_count}: {len(chunk)} bytes, ~{records_in_chunk} records")
    
    print(f"   • Total chunks: {chunk_count}")
    print(f"   • Total records recovered: {total_records:,}")


async def demo_memory_aware_chunking():
    """Demonstrate memory-aware chunking strategies."""
    print("\n" + "="*60)
    print("DEMO 3: Memory-Aware Chunking")
    print("="*60)
    
    # Generate different types of data
    datasets = {
        "small_records": [{"id": i, "value": f"item_{i}"} for i in range(1000)],
        "large_records": [
            {
                "id": i,
                "data": list(range(i, i + 100)),
                "description": f"Large record {i} with lots of data " * 10,
                "metadata": {"created": datetime.now().isoformat(), "size": "large"}
            }
            for i in range(500)
        ],
        "text_data": "This is a large text document. " * 10000,
        "binary_data": b"Binary chunk data. " * 5000,
    }
    
    # Test 1: Adaptive chunking with different data types
    print("\n1. Testing adaptive chunking...")
    
    chunking_config = ChunkingConfig(
        strategy=ChunkingStrategy.ADAPTIVE,
        target_chunk_size=32 * 1024,  # 32KB target
        memory_limit_mb=50,
        parallel_processing=True,
        max_concurrent_chunks=4,
    )
    
    chunker = AdaptiveChunker(chunking_config)
    
    for name, data in datasets.items():
        print(f"\n   Processing {name}...")
        chunk_count = 0
        total_size = 0
        
        async for chunk_id, chunk in chunker.chunk_data(data):
            chunk_count += 1
            if isinstance(chunk, (str, bytes)):
                chunk_size = len(chunk)
            elif isinstance(chunk, list):
                chunk_size = len(str(chunk))
            else:
                chunk_size = len(str(chunk))
            
            total_size += chunk_size
            
            if chunk_count <= 3:
                preview = str(chunk)[:50] if len(str(chunk)) > 50 else str(chunk)
                print(f"     • Chunk {chunk_id}: {chunk_size} bytes - {preview}...")
        
        print(f"     • Total chunks: {chunk_count}")
        print(f"     • Total size processed: {total_size:,} bytes")
    
    # Show performance metrics
    print(f"\n   Performance Metrics:")
    all_metrics = chunker.get_all_metrics()
    for data_type, metrics in all_metrics.items():
        if metrics:  # Only show if we have metrics
            print(f"     • {data_type}:")
            print(f"       - Throughput: {metrics.get('throughput_mbps', 0):.2f} MB/s")
            print(f"       - Avg chunk size: {metrics.get('avg_chunk_size', 0):,.0f} bytes")
            print(f"       - Processing time: {metrics.get('total_time_ms', 0):.1f} ms")
    
    # Test 2: Parallel chunk processing
    print("\n2. Testing parallel chunk processing...")
    
    large_dataset = list(range(10000))  # Simple numeric data
    
    def process_chunk(chunk):
        """Simple processing function - compute sum and stats."""
        return {
            "chunk_sum": sum(chunk),
            "chunk_min": min(chunk),
            "chunk_max": max(chunk),
            "chunk_count": len(chunk),
        }
    
    # Process in parallel
    start_time = asyncio.get_event_loop().time()
    results_parallel = await chunk_and_process(
        large_dataset, 
        process_chunk, 
        chunking_config, 
        parallel=True
    )
    parallel_time = asyncio.get_event_loop().time() - start_time
    
    # Process sequentially for comparison
    start_time = asyncio.get_event_loop().time()
    results_sequential = await chunk_and_process(
        large_dataset, 
        process_chunk, 
        chunking_config, 
        parallel=False
    )
    sequential_time = asyncio.get_event_loop().time() - start_time
    
    print(f"   • Parallel processing: {len(results_parallel)} chunks in {parallel_time:.3f}s")
    print(f"   • Sequential processing: {len(results_sequential)} chunks in {sequential_time:.3f}s")
    print(f"   • Speed improvement: {sequential_time / parallel_time:.2f}x faster")
    
    # Verify results are the same
    total_parallel = sum(r["chunk_sum"] for r in results_parallel)
    total_sequential = sum(r["chunk_sum"] for r in results_sequential)
    print(f"   • Results match: {total_parallel == total_sequential}")
    
    # Test 3: Optimal configuration
    print("\n3. Testing optimal configuration selection...")
    
    test_scenarios = [
        (10, DataType.JSON, 512, "Low memory"),
        (1000, DataType.TEXT, 2048, "Normal scenario"),
        (5000, DataType.PARQUET, 8192, "Large data, high memory"),
    ]
    
    for data_size_mb, data_type, available_memory_mb, scenario in test_scenarios:
        config = get_optimal_chunk_config(data_size_mb, data_type, available_memory_mb)
        print(f"   • {scenario}:")
        print(f"     - Data: {data_size_mb}MB {data_type.value}")
        print(f"     - Memory: {available_memory_mb}MB available")
        print(f"     - Strategy: {config.strategy.value}")
        print(f"     - Chunk size: {config.target_chunk_size // 1024}KB")
        print(f"     - Concurrent: {config.max_concurrent_chunks}")


async def demo_integration_with_trading():
    """Demonstrate integration with existing wheel trading patterns."""
    print("\n" + "="*60)
    print("DEMO 4: Integration with Trading Patterns")
    print("="*60)
    
    # Simulate a large options analysis result
    options_analysis = {
        "analysis_timestamp": datetime.now().isoformat(),
        "underlying_symbol": "AAPL",
        "market_conditions": {
            "volatility_regime": "normal",
            "trend": "bullish",
            "support_levels": [150.0, 145.0, 140.0],
            "resistance_levels": [165.0, 170.0, 175.0],
        },
        "wheel_opportunities": []
    }
    
    # Generate many wheel opportunities
    for i in range(2000):
        opportunity = {
            "id": f"wheel_{i:04d}",
            "strike_price": round(140 + (i * 0.5), 2),
            "expiration": (datetime.now() + timedelta(days=30 + i % 60)).isoformat(),
            "put_premium": round(random.uniform(1.5, 8.0), 2),
            "call_premium": round(random.uniform(2.0, 12.0), 2),
            "probability_profit": round(random.uniform(0.55, 0.85), 3),
            "max_profit": round(random.uniform(100, 800), 2),
            "max_loss": round(random.uniform(-2000, -500), 2),
            "greeks": {
                "delta": round(random.uniform(-0.5, 0.5), 3),
                "gamma": round(random.uniform(0, 0.1), 4),
                "theta": round(random.uniform(-0.5, 0), 3),
                "vega": round(random.uniform(0, 2), 3),
            },
            "risk_metrics": {
                "var_95": round(random.uniform(-1000, -100), 2),
                "expected_return": round(random.uniform(50, 300), 2),
                "sharpe_ratio": round(random.uniform(0.5, 2.5), 2),
                "kelly_fraction": round(random.uniform(0.1, 0.4), 3),
            },
            "backtesting_results": {
                "win_rate": round(random.uniform(0.6, 0.9), 3),
                "avg_win": round(random.uniform(100, 400), 2),
                "avg_loss": round(random.uniform(-300, -50), 2),
                "total_trades": random.randint(50, 200),
                "profit_factor": round(random.uniform(1.2, 3.0), 2),
            }
        }
        options_analysis["wheel_opportunities"].append(opportunity)
    
    print(f"Generated analysis with {len(options_analysis['wheel_opportunities']):,} opportunities")
    
    # Test 1: Safe output for Claude Code
    print("\n1. Testing safe output for Claude Code consumption...")
    config = OutputConfig(
        max_string_length=200000,  # 200KB - typical Claude limit
        use_temp_files=True,
        include_metadata=True,
        preview_lines=30,
    )
    
    result = safe_output(options_analysis, config, "wheel_analysis")
    print(f"   • Analysis size: {result.original_size:,} bytes")
    print(f"   • Fits in Claude: {not result.is_truncated}")
    print(f"   • File backup: {result.file_path is not None}")
    
    if result.is_truncated:
        print(f"   • Preview length: {len(result.content):,} characters")
        print("   • Claude will see truncated version with file reference")
    
    # Test 2: Stream processing for analysis
    print("\n2. Testing streaming analysis for processing...")
    
    processed_opportunities = []
    high_profit_count = 0
    
    async for opportunity in stream_large_json(options_analysis["wheel_opportunities"]):
        # Simulate processing each opportunity
        if opportunity["probability_profit"] > 0.75:
            high_profit_count += 1
            if len(processed_opportunities) < 10:  # Keep first 10 high-profit ones
                processed_opportunities.append(opportunity)
    
    print(f"   • High-probability opportunities: {high_profit_count:,}")
    print(f"   • Sample opportunities retained: {len(processed_opportunities)}")
    
    # Test 3: Memory-efficient backtesting data
    print("\n3. Testing with backtesting data streams...")
    
    # Simulate large backtesting dataset
    backtest_data = []
    for day in range(1000):  # 1000 trading days
        date = datetime.now() - timedelta(days=1000 - day)
        for i in range(50):  # 50 trades per day
            trade = {
                "date": date.isoformat(),
                "trade_id": f"{date.strftime('%Y%m%d')}_{i:03d}",
                "symbol": random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA']),
                "strategy": "wheel",
                "entry_price": round(random.uniform(100, 500), 2),
                "exit_price": round(random.uniform(95, 510), 2),
                "quantity": random.randint(1, 10),
                "pnl": round(random.uniform(-500, 800), 2),
                "duration_days": random.randint(1, 45),
                "market_conditions": {
                    "vix": round(random.uniform(12, 35), 1),
                    "spy_price": round(random.uniform(350, 450), 2),
                    "treasury_10y": round(random.uniform(1.5, 4.5), 2),
                }
            }
            backtest_data.append(trade)
    
    print(f"   • Generated backtest data: {len(backtest_data):,} trades")
    
    # Process in chunks to calculate statistics
    def calculate_chunk_stats(chunk):
        total_pnl = sum(trade["pnl"] for trade in chunk)
        winning_trades = sum(1 for trade in chunk if trade["pnl"] > 0)
        return {
            "chunk_pnl": total_pnl,
            "winning_trades": winning_trades,
            "total_trades": len(chunk),
            "win_rate": winning_trades / len(chunk) if chunk else 0,
        }
    
    chunk_config = ChunkingConfig(target_chunk_size=100 * 1024)  # 100KB chunks
    chunk_stats = await chunk_and_process(
        backtest_data, 
        calculate_chunk_stats, 
        chunk_config, 
        parallel=True
    )
    
    # Aggregate results
    total_pnl = sum(stat["chunk_pnl"] for stat in chunk_stats)
    total_winning = sum(stat["winning_trades"] for stat in chunk_stats)
    total_trades = sum(stat["total_trades"] for stat in chunk_stats)
    overall_win_rate = total_winning / total_trades
    
    print(f"   • Processed {len(chunk_stats)} chunks")
    print(f"   • Total P&L: ${total_pnl:,.2f}")
    print(f"   • Win rate: {overall_win_rate:.1%}")
    print(f"   • Avg P&L per trade: ${total_pnl / total_trades:.2f}")


async def main():
    """Run all demos."""
    print("🚀 STREAMING DATA PROCESSORS COMPREHENSIVE DEMO")
    print("=" * 60)
    print("Preventing Claude Code string overflow with intelligent streaming!")
    print()
    
    # Run all demos
    await demo_safe_output_handling()
    await demo_streaming_processors()
    await demo_memory_aware_chunking()
    await demo_integration_with_trading()
    
    print("\n" + "="*60)
    print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Key Features Demonstrated:")
    print("   ✅ Safe output handling with automatic file fallback")
    print("   ✅ Memory-aware streaming for large datasets")
    print("   ✅ Intelligent chunking strategies")
    print("   ✅ Parallel processing capabilities")
    print("   ✅ Integration with wheel trading patterns")
    print("   ✅ Error recovery and performance monitoring")
    print("\n🛡️  Your Claude Code sessions are now protected from string overflow!")


if __name__ == "__main__":
    asyncio.run(main())