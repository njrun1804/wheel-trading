#!/usr/bin/env python3
"""
Production integration example for streaming processors with wheel trading system.

Shows how to integrate streaming processors with existing components:
- DuckDB query results
- Options chain data
- Risk analysis results  
- Backtesting output
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

# Add src to path for demo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unity_wheel.utils import (
    safe_output,
    safe_query_output,
    safe_dataframe_output,
    stream_large_json,
    get_safe_output_logger,
    OutputConfig,
    StreamConfig,
)


class ProductionStreamingIntegration:
    """Production example of streaming integration with wheel trading."""
    
    def __init__(self):
        # Configure for production use
        self.output_config = OutputConfig(
            max_string_length=500_000,  # 500KB Claude limit
            max_memory_mb=100,          # 100MB memory limit
            use_temp_files=True,
            compress_files=True,
            auto_cleanup=True,
            include_metadata=True,
        )
        
        self.stream_config = StreamConfig(
            max_memory_mb=50,
            max_total_memory_mb=200,
            adaptive_chunking=True,
            prefetch_size=3,
        )
        
        # Initialize safe output logger
        self.logger = get_safe_output_logger("trading.streaming", self.output_config)
    
    def handle_large_query_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Handle large DuckDB query results safely."""
        print(f"Processing query results: {len(results):,} records")
        
        # Use safe query output
        result = safe_query_output(
            results, 
            query=query, 
            config=self.output_config,
            max_results=10000
        )
        
        # Log the operation
        self.logger.log_large_data(
            results,
            level="INFO", 
            message=f"Query results processed: {query[:100]}...",
            query_record_count=len(results),
            truncated=result.is_truncated,
        )
        
        return result.content
    
    def handle_options_chain_analysis(self, options_data: Dict[str, Any]) -> str:
        """Handle large options chain analysis results."""
        print("Processing options chain analysis...")
        
        # Add processing metadata
        analysis_metadata = {
            "processed_at": datetime.now().isoformat(),
            "processor": "wheel_trading_streaming",
            "data_size_estimate": len(str(options_data)),
        }
        
        enhanced_data = {
            "metadata": analysis_metadata,
            "analysis": options_data,
            "summary": self._create_options_summary(options_data)
        }
        
        result = safe_output(enhanced_data, self.output_config, "options_analysis")
        
        # Log with trading-specific context
        self.logger.log_large_data(
            enhanced_data,
            message="Options chain analysis completed",
            symbol=options_data.get("symbol", "UNKNOWN"),
            chain_size=len(options_data.get("options", [])),
            analysis_type="wheel_strategy",
        )
        
        return result.content
    
    def _create_options_summary(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of options data for quick reference."""
        options = options_data.get("options", [])
        if not options:
            return {"error": "No options data available"}
        
        # Calculate summary statistics
        strikes = [opt.get("strike", 0) for opt in options if opt.get("strike")]
        premiums = [opt.get("premium", 0) for opt in options if opt.get("premium")]
        
        return {
            "total_options": len(options),
            "strike_range": {
                "min": min(strikes) if strikes else 0,
                "max": max(strikes) if strikes else 0,
            },
            "premium_range": {
                "min": min(premiums) if premiums else 0,
                "max": max(premiums) if premiums else 0,
                "avg": sum(premiums) / len(premiums) if premiums else 0,
            },
            "expiration_dates": list(set(
                opt.get("expiration") for opt in options 
                if opt.get("expiration")
            ))[:10],  # First 10 unique expirations
        }
    
    async def handle_streaming_backtest_results(self, backtest_data: List[Dict[str, Any]]) -> str:
        """Handle streaming backtest results processing."""
        print(f"Streaming backtest results: {len(backtest_data):,} trades")
        
        # Process trades in streaming fashion
        processed_results = {
            "summary": {"total_trades": len(backtest_data)},
            "performance_metrics": {},
            "sample_trades": [],
            "risk_analysis": {},
        }
        
        # Stream process the trades
        total_pnl = 0
        winning_trades = 0
        trade_count = 0
        
        async for trade in stream_large_json(backtest_data, config=self.stream_config):
            trade_count += 1
            pnl = trade.get("pnl", 0)
            total_pnl += pnl
            
            if pnl > 0:
                winning_trades += 1
            
            # Keep sample of first few trades
            if len(processed_results["sample_trades"]) < 20:
                processed_results["sample_trades"].append(trade)
            
            # Show progress for large datasets
            if trade_count % 5000 == 0:
                print(f"   Processed {trade_count:,} trades...")
        
        # Calculate final metrics
        processed_results["performance_metrics"] = {
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(winning_trades / trade_count, 3) if trade_count > 0 else 0,
            "avg_pnl_per_trade": round(total_pnl / trade_count, 2) if trade_count > 0 else 0,
            "total_trades_processed": trade_count,
        }
        
        # Risk analysis
        sample_pnls = [trade.get("pnl", 0) for trade in processed_results["sample_trades"]]
        if sample_pnls:
            processed_results["risk_analysis"] = {
                "max_win": max(sample_pnls),
                "max_loss": min(sample_pnls),
                "sample_volatility": round(
                    (sum((pnl - total_pnl/len(sample_pnls))**2 for pnl in sample_pnls) / len(sample_pnls))**0.5, 
                    2
                ) if len(sample_pnls) > 1 else 0,
            }
        
        # Use safe output for the final results
        result = safe_output(processed_results, self.output_config, "backtest_results")
        
        self.logger.log_large_data(
            processed_results,
            message="Backtest results processed via streaming",
            total_trades=trade_count,
            final_pnl=total_pnl,
            win_rate=processed_results["performance_metrics"]["win_rate"],
        )
        
        return result.content
    
    def handle_dataframe_output(self, df, description: str = "dataframe") -> str:
        """Handle pandas DataFrame output safely."""
        print(f"Processing DataFrame: {description}")
        
        try:
            # Get DataFrame info
            rows, cols = df.shape
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            
            print(f"   DataFrame: {rows:,} rows × {cols} columns ({memory_usage:.1f} MB)")
            
            # Use safe DataFrame output
            result = safe_dataframe_output(df, self.output_config, max_rows=5000)
            
            self.logger.log_large_data(
                f"DataFrame processed: {description}",
                message=f"DataFrame output: {description}",
                shape=[rows, cols],
                memory_mb=memory_usage,
                truncated=result.is_truncated,
            )
            
            return result.content
            
        except Exception as e:
            error_msg = f"Error processing DataFrame {description}: {e}"
            self.logger.log_large_data(
                error_msg,
                level="ERROR",
                message="DataFrame processing failed",
                error=str(e),
            )
            return error_msg
    
    def create_claude_safe_response(self, data: Any, context: str = "analysis") -> Dict[str, Any]:
        """Create a Claude-safe response with metadata."""
        result = safe_output(data, self.output_config, context)
        
        response = {
            "status": "success",
            "context": context,
            "content": result.content,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "original_size_bytes": result.original_size,
                "is_truncated": result.is_truncated,
                "content_hash": result.content_hash,
                "processor_version": "1.0.0",
            }
        }
        
        if result.file_path:
            response["metadata"]["file_backup"] = {
                "path": str(result.file_path),
                "size_bytes": result.compressed_size,
                "note": "Full data available in backup file",
            }
        
        if result.is_truncated:
            response["metadata"]["truncation_info"] = {
                "reason": "Content exceeds Claude Code limits",
                "preview_size": len(result.content),
                "full_size": result.original_size,
                "recommendation": "Use file backup for complete analysis",
            }
        
        return response


# Demo functions for production integration
async def demo_query_results_integration():
    """Demo integration with DuckDB query results."""
    print("\n" + "="*50)
    print("PRODUCTION DEMO 1: Query Results Integration")
    print("="*50)
    
    integration = ProductionStreamingIntegration()
    
    # Simulate large query results
    query = """
    SELECT 
        symbol, 
        date, 
        close_price,
        volume,
        sma_20,
        sma_50,
        rsi,
        bollinger_upper,
        bollinger_lower,
        options_volume,
        implied_volatility
    FROM daily_data 
    WHERE date >= '2023-01-01' 
    AND symbol IN ('AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA')
    ORDER BY date DESC
    """
    
    # Simulate 50,000 database records
    results = []
    import random
    from datetime import timedelta
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    base_date = datetime.now() - timedelta(days=1000)
    
    for i in range(10000):  # 10K records for demo
        date = base_date + timedelta(days=i % 365)
        symbol = random.choice(symbols)
        base_price = {'AAPL': 150, 'GOOGL': 2500, 'MSFT': 300, 'TSLA': 800, 'NVDA': 400}[symbol]
        
        record = {
            'symbol': symbol,
            'date': date.strftime('%Y-%m-%d'),
            'close_price': round(base_price * (1 + random.uniform(-0.1, 0.1)), 2),
            'volume': random.randint(1000000, 50000000),
            'sma_20': round(base_price * (1 + random.uniform(-0.05, 0.05)), 2),
            'sma_50': round(base_price * (1 + random.uniform(-0.08, 0.08)), 2),
            'rsi': round(random.uniform(20, 80), 1),
            'bollinger_upper': round(base_price * 1.02, 2),
            'bollinger_lower': round(base_price * 0.98, 2),
            'options_volume': random.randint(10000, 500000),
            'implied_volatility': round(random.uniform(0.15, 0.45), 3),
        }
        results.append(record)
    
    # Process with streaming integration
    safe_output = integration.handle_large_query_results(query, results)
    print(f"Safe output length: {len(safe_output):,} characters")
    print("Query results processed successfully for Claude Code!")


async def demo_options_analysis_integration():
    """Demo integration with options analysis."""
    print("\n" + "="*50)
    print("PRODUCTION DEMO 2: Options Analysis Integration")
    print("="*50)
    
    integration = ProductionStreamingIntegration()
    
    # Simulate large options analysis
    options_analysis = {
        "symbol": "AAPL",
        "underlying_price": 155.50,
        "analysis_timestamp": datetime.now().isoformat(),
        "market_conditions": {
            "implied_volatility_rank": 0.65,
            "historical_volatility": 0.28,
            "volume_ratio": 1.35,
            "put_call_ratio": 0.85,
        },
        "options": []
    }
    
    # Generate many options
    import random
    from datetime import timedelta
    
    strikes = [round(130 + i * 2.5, 2) for i in range(40)]  # 130 to 227.5
    expirations = [(datetime.now() + timedelta(days=7 + i*7)).strftime('%Y-%m-%d') for i in range(8)]
    
    for expiration in expirations:
        for strike in strikes:
            for option_type in ['call', 'put']:
                option = {
                    "strike": strike,
                    "expiration": expiration,
                    "type": option_type,
                    "bid": round(random.uniform(0.5, 25.0), 2),
                    "ask": round(random.uniform(0.6, 26.0), 2),
                    "last": round(random.uniform(0.55, 25.5), 2),
                    "volume": random.randint(0, 5000),
                    "open_interest": random.randint(0, 10000),
                    "implied_volatility": round(random.uniform(0.15, 0.8), 3),
                    "delta": round(random.uniform(-1, 1), 3),
                    "gamma": round(random.uniform(0, 0.1), 4),
                    "theta": round(random.uniform(-2, 0), 3),
                    "vega": round(random.uniform(0, 3), 3),
                    "wheel_score": round(random.uniform(0, 100), 1),
                    "expected_return": round(random.uniform(-20, 40), 2),
                    "probability_profit": round(random.uniform(0.3, 0.9), 3),
                }
                options_analysis["options"].append(option)
    
    print(f"Generated options analysis: {len(options_analysis['options']):,} options")
    
    # Process with streaming integration
    safe_output = integration.handle_options_chain_analysis(options_analysis)
    print(f"Safe output length: {len(safe_output):,} characters")
    print("Options analysis processed successfully for Claude Code!")


async def demo_backtest_streaming_integration():
    """Demo integration with backtest streaming."""
    print("\n" + "="*50)
    print("PRODUCTION DEMO 3: Backtest Streaming Integration")
    print("="*50)
    
    integration = ProductionStreamingIntegration()
    
    # Generate large backtest dataset
    backtest_trades = []
    import random
    from datetime import timedelta
    
    strategies = ['wheel', 'covered_call', 'cash_secured_put', 'iron_condor']
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN']
    
    base_date = datetime.now() - timedelta(days=730)  # 2 years of data
    
    for i in range(20000):  # 20K trades
        entry_date = base_date + timedelta(days=random.randint(0, 729))
        exit_date = entry_date + timedelta(days=random.randint(1, 45))
        
        trade = {
            "trade_id": f"BT_{i:06d}",
            "strategy": random.choice(strategies),
            "symbol": random.choice(symbols),
            "entry_date": entry_date.strftime('%Y-%m-%d'),
            "exit_date": exit_date.strftime('%Y-%m-%d'),
            "entry_price": round(random.uniform(100, 500), 2),
            "exit_price": round(random.uniform(95, 510), 2),
            "quantity": random.randint(1, 10),
            "pnl": round(random.uniform(-800, 1200), 2),
            "pnl_percent": round(random.uniform(-15, 25), 2),
            "days_in_trade": (exit_date - entry_date).days,
            "max_profit": round(random.uniform(50, 1000), 2),
            "max_loss": round(random.uniform(-1500, -100), 2),
            "commissions": round(random.uniform(1.5, 15.0), 2),
            "slippage": round(random.uniform(0, 5.0), 2),
            "market_conditions": {
                "vix_entry": round(random.uniform(12, 45), 1),
                "vix_exit": round(random.uniform(12, 45), 1),
                "spy_price": round(random.uniform(350, 480), 2),
                "treasury_10y": round(random.uniform(1.5, 5.0), 2),
            },
            "greeks_at_entry": {
                "delta": round(random.uniform(-1, 1), 3),
                "gamma": round(random.uniform(0, 0.1), 4),
                "theta": round(random.uniform(-2, 0), 3),
                "vega": round(random.uniform(0, 3), 3),
            },
            "win": random.choice([True, False]),
            "tags": random.sample(['high_iv', 'earnings', 'dividend', 'technical', 'fundamental'], k=random.randint(1, 3)),
        }
        backtest_trades.append(trade)
    
    print(f"Generated backtest dataset: {len(backtest_trades):,} trades")
    
    # Process with streaming integration
    safe_output = await integration.handle_streaming_backtest_results(backtest_trades)
    print(f"Safe output length: {len(safe_output):,} characters")
    print("Backtest results processed successfully for Claude Code!")


async def demo_dataframe_integration():
    """Demo integration with DataFrame processing."""
    print("\n" + "="*50)
    print("PRODUCTION DEMO 4: DataFrame Integration")
    print("="*50)
    
    integration = ProductionStreamingIntegration()
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create large DataFrame
        dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'] * 20  # 100 symbols
        
        data = []
        for date in dates:
            for symbol in symbols:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'price': np.random.uniform(50, 500),
                    'volume': np.random.randint(100000, 10000000),
                    'returns': np.random.normal(0.001, 0.02),
                    'volatility': np.random.uniform(0.1, 0.5),
                    'rsi': np.random.uniform(20, 80),
                    'macd': np.random.uniform(-5, 5),
                    'bollinger_position': np.random.uniform(0, 1),
                })
        
        df = pd.DataFrame(data)
        print(f"Created DataFrame: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Process with streaming integration  
        safe_output = integration.handle_dataframe_output(df, "Market Data Analysis")
        print(f"Safe output length: {len(safe_output):,} characters")
        print("DataFrame processed successfully for Claude Code!")
        
    except ImportError:
        print("Pandas not available - skipping DataFrame demo")


async def demo_claude_response_formatting():
    """Demo Claude-safe response formatting."""
    print("\n" + "="*50)
    print("PRODUCTION DEMO 5: Claude Response Formatting")
    print("="*50)
    
    integration = ProductionStreamingIntegration()
    
    # Create complex analysis result
    analysis_result = {
        "wheel_strategy_analysis": {
            "recommended_positions": [
                {
                    "symbol": "AAPL",
                    "strategy": "cash_secured_put",
                    "strike": 145.0,
                    "expiration": "2024-02-16",
                    "premium": 3.50,
                    "probability_profit": 0.75,
                    "max_profit": 350,
                    "max_loss": -14150,
                    "expected_return": 2.4,
                    "rationale": "High implied volatility with strong support at $145 level",
                }
                for i in range(50)  # 50 recommendations
            ],
            "market_analysis": {
                "overall_sentiment": "bullish",
                "volatility_regime": "elevated",
                "key_levels": {
                    "support": [140, 135, 130],
                    "resistance": [160, 165, 170],
                },
                "economic_factors": [
                    "Fed policy neutral",
                    "Earnings season approaching", 
                    "Technical breakout pending",
                ],
            },
            "risk_assessment": {
                "portfolio_var": -15000,
                "max_drawdown": -25000,
                "sharpe_ratio": 1.85,
                "correlation_analysis": "Low correlation with existing positions",
            },
        }
    }
    
    # Create Claude-safe response
    claude_response = integration.create_claude_safe_response(
        analysis_result, 
        context="wheel_strategy_recommendations"
    )
    
    print("Claude-safe response created:")
    print(f"   Status: {claude_response['status']}")
    print(f"   Content length: {len(claude_response['content']):,} chars")
    print(f"   Truncated: {claude_response['metadata']['is_truncated']}")
    print(f"   Original size: {claude_response['metadata']['original_size_bytes']:,} bytes")
    
    if claude_response['metadata'].get('file_backup'):
        print(f"   Backup file: {claude_response['metadata']['file_backup']['path']}")
    
    # Show how Claude would see this
    print("\nClaude would receive:")
    print(claude_response['content'][:500] + "..." if len(claude_response['content']) > 500 else claude_response['content'])


async def main():
    """Run all production integration demos."""
    print("🏭 PRODUCTION STREAMING INTEGRATION DEMOS")
    print("="*60)
    print("Real-world integration with wheel trading system components")
    print()
    
    await demo_query_results_integration()
    await demo_options_analysis_integration() 
    await demo_backtest_streaming_integration()
    await demo_dataframe_integration()
    await demo_claude_response_formatting()
    
    print("\n" + "="*60)
    print("🎉 ALL PRODUCTION DEMOS COMPLETED!")
    print("="*60)
    print("\n✅ Production Integration Features:")
    print("   • DuckDB query result streaming")
    print("   • Options chain analysis processing")
    print("   • Backtest data streaming")
    print("   • DataFrame safe output")
    print("   • Claude-optimized response formatting")
    print("   • Comprehensive logging and monitoring")
    print("   • Error recovery and fallback mechanisms")
    print("\n🚀 Ready for production deployment in wheel trading system!")


if __name__ == "__main__":
    asyncio.run(main())