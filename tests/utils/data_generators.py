#!/usr/bin/env python3
"""
Data Generators for Overflow Prevention Testing

Generates various types of large datasets for testing string overflow
prevention mechanisms across different scenarios.
"""

import json
import random
import string
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Generator
from dataclasses import dataclass, asdict


@dataclass
class OptionsData:
    """Mock options data structure"""
    symbol: str
    strike: float
    expiry: str
    option_type: str
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def generate_random_string(length: int) -> str:
    """Generate random string of specified length"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_large_dataset(size: int) -> List[Dict[str, Any]]:
    """
    Generate large dataset for testing memory allocation and processing
    
    Args:
        size: Number of records to generate
        
    Returns:
        List of dictionaries representing dataset records
    """
    dataset = []
    
    # Generate variety of record types
    for i in range(size):
        record_type = i % 5
        
        if record_type == 0:
            # Options data
            record = {
                "id": i,
                "type": "options",
                "symbol": f"SPY{random.randint(100, 600)}C{random.randint(1, 12):02d}{random.randint(1, 28):02d}",
                "strike": round(random.uniform(200, 600), 2),
                "expiry": (datetime.now() + timedelta(days=random.randint(1, 365))).isoformat(),
                "bid": round(random.uniform(0.01, 50), 2),
                "ask": round(random.uniform(0.01, 50), 2),
                "volume": random.randint(0, 10000),
                "open_interest": random.randint(0, 50000),
                "implied_volatility": round(random.uniform(0.1, 2.0), 4),
                "greeks": {
                    "delta": round(random.uniform(-1, 1), 4),
                    "gamma": round(random.uniform(0, 0.1), 4),
                    "theta": round(random.uniform(-1, 0), 4),
                    "vega": round(random.uniform(0, 1), 4),
                    "rho": round(random.uniform(-1, 1), 4),
                },
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "data_source": "test_generator",
                    "quality_score": random.uniform(0.8, 1.0),
                }
            }
        elif record_type == 1:
            # Price history data
            record = {
                "id": i,
                "type": "price_history",
                "symbol": random.choice(["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL"]),
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 10000))).isoformat(),
                "open": round(random.uniform(100, 500), 2),
                "high": round(random.uniform(100, 500), 2),
                "low": round(random.uniform(100, 500), 2),
                "close": round(random.uniform(100, 500), 2),
                "volume": random.randint(1000, 1000000),
                "vwap": round(random.uniform(100, 500), 2),
                "indicators": {
                    "sma_20": round(random.uniform(100, 500), 2),
                    "sma_50": round(random.uniform(100, 500), 2),
                    "rsi": round(random.uniform(0, 100), 2),
                    "macd": round(random.uniform(-10, 10), 4),
                    "bollinger_upper": round(random.uniform(100, 500), 2),
                    "bollinger_lower": round(random.uniform(100, 500), 2),
                }
            }
        elif record_type == 2:
            # Portfolio positions
            record = {
                "id": i,
                "type": "position",
                "account_id": f"account_{random.randint(1, 100)}",
                "symbol": f"SPY{random.randint(100, 600)}C{random.randint(1, 12):02d}{random.randint(1, 28):02d}",
                "quantity": random.randint(-100, 100),
                "entry_price": round(random.uniform(0.01, 50), 2),
                "current_price": round(random.uniform(0.01, 50), 2),
                "unrealized_pnl": round(random.uniform(-10000, 10000), 2),
                "realized_pnl": round(random.uniform(-5000, 5000), 2),
                "position_value": round(random.uniform(100, 50000), 2),
                "risk_metrics": {
                    "portfolio_delta": round(random.uniform(-1000, 1000), 2),
                    "portfolio_gamma": round(random.uniform(0, 100), 2),
                    "portfolio_theta": round(random.uniform(-1000, 0), 2),
                    "portfolio_vega": round(random.uniform(0, 1000), 2),
                    "var_1d": round(random.uniform(0, 10000), 2),
                    "var_10d": round(random.uniform(0, 50000), 2),
                }
            }
        elif record_type == 3:
            # Market analysis
            record = {
                "id": i,
                "type": "market_analysis",
                "timestamp": datetime.now().isoformat(),
                "market_conditions": random.choice(["bullish", "bearish", "neutral", "volatile"]),
                "vix_level": round(random.uniform(10, 80), 2),
                "sector_performance": {
                    "technology": round(random.uniform(-5, 5), 2),
                    "healthcare": round(random.uniform(-5, 5), 2),
                    "finance": round(random.uniform(-5, 5), 2),
                    "energy": round(random.uniform(-5, 5), 2),
                    "consumer": round(random.uniform(-5, 5), 2),
                },
                "economic_indicators": {
                    "gdp_growth": round(random.uniform(-2, 5), 2),
                    "inflation_rate": round(random.uniform(0, 10), 2),
                    "unemployment_rate": round(random.uniform(3, 15), 2),
                    "fed_funds_rate": round(random.uniform(0, 10), 2),
                },
                "sentiment_analysis": {
                    "social_media_sentiment": random.uniform(-1, 1),
                    "news_sentiment": random.uniform(-1, 1),
                    "analyst_sentiment": random.uniform(-1, 1),
                    "put_call_ratio": round(random.uniform(0.5, 2.0), 2),
                }
            }
        else:
            # Debug/log data (typically very verbose)
            record = {
                "id": i,
                "type": "debug_log",
                "timestamp": datetime.now().isoformat(),
                "level": random.choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
                "component": random.choice(["duckdb", "jarvis", "einstein", "meta", "api"]),
                "message": generate_random_string(random.randint(50, 500)),
                "stack_trace": generate_random_string(random.randint(100, 2000)) if random.random() < 0.1 else None,
                "context": {
                    "user_id": f"user_{random.randint(1, 1000)}",
                    "session_id": generate_random_string(32),
                    "request_id": generate_random_string(16),
                    "function_name": f"function_{random.randint(1, 100)}",
                    "line_number": random.randint(1, 1000),
                    "execution_time_ms": round(random.uniform(0.1, 1000), 2),
                },
                "additional_data": {
                    "memory_usage_mb": round(random.uniform(10, 2000), 2),
                    "cpu_usage_percent": round(random.uniform(0, 100), 2),
                    "query_count": random.randint(0, 1000),
                    "cache_hit_rate": round(random.uniform(0, 1), 2),
                }
            }
            
        dataset.append(record)
        
    return dataset


def generate_massive_search_results(count: int) -> List[Dict[str, Any]]:
    """Generate massive search results for testing"""
    results = []
    
    for i in range(count):
        result = {
            "id": i,
            "score": random.uniform(0, 1),
            "title": generate_random_string(random.randint(20, 100)),
            "content": generate_random_string(random.randint(100, 2000)),
            "metadata": {
                "source": random.choice(["database", "api", "cache", "file"]),
                "timestamp": datetime.now().isoformat(),
                "size": random.randint(100, 100000),
                "type": random.choice(["options", "prices", "analysis", "news", "research"]),
            },
            "highlights": [
                generate_random_string(random.randint(10, 50)) 
                for _ in range(random.randint(1, 10))
            ],
            "related_items": [
                f"item_{random.randint(1, 10000)}" 
                for _ in range(random.randint(0, 20))
            ]
        }
        results.append(result)
        
    return results


def generate_streaming_data(chunk_count: int) -> List[str]:
    """Generate streaming data chunks"""
    chunks = []
    
    for i in range(chunk_count):
        chunk_type = i % 4
        
        if chunk_type == 0:
            # Price tick data
            chunk = json.dumps({
                "type": "price_tick",
                "symbol": random.choice(["SPY", "QQQ", "IWM"]),
                "price": round(random.uniform(100, 500), 2),
                "volume": random.randint(100, 10000),
                "timestamp": datetime.now().isoformat(),
            })
        elif chunk_type == 1:
            # Options quote
            chunk = json.dumps({
                "type": "options_quote",
                "symbol": f"SPY{random.randint(100, 600)}C{random.randint(1, 12):02d}{random.randint(1, 28):02d}",
                "bid": round(random.uniform(0.01, 50), 2),
                "ask": round(random.uniform(0.01, 50), 2),
                "volume": random.randint(0, 1000),
                "timestamp": datetime.now().isoformat(),
            })
        elif chunk_type == 2:
            # Market data
            chunk = json.dumps({
                "type": "market_data",
                "indicators": {
                    "vix": round(random.uniform(10, 80), 2),
                    "spy_price": round(random.uniform(300, 500), 2),
                    "volume": random.randint(10000, 1000000),
                },
                "timestamp": datetime.now().isoformat(),
            })
        else:
            # Large text chunk (simulating news/analysis)
            chunk = json.dumps({
                "type": "analysis",
                "content": generate_random_string(random.randint(500, 5000)),
                "timestamp": datetime.now().isoformat(),
            })
            
        chunks.append(chunk)
        
    return chunks


def generate_options_chain(symbol: str, expiry_count: int = 10, strikes_per_expiry: int = 50) -> List[OptionsData]:
    """Generate realistic options chain data"""
    options_chain = []
    
    base_price = random.uniform(300, 500)  # Base stock price
    
    for expiry_idx in range(expiry_count):
        expiry_date = datetime.now() + timedelta(days=random.randint(1, 365))
        
        for strike_idx in range(strikes_per_expiry):
            strike = base_price + (strike_idx - strikes_per_expiry // 2) * 5
            
            for option_type in ["call", "put"]:
                # Generate realistic option data
                moneyness = strike / base_price
                time_to_expiry = (expiry_date - datetime.now()).days / 365
                
                # Simple option pricing simulation
                intrinsic_value = max(0, base_price - strike) if option_type == "call" else max(0, strike - base_price)
                time_value = random.uniform(0.5, 5.0) * time_to_expiry
                
                bid = max(0.01, intrinsic_value + time_value - random.uniform(0, 0.5))
                ask = bid + random.uniform(0.01, 0.5)
                
                option = OptionsData(
                    symbol=f"{symbol}{int(strike)}{'C' if option_type == 'call' else 'P'}{expiry_date.strftime('%m%d')}",
                    strike=round(strike, 2),
                    expiry=expiry_date.isoformat(),
                    option_type=option_type,
                    bid=round(bid, 2),
                    ask=round(ask, 2),
                    volume=random.randint(0, 1000),
                    open_interest=random.randint(0, 10000),
                    implied_volatility=round(random.uniform(0.1, 2.0), 4),
                    delta=round(random.uniform(-1, 1), 4),
                    gamma=round(random.uniform(0, 0.1), 4),
                    theta=round(random.uniform(-1, 0), 4),
                    vega=round(random.uniform(0, 1), 4),
                    rho=round(random.uniform(-1, 1), 4),
                )
                
                options_chain.append(option)
                
    return options_chain


def generate_portfolio_data(position_count: int = 100) -> List[Dict[str, Any]]:
    """Generate realistic portfolio data"""
    positions = []
    
    for i in range(position_count):
        position = {
            "id": i,
            "account_id": f"account_{random.randint(1, 10)}",
            "symbol": f"SPY{random.randint(100, 600)}{'C' if random.random() > 0.5 else 'P'}{random.randint(1, 12):02d}{random.randint(1, 28):02d}",
            "quantity": random.randint(-100, 100),
            "entry_price": round(random.uniform(0.01, 50), 2),
            "current_price": round(random.uniform(0.01, 50), 2),
            "entry_date": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            "strategy": random.choice(["wheel", "covered_call", "cash_secured_put", "iron_condor", "straddle"]),
            "risk_metrics": {
                "delta": round(random.uniform(-1000, 1000), 2),
                "gamma": round(random.uniform(0, 100), 2),
                "theta": round(random.uniform(-100, 0), 2),
                "vega": round(random.uniform(0, 1000), 2),
                "position_size_percent": round(random.uniform(0.1, 10), 2),
                "days_to_expiry": random.randint(1, 365),
                "implied_volatility": round(random.uniform(0.1, 2.0), 4),
            },
            "performance": {
                "unrealized_pnl": round(random.uniform(-10000, 10000), 2),
                "realized_pnl": round(random.uniform(-5000, 5000), 2),
                "total_return_percent": round(random.uniform(-50, 100), 2),
                "max_drawdown_percent": round(random.uniform(0, 50), 2),
                "win_rate": round(random.uniform(0.3, 0.8), 2),
            }
        }
        positions.append(position)
        
    return positions


def generate_performance_data(days: int = 365) -> List[Dict[str, Any]]:
    """Generate performance history data"""
    performance_data = []
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        
        data_point = {
            "date": date.isoformat(),
            "portfolio_value": round(random.uniform(50000, 200000), 2),
            "daily_pnl": round(random.uniform(-5000, 5000), 2),
            "positions_count": random.randint(10, 100),
            "cash_balance": round(random.uniform(10000, 50000), 2),
            "margin_used": round(random.uniform(0, 100000), 2),
            "buying_power": round(random.uniform(0, 200000), 2),
            "risk_metrics": {
                "portfolio_delta": round(random.uniform(-1000, 1000), 2),
                "portfolio_gamma": round(random.uniform(0, 100), 2),
                "portfolio_theta": round(random.uniform(-1000, 0), 2),
                "portfolio_vega": round(random.uniform(0, 1000), 2),
                "beta_weighted_delta": round(random.uniform(-500, 500), 2),
                "var_1d": round(random.uniform(0, 10000), 2),
                "sharpe_ratio": round(random.uniform(-2, 3), 2),
                "max_drawdown": round(random.uniform(0, 50), 2),
            },
            "market_data": {
                "spy_price": round(random.uniform(300, 500), 2),
                "vix_level": round(random.uniform(10, 80), 2),
                "volume": random.randint(100000, 10000000),
                "market_cap": round(random.uniform(30000000000, 50000000000), 2),
            }
        }
        performance_data.append(data_point)
        
    return performance_data


def generate_memory_stress_data(target_size_mb: int) -> Dict[str, Any]:
    """Generate data specifically designed to stress memory systems"""
    target_bytes = target_size_mb * 1024 * 1024
    
    # Create different types of memory-intensive data
    data = {
        "large_strings": [],
        "nested_structures": [],
        "repeated_patterns": [],
        "random_data": [],
    }
    
    # Large strings (25% of target size)
    string_size = target_bytes // 4
    chunk_size = 100000  # 100KB chunks
    
    for i in range(string_size // chunk_size):
        data["large_strings"].append(generate_random_string(chunk_size))
        
    # Nested structures (25% of target size)
    def create_nested_data(depth: int, size: int) -> Dict[str, Any]:
        if depth == 0:
            return {"data": generate_random_string(size)}
        
        return {
            f"level_{depth}": create_nested_data(depth - 1, size // 2),
            "metadata": generate_random_string(size // 4),
            "items": [generate_random_string(size // 10) for _ in range(10)]
        }
    
    data["nested_structures"] = create_nested_data(10, target_bytes // 4)
    
    # Repeated patterns (25% of target size)
    pattern = generate_random_string(1000)
    repeat_count = (target_bytes // 4) // len(pattern)
    data["repeated_patterns"] = [pattern] * repeat_count
    
    # Random data (25% of target size)
    data["random_data"] = generate_large_dataset(target_bytes // (4 * 1000))  # Assuming ~1KB per record
    
    return data


# Streaming data generator
class StreamingDataGenerator:
    """Generator for continuous streaming data"""
    
    def __init__(self, data_types: List[str] = None):
        self.data_types = data_types or ["prices", "options", "news", "analysis"]
        self.running = False
        
    def generate_stream(self, duration_seconds: int = 60) -> Generator[str, None, None]:
        """Generate streaming data for specified duration"""
        start_time = time.time()
        self.running = True
        
        while self.running and (time.time() - start_time) < duration_seconds:
            data_type = random.choice(self.data_types)
            
            if data_type == "prices":
                data = {
                    "type": "price_tick",
                    "symbol": random.choice(["SPY", "QQQ", "IWM", "AAPL", "MSFT"]),
                    "price": round(random.uniform(100, 500), 2),
                    "volume": random.randint(100, 10000),
                    "timestamp": datetime.now().isoformat(),
                }
            elif data_type == "options":
                data = {
                    "type": "options_quote",
                    "symbol": f"SPY{random.randint(100, 600)}C{random.randint(1, 12):02d}{random.randint(1, 28):02d}",
                    "bid": round(random.uniform(0.01, 50), 2),
                    "ask": round(random.uniform(0.01, 50), 2),
                    "volume": random.randint(0, 1000),
                    "timestamp": datetime.now().isoformat(),
                }
            elif data_type == "news":
                data = {
                    "type": "news",
                    "headline": generate_random_string(random.randint(20, 100)),
                    "content": generate_random_string(random.randint(200, 2000)),
                    "sentiment": random.uniform(-1, 1),
                    "timestamp": datetime.now().isoformat(),
                }
            else:  # analysis
                data = {
                    "type": "analysis",
                    "content": generate_random_string(random.randint(500, 5000)),
                    "indicators": {
                        "bullish_signals": random.randint(0, 10),
                        "bearish_signals": random.randint(0, 10),
                        "confidence": random.uniform(0, 1),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            
            yield json.dumps(data)
            time.sleep(random.uniform(0.01, 0.1))  # Simulate variable timing
            
    def stop(self):
        """Stop the streaming generator"""
        self.running = False


# Utility functions
def estimate_data_size(data: Any) -> int:
    """Estimate size of data in bytes"""
    if isinstance(data, str):
        return len(data.encode('utf-8'))
    elif isinstance(data, (list, dict)):
        return len(json.dumps(data).encode('utf-8'))
    else:
        return len(str(data).encode('utf-8'))


def create_test_database_content(size_mb: int) -> Dict[str, List[Dict[str, Any]]]:
    """Create test database content of specified size"""
    target_bytes = size_mb * 1024 * 1024
    current_bytes = 0
    
    content = {
        "options": [],
        "prices": [],
        "positions": [],
        "analysis": [],
        "performance": [],
    }
    
    while current_bytes < target_bytes:
        # Add options data
        options_batch = generate_options_chain("SPY", 5, 20)
        content["options"].extend([asdict(opt) for opt in options_batch])
        
        # Add price data
        prices_batch = generate_large_dataset(100)
        content["prices"].extend([p for p in prices_batch if p["type"] == "price_history"])
        
        # Add positions
        positions_batch = generate_portfolio_data(50)
        content["positions"].extend(positions_batch)
        
        # Add analysis
        analysis_batch = generate_large_dataset(20)
        content["analysis"].extend([a for a in analysis_batch if a["type"] == "market_analysis"])
        
        # Add performance data
        performance_batch = generate_performance_data(30)
        content["performance"].extend(performance_batch)
        
        # Estimate current size
        current_bytes = estimate_data_size(content)
        
    return content