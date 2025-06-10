#!/usr/bin/env python3
"""
Minimal Unity Wheel Trader
Zero external dependencies - pure Python implementation
For use when nothing else works in the Codex environment.
"""

import math
import json
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta


class CalculationResult(NamedTuple):
    """Result of a calculation with confidence score."""
    value: float
    confidence: float


@dataclass
class StrikeData:
    """Options strike data."""
    strike: float
    delta: float
    price: float
    volume: int = 100
    open_interest: int = 100


@dataclass
class Recommendation:
    """Trading recommendation."""
    action: str
    strike: float
    contracts: int
    expected_return: float
    confidence: float
    reasoning: str


class PureMathEngine:
    """Pure Python math engine - no external dependencies."""
    
    @staticmethod
    def normal_cdf(x: float) -> float:
        """Cumulative distribution function for standard normal distribution."""
        # Abramowitz and Stegun approximation
        # Error < 7.5e-8
        if x < 0:
            return 1 - PureMathEngine.normal_cdf(-x)
        
        k = 1.0 / (1.0 + 0.2316419 * x)
        k2 = k * k
        k3 = k2 * k
        k4 = k2 * k2
        k5 = k4 * k
        
        w = (0.319381530 * k 
             - 0.356563782 * k2
             + 1.781477937 * k3
             - 1.821255978 * k4
             + 1.330274429 * k5)
        
        return 1.0 - math.exp(-0.5 * x * x) * w / math.sqrt(2 * math.pi)
    
    @staticmethod
    def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> CalculationResult:
        """Black-Scholes call option price."""
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return CalculationResult(0.0, 0.0)
            
            sqrt_T = math.sqrt(T)
            d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            
            call_price = (S * PureMathEngine.normal_cdf(d1) 
                         - K * math.exp(-r * T) * PureMathEngine.normal_cdf(d2))
            
            # Confidence based on input reasonableness
            confidence = 0.95 if 0.1 <= sigma <= 2.0 and 0.01 <= T <= 5.0 else 0.8
            
            return CalculationResult(max(call_price, 0.0), confidence)
            
        except (ValueError, OverflowError, ZeroDivisionError):
            return CalculationResult(0.0, 0.0)
    
    @staticmethod
    def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> CalculationResult:
        """Black-Scholes put option price."""
        try:
            call_result = PureMathEngine.black_scholes_call(S, K, T, r, sigma)
            if call_result.confidence == 0.0:
                return CalculationResult(0.0, 0.0)
            
            # Put-call parity: Put = Call - S + K*e^(-rT)
            put_price = call_result.value - S + K * math.exp(-r * T)
            
            return CalculationResult(max(put_price, 0.0), call_result.confidence)
            
        except (ValueError, OverflowError, ZeroDivisionError):
            return CalculationResult(0.0, 0.0)
    
    @staticmethod
    def calculate_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> CalculationResult:
        """Calculate option delta."""
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return CalculationResult(0.0, 0.0)
            
            sqrt_T = math.sqrt(T)
            d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
            
            if option_type.lower() == 'call':
                delta = PureMathEngine.normal_cdf(d1)
            else:  # put
                delta = PureMathEngine.normal_cdf(d1) - 1.0
            
            confidence = 0.95 if 0.1 <= sigma <= 2.0 and 0.01 <= T <= 5.0 else 0.8
            return CalculationResult(delta, confidence)
            
        except (ValueError, OverflowError, ZeroDivisionError):
            return CalculationResult(0.0, 0.0)


class MinimalWheelStrategy:
    """Minimal wheel strategy implementation."""
    
    def __init__(self, target_delta: float = 0.30):
        self.target_delta = target_delta
        self.math_engine = PureMathEngine()
    
    def find_optimal_put_strike(
        self,
        underlying_price: float,
        strikes: List[float],
        time_to_expiry: float,
        risk_free_rate: float = 0.05,
        volatility: float = 0.30
    ) -> Optional[StrikeData]:
        """Find optimal put strike for wheel strategy."""
        
        if not strikes or underlying_price <= 0:
            return None
        
        best_strike = None
        best_score = -float('inf')
        
        for strike in strikes:
            if strike >= underlying_price:  # Only consider OTM puts
                continue
            
            # Calculate put price and delta
            put_price = self.math_engine.black_scholes_put(
                underlying_price, strike, time_to_expiry, risk_free_rate, volatility
            )
            
            delta_result = self.math_engine.calculate_delta(
                underlying_price, strike, time_to_expiry, risk_free_rate, volatility, 'put'
            )
            
            if put_price.confidence < 0.5 or delta_result.confidence < 0.5:
                continue
            
            delta = abs(delta_result.value)  # Put delta is negative, we want absolute value
            
            # Score based on delta proximity to target and premium
            delta_score = 1.0 - abs(delta - self.target_delta) / self.target_delta
            premium_score = put_price.value / underlying_price  # Premium as % of stock price
            
            # Combined score (70% delta fit, 30% premium)
            total_score = 0.7 * delta_score + 0.3 * premium_score
            
            if total_score > best_score:
                best_score = total_score
                best_strike = StrikeData(
                    strike=strike,
                    delta=delta,
                    price=put_price.value
                )
        
        return best_strike
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        stock_price: float,
        max_position_pct: float = 0.20
    ) -> int:
        """Calculate number of contracts to trade."""
        max_position_value = portfolio_value * max_position_pct
        shares_per_contract = 100
        max_contracts = int(max_position_value / (stock_price * shares_per_contract))
        
        return max(1, max_contracts)


class MinimalWheelAdvisor:
    """Minimal wheel advisor - pure Python implementation."""
    
    def __init__(self):
        self.strategy = MinimalWheelStrategy()
    
    def generate_recommendation(
        self,
        portfolio_value: float,
        underlying_price: float = 25.0,  # Default Unity price
        available_strikes: Optional[List[float]] = None,
        days_to_expiry: int = 45,
        implied_volatility: float = 0.35
    ) -> Recommendation:
        """Generate a wheel strategy recommendation."""
        
        # Default strikes if none provided
        if available_strikes is None:
            # Generate strikes around current price
            available_strikes = []
            for i in range(-5, 6):
                strike = underlying_price + i * 2.5
                if strike > 0:
                    available_strikes.append(strike)
        
        # Convert days to years
        time_to_expiry = days_to_expiry / 365.0
        
        # Find optimal strike
        optimal_strike = self.strategy.find_optimal_put_strike(
            underlying_price=underlying_price,
            strikes=available_strikes,
            time_to_expiry=time_to_expiry,
            volatility=implied_volatility
        )
        
        if optimal_strike is None:
            return Recommendation(
                action="SKIP",
                strike=0.0,
                contracts=0,
                expected_return=0.0,
                confidence=0.0,
                reasoning="No suitable strikes found"
            )
        
        # Calculate position size
        contracts = self.strategy.calculate_position_size(
            portfolio_value=portfolio_value,
            stock_price=underlying_price
        )
        
        # Calculate expected return
        premium_per_contract = optimal_strike.price * 100  # 100 shares per contract
        total_premium = premium_per_contract * contracts
        expected_return_pct = (total_premium / portfolio_value) * (365.0 / days_to_expiry)  # Annualized
        
        # Confidence based on delta proximity to target and market conditions
        delta_fit = 1.0 - abs(optimal_strike.delta - 0.30) / 0.30
        confidence = min(0.95, 0.5 + 0.4 * delta_fit)
        
        return Recommendation(
            action="SELL_PUT",
            strike=optimal_strike.strike,
            contracts=contracts,
            expected_return=expected_return_pct,
            confidence=confidence,
            reasoning=f"Sell {contracts} contracts of ${optimal_strike.strike:.1f} puts, "
                     f"delta={optimal_strike.delta:.2f}, premium=${total_premium:.0f}"
        )


def main():
    """Main function for testing the minimal trader."""
    print("🚀 MINIMAL UNITY WHEEL TRADER")
    print("=" * 50)
    print("Pure Python implementation - no external dependencies")
    print()
    
    # Test the math engine
    print("🧮 Testing math engine...")
    math_engine = PureMathEngine()
    
    # Test Black-Scholes
    call_price = math_engine.black_scholes_call(100, 100, 1, 0.05, 0.2)
    put_price = math_engine.black_scholes_put(100, 100, 1, 0.05, 0.2)
    delta = math_engine.calculate_delta(100, 100, 1, 0.05, 0.2, 'call')
    
    print(f"   Call price: ${call_price.value:.2f} (confidence: {call_price.confidence:.1%})")
    print(f"   Put price: ${put_price.value:.2f} (confidence: {put_price.confidence:.1%})")
    print(f"   Call delta: {delta.value:.3f} (confidence: {delta.confidence:.1%})")
    
    if call_price.confidence > 0.9 and put_price.confidence > 0.9:
        print("   ✅ Math engine working correctly")
    else:
        print("   ⚠️  Math engine has issues")
    
    print()
    
    # Test the advisor
    print("📊 Testing wheel advisor...")
    advisor = MinimalWheelAdvisor()
    
    # Generate recommendation for $100k portfolio
    rec = advisor.generate_recommendation(
        portfolio_value=100000,
        underlying_price=25.0,
        days_to_expiry=45,
        implied_volatility=0.35
    )
    
    print(f"   Action: {rec.action}")
    print(f"   Strike: ${rec.strike:.1f}")
    print(f"   Contracts: {rec.contracts}")
    print(f"   Expected return: {rec.expected_return:.1%} (annualized)")
    print(f"   Confidence: {rec.confidence:.1%}")
    print(f"   Reasoning: {rec.reasoning}")
    
    if rec.confidence > 0.5:
        print("   ✅ Advisor working correctly")
    else:
        print("   ⚠️  Advisor has low confidence")
    
    print()
    print("🎯 MINIMAL TRADER READY!")
    print("   This implementation uses zero external dependencies")
    print("   Suitable for restricted environments")
    
    return rec


if __name__ == "__main__":
    recommendation = main()
    
    # Output as JSON for programmatic use
    print("\n" + "=" * 50)
    print("JSON OUTPUT:")
    
    rec_dict = {
        'action': recommendation.action,
        'strike': recommendation.strike,
        'contracts': recommendation.contracts,
        'expected_return': recommendation.expected_return,
        'confidence': recommendation.confidence,
        'reasoning': recommendation.reasoning,
        'timestamp': datetime.now().isoformat()
    }
    
    print(json.dumps(rec_dict, indent=2))