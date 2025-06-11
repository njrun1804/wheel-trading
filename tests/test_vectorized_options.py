"""Tests for vectorized options calculations."""

import time

import numpy as np
import pytest

from src.unity_wheel.math.vectorized_options import (
    VectorizedResults,
    compare_scenario_analysis,
    quick_strike_comparison,
    quick_vol_sensitivity,
    vectorized_black_scholes,
    vectorized_greeks,
    vectorized_wheel_analysis,
)


class TestVectorizedResults:
    """Test the VectorizedResults container."""
    
    def test_vectorized_results_creation(self):
        """Test creating VectorizedResults."""
        values = np.array([1.5, 2.0, 2.5])
        confidence = 0.95
        shape_info = {"result_shape": (3,), "total_calculations": 3}
        
        result = VectorizedResults(values, confidence, shape_info)
        
        assert result.size == 3
        assert len(result) == 3
        assert result.confidence == 0.95
        assert result[0] == 1.5
        assert result[1] == 2.0
        assert result[2] == 2.5
    
    def test_vectorized_results_methods(self):
        """Test VectorizedResults methods."""
        values = np.array([[1, 2], [3, 4]])
        result = VectorizedResults(values, 0.90, {})
        
        # Test to_list
        as_list = result.to_list()
        assert as_list == [1, 2, 3, 4]
        
        # Test reshape
        reshaped = result.reshape((4,))
        assert reshaped.shape == (4,)
        np.testing.assert_array_equal(reshaped, [1, 2, 3, 4])


class TestVectorizedBlackScholes:
    """Test vectorized Black-Scholes calculations."""
    
    def test_single_calculation(self):
        """Test single option calculation."""
        result = vectorized_black_scholes(100, 100, 0.25, 0.05, 0.20, "call")
        
        assert isinstance(result, VectorizedResults)
        assert result.size == 1
        assert result.confidence > 0.95
        
        # Should be close to standard Black-Scholes result
        expected_price = 5.57  # Approximate for these parameters
        assert abs(result.values - expected_price) < 1.0
    
    def test_multiple_strikes(self):
        """Test calculation across multiple strikes."""
        strikes = [95, 100, 105, 110]
        result = vectorized_black_scholes(100, strikes, 0.25, 0.05, 0.20, "call")
        
        assert result.size == 4
        assert result.values.shape == (4,)
        
        # ITM options should be more expensive than OTM
        assert result.values[0] > result.values[-1]  # 95 strike > 110 strike
        
        # Check that all results are reasonable
        assert np.all(result.values > 0)
        assert np.all(result.values < 100)  # Sanity check
    
    def test_multiple_spots_and_strikes(self):
        """Test calculation with multiple spots and strikes."""
        spots = [98, 100, 102]
        strikes = [95, 100, 105]
        result = vectorized_black_scholes(spots, strikes, 0.25, 0.05, 0.20, "call")
        
        assert result.size == 3
        assert result.values.shape == (3,)
        
        # Higher spot should give higher call price (for same strike)
        # But we're pairing spots with strikes, so relationship is more complex
        assert np.all(result.values > 0)
    
    def test_broadcast_compatibility(self):
        """Test that arrays broadcast correctly."""
        # Single spot, multiple strikes
        result1 = vectorized_black_scholes(100, [95, 100, 105], 0.25, 0.05, 0.20, "call")
        assert result1.size == 3
        
        # Multiple spots, single strike  
        result2 = vectorized_black_scholes([95, 100, 105], 100, 0.25, 0.05, 0.20, "call")
        assert result2.size == 3
        
        # Full grid (should broadcast to 3x3)
        spots = [95, 100, 105]
        strikes = [95, 100, 105]
        spots_grid, strikes_grid = np.meshgrid(spots, strikes)
        result3 = vectorized_black_scholes(spots_grid, strikes_grid, 0.25, 0.05, 0.20, "call")
        assert result3.size == 9
        assert result3.values.shape == (3, 3)
    
    def test_put_call_parity_vectorized(self):
        """Test put-call parity across multiple calculations."""
        strikes = [95, 100, 105]
        
        call_result = vectorized_black_scholes(100, strikes, 0.25, 0.05, 0.20, "call")
        put_result = vectorized_black_scholes(100, strikes, 0.25, 0.05, 0.20, "put")
        
        # Put-call parity: C - P = S - K*exp(-rT)
        spot = 100
        strikes_array = np.array(strikes)
        discount_factor = np.exp(-0.05 * 0.25)
        
        parity_left = call_result.values - put_result.values
        parity_right = spot - strikes_array * discount_factor
        
        np.testing.assert_allclose(parity_left, parity_right, rtol=1e-10)
    
    def test_expired_options(self):
        """Test handling of expired options."""
        result_call = vectorized_black_scholes(100, [95, 105], 0, 0.05, 0.20, "call")
        result_put = vectorized_black_scholes(100, [95, 105], 0, 0.05, 0.20, "put")
        
        # Expired call intrinsic values
        assert result_call.values[0] == 5.0  # max(100-95, 0)
        assert result_call.values[1] == 0.0  # max(100-105, 0)
        
        # Expired put intrinsic values  
        assert result_put.values[0] == 0.0   # max(95-100, 0)
        assert result_put.values[1] == 5.0   # max(105-100, 0)
    
    def test_zero_volatility(self):
        """Test handling of zero volatility."""
        result = vectorized_black_scholes(100, [95, 105], 0.25, 0.05, 0, "call")
        
        # Should return intrinsic value for zero vol
        discount_factor = np.exp(-0.05 * 0.25)
        expected_95 = max(100 - 95 * discount_factor, 0)
        expected_105 = max(100 - 105 * discount_factor, 0)
        
        assert abs(result.values[0] - expected_95) < 1e-10
        assert abs(result.values[1] - expected_105) < 1e-10


class TestVectorizedGreeks:
    """Test vectorized Greeks calculations."""
    
    def test_greeks_structure(self):
        """Test that Greeks are returned with correct structure."""
        result = vectorized_greeks(100, [95, 100, 105], 0.25, 0.05, 0.20, "call")
        
        expected_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        assert set(result.keys()) == set(expected_greeks)
        
        for greek_name, greek_result in result.items():
            assert isinstance(greek_result, VectorizedResults)
            assert greek_result.size == 3
    
    def test_greeks_ranges(self):
        """Test that Greeks are within expected ranges."""
        result = vectorized_greeks(100, [95, 100, 105], 0.25, 0.05, 0.20, "call")
        
        # Delta should be between 0 and 1 for calls
        deltas = result['delta'].values
        assert np.all(deltas >= 0) and np.all(deltas <= 1)
        
        # Gamma should be positive
        gammas = result['gamma'].values
        assert np.all(gammas >= 0)
        
        # Vega should be positive  
        vegas = result['vega'].values
        assert np.all(vegas >= 0)
    
    def test_put_greeks_ranges(self):
        """Test that put Greeks are within expected ranges."""
        result = vectorized_greeks(100, [95, 100, 105], 0.25, 0.05, 0.20, "put")
        
        # Delta should be between -1 and 0 for puts
        deltas = result['delta'].values
        assert np.all(deltas >= -1) and np.all(deltas <= 0)
        
        # Gamma should still be positive for puts
        gammas = result['gamma'].values
        assert np.all(gammas >= 0)
    
    def test_atm_greeks(self):
        """Test Greeks for at-the-money options."""
        result = vectorized_greeks(100, 100, 0.25, 0.05, 0.20, "call")
        
        # ATM call delta should be around 0.5
        delta = result['delta'].values[0]
        assert 0.4 < delta < 0.6
        
        # ATM options should have highest gamma
        multi_result = vectorized_greeks(100, [95, 100, 105], 0.25, 0.05, 0.20, "call")
        gammas = multi_result['gamma'].values
        atm_gamma_idx = np.argmax(gammas)
        assert atm_gamma_idx == 1  # Middle strike (ATM) should have highest gamma
    
    def test_expired_greeks(self):
        """Test Greeks for expired options."""
        result = vectorized_greeks(100, [95, 105], 0, 0.05, 0.20, "call")
        
        # Expired ITM call should have delta = 1
        assert result['delta'].values[0] == 1.0  # 95 strike, ITM
        
        # Expired OTM call should have delta = 0
        assert result['delta'].values[1] == 0.0  # 105 strike, OTM
        
        # All other Greeks should be 0 for expired options
        for greek in ['gamma', 'theta', 'vega', 'rho']:
            assert np.all(result[greek].values == 0.0)


class TestVectorizedWheelAnalysis:
    """Test vectorized wheel strategy analysis."""
    
    def test_wheel_analysis_basic(self):
        """Test basic wheel analysis functionality."""
        strikes = [30, 32.5, 35, 37.5]
        expirations = [30/365, 45/365, 60/365]  # 30, 45, 60 days
        
        analysis = vectorized_wheel_analysis(
            spot_price=35.0,
            strikes=strikes,
            expirations=expirations,
            volatility=0.60,
            target_delta=0.30
        )
        
        assert 'candidates' in analysis
        assert 'summary' in analysis
        assert 'market_data' in analysis
        
        # Should have some candidates
        assert len(analysis['candidates']) > 0
        
        # Check candidate structure
        candidate = analysis['candidates'][0]
        required_fields = [
            'strike', 'expiration_years', 'dte_days', 'option_price',
            'delta', 'premium_pct', 'expected_return', 'score'
        ]
        for field in required_fields:
            assert field in candidate
    
    def test_wheel_analysis_filtering(self):
        """Test that wheel analysis filters candidates properly."""
        # Use high target delta to filter out most candidates
        analysis = vectorized_wheel_analysis(
            spot_price=35.0,
            strikes=[30, 35, 40],  # Include some far OTM
            expirations=[45/365],
            target_delta=0.50,  # Very high target delta
            min_premium_pct=5.0   # High minimum premium
        )
        
        # Should have fewer candidates due to strict filtering
        candidates = analysis['candidates']
        
        # All candidates should meet delta criteria  
        for candidate in candidates:
            assert abs(candidate['delta'] - (-0.50)) <= 0.10
            assert candidate['premium_pct'] >= 5.0
    
    def test_wheel_analysis_sorting(self):
        """Test that candidates are sorted by score."""
        analysis = vectorized_wheel_analysis(
            spot_price=35.0,
            strikes=[30, 32.5, 35],
            expirations=[30/365, 45/365],
            target_delta=0.30
        )
        
        candidates = analysis['candidates']
        if len(candidates) > 1:
            # Scores should be in descending order
            scores = [c['score'] for c in candidates]
            assert scores == sorted(scores, reverse=True)
    
    def test_wheel_analysis_performance(self):
        """Test performance of vectorized wheel analysis."""
        # Large number of combinations
        strikes = list(range(25, 46))  # 21 strikes  
        expirations = [d/365 for d in range(15, 91, 5)]  # 16 expirations
        
        start_time = time.time()
        analysis = vectorized_wheel_analysis(
            spot_price=35.0,
            strikes=strikes,
            expirations=expirations,
            target_delta=0.30
        )
        computation_time = time.time() - start_time
        
        total_combinations = len(strikes) * len(expirations)
        assert analysis['summary']['total_combinations'] == total_combinations
        
        # Should complete quickly even with many combinations
        assert computation_time < 1.0  # Less than 1 second
        
        print(f"Analyzed {total_combinations} combinations in {computation_time:.3f}s")


class TestScenarioAnalysis:
    """Test scenario comparison functionality."""
    
    def test_scenario_comparison(self):
        """Test comparing multiple scenarios."""
        base_case = {
            'spot_price': 35.0,
            'strike': 32.5,
            'expiration': 45/365,
            'volatility': 0.60,
            'risk_free_rate': 0.05
        }
        
        scenarios = [
            {'spot_price': 33.0, 'volatility': 0.70},  # Lower spot, higher vol
            {'spot_price': 37.0, 'volatility': 0.50},  # Higher spot, lower vol
            {'volatility': 0.80},                       # High vol scenario
        ]
        
        comparison = compare_scenario_analysis(scenarios, base_case)
        
        assert 'base_case' in comparison
        assert 'scenarios' in comparison
        assert 'summary' in comparison
        
        assert len(comparison['scenarios']) == 3
        
        # Check that scenarios have required fields
        for scenario_result in comparison['scenarios']:
            assert 'option_price' in scenario_result
            assert 'price_diff' in scenario_result
            assert 'price_diff_pct' in scenario_result
            assert 'delta' in scenario_result
    
    def test_empty_scenarios(self):
        """Test handling of empty scenarios list."""
        base_case = {'spot_price': 35.0, 'strike': 32.5, 'expiration': 45/365}
        comparison = compare_scenario_analysis([], base_case)
        
        assert 'error' in comparison


class TestConvenienceFunctions:
    """Test convenience functions for common use cases."""
    
    def test_quick_strike_comparison(self):
        """Test quick strike comparison."""
        strikes = [30, 32.5, 35, 37.5]
        result = quick_strike_comparison(
            spot=35.0,
            strikes=strikes,
            dte=45,
            vol=0.60,
            option_type="put"
        )
        
        assert len(result) == 4
        
        for strike in strikes:
            assert strike in result
            assert 'price' in result[strike]
            assert 'delta' in result[strike]
            assert 'premium_pct' in result[strike]
            
            # Put prices should generally decrease as strikes decrease
            assert result[strike]['price'] > 0
    
    def test_quick_vol_sensitivity(self):
        """Test quick volatility sensitivity analysis."""
        result = quick_vol_sensitivity(
            spot=35.0,
            strike=32.5,
            dte=45,
            vol_range=(0.30, 0.90),
            num_points=10
        )
        
        assert 'volatilities' in result
        assert 'prices' in result
        assert 'vol_sensitivity' in result
        
        assert len(result['volatilities']) == 10
        assert len(result['prices']) == 10
        assert len(result['vol_sensitivity']) == 10
        
        # Prices should increase with volatility
        prices = result['prices']
        assert prices[-1] > prices[0]  # Highest vol > lowest vol
        
        # Vol sensitivity should be positive (vega > 0)
        sensitivities = result['vol_sensitivity']
        assert np.all(np.array(sensitivities) > 0)


@pytest.mark.performance
class TestVectorizedPerformance:
    """Performance tests for vectorized operations."""
    
    def test_vectorized_vs_loop_performance(self):
        """Compare vectorized performance against loop-based calculations."""
        # Test parameters
        spot = 35.0
        strikes = list(np.linspace(25, 45, 50))  # 50 strikes
        dte = 45/365
        vol = 0.60
        
        # Time vectorized calculation
        start_time = time.time()
        vectorized_result = vectorized_black_scholes(spot, strikes, dte, 0.05, vol, "put")
        vectorized_time = time.time() - start_time
        
        # Time equivalent loop calculation (simplified)
        from src.unity_wheel.math.options_enhanced import black_scholes_price_enhanced
        
        start_time = time.time()
        loop_results = []
        for strike in strikes:
            result = black_scholes_price_enhanced(spot, strike, dte, 0.05, vol, "put")
            loop_results.append(result.value)
        loop_time = time.time() - start_time
        
        # Vectorized should be faster
        speedup = loop_time / vectorized_time if vectorized_time > 0 else float('inf')
        
        print(f"Vectorized time: {vectorized_time:.4f}s")
        print(f"Loop time: {loop_time:.4f}s") 
        print(f"Speedup: {speedup:.1f}x")
        
        # Results should be very close
        np.testing.assert_allclose(
            vectorized_result.values, loop_results, rtol=1e-10
        )
        
        # Vectorized should be faster (allow for some variance)
        assert speedup > 0.5  # At least not significantly slower
    
    def test_large_scale_analysis(self):
        """Test performance with large-scale analysis."""
        # Simulate analyzing a large number of options
        num_strikes = 100
        num_expirations = 20
        
        strikes = list(np.linspace(20, 50, num_strikes))
        expirations = [d/365 for d in range(15, 365, 18)][:num_expirations]
        
        start_time = time.time()
        analysis = vectorized_wheel_analysis(
            spot_price=35.0,
            strikes=strikes,
            expirations=expirations,
            target_delta=0.30
        )
        elapsed_time = time.time() - start_time
        
        total_combinations = num_strikes * num_expirations
        rate = total_combinations / elapsed_time if elapsed_time > 0 else float('inf')
        
        print(f"Analyzed {total_combinations} combinations in {elapsed_time:.3f}s")
        print(f"Rate: {rate:.0f} calculations/second")
        
        # Should handle large scale efficiently
        assert elapsed_time < 5.0  # Should complete within 5 seconds
        assert rate > 1000  # Should process at least 1000 calculations/second
        
        # Should still produce valid results
        assert len(analysis['candidates']) > 0
        assert analysis['summary']['total_combinations'] == total_combinations