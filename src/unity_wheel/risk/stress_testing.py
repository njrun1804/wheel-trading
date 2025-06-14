"""Stress testing scenarios for wheel trading positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from unity_wheel.utils import get_logger

logger = get_logger(__name__)


@dataclass
class StressTestResult:
    """Results from a stress test scenario."""
    scenario_name: str
    portfolio_value: float
    profit_loss: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    recovery_time: int
    probability: float


class StressTestScenarios:
    """Run various stress test scenarios on wheel trading positions."""
    
    def __init__(self):
        """Initialize stress test scenarios."""
        self.scenarios = self._define_scenarios()
        
    def _define_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Define standard stress test scenarios."""
        return {
            'market_crash': {
                'description': '20% market drop over 5 days',
                'price_change': -0.20,
                'volatility_multiplier': 2.5,
                'correlation': 0.8,
                'probability': 0.05
            },
            'flash_crash': {
                'description': '10% intraday drop and recovery',
                'price_change': -0.10,
                'volatility_multiplier': 3.0,
                'correlation': 0.9,
                'probability': 0.02
            },
            'volatility_spike': {
                'description': 'VIX doubles without price movement',
                'price_change': 0.0,
                'volatility_multiplier': 2.0,
                'correlation': 0.5,
                'probability': 0.10
            },
            'earnings_miss': {
                'description': '15% drop on earnings disappointment',
                'price_change': -0.15,
                'volatility_multiplier': 1.5,
                'correlation': 0.3,
                'probability': 0.15
            },
            'sector_rotation': {
                'description': 'Tech sector 10% underperformance',
                'price_change': -0.10,
                'volatility_multiplier': 1.3,
                'correlation': 0.6,
                'probability': 0.20
            },
            'rate_hike': {
                'description': 'Surprise 50bp rate increase',
                'price_change': -0.05,
                'volatility_multiplier': 1.4,
                'correlation': 0.7,
                'probability': 0.25
            }
        }
        
    def run_scenarios(self, portfolio_value: float, 
                     positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run all stress test scenarios on current positions.
        
        Args:
            portfolio_value: Total portfolio value
            positions: List of current positions
            
        Returns:
            Dictionary with scenario results and summary statistics
        """
        results = []
        
        for scenario_name, scenario in self.scenarios.items():
            result = self._run_single_scenario(
                scenario_name, scenario, portfolio_value, positions
            )
            results.append(result)
            
        # Calculate summary statistics
        total_var_95 = self._calculate_portfolio_var(results)
        total_cvar_95 = self._calculate_portfolio_cvar(results)
        worst_case = min(results, key=lambda r: r.profit_loss)
        
        return {
            'scenarios': results,
            'summary': {
                'var_95': total_var_95,
                'cvar_95': total_cvar_95,
                'worst_case_scenario': worst_case.scenario_name,
                'worst_case_loss': worst_case.profit_loss,
                'expected_shortfall': self._calculate_expected_shortfall(results)
            }
        }
        
    def _run_single_scenario(self, name: str, scenario: Dict[str, Any],
                           portfolio_value: float, 
                           positions: List[Dict[str, Any]]) -> StressTestResult:
        """Run a single stress test scenario."""
        total_pnl = 0
        
        for position in positions:
            # Calculate position P&L under scenario
            position_pnl = self._calculate_position_pnl(position, scenario)
            total_pnl += position_pnl
            
        # Calculate risk metrics
        var_95 = self._calculate_var_for_scenario(total_pnl, scenario)
        cvar_95 = self._calculate_cvar_for_scenario(total_pnl, scenario)
        
        # Estimate recovery time (simplified)
        recovery_days = int(abs(total_pnl) / (portfolio_value * 0.001))  # 0.1% daily return
        
        return StressTestResult(
            scenario_name=name,
            portfolio_value=portfolio_value + total_pnl,
            profit_loss=total_pnl,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=min(0, total_pnl / portfolio_value),
            recovery_time=recovery_days,
            probability=scenario['probability']
        )
        
    def _calculate_position_pnl(self, position: Dict[str, Any], 
                              scenario: Dict[str, Any]) -> float:
        """Calculate P&L for a single position under scenario."""
        strike = position.get('strike', 0)
        current_price = position.get('current_price', strike)
        contracts = position.get('contracts', 1)
        position_type = position.get('type', 'put')
        
        # New price under scenario
        new_price = current_price * (1 + scenario['price_change'])
        
        if position_type == 'put':
            # Short put P&L
            if new_price < strike:
                # Assigned - loss
                pnl = (new_price - strike) * 100 * contracts
            else:
                # Expires worthless - keep premium
                pnl = position.get('premium', 0) * 100 * contracts
        else:
            # Add logic for calls/stocks if needed
            pnl = 0
            
        return pnl
        
    def _calculate_var_for_scenario(self, pnl: float, 
                                  scenario: Dict[str, Any]) -> float:
        """Calculate VaR for a specific scenario."""
        # Use normal distribution with increased volatility
        vol_adjustment = scenario['volatility_multiplier']
        
        # Simple VaR calculation
        var_95 = abs(pnl) * vol_adjustment * stats.norm.ppf(0.05)
        
        return var_95
        
    def _calculate_cvar_for_scenario(self, pnl: float,
                                   scenario: Dict[str, Any]) -> float:
        """Calculate CVaR (expected shortfall) for scenario."""
        # CVaR is typically 20-40% worse than VaR
        var_95 = self._calculate_var_for_scenario(pnl, scenario)
        cvar_95 = var_95 * 1.3
        
        return cvar_95
        
    def _calculate_portfolio_var(self, results: List[StressTestResult]) -> float:
        """Calculate portfolio VaR across all scenarios."""
        # Weight by probability
        weighted_vars = []
        
        for result in results:
            weighted_vars.append(result.var_95 * result.probability)
            
        # Sum weighted VaRs (simplified - ignores correlation)
        portfolio_var = sum(weighted_vars)
        
        return portfolio_var
        
    def _calculate_portfolio_cvar(self, results: List[StressTestResult]) -> float:
        """Calculate portfolio CVaR across all scenarios."""
        # Weight by probability
        weighted_cvars = []
        
        for result in results:
            weighted_cvars.append(result.cvar_95 * result.probability)
            
        portfolio_cvar = sum(weighted_cvars)
        
        return portfolio_cvar
        
    def _calculate_expected_shortfall(self, results: List[StressTestResult]) -> float:
        """Calculate expected shortfall (average of worst 5% outcomes)."""
        # Sort by P&L
        sorted_results = sorted(results, key=lambda r: r.profit_loss)
        
        # Take worst outcomes based on probability
        cumulative_prob = 0
        shortfall_sum = 0
        count = 0
        
        for result in sorted_results:
            if cumulative_prob < 0.05:
                shortfall_sum += result.profit_loss
                cumulative_prob += result.probability
                count += 1
                
        return shortfall_sum / max(count, 1)