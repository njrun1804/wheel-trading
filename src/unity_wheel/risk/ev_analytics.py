"""Expected Value (EV) based risk analytics."""
from __future__ import annotations


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config.unified_config import get_config
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EVAnalysis:
    """Results from EV-based risk analysis."""
    expected_value: float
    win_probability: float
    loss_probability: float
    expected_return: float
    risk_adjusted_ev: float
    confidence_interval: Tuple[float, float]
    recommendation: str


class EVRiskAnalyzer:
    """Analyzes risk using expected value calculations."""
    
    def __init__(self):
        """Initialize EV risk analyzer."""
        self.config = get_config()
        
    def analyze_position(
        self,
        strike: float,
        premium: float,
        underlying_price: float,
        volatility: float,
        days_to_expiry: int,
        position_type: str = 'CSP'
    ) -> EVAnalysis:
        """Analyze a position's expected value and risk."""
        # Calculate probabilities
        if position_type == 'CSP':
            # Probability of assignment (stock falling below strike)
            assignment_prob = self._calculate_assignment_probability(
                underlying_price, strike, volatility, days_to_expiry
            )
            win_prob = 1 - assignment_prob
        else:  # CC
            # Probability of assignment (stock rising above strike)
            assignment_prob = self._calculate_assignment_probability(
                underlying_price, strike, volatility, days_to_expiry, is_call=True
            )
            win_prob = 1 - assignment_prob
        
        # Calculate expected values
        if position_type == 'CSP':
            # Win: Keep premium
            win_value = premium * 100  # Per contract
            # Loss: Assigned at strike, own stock worth less
            avg_loss = (strike - underlying_price * 0.95) * 100  # Assume 5% avg drop
            loss_value = -avg_loss + premium * 100
        else:  # CC
            # Win: Keep premium + limited upside
            win_value = premium * 100
            # Loss: Stock called away, miss upside
            avg_gain_missed = (underlying_price * 1.05 - strike) * 100  # Assume 5% avg rise
            loss_value = -max(0, avg_gain_missed) + premium * 100
        
        # Calculate EV
        expected_value = win_prob * win_value + assignment_prob * loss_value
        
        # Calculate return metrics
        capital_required = strike * 100 * self.config.risk.margin_requirement
        expected_return = expected_value / capital_required
        
        # Risk adjustment
        risk_factor = 1 - (assignment_prob * self.config.risk.risk_aversion_factor)
        risk_adjusted_ev = expected_value * risk_factor
        
        # Confidence interval (simplified)
        std_dev = abs(win_value - loss_value) * np.sqrt(win_prob * assignment_prob)
        confidence_interval = (
            expected_value - 1.96 * std_dev,
            expected_value + 1.96 * std_dev
        )
        
        # Recommendation
        if risk_adjusted_ev > 0 and expected_return > self.config.risk.min_expected_return:
            recommendation = "PROCEED"
        elif expected_value > 0 and risk_adjusted_ev <= 0:
            recommendation = "CAUTION"
        else:
            recommendation = "AVOID"
        
        return EVAnalysis(
            expected_value=expected_value,
            win_probability=win_prob,
            loss_probability=assignment_prob,
            expected_return=expected_return,
            risk_adjusted_ev=risk_adjusted_ev,
            confidence_interval=confidence_interval,
            recommendation=recommendation
        )
    
    def _calculate_assignment_probability(
        self,
        spot: float,
        strike: float,
        volatility: float,
        days: int,
        is_call: bool = False
    ) -> float:
        """Calculate probability of assignment using simplified model."""
        # Convert to annualized
        time_to_expiry = days / 365.0
        
        # Calculate z-score
        if is_call:
            # Probability that S_T > K
            z = (np.log(spot / strike) + (0.02 - 0.5 * volatility**2) * time_to_expiry) / (
                volatility * np.sqrt(time_to_expiry)
            )
            prob = 1 - self._normal_cdf(z)
        else:
            # Probability that S_T < K  
            z = (np.log(spot / strike) + (0.02 - 0.5 * volatility**2) * time_to_expiry) / (
                volatility * np.sqrt(time_to_expiry)
            )
            prob = self._normal_cdf(-z)
        
        return prob
    
    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function for standard normal."""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))
    
    def analyze_portfolio(
        self,
        positions: List[Dict]
    ) -> Dict:
        """Analyze entire portfolio EV and risk."""
        total_ev = 0
        total_risk_adjusted_ev = 0
        position_analyses = []
        
        for pos in positions:
            analysis = self.analyze_position(
                strike=pos['strike'],
                premium=pos['premium'],
                underlying_price=pos['underlying_price'],
                volatility=pos['volatility'],
                days_to_expiry=pos['days_to_expiry'],
                position_type=pos.get('position_type', 'CSP')
            )
            position_analyses.append(analysis)
            total_ev += analysis.expected_value
            total_risk_adjusted_ev += analysis.risk_adjusted_ev
        
        return {
            'total_expected_value': total_ev,
            'total_risk_adjusted_ev': total_risk_adjusted_ev,
            'average_win_probability': np.mean([a.win_probability for a in position_analyses]),
            'positions': position_analyses
        }