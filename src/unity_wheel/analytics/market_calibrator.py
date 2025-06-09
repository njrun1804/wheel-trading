"""
Market calibrator that uses historical data to optimize wheel strategy parameters.
Adapts settings based on current market regime and historical performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from ..utils import get_logger
from ..risk.regime_detector import RegimeDetector, RegimeInfo

logger = get_logger(__name__)


@dataclass
class OptimalParameters:
    """Optimal parameters for wheel strategy based on market conditions."""
    # Strike selection
    put_delta_target: float
    call_delta_target: float
    delta_range: Tuple[float, float]  # Min/max acceptable
    
    # Expiration selection
    dte_target: int
    dte_range: Tuple[int, int]
    
    # Position sizing
    kelly_fraction: float
    max_position_pct: float
    
    # Risk thresholds
    max_var_95: float
    profit_target: float  # When to close early
    stop_loss: float  # Max acceptable loss
    
    # Rolling thresholds
    roll_at_dte: int
    roll_at_profit_pct: float
    roll_at_delta: float
    
    # Confidence in parameters
    confidence_score: float
    regime: str
    

class MarketCalibrator:
    """Calibrates strategy parameters using historical data."""
    
    def __init__(self, symbol: str = "U"):
        self.symbol = symbol
        self.regime_detector = RegimeDetector()
        self.historical_performance: Dict = {}
        
    async def calibrate_from_history(
        self,
        returns: np.ndarray,
        prices: pd.DataFrame,
        iv_history: Optional[pd.DataFrame] = None
    ) -> OptimalParameters:
        """
        Calibrate optimal parameters from historical data.
        
        Args:
            returns: Daily returns array
            prices: DataFrame with OHLC data
            iv_history: Optional implied volatility history
        """
        
        # 1. Detect current regime
        self.regime_detector.fit(returns)
        current_regime, confidence = self.regime_detector.get_current_regime(returns)
        
        logger.info(
            f"Calibrating for {current_regime.name}",
            volatility=f"{current_regime.volatility*100:.1f}%",
            confidence=f"{confidence:.1%}"
        )
        
        # 2. Calculate IV rank if available
        iv_rank = self._calculate_iv_rank(iv_history) if iv_history is not None else 50.0
        
        # 3. Analyze historical performance by regime
        regime_performance = self._analyze_regime_performance(returns, prices)
        
        # 4. Calculate optimal parameters based on regime
        optimal_params = self._calculate_optimal_parameters(
            current_regime,
            regime_performance,
            iv_rank
        )
        
        return optimal_params
    
    def _calculate_iv_rank(self, iv_history: pd.DataFrame) -> float:
        """Calculate current IV rank (0-100)."""
        if len(iv_history) < 252:  # Need 1 year
            return 50.0  # Default to middle
            
        current_iv = iv_history.iloc[-1]
        one_year_ivs = iv_history.iloc[-252:]
        
        # Calculate percentile
        iv_rank = stats.percentileofscore(one_year_ivs, current_iv)
        
        logger.info(f"IV Rank: {iv_rank:.0f}")
        return iv_rank
    
    def _analyze_regime_performance(
        self,
        returns: np.ndarray,
        prices: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Analyze performance metrics by regime."""
        
        performance = {}
        
        for regime_id, regime_info in self.regime_detector.regime_info.items():
            # Calculate regime-specific metrics
            regime_metrics = {
                'avg_move_30d': self._calculate_avg_move(prices, 30, regime_info.volatility),
                'avg_move_45d': self._calculate_avg_move(prices, 45, regime_info.volatility),
                'avg_move_60d': self._calculate_avg_move(prices, 60, regime_info.volatility),
                'breach_prob_20': self._calculate_breach_probability(returns, 0.20, 30),
                'breach_prob_30': self._calculate_breach_probability(returns, 0.30, 30),
                'optimal_dte': self._find_optimal_dte(prices, regime_info.volatility),
                'win_rate': self._estimate_win_rate(returns, regime_info.volatility)
            }
            
            performance[regime_info.name] = regime_metrics
            
        return performance
    
    def _calculate_avg_move(
        self,
        prices: pd.DataFrame,
        days: int,
        volatility: float
    ) -> float:
        """Calculate average price movement over N days."""
        # Use volatility to estimate expected move
        # Annual vol * sqrt(days/252) = expected move
        return volatility * np.sqrt(days / 252)
    
    def _calculate_breach_probability(
        self,
        returns: np.ndarray,
        threshold: float,
        days: int
    ) -> float:
        """Calculate probability of breaching a threshold in N days."""
        # Calculate N-day returns
        cumulative_returns = pd.Series(returns).rolling(days).sum()
        breaches = (cumulative_returns < -threshold).sum()
        
        return breaches / len(cumulative_returns.dropna())
    
    def _find_optimal_dte(
        self,
        prices: pd.DataFrame,
        volatility: float
    ) -> int:
        """Find optimal days to expiration based on volatility."""
        # Higher volatility = shorter DTE preferred
        if volatility < 0.30:  # 30% annual
            return 45
        elif volatility < 0.60:  # 60% annual
            return 35
        else:
            return 25  # High vol - stay short
    
    def _estimate_win_rate(
        self,
        returns: np.ndarray,
        volatility: float
    ) -> float:
        """Estimate win rate for wheel strategy in this volatility regime."""
        # Simplified: Lower volatility = higher win rate
        # This would be refined with actual backtest data
        base_win_rate = 0.85
        vol_adjustment = max(0, (0.50 - volatility) * 0.3)
        
        return min(0.95, base_win_rate + vol_adjustment)
    
    def _calculate_optimal_parameters(
        self,
        current_regime: RegimeInfo,
        regime_performance: Dict,
        iv_rank: float
    ) -> OptimalParameters:
        """Calculate optimal parameters for current conditions."""
        
        perf = regime_performance[current_regime.name]
        
        # Delta targets based on regime and IV rank
        if current_regime.volatility < 0.40:  # Low vol
            put_delta = 0.30 if iv_rank > 50 else 0.25
            call_delta = 0.20
        elif current_regime.volatility < 0.80:  # Medium vol
            put_delta = 0.25 if iv_rank > 50 else 0.20
            call_delta = 0.15
        else:  # High vol
            put_delta = 0.20 if iv_rank > 50 else 0.15
            call_delta = 0.10
        
        # DTE based on optimal finding
        dte_target = perf['optimal_dte']
        
        # Position sizing from regime
        kelly = current_regime.kelly_fraction
        
        # Adjust for IV rank
        if iv_rank > 75:  # High IV
            kelly *= 1.2  # Can be more aggressive
            put_delta += 0.05
        elif iv_rank < 25:  # Low IV
            kelly *= 0.8  # Be conservative
            put_delta -= 0.05
        
        # Risk limits based on regime
        max_var = abs(current_regime.var_95) * 1.5  # 1.5x daily VaR
        
        # Rolling thresholds
        if current_regime.volatility > 0.60:  # High vol
            roll_profit = 0.25  # Take profits early
            roll_dte = 14  # Roll sooner
        else:
            roll_profit = 0.50  # Standard
            roll_dte = 21
        
        return OptimalParameters(
            # Strike selection
            put_delta_target=put_delta,
            call_delta_target=call_delta,
            delta_range=(put_delta - 0.10, put_delta + 0.10),
            
            # Expiration
            dte_target=dte_target,
            dte_range=(dte_target - 7, dte_target + 14),
            
            # Position sizing
            kelly_fraction=kelly,
            max_position_pct=0.20,  # Never more than 20%
            
            # Risk thresholds
            max_var_95=max_var,
            profit_target=roll_profit,
            stop_loss=max_var * 2,  # 2x daily VaR
            
            # Rolling
            roll_at_dte=roll_dte,
            roll_at_profit_pct=roll_profit,
            roll_at_delta=put_delta + 0.20,  # Roll if delta increases by 0.20
            
            # Meta
            confidence_score=confidence,
            regime=current_regime.name
        )
    
    def generate_trading_rules(self, params: OptimalParameters) -> List[str]:
        """Generate human-readable trading rules."""
        
        rules = [
            f"=== WHEEL STRATEGY RULES FOR {params.regime.upper()} ===",
            "",
            "ENTRY RULES:",
            f"1. Sell puts at {params.put_delta_target:.2f} delta (range: {params.delta_range[0]:.2f}-{params.delta_range[1]:.2f})",
            f"2. Target {params.dte_target} DTE (range: {params.dte_range[0]}-{params.dte_range[1]})",
            f"3. Position size: {params.kelly_fraction*100:.0f}% Kelly (max {params.max_position_pct*100:.0f}% of portfolio)",
            "",
            "MANAGEMENT RULES:",
            f"4. Take profits at {params.profit_target*100:.0f}% of max profit",
            f"5. Roll at {params.roll_at_dte} DTE or earlier",
            f"6. Roll if delta reaches {params.roll_at_delta:.2f}",
            f"7. Stop loss at {params.stop_loss*100:.0f}% of position",
            "",
            "RISK LIMITS:",
            f"8. Max daily VaR: {params.max_var_95*100:.1f}%",
            f"9. If assigned, sell calls at {params.call_delta_target:.2f} delta",
            "",
            f"Confidence: {params.confidence_score:.1%}"
        ]
        
        return rules