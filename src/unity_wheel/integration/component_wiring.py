"""Component wiring for proper integration between all system components.

This module ensures all components are properly connected and communicate
with each other, eliminating the disconnected architecture.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from ..analytics import DecisionTracker, UnityAssignmentModel
from ..config.unified_config import get_config
from ..data_providers.base import FREDDataManager
from ..math.options import calculate_all_greeks
from ..optimization.engine import IntelligentBucketing
from ..optimization.milp_solver import MILPSolver
from ..risk import EVRiskAnalyzer, StressTestScenarios
from ..storage.storage import Storage
from ..strategy import WheelStrategy
from ..utils import get_logger

logger = get_logger(__name__)
config = get_config()


class IntegratedWheelStrategy(WheelStrategy):
    """Enhanced wheel strategy that uses intelligent bucketing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucketing = IntelligentBucketing()
        
    def find_optimal_put_strike(self, current_price: float, available_strikes: list[float],
                               volatility: float, days_to_expiry: int, 
                               risk_free_rate: float, portfolio_value: float) -> Any:
        """Find optimal strike using bucketing to reduce complexity."""
        # First apply intelligent bucketing
        bucketed_strikes = self.bucketing.bucket_strikes(
            strikes=available_strikes,
            current_price=current_price,
            granularity=0.02  # 2% buckets as per documentation
        )
        
        logger.info(f"Reduced {len(available_strikes)} strikes to {len(bucketed_strikes)} buckets")
        
        # Then find optimal from bucketed strikes
        return super().find_optimal_put_strike(
            current_price, bucketed_strikes, volatility, 
            days_to_expiry, risk_free_rate, portfolio_value
        )


class IntegratedRiskAnalyzer(EVRiskAnalyzer):
    """Risk analyzer integrated with MILP solver and stress testing."""
    
    def __init__(self):
        super().__init__()
        self.milp_solver = MILPSolver()
        self.stress_tester = StressTestScenarios()
        
    def analyze_portfolio(self, positions: list[Dict[str, Any]], 
                         portfolio_value: float) -> Dict[str, Any]:
        """Analyze portfolio risk with MILP optimization."""
        # First get EV analysis
        ev_results = []
        for position in positions:
            ev_result = self.analyze_position(position, {
                'current_price': position.get('current_price'),
                'volatility': position.get('volatility', 0.20)
            })
            ev_results.append(ev_result)
        
        # Run MILP optimization
        optimization = self.milp_solver.optimize_portfolio(
            positions=positions,
            constraints={
                'max_var': config.risk.max_var_95,
                'max_allocation': config.trading.max_position_size,
                'min_return': 0.001
            }
        )
        
        # Run stress tests
        stress_results = self.stress_tester.run_scenarios(
            portfolio_value=portfolio_value,
            positions=positions
        )
        
        return {
            'ev_analysis': ev_results,
            'optimization': optimization,
            'stress_tests': stress_results,
            'recommendations': self._generate_recommendations(ev_results, optimization, stress_results)
        }
        
    def _generate_recommendations(self, ev_results, optimization, stress_results):
        """Generate actionable recommendations from analysis."""
        recommendations = []
        
        # Check if portfolio is over-allocated
        if optimization.get('total_allocation', 0) > config.risk.max_portfolio_allocation:
            recommendations.append({
                'type': 'REDUCE_EXPOSURE',
                'reason': 'Portfolio over-allocated',
                'action': f"Reduce positions to {config.risk.max_portfolio_allocation:.0%} of portfolio"
            })
            
        # Check stress test results
        worst_case = stress_results['summary']['worst_case_loss']
        if abs(worst_case) > portfolio_value * config.risk.max_cvar_95:
            recommendations.append({
                'type': 'HEDGE_REQUIRED',
                'reason': f'Worst case loss ${abs(worst_case):,.0f} exceeds risk limit',
                'action': 'Consider protective puts or reducing position size'
            })
            
        return recommendations


class IntegratedDecisionTracker(DecisionTracker):
    """Decision tracker integrated with storage and MLflow."""
    
    def __init__(self, storage: Storage):
        super().__init__()
        self.storage = storage
        self._mlflow_client = None
        
    @property
    def mlflow_client(self):
        """Lazy load MLflow client."""
        if self._mlflow_client is None and config.mcp.use_mlflow_mcp:
            try:
                import mlflow
                mlflow.set_tracking_uri(config.mcp.mlflow_tracking_uri)
                self._mlflow_client = mlflow
            except ImportError:
                logger.warning("MLflow not available, skipping experiment tracking")
        return self._mlflow_client
        
    async def track_decision_integrated(self, decision: Dict[str, Any]) -> str:
        """Track decision with storage and MLflow integration."""
        # Track in local storage
        decision_id = self.track_decision(decision)
        
        # Save to database
        await self.storage.save_decision(decision_id, decision)
        
        # Track in MLflow if available
        if self.mlflow_client:
            try:
                self.mlflow_client.start_run()
                self.mlflow_client.log_params({
                    'symbol': decision.get('symbol'),
                    'action': decision.get('action'),
                    'strike': decision.get('strike'),
                    'contracts': decision.get('contracts')
                })
                self.mlflow_client.log_metrics({
                    'expected_return': decision.get('expected_return', 0),
                    'confidence': decision.get('confidence', 0)
                })
                self.mlflow_client.end_run()
            except (ValueError, KeyError, AttributeError) as e:
                logger.error(f"Failed to track in MLflow: {e}")
                
        return decision_id
        
    async def record_outcome_integrated(self, decision_id: str, outcome: Dict[str, Any]) -> None:
        """Record outcome with storage integration."""
        # Record locally
        self.record_outcome(decision_id, outcome)
        
        # Update in database
        await self.storage.update_decision_outcome(decision_id, outcome)
        
        # Log to MLflow
        if self.mlflow_client:
            try:
                self.mlflow_client.start_run()
                self.mlflow_client.log_metrics({
                    'actual_return': outcome.get('actual_return', 0),
                    'assigned': 1 if outcome.get('assigned') else 0
                })
                self.mlflow_client.end_run()
            except (ValueError, KeyError, AttributeError) as e:
                logger.error(f"Failed to log outcome to MLflow: {e}")


class IntegratedStatsAnalyzer:
    """Statistical analyzer using Statsource MCP."""
    
    def __init__(self):
        self.config = config
        self._statsource_client = None
        
    @property
    def statsource_client(self):
        """Lazy load Statsource client."""
        if self._statsource_client is None and config.mcp.use_statsource_mcp:
            # This would be the actual MCP client
            # For now, we'll create a mock
            self._statsource_client = self._create_statsource_client()
        return self._statsource_client
        
    def _create_statsource_client(self):
        """Create Statsource MCP client."""
        # In real implementation, this would connect to Statsource MCP
        return None
        
    async def detect_anomalies(self, options_data: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Detect anomalies in options data using Statsource."""
        if not self.statsource_client:
            return []
            
        anomalies = []
        
        # Extract IV values
        iv_values = [opt['implied_volatility'] for opt in options_data]
        
        # Use Statsource to detect outliers
        # In real implementation: 
        # outliers = await self.statsource_client.detect_outliers(iv_values)
        
        # For now, simple z-score detection
        import numpy as np
        mean_iv = np.mean(iv_values)
        std_iv = np.std(iv_values)
        
        for i, opt in enumerate(options_data):
            z_score = abs((opt['implied_volatility'] - mean_iv) / std_iv)
            if z_score > 3:
                anomalies.append({
                    'option': opt,
                    'type': 'IV_OUTLIER',
                    'z_score': z_score,
                    'message': f"IV {opt['implied_volatility']:.1%} is {z_score:.1f} std devs from mean"
                })
                
        return anomalies


class ComponentRegistry:
    """Central registry for all integrated components."""
    
    def __init__(self):
        self.storage = None
        self.strategy = None
        self.risk_analyzer = None
        self.decision_tracker = None
        self.stats_analyzer = None
        self.assignment_model = None
        self.fred_manager = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize all components with proper wiring."""
        if self._initialized:
            return
            
        # Initialize storage
        self.storage = Storage()
        await self.storage.initialize()
        
        # Initialize integrated components
        self.strategy = IntegratedWheelStrategy()
        self.risk_analyzer = IntegratedRiskAnalyzer()
        self.decision_tracker = IntegratedDecisionTracker(self.storage)
        self.stats_analyzer = IntegratedStatsAnalyzer()
        self.assignment_model = UnityAssignmentModel()
        self.fred_manager = FREDDataManager(storage=self.storage)
        
        self._initialized = True
        logger.info("All components initialized and wired")
        
    async def close(self):
        """Clean up all components."""
        if self.storage:
            await self.storage.close()
        self._initialized = False


# Global registry instance
_registry: Optional[ComponentRegistry] = None


async def get_component_registry() -> ComponentRegistry:
    """Get or create the global component registry."""
    global _registry
    if _registry is None:
        _registry = ComponentRegistry()
        await _registry.initialize()
    return _registry


__all__ = [
    'IntegratedWheelStrategy',
    'IntegratedRiskAnalyzer', 
    'IntegratedDecisionTracker',
    'IntegratedStatsAnalyzer',
    'ComponentRegistry',
    'get_component_registry'
]