"""Parameter evolution system for tracking and updating all static values."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config.unified_config import get_config
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvolvingParameter:
    """Represents a parameter that can evolve over time."""
    name: str
    current_value: float
    initial_value: float
    bounds: Tuple[float, float]
    description: str
    component: str
    last_updated: datetime = field(default_factory=datetime.now)
    update_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_impact: float = 0.0
    
    def update(self, new_value: float, reason: str, performance_delta: float = 0.0):
        """Update parameter value."""
        old_value = self.current_value
        self.current_value = np.clip(new_value, self.bounds[0], self.bounds[1])
        self.last_updated = datetime.now()
        
        # Track update
        self.update_history.append({
            'timestamp': self.last_updated,
            'old_value': old_value,
            'new_value': self.current_value,
            'reason': reason,
            'performance_delta': performance_delta
        })
        
        # Update performance impact
        self.performance_impact += performance_delta
        
        logger.info(
            f"Parameter {self.name} updated: {old_value:.4f} → {self.current_value:.4f} "
            f"(reason: {reason})"
        )


class ParameterEvolution:
    """
    Central system for managing all evolving parameters.
    
    This replaces static values throughout the codebase with dynamically
    learned values based on trading outcomes.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize parameter evolution system."""
        self.config = get_config()
        self.storage_path = storage_path or Path("data/parameter_evolution.json")
        
        # Initialize all evolving parameters
        self.parameters: Dict[str, EvolvingParameter] = self._initialize_parameters()
        
        # Load saved state if exists
        self._load_state()
        
        # Track global performance
        self.global_performance = {
            'total_updates': 0,
            'positive_updates': 0,
            'negative_updates': 0,
            'cumulative_impact': 0.0
        }
    
    def _initialize_parameters(self) -> Dict[str, EvolvingParameter]:
        """Initialize all parameters that should evolve."""
        params = {}
        
        # Anomaly Detection Parameters
        params['anomaly.z_score_threshold'] = EvolvingParameter(
            name='anomaly.z_score_threshold',
            current_value=2.5,
            initial_value=2.5,
            bounds=(1.5, 4.0),
            description='Z-score threshold for anomaly detection',
            component='AnomalyDetector'
        )
        
        params['anomaly.volume_spike_ratio'] = EvolvingParameter(
            name='anomaly.volume_spike_ratio',
            current_value=5.0,
            initial_value=5.0,
            bounds=(2.0, 10.0),
            description='Volume ratio for spike detection',
            component='AnomalyDetector'
        )
        
        params['anomaly.gap_threshold'] = EvolvingParameter(
            name='anomaly.gap_threshold',
            current_value=0.10,
            initial_value=0.10,
            bounds=(0.05, 0.20),
            description='Price gap threshold for anomaly',
            component='AnomalyDetector'
        )
        
        # Position Sizing Parameters
        params['sizing.base_confidence'] = EvolvingParameter(
            name='sizing.base_confidence',
            current_value=0.95,
            initial_value=0.95,
            bounds=(0.80, 0.99),
            description='Base confidence level for position sizing',
            component='PositionSizer'
        )
        
        params['sizing.kelly_min'] = EvolvingParameter(
            name='sizing.kelly_min',
            current_value=0.10,
            initial_value=0.10,
            bounds=(0.05, 0.20),
            description='Minimum Kelly fraction',
            component='PositionSizer'
        )
        
        params['sizing.kelly_max'] = EvolvingParameter(
            name='sizing.kelly_max',
            current_value=0.40,
            initial_value=0.40,
            bounds=(0.30, 0.60),
            description='Maximum Kelly fraction',
            component='PositionSizer'
        )
        
        # Wheel Strategy Parameters
        params['wheel.delta_threshold'] = EvolvingParameter(
            name='wheel.delta_threshold',
            current_value=0.05,
            initial_value=0.05,
            bounds=(0.01, 0.10),
            description='Delta threshold for confidence adjustment',
            component='WheelStrategy'
        )
        
        params['wheel.premium_ratio_threshold'] = EvolvingParameter(
            name='wheel.premium_ratio_threshold',
            current_value=0.02,
            initial_value=0.02,
            bounds=(0.01, 0.05),
            description='Minimum premium/strike ratio',
            component='WheelStrategy'
        )
        
        params['wheel.theta_threshold'] = EvolvingParameter(
            name='wheel.theta_threshold',
            current_value=-1.0,
            initial_value=-1.0,
            bounds=(-5.0, -0.1),
            description='Minimum theta ($/day)',
            component='WheelStrategy'
        )
        
        # Stress Testing Parameters
        params['stress.market_crash_magnitude'] = EvolvingParameter(
            name='stress.market_crash_magnitude',
            current_value=-0.20,
            initial_value=-0.20,
            bounds=(-0.40, -0.10),
            description='Market crash scenario magnitude',
            component='StressTestScenarios'
        )
        
        params['stress.var_multiplier'] = EvolvingParameter(
            name='stress.var_multiplier',
            current_value=1.65,
            initial_value=1.65,
            bounds=(1.2, 2.5),
            description='VaR to capital multiplier',
            component='StressTestScenarios'
        )
        
        # Transaction Cost Parameters
        params['txcost.typical_spread_min'] = EvolvingParameter(
            name='txcost.typical_spread_min',
            current_value=0.05,
            initial_value=0.05,
            bounds=(0.01, 0.10),
            description='Typical minimum spread',
            component='TransactionCostModel'
        )
        
        params['txcost.size_impact_per_contract'] = EvolvingParameter(
            name='txcost.size_impact_per_contract',
            current_value=0.01,
            initial_value=0.01,
            bounds=(0.001, 0.05),
            description='Price impact per contract',
            component='TransactionCostModel'
        )
        
        # Regime Detection Parameters
        params['regime.low_vol_kelly'] = EvolvingParameter(
            name='regime.low_vol_kelly',
            current_value=0.50,
            initial_value=0.50,
            bounds=(0.30, 0.70),
            description='Kelly fraction for low volatility regime',
            component='RegimeDetector'
        )
        
        params['regime.mid_vol_kelly'] = EvolvingParameter(
            name='regime.mid_vol_kelly',
            current_value=0.33,
            initial_value=0.33,
            bounds=(0.20, 0.50),
            description='Kelly fraction for medium volatility regime',
            component='RegimeDetector'
        )
        
        params['regime.high_vol_kelly'] = EvolvingParameter(
            name='regime.high_vol_kelly',
            current_value=0.25,
            initial_value=0.25,
            bounds=(0.10, 0.40),
            description='Kelly fraction for high volatility regime',
            component='RegimeDetector'
        )
        
        # Bucketing Parameters
        params['bucketing.granularity'] = EvolvingParameter(
            name='bucketing.granularity',
            current_value=0.02,
            initial_value=0.02,
            bounds=(0.01, 0.05),
            description='Strike bucket granularity',
            component='IntelligentBucketing'
        )
        
        return params
    
    def get_parameter(self, name: str) -> float:
        """Get current value of a parameter."""
        if name in self.parameters:
            return self.parameters[name].current_value
        else:
            logger.warning(f"Unknown parameter: {name}")
            return None
    
    def update_parameter(
        self,
        name: str,
        new_value: float,
        reason: str,
        performance_delta: float = 0.0
    ) -> bool:
        """
        Update a parameter value.
        
        Args:
            name: Parameter name
            new_value: New value
            reason: Reason for update
            performance_delta: Performance impact of this change
            
        Returns:
            True if update was successful
        """
        if name not in self.parameters:
            logger.error(f"Unknown parameter: {name}")
            return False
        
        param = self.parameters[name]
        param.update(new_value, reason, performance_delta)
        
        # Update global stats
        self.global_performance['total_updates'] += 1
        if performance_delta > 0:
            self.global_performance['positive_updates'] += 1
        elif performance_delta < 0:
            self.global_performance['negative_updates'] += 1
        self.global_performance['cumulative_impact'] += performance_delta
        
        # Save state
        self._save_state()
        
        return True
    
    def batch_update(self, updates: List[Dict[str, Any]]) -> int:
        """
        Apply multiple parameter updates.
        
        Args:
            updates: List of update dicts with keys: name, value, reason, performance_delta
            
        Returns:
            Number of successful updates
        """
        successful = 0
        for update in updates:
            if self.update_parameter(
                update['name'],
                update['value'],
                update['reason'],
                update.get('performance_delta', 0.0)
            ):
                successful += 1
        
        return successful
    
    def get_component_parameters(self, component: str) -> Dict[str, float]:
        """Get all parameters for a specific component."""
        return {
            name: param.current_value
            for name, param in self.parameters.items()
            if param.component == component
        }
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """Generate report on parameter evolution."""
        report = {
            'global_stats': self.global_performance,
            'parameters': {}
        }
        
        for name, param in self.parameters.items():
            report['parameters'][name] = {
                'current_value': param.current_value,
                'initial_value': param.initial_value,
                'change_pct': (param.current_value - param.initial_value) / param.initial_value * 100,
                'update_count': len(param.update_history),
                'last_updated': param.last_updated.isoformat(),
                'performance_impact': param.performance_impact,
                'bounds': param.bounds
            }
        
        return report
    
    def reset_parameter(self, name: str) -> bool:
        """Reset a parameter to its initial value."""
        if name not in self.parameters:
            return False
        
        param = self.parameters[name]
        param.update(
            param.initial_value,
            "Reset to initial value",
            -param.performance_impact  # Undo performance impact
        )
        
        return True
    
    def _save_state(self) -> None:
        """Save current state to disk."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'global_performance': self.global_performance,
            'parameters': {}
        }
        
        for name, param in self.parameters.items():
            state['parameters'][name] = {
                'current_value': param.current_value,
                'last_updated': param.last_updated.isoformat(),
                'performance_impact': param.performance_impact,
                'update_count': len(param.update_history)
            }
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self) -> None:
        """Load saved state from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                state = json.load(f)
            
            # Restore global performance
            self.global_performance.update(state.get('global_performance', {}))
            
            # Restore parameter values
            for name, param_state in state.get('parameters', {}).items():
                if name in self.parameters:
                    param = self.parameters[name]
                    param.current_value = param_state['current_value']
                    param.last_updated = datetime.fromisoformat(param_state['last_updated'])
                    param.performance_impact = param_state['performance_impact']
            
            logger.info(f"Loaded parameter evolution state from {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to load parameter evolution state: {e}")