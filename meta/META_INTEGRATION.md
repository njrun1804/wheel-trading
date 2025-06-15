# Meta Integration Guide for Claude Code CLI

## Core Principle
Code can be made meta-aware for development workflow automation. The meta system provides file monitoring, quality checking, and template-based improvements.

## Required Meta Imports

### For Every Module
```python
# Required at top of every significant Python file
from meta_prime import MetaPrime
meta = MetaPrime()
meta.observe("module_load", {"module": __name__})
```

### For Critical Functions
```python
# For functions that make trading decisions
def make_trade_decision(self, data):
    meta.observe("trade_decision_start", {"data_points": len(data)})
    # Your logic here
    meta.observe("trade_decision_complete", {"decision": result})
    return result
```

## Meta Decorators

### Standard Meta Decorators
```python
# Use these decorators on key functions

@meta.observe_function("strategy_execution")
def execute_wheel_strategy(self):
    pass

@meta.observe_performance("risk_calculation")  
def calculate_portfolio_risk(self):
    pass

@meta.observe_errors("data_processing")
def process_market_data(self):
    pass
```

## File Structure Integration

### Required __meta__.py Files
Create `__meta__.py` in each package:

```python
# src/unity_wheel/__meta__.py
from meta_prime import MetaPrime

class UnityWheelMeta(MetaPrime):
    domain = "options_trading"
    component = "main_application"
    
    def observe_trading_session(self):
        pass

# Export for easy import
meta = UnityWheelMeta()
```

### Module-Level Integration
```python
# In each significant module (wheel.py, advisor.py, etc.)
from . import __meta__
meta = __meta__.meta

class WheelStrategy:
    def __init__(self):
        meta.observe("wheel_strategy_init", {"timestamp": time.time()})
```

## Event Observation Patterns

### Trading Events
```python
# Strategy events
meta.observe("strategy_start", {"strategy": "wheel"})
meta.observe("strategy_decision", {"action": "sell_put", "strike": 100})
meta.observe("strategy_complete", {"profit": 150.00})

# Risk events  
meta.observe("risk_check", {"portfolio_value": 10000})
meta.observe("risk_breach", {"limit": "max_loss", "current": -500})

# Market events
meta.observe("market_data", {"symbol": "U", "price": 105.50})
meta.observe("volatility_spike", {"iv": 0.35, "threshold": 0.30})
```

### System Events
```python
# Performance events
meta.observe("slow_query", {"duration": 2.5, "query": "options_chain"})
meta.observe("cache_hit", {"key": "unity_options", "hit_rate": 0.85})

# Error events
meta.observe("api_error", {"provider": "databento", "error": "rate_limit"})
meta.observe("data_quality", {"missing_records": 50})
```

## Code Evolution Patterns

### Self-Improving Functions
```python
def calculate_optimal_strike(self, data):
    # Meta system observes performance and outcomes
    meta.observe("strike_calculation_start", {"market_data": data})
    
    # Your calculation logic
    strike = self._calculate(data)
    
    # Meta system learns from results
    meta.observe("strike_calculation_result", {
        "strike": strike,
        "market_price": data.current_price,
        "iv": data.implied_volatility
    })
    
    return strike
```

### Adaptive Parameters
```python
class WheelStrategy:
    def __init__(self):
        # Meta system can evolve these parameters
        self.delta_threshold = meta.get_evolved_parameter("delta_threshold", default=0.30)
        self.dte_target = meta.get_evolved_parameter("dte_target", default=30)
        
    def adjust_parameters(self):
        # Meta system suggests parameter improvements
        suggestions = meta.get_parameter_suggestions()
        for param, value in suggestions.items():
            if meta.validate_parameter_change(param, value):
                setattr(self, param, value)
                meta.observe("parameter_evolution", {"param": param, "new_value": value})
```

## Integration with Existing Systems

### Database Integration
```python
# Meta-aware database operations
class MetaStorage(Storage):
    def query(self, sql):
        meta.observe("db_query_start", {"sql": sql[:100]})
        start_time = time.time()
        
        result = super().query(sql)
        
        meta.observe("db_query_complete", {
            "duration": time.time() - start_time,
            "rows": len(result) if result else 0
        })
        return result
```

### API Integration
```python
# Meta-aware API calls
class MetaAdvisor(Advisor):
    def get_recommendation(self, position_size):
        meta.observe("recommendation_request", {"position_size": position_size})
        
        recommendation = super().get_recommendation(position_size)
        
        meta.observe("recommendation_generated", {
            "action": recommendation.action,
            "confidence": recommendation.confidence,
            "expected_return": recommendation.expected_return
        })
        
        return recommendation
```

### Testing Integration
```python
# Meta-aware testing
class TestWheelStrategy:
    def setup_method(self):
        meta.observe("test_start", {"test": self._testMethodName})
        
    def teardown_method(self):
        meta.observe("test_complete", {"test": self._testMethodName})
        
    def test_wheel_execution(self):
        meta.observe("test_scenario", {"type": "wheel_execution"})
        # Test logic here
        assert result == expected
        meta.observe("test_assertion", {"passed": True})
```

## Hardware Integration

### M4 Pro Optimization
```python
# Meta system coordinates with hardware optimization
from src.unity_wheel.accelerated_tools.sequential_thinking_mac_optimized import get_sequential_thinking_turbo

class MetaOptimizedStrategy:
    def __init__(self):
        self.thinking_engine = get_sequential_thinking_turbo()
        meta.observe("hardware_optimization", {"cores": 12, "gpu_cores": 20})
        
    def think_through_decision(self, context):
        meta.observe("thinking_start", {"context_size": len(context)})
        
        result = self.thinking_engine.process(context)
        
        meta.observe("thinking_complete", {
            "processing_time": result.duration,
            "confidence": result.confidence
        })
        
        return result
```

## Error Handling and Recovery

### Meta-Aware Error Handling
```python
def safe_trading_operation(self):
    try:
        meta.observe("operation_start", {"type": "trading"})
        result = self.execute_trade()
        meta.observe("operation_success", {"result": result})
        return result
        
    except Exception as e:
        meta.observe("operation_error", {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "context": "trading_operation"
        })
        
        # Meta system can suggest recovery strategies
        recovery = meta.suggest_recovery("trading_operation", e)
        if recovery:
            return self.execute_recovery(recovery)
        raise
```

## Configuration Integration

### Meta-Driven Configuration
```python
# config.yaml with meta awareness
meta_config:
  enabled: true
  observation_level: "detailed"  # basic, detailed, comprehensive
  evolution_enabled: true
  hardware_optimization: "m4_pro"
  
trading_config:
  # Meta system can evolve these values
  delta_threshold: 0.30  # meta_evolvable
  max_position_size: 10000  # meta_evolvable
  risk_limit: 0.02  # meta_evolvable
```

### Loading Meta-Aware Config
```python
from meta_prime import MetaPrime
import yaml

def load_config():
    meta = MetaPrime()
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Meta system can override config values based on learning
    evolved_config = meta.evolve_configuration(config)
    
    meta.observe("config_loaded", {
        "original_values": len(config),
        "evolved_values": len(evolved_config)
    })
    
    return evolved_config
```

## Development Workflow

### Meta-Integrated Development Loop
1. **Code Change**: Make your change
2. **Meta Observation**: `meta.observe("code_change", details)`
3. **Validation**: `python meta_auditor.py --validate`
4. **Testing**: Run tests with meta observation active
5. **Evolution**: Let meta system suggest improvements
6. **Commit**: Include meta observations in commit

### Daily Development Routine
```bash
# Start of day
python meta_coordinator.py --daily-start

# During development
# Meta system automatically observes file changes

# End of day
python meta_coordinator.py --daily-summary
python meta_auditor.py --day-report
```

## Quality Assurance

### Meta-Enhanced QA
```python
# Quality checks with meta intelligence
def meta_quality_check():
    issues = meta.detect_quality_issues()
    
    for issue in issues:
        if issue.severity == "high":
            print(f"❌ {issue.description}")
            print(f"   Suggestion: {issue.suggested_fix}")
        elif issue.severity == "medium":
            print(f"⚠️  {issue.description}")
        else:
            print(f"ℹ️  {issue.description}")
    
    return len([i for i in issues if i.severity == "high"]) == 0
```

## Success Metrics

The meta integration is successful when:
- All major functions have meta observation
- System learns and improves from usage patterns  
- Performance metrics show continuous improvement
- Code quality metrics trend upward
- Meta system provides valuable insights and suggestions

## Troubleshooting

### Common Integration Issues
1. **Missing Meta Imports**: Add `from meta_prime import MetaPrime`
2. **No Observations**: Ensure `meta.observe()` calls are present
3. **Performance Impact**: Use `meta.set_observation_level("basic")`
4. **Meta System Unresponsive**: Restart with `python meta_coordinator.py --restart`

## Advanced Integration

### Custom Meta Extensions
```python
# Create domain-specific meta extensions
class TradingMeta(MetaPrime):
    def observe_trade_outcome(self, trade):
        # Custom trading-specific observations
        pass
        
    def suggest_strategy_improvements(self):
        # AI-driven strategy suggestions
        pass
        
    def detect_market_regime_change(self):
        # Market condition analysis
        pass
```

This integration ensures every part of the codebase benefits from meta-intelligence and continuous improvement.