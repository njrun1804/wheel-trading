# Unity Wheel Trading Bot - Critical Follow-On Work Plan

Based on unified maximum compute analysis, here's the prioritized work needed to complete the integration:

## 🚨 CRITICAL - Integration Blockers

### 1. Eliminate Hardcoded Values (95 instances)
**Impact**: Prevents true data flow and configuration management

```python
# Current (BAD):
symbol = "U"
portfolio_value = 100000
delta_target = 0.30

# Target (GOOD):
symbol = config.trading.symbol
portfolio_value = account.total_value
delta_target = strategy_params.target_delta
```

**Files to fix**:
- 44 instances of hardcoded "U" symbol
- 24 instances of hardcoded delta values
- 10 instances of hardcoded portfolio values

### 2. Complete Deprecated File Removal
**Impact**: Old code still being imported, causing confusion

- `src/unity_wheel/utils/__init__.py` - Still imports position_sizing_deprecated
- `src/unity_wheel/math/__init__.py` - Still imports options_deprecated
- Move all `*_deprecated.py` files to archive

### 3. Wire Up Missing Component Integrations
**Impact**: Components exist but don't communicate

Critical missing connections:
- ❌ **Bucketing → Strategy**: IntelligentBucketing not used by WheelStrategy
- ❌ **MILP → Risk**: MILPSolver not integrated with RiskAnalyzer  
- ❌ **Decision → Storage**: DecisionTracker doesn't save to Storage
- ❌ **Analytics → MLflow**: No experiment tracking
- ❌ **Risk → Statsource**: Stress tests don't use statistical analysis

## ⚡ HIGH PRIORITY - Performance & Reliability

### 4. Implement Arrow/Polars for Data Processing
**Current**: Using basic Python/pandas
**Target**: 13.5ms → <5ms query time

```python
# Add to storage layer
import polars as pl
import pyarrow.parquet as pq

class OptimizedStorage(Storage):
    def get_options_arrow(self, symbol: str) -> pa.Table:
        # Use Arrow for 10x faster queries
        pass
```

### 5. Add Comprehensive Integration Tests
**Current**: No integration tests
**Needed**:
- End-to-end workflow test
- Component integration tests
- MCP server integration tests
- Performance regression tests

### 6. Unify Configuration Management
**Current**: Multiple config files and hardcoded values
**Target**: Single source of truth

```python
# Unified config structure
class UnifiedConfig:
    trading: TradingConfig
    risk: RiskConfig
    storage: StorageConfig
    mcp: MCPConfig
    
    @classmethod
    def from_env(cls) -> UnifiedConfig:
        # Load from environment with validation
        pass
```

## 📊 MEDIUM PRIORITY - Optimization & Enhancement

### 7. Connect MCP Servers to Main Flow
- **DuckDB MCP**: Use for all options queries
- **MLflow MCP**: Track every decision and outcome
- **Statsource MCP**: Detect IV outliers in real-time
- **Memory MCP**: Store and recall successful patterns

### 8. Implement Decision Feedback Loop
```python
# Current: Decision made, outcome unknown
# Target: Complete learning cycle
async def complete_decision_cycle():
    decision = await system.make_decision()
    await tracker.track_decision(decision)
    
    # Later...
    outcome = await monitor.get_outcome(decision.id)
    await tracker.record_outcome(decision.id, outcome)
    await mlflow.log_metrics(outcome.metrics)
    
    # Learn and improve
    patterns = await analyzer.extract_patterns()
    await system.update_strategy(patterns)
```

### 9. Performance Benchmarking Suite
- Query performance tests
- Optimization algorithm benchmarks
- End-to-end latency measurements
- Memory usage profiling

## 🔧 Implementation Approach

### Phase 1: Foundation (1-2 days)
1. Create unified config system
2. Replace all hardcoded values
3. Complete deprecated removals

### Phase 2: Integration (2-3 days)
1. Wire component connections
2. Add integration tests
3. Connect MCP servers

### Phase 3: Optimization (2-3 days)
1. Implement Arrow/Polars
2. Add performance benchmarks
3. Optimize critical paths

### Phase 4: Validation (1-2 days)
1. Run full integration tests
2. Performance validation
3. Documentation update

## Success Metrics

- ✅ Zero hardcoded values
- ✅ All components connected
- ✅ <5ms query performance
- ✅ 100% deprecated code removed
- ✅ Full test coverage for integrations
- ✅ MCP servers actively used
- ✅ Decision learning cycle complete

## Validation with PyREPL

Use PyREPL to validate each phase:
```python
# Test configuration unification
await pyrepl.execute("""
from unity_wheel.config import UnifiedConfig
config = UnifiedConfig.from_env()
assert config.trading.symbol != "U"  # Not hardcoded
assert config.validate()  # All required fields
""")

# Test component integration
await pyrepl.execute("""
from unity_wheel.analytics import EnhancedWheelSystem
system = EnhancedWheelSystem(portfolio_value=200000)
rec = await system.generate_recommendation()
assert rec.optimization_details['method'] == 'intelligent_bucketing'
assert rec.decision_id in system.decision_tracker.decisions
""")
```

This plan addresses all critical gaps found in the unified maximum compute analysis and ensures true integration of all components.