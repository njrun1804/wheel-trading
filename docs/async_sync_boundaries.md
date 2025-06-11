# Async/Sync Boundaries Architecture

This document defines the clear boundaries between async and sync code in the Unity Wheel Trading Bot.

## Design Principles

1. **Async at I/O Boundaries**: All external I/O operations (API calls, database access) should be async
2. **Sync for Compute**: Pure computational functions (math, analysis) remain synchronous
3. **Clear Interfaces**: Async modules provide sync wrappers for easy integration
4. **No Mixed Execution**: A module is either fully async or fully sync, not mixed

## Module Classification

### Async Modules (I/O Bound)

These modules handle external I/O and should remain async:

```
src/unity_wheel/
├── data_providers/          # All async - external APIs
│   ├── databento/          # Market data API (async)
│   ├── schwab/             # Broker API (async)
│   └── fred/               # Economic data API (async)
├── storage/                # Database operations (async)
│   ├── storage.py          # Base storage interface
│   └── duckdb_cache.py     # Cache operations
├── auth/                   # Authentication (async)
│   ├── oauth.py            # OAuth flows
│   └── auth_client.py      # Auth client
└── monitoring/scripts/     # Live monitoring (async)
```

### Sync Modules (Compute Bound)

These modules perform calculations and should remain sync:

```
src/unity_wheel/
├── math/                   # Pure calculations (sync)
│   └── options.py          # Black-Scholes, Greeks
├── risk/                   # Risk calculations (sync)
│   ├── analytics.py        # VaR, CVaR calculations
│   └── limits.py           # Risk limit checks
├── strategy/               # Strategy logic (sync)
│   └── wheel.py            # Wheel strategy calculations
├── adaptive/               # Adaptive logic (sync)
│   └── adaptive_wheel.py   # Adaptive adjustments
└── utils/                  # Utilities (sync)
    └── position_sizing.py  # Position calculations
```

### Bridge Modules

These modules bridge async and sync code:

```
src/unity_wheel/
├── api/                    # API layer (sync with async calls)
│   └── advisor.py          # Main advisor (sync interface)
├── cli/                    # CLI interface (sync entry)
│   └── run.py              # Main entry point
└── analytics/              # Mixed analytics
    ├── decision_engine.py  # Async data fetch, sync analysis
    └── market_calibrator.py # Async data, sync calibration
```

## Implementation Pattern

### 1. Async Module Pattern

```python
# data_providers/databento/client.py
class DatabentoClient:
    async def get_option_chain(self, symbol: str) -> OptionChain:
        """Async method for API calls."""
        response = await self._api_call(...)
        return response

    def get_option_chain_sync(self, symbol: str) -> OptionChain:
        """Sync wrapper for easy integration."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.get_option_chain(symbol))
        finally:
            loop.close()
```

### 2. Sync Module Pattern

```python
# math/options.py
def black_scholes_price_validated(S, K, T, r, sigma, option_type):
    """Pure sync calculation - no async needed."""
    # Mathematical calculations only
    return CalculationResult(value=price, confidence=0.99)
```

### 3. Bridge Module Pattern

```python
# api/advisor.py
class WheelAdvisor:
    def advise_position(self, market_snapshot):
        """Sync interface that may call async data providers."""
        # Use sync wrappers when needed
        if need_fresh_data:
            data = self.databento_client.get_option_chain_sync(symbol)

        # All calculations remain sync
        recommendation = self.strategy.find_optimal_strike(...)
        return recommendation
```

## Migration Strategy

### Phase 1: Create Sync Wrappers (Completed)
- ✅ Add sync wrappers to async data providers
- ✅ Ensure all modules can be called synchronously

### Phase 2: Define Clear Interfaces (In Progress)
- Create `AsyncDataProvider` and `SyncDataProvider` base classes
- Enforce interface contracts

### Phase 3: Separate Mixed Modules
- Split modules that mix async/sync code
- Move async parts to data layer
- Keep sync parts in business logic

### Phase 4: Document Dependencies
- Create dependency graph showing async/sync boundaries
- Ensure no circular dependencies cross boundaries

## Best Practices

1. **Never mix async/sync in same class**: A class should be either fully async or fully sync
2. **Use sync wrappers sparingly**: Only at integration boundaries
3. **Prefer composition**: Async data providers + sync business logic
4. **Test both interfaces**: Ensure sync wrappers work correctly
5. **Document async requirements**: Make it clear which modules require async context

## Common Patterns

### Getting Data Synchronously
```python
# Instead of forcing async everywhere:
# BAD: async def calculate_position_size(...):

# GOOD: Sync calculation with sync data access
def calculate_position_size(portfolio_value: float, ...):
    # If fresh data needed, use sync wrapper
    if need_market_data:
        data = data_provider.get_data_sync()
    # Rest remains sync
    return calculate_size(data, portfolio_value)
```

### Async Data Collection
```python
# Keep async at data collection boundary
async def collect_market_data(symbols: List[str]):
    tasks = [get_option_chain(s) for s in symbols]
    return await asyncio.gather(*tasks)

# Convert to sync for business logic
def analyze_market(symbols: List[str]):
    # Use sync wrapper
    data = collect_market_data_sync(symbols)
    # Sync analysis
    return analyze(data)
```

## Testing Strategy

1. **Unit tests**: Test async and sync interfaces separately
2. **Integration tests**: Test sync wrappers work correctly
3. **Performance tests**: Ensure sync wrappers don't create bottlenecks
4. **Error handling**: Test timeout and error propagation across boundaries
