"""Critical context for Codex when working across files.

This file serves as a bridge for AI code generation tools to understand
the key architectural decisions, current focus areas, and invariants
that must be maintained across the codebase.

IMPORTANT: This file is specifically for AI/Codex consumption.
"""

# ============================================================================
# ARCHITECTURE INVARIANTS (Never violate these)
# ============================================================================

INVARIANTS = {
    "ticker_configuration": """
    Unity ticker is ALWAYS configurable via config.unity.ticker
    NEVER hardcode 'U' - always use config.unity.ticker
    Example: ticker = get_config().unity.ticker
    """,
    "confidence_scores": """
    ALL calculations must return confidence scores
    Use CalculationResult(value, confidence, warnings) or (value, confidence) tuples
    Minimum confidence threshold: 0.30 (30%)
    """,
    "position_sizing": """
    Position sizing EXCLUSIVELY through DynamicPositionSizer
    NO other module should calculate position sizes
    Located at: src/unity_wheel/utils/position_sizing.py
    """,
    "error_handling": """
    NO bare except clauses - catch specific exceptions only
    Always log with context using extra={} dict
    Use @with_recovery decorator for external calls
    """,
    "risk_limits": """
    Hard limits enforced at multiple levels:
    - MAX_POSITION_SIZE = 0.20 (20% of portfolio)
    - MAX_CONCURRENT_PUTS = 3 (Unity-specific)
    - MAX_VOLATILITY = 1.50 (150% - stop trading above)
    - MAX_DRAWDOWN = -0.20 (-20% circuit breaker)
    """,
}

# ============================================================================
# CURRENT OPTIMIZATION FOCUS (January 2025)
# ============================================================================

OPTIMIZATIONS = {
    "performance": """
    Recent optimizations achieving 5x speedup:
    - Vectorized option calculations (process all strikes at once)
    - Removed lazy imports from hot paths
    - Early filtering by confidence scores
    Key method: find_optimal_put_strike_vectorized() in wheel.py
    """,
    "borrowing_analysis": """
    New advanced financial modeling features:
    - BorrowingCostAnalyzer: Determines when to use margin
    - AdvancedFinancialModeling: Monte Carlo, multi-period optimization
    - Integration with position sizing decisions
    See: src/unity_wheel/risk/borrowing_cost_analyzer.py
    """,
    "unity_adaptations": """
    Unity-specific behavior modeling:
    - Assignment probability near earnings (±15-25% moves)
    - Friday assignment patterns
    - Volatility regime detection
    See: UnityAssignmentModel in analytics module
    """,
}

# ============================================================================
# KEY ENTRY POINTS AND FLOWS
# ============================================================================

ENTRY_POINTS = {
    "main_flow": """
    1. run.py -> Simple wrapper
    2. src/unity_wheel/cli/run.py:264 -> main() function
    3. src/unity_wheel/cli/run.py:112 -> generate_recommendation()
    4. src/unity_wheel/api/advisor.py:106 -> advise_position()
    """,
    "recommendation_pipeline": """
    MarketSnapshot -> WheelAdvisor.advise_position() ->
    1. Validate market data
    2. Find optimal strike (vectorized)
    3. Calculate position size (DynamicPositionSizer)
    4. Assess risk (with confidence)
    5. Return Recommendation object
    """,
    "risk_calculation_flow": """
    RiskAnalytics.calculate_portfolio_risk() ->
    1. calculate_var() - Value at Risk
    2. calculate_cvar() - Conditional VaR
    3. calculate_kelly_fraction() - Position sizing
    4. check_risk_limits() - Circuit breakers
    All return (value, confidence) tuples
    """,
}

# ============================================================================
# CRITICAL FILES AND THEIR PURPOSES
# ============================================================================

CRITICAL_FILES = {
    "src/unity_wheel/api/advisor.py": """
    Main recommendation engine - orchestrates entire flow
    Key method: advise_position() - returns trading recommendations
    Recent: Added borrowing cost analysis integration
    """,
    "src/unity_wheel/strategy/wheel.py": """
    Core wheel strategy implementation
    Key method: find_optimal_put_strike_vectorized() - 10x faster
    Handles strike selection with multi-factor scoring
    """,
    "src/unity_wheel/math/options.py": """
    Black-Scholes pricing and Greeks calculations
    ALL functions return CalculationResult with confidence
    Heavily optimized with vectorization support
    """,
    "src/unity_wheel/risk/analytics.py": """
    Portfolio risk calculations and metrics
    Implements VaR, CVaR, Kelly criterion
    Unity-specific assignment probability models
    """,
    "src/unity_wheel/utils/position_sizing.py": """
    SINGLE source of truth for position sizing
    Enforces all risk limits and constraints
    Returns PositionSizeResult with confidence
    """,
    "src/config/schema.py": """
    Complete configuration validation schemas
    924 lines of Pydantic models
    Defines ALL configurable parameters
    """,
}

# ============================================================================
# PATTERNS TO FOLLOW
# ============================================================================

CODING_PATTERNS = {
    "imports": """
    # Always use absolute imports
    from src.unity_wheel.math.options import black_scholes_price_validated
    # NEVER use relative imports like:
    # from ..math import options  # DON'T DO THIS
    """,
    "validation": """
    # For required data (will raise if missing)
    ticker = die(data.get('ticker'), 'Ticker required')

    # For optional data with fallback
    volatility = data.get('volatility', 0.20)  # Default 20%
    """,
    "logging": """
    # Always use structured logging
    logger.info(
        "Operation completed",
        extra={
            "function": "calculate_risk",
            "ticker": ticker,
            "execution_time_ms": elapsed * 1000,
            "confidence": result.confidence
        }
    )
    """,
    "decorators": """
    # Standard decorator stack for calculations
    @timed_operation(threshold_ms=10.0)  # Performance tracking
    @cached(ttl=timedelta(minutes=5))    # Result caching
    @with_recovery(strategy=RecoveryStrategy.FALLBACK)  # Error recovery
    def calculate_metric(...) -> CalculationResult:
        pass
    """,
}

# ============================================================================
# COMMON GOTCHAS AND SOLUTIONS
# ============================================================================

GOTCHAS = {
    "math_module_conflict": """
    PROBLEM: 'from math import sqrt' conflicts with our math module
    SOLUTION: Always use 'import math' or 'from src.unity_wheel.math import ...'
    """,
    "confidence_propagation": """
    PROBLEM: Forgetting to propagate confidence through calculations
    SOLUTION: Every function that calls another must multiply confidences
    Example: final_confidence = calc1_confidence * calc2_confidence * 0.95
    """,
    "option_pricing_validation": """
    PROBLEM: Black-Scholes can return NaN for edge cases
    SOLUTION: Always check result.confidence before using result.value
    if result.confidence > 0.5:
        price = result.value
    else:
        # Handle low confidence case
    """,
    "unity_specific_limits": """
    PROBLEM: Generic position sizing ignores Unity's 3-put limit
    SOLUTION: Always use DynamicPositionSizer which enforces this
    """,
}

# ============================================================================
# TESTING PATTERNS
# ============================================================================

TESTING_PATTERNS = {
    "property_based": """
    Use Hypothesis for mathematical properties:
    @given(
        price=st.floats(min_value=0.01, max_value=1000),
        strike=st.floats(min_value=0.01, max_value=1000)
    )
    def test_black_scholes_bounds(price, strike):
        # Test mathematical properties hold
    """,
    "confidence_testing": """
    Always test confidence scores:
    assert 0 <= result.confidence <= 1
    if inputs_are_valid:
        assert result.confidence > 0.5
    """,
    "mock_patterns": """
    Mock external services consistently:
    @patch('src.unity_wheel.schwab.client.SchwabClient.get_positions')
    async def test_with_mock(mock_get_positions):
        mock_get_positions.return_value = test_positions
    """,
}

# ============================================================================
# PERFORMANCE TARGETS AND SLAS
# ============================================================================

PERFORMANCE_SLAS = {
    "black_scholes": "0.2ms per calculation",
    "greeks_calculation": "0.3ms for all Greeks",
    "var_calculation": "10ms for 1000 data points",
    "strike_selection": "100ms for full option chain",
    "full_recommendation": "200ms end-to-end",
    "api_calls": "1000ms timeout with retry",
}

# ============================================================================
# CURRENT DEVELOPMENT FOCUS
# ============================================================================

CURRENT_FOCUS = """
As of January 2025, the main development focus areas are:

1. Performance Optimization
   - Vectorizing remaining calculations
   - Reducing memory allocations
   - Optimizing hot paths

2. Advanced Financial Modeling
   - Monte Carlo simulations for risk
   - Multi-period optimization
   - Correlation analysis

3. Unity-Specific Enhancements
   - Better earnings volatility modeling
   - Friday assignment patterns
   - Adaptive position sizing

4. Code Quality
   - Eliminating remaining bare excepts
   - Improving test coverage
   - Adding more property-based tests

When generating code, prioritize these areas and maintain
consistency with existing patterns.
"""

# ============================================================================
# CODEX INSTRUCTIONS
# ============================================================================

CODEX_INSTRUCTIONS = """
When generating code for this project:

1. ALWAYS check config.unity.ticker instead of hardcoding 'U'
2. ALWAYS return confidence scores from calculations
3. ALWAYS use DynamicPositionSizer for position sizing
4. ALWAYS catch specific exceptions, never bare except
5. ALWAYS use structured logging with extra={} context
6. ALWAYS validate inputs with die() or Pydantic models
7. ALWAYS add type hints and docstrings
8. ALWAYS write tests for new functionality
9. ALWAYS use absolute imports from src.unity_wheel
10. ALWAYS respect the performance SLAs listed above

Refer to the pattern files in src/patterns/ for examples.
"""
