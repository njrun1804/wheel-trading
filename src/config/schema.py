"""
from __future__ import annotations

Configuration schema and validation using Pydantic.
Provides comprehensive validation, reasonableness checks, and type safety.
"""

from datetime import datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import SecretStr

# Removed circular import - use local config instead
config = get_config()



class RollTriggers(BaseModel):
    """Roll trigger configuration."""

    profit_target_percent: float = Field(
        0.50, ge=0.0, le=1.0, description="Roll when profit target reached (0-1)"
    )
    delta_breach_threshold: float = Field(
        0.70, ge=0.0, le=1.0, description="Roll when delta breached"
    )
    dte_threshold: int = Field(7, ge=0, le=365, description="Roll when DTE below threshold")
    profit_threshold_1: float = Field(0.50, ge=0.0, le=1.0, description="First profit trigger")
    profit_threshold_2: float = Field(0.95, ge=0.0, le=1.0, description="Second profit trigger")


class StrikeRange(BaseModel):
    """Strike range configuration."""

    min_moneyness: float = Field(0.80, ge=0.0, le=1.0, description="Minimum moneyness (% of spot)")
    max_moneyness: float = Field(1.10, ge=1.0, le=2.0, description="Maximum moneyness (% of spot)")


class StrategyConfig(BaseModel):
    """Strategy parameters configuration."""

    delta_target: float = Field(0.30, ge=0.0, le=1.0, description="Target delta for short puts")
    days_to_expiry_target: int = Field(35, ge=1, le=365, description="Target days to expiry")
    min_days_to_expiry: int = Field(
        21, ge=0, le=365, description="Minimum DTE before considering roll"
    )
    max_delta_short_put: float = Field(
        0.35, ge=0.0, le=1.0, description="Maximum delta for short puts"
    )
    strike_intervals: List[float] = Field([1.0, 2.5, 5.0], description="Strike selection intervals")
    strike_range: StrikeRange = Field(default_factory=StrikeRange)
    min_premium_yield: float = Field(0.01, ge=0.0, description="Minimum premium yield")
    roll_triggers: RollTriggers = Field(default_factory=RollTriggers)

    @field_validator("strike_intervals")
    @classmethod
    def validate_strike_intervals(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("Strike intervals cannot be empty")
        if any(x <= 0 for x in v):
            raise ValueError("Strike intervals must be positive")
        return sorted(v)

    @model_validator(mode="after")
    def validate_dte_consistency(self) -> "StrategyConfig":
        if self.min_days_to_expiry >= self.days_to_expiry_target:
            raise ValueError("min_days_to_expiry must be less than days_to_expiry_target")
        if self.roll_triggers.dte_threshold > self.min_days_to_expiry:
            raise ValueError("roll dte_threshold cannot exceed min_days_to_expiry")
        return self


class RiskLimits(BaseModel):
    """Risk limit configuration."""

    max_var_95: float = Field(0.05, ge=0.0, le=1.0, description="Max VaR at 95% confidence")
    max_cvar_95: float = Field(0.075, ge=0.0, le=1.0, description="Max CVaR at 95% confidence")
    max_kelly_fraction: float = Field(
        0.25, ge=0.0, le=1.0, description="Max Kelly fraction per position"
    )
    max_delta_exposure: float = Field(100.0, ge=0.0, description="Max delta exposure")
    max_gamma_exposure: float = Field(10.0, ge=0.0, description="Max gamma exposure")
    max_vega_exposure: float = Field(1000.0, ge=0.0, description="Max vega exposure")
    max_theta_decay: float = Field(100.0, ge=0.0, description="Max daily theta decay")
    max_contracts_per_trade: int = Field(100, ge=1, description="Max contracts per trade")
    max_notional_percent: float = Field(
        2.00, ge=0.0, le=10.0, description="Max notional as percent of portfolio"
    )

    @model_validator(mode="after")
    def validate_risk_hierarchy(self) -> "RiskLimits":
        if self.max_cvar_95 < self.max_var_95:
            raise ValueError("max_cvar_95 must be >= max_var_95")
        return self


class GreekLimits(BaseModel):
    """Greek exposure limits."""

    max_delta_exposure: float = Field(100.0, ge=0.0, description="Maximum delta exposure")
    max_gamma_exposure: float = Field(10.0, ge=0.0, description="Maximum gamma exposure")
    max_vega_exposure: float = Field(1000.0, ge=0.0, description="Maximum vega exposure ($)")
    max_theta_decay: float = Field(100.0, ge=0.0, description="Maximum daily theta decay")
    max_rho_exposure: float = Field(500.0, ge=0.0, description="Maximum rho exposure")


class MarginConfig(BaseModel):
    """Margin and leverage configuration."""

    max_utilization: float = Field(0.50, ge=0.0, le=1.0, description="Maximum margin utilization")
    safety_factor: float = Field(0.80, ge=0.0, le=1.0, description="Safety factor for margin")
    margin_requirement: float = Field(
        0.20, ge=0.0, le=1.0, description="Simplified margin requirement"
    )
    margin_multiplier: float = Field(0.50, ge=0.0, description="Safety multiplier for margin")


class CircuitBreakers(BaseModel):
    """Circuit breaker configuration for safety stops."""

    # Position-level circuit breakers
    max_position_pct: float = Field(
        0.20, ge=0.0, le=1.0, description="Max % of portfolio in one position"
    )
    max_contracts: int = Field(10, ge=1, description="Max contracts per trade")
    min_portfolio_value: float = Field(10000, ge=0.0, description="Stop trading below this value")

    # Market condition circuit breakers
    max_volatility: float = Field(1.5, ge=0.0, description="Max annual volatility (150%)")
    max_gap_percent: float = Field(0.10, ge=0.0, le=1.0, description="Max price gap (10%)")
    min_volume_ratio: float = Field(0.5, ge=0.0, le=1.0, description="Min volume vs average")

    # Loss circuit breakers
    max_daily_loss_pct: float = Field(0.02, ge=0.0, le=1.0, description="Max daily loss %")
    max_weekly_loss_pct: float = Field(0.05, ge=0.0, le=1.0, description="Max weekly loss %")
    max_consecutive_losses: int = Field(3, ge=1, description="Stop after N consecutive losses")

    # System health circuit breakers
    min_confidence: float = Field(0.30, ge=0.0, le=1.0, description="Min confidence to trade")
    max_warnings: int = Field(3, ge=0, description="Max warnings before stopping")
    blackout_hours: List[int] = Field([], description="Hours to avoid trading (24h format)")


class AdaptiveLimits(BaseModel):
    """Adaptive parameter adjustment configuration."""

    enabled: bool = Field(True, description="Enable adaptive adjustments")
    base_max_position_pct: float = Field(0.20, ge=0.0, le=1.0, description="Base position size %")
    base_min_confidence: float = Field(0.30, ge=0.0, le=1.0, description="Base minimum confidence")

    # Scaling features
    volatility_scaling: bool = Field(True, description="Scale with volatility")
    confidence_scaling: bool = Field(True, description="Scale with confidence")
    performance_scaling: bool = Field(True, description="Scale with performance")
    regime_scaling: bool = Field(True, description="Scale with market regime")

    # ML integration
    ml_blend_max: float = Field(0.80, ge=0.0, le=1.0, description="Max ML weight")
    ml_confidence_threshold: float = Field(0.60, ge=0.0, le=1.0, description="Min ML confidence")


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_size: float = Field(
        1.00, ge=0.0, le=1.0, description="Max position size as fraction of portfolio"
    )
    max_margin_percent: float = Field(
        0.95, ge=0.0, le=1.0, description="Maximum margin utilization"
    )
    max_drawdown_percent: float = Field(
        0.50, ge=0.0, le=1.0, description="Maximum drawdown before halting"
    )
    cvar_percentile: float = Field(
        0.95, ge=0.0, le=1.0, description="CVaR percentile for risk calculations"
    )
    objective_risk_weight: float = Field(
        0.20, ge=0.0, description="Risk weight in objective function"
    )
    kelly_fraction: float = Field(0.50, ge=0.0, le=1.0, description="Kelly criterion fraction")
    limits: RiskLimits = Field(default_factory=RiskLimits)
    greeks: GreekLimits = Field(default_factory=GreekLimits)
    margin: MarginConfig = Field(default_factory=MarginConfig)
    circuit_breakers: CircuitBreakers = Field(default_factory=CircuitBreakers)
    adaptive_limits: AdaptiveLimits = Field(default_factory=AdaptiveLimits)

    @model_validator(mode="after")
    def validate_risk_consistency(self) -> "RiskConfig":
        if self.max_position_size > self.limits.max_notional_percent:
            raise ValueError("max_position_size cannot exceed max_notional_percent")
        if self.kelly_fraction > 1.0:
            raise ValueError("kelly_fraction should not exceed 1.0 (full Kelly)")
        return self


class CacheTTL(BaseModel):
    """Cache TTL configuration."""

    options_chain: int = Field(900, ge=0, description="Options chain cache TTL in seconds")
    stock_quotes: int = Field(60, ge=0, description="Stock quotes cache TTL in seconds")
    volatility: int = Field(900, ge=0, description="Volatility cache TTL in seconds")
    greeks: int = Field(300, ge=0, description="Greeks cache TTL in seconds")
    account_data: int = Field(30, ge=0, description="Account data cache TTL in seconds")
    historical: int = Field(86400, ge=0, description="Historical data cache TTL in seconds")
    market_data: int = Field(60, ge=0, description="Market data cache TTL in seconds")
    intraday: int = Field(900, ge=0, description="Intraday data cache TTL in seconds")


class APITimeouts(BaseModel):
    """API timeout configuration."""

    connect: float = Field(5.0, ge=0.0, description="Connection timeout in seconds")
    read: float = Field(30.0, ge=0.0, description="Read timeout in seconds")
    write: float = Field(10.0, ge=0.0, description="Write timeout in seconds")
    total: float = Field(60.0, ge=0.0, description="Total timeout in seconds")

    @model_validator(mode="after")
    def validate_timeout_consistency(self) -> "APITimeouts":
        if self.total < max(self.connect, self.read, self.write):
            raise ValueError("total timeout must be >= max of individual timeouts")
        return self


class RetryConfig(BaseModel):
    """Retry configuration for API calls."""

    max_attempts: int = Field(3, ge=1, le=10, description="Maximum retry attempts")
    delays: List[int] = Field([1, 2, 5], description="Exponential backoff delays in seconds")
    rate_limit_wait: int = Field(60, ge=0, description="Wait after rate limit in seconds")


class LiquidityRequirements(BaseModel):
    """Minimum liquidity requirements."""

    volume: int = Field(10, ge=0, description="Minimum daily volume")
    open_interest: int = Field(100, ge=0, description="Minimum open interest")
    bid_size: int = Field(1, ge=0, description="Minimum bid size")
    max_bid_ask_spread: float = Field(0.50, ge=0.0, description="Maximum bid-ask spread")


class DataQuality(BaseModel):
    """Data quality configuration."""

    stale_data_seconds: int = Field(30, ge=0, description="Stale data threshold in seconds")
    min_confidence_score: float = Field(
        0.80, ge=0.0, le=1.0, description="Minimum confidence score"
    )
    max_spread_pct: float = Field(10.0, ge=0.0, description="Maximum bid-ask spread %")
    min_quote_size: int = Field(1, ge=0, description="Minimum quote size")
    max_price_change_pct: float = Field(50.0, ge=0.0, description="Maximum price change %")
    min_options_per_expiry: int = Field(10, ge=0, description="Minimum option contracts")
    stale_data_minutes: int = Field(15, ge=0, description="Data considered stale after")
    min_liquidity: LiquidityRequirements = Field(default_factory=LiquidityRequirements)


class DataConfig(BaseModel):
    """Market data configuration."""

    cache_ttl: CacheTTL = Field(default_factory=CacheTTL)
    api_timeouts: APITimeouts = Field(default_factory=APITimeouts)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    quality: DataQuality = Field(default_factory=DataQuality)


class BrokerConfig(BaseModel):
    """Broker configuration."""

    name: str = Field("schwab", description="Broker name")
    api_key: Optional[SecretStr] = Field(None, description="API key")
    api_secret: Optional[SecretStr] = Field(None, description="API secret")


class MarketHours(BaseModel):
    """Market hours configuration."""

    open: str = Field("09:30:00", description="Market open time")
    close: str = Field("16:00:00", description="Market close time")
    pre_market_open: str = Field("07:00:00", description="Pre-market open time")
    after_hours_close: str = Field("20:00:00", description="After-hours close time")

    @field_validator("open", "close", "pre_market_open", "after_hours_close")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        try:
            time.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid time format: {v}, expected HH:MM:SS")
        return v


class TradingExecutionConfig(BaseModel):
    """Trading execution parameters."""

    commission_per_contract: float = Field(
        0.65, ge=0.0, description="Commission per options contract"
    )
    contracts_per_trade: int = Field(100, ge=1, description="Standard lot size")
    max_concurrent_puts: int = Field(3, ge=1, description="Max concurrent put positions")


class TradingConfig(BaseModel):
    """Trading environment configuration."""

    mode: str = Field(
        "paper",
        pattern="^(live|paper|backtest)$",
        description="Trading mode",
    )
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    market_hours: MarketHours = Field(default_factory=MarketHours)
    execution: TradingExecutionConfig = Field(default_factory=TradingExecutionConfig)


class VolatilityRegimes(BaseModel):
    """Volatility regime thresholds."""

    low: float = Field(0.40, ge=0.0, description="Low volatility threshold")
    normal: float = Field(0.65, ge=0.0, description="Normal volatility threshold")
    high: float = Field(0.80, ge=0.0, description="High volatility threshold")
    extreme: float = Field(1.00, ge=0.0, description="Extreme volatility threshold")

    @model_validator(mode="after")
    def validate_regime_order(self) -> "VolatilityRegimes":
        if not (self.low < self.normal < self.high < self.extreme):
            raise ValueError("Volatility regimes must be in ascending order")
        return self


class UnityVolatility(BaseModel):
    """Unity volatility configuration."""

    typical_range: List[float] = Field([0.40, 0.90], description="Typical IV range")
    average: float = Field(0.65, ge=0.0, description="Average IV")
    regimes: VolatilityRegimes = Field(default_factory=VolatilityRegimes)

    @field_validator("typical_range")
    @classmethod
    def validate_range(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("typical_range must have exactly 2 values [min, max]")
        if v[0] >= v[1]:
            raise ValueError("typical_range[0] must be less than typical_range[1]")
        return v


class UnityEarnings(BaseModel):
    """Unity earnings behavior configuration."""

    typical_move_pct: float = Field(0.20, ge=0.0, le=1.0, description="Typical earnings move %")
    blackout_days_before: int = Field(7, ge=0, description="Days before earnings to stop trading")
    iv_expansion_factor: float = Field(1.5, ge=1.0, description="IV expansion factor into earnings")


class UnityGapBehavior(BaseModel):
    """Unity gap behavior configuration."""

    typical_overnight_move: float = Field(0.05, ge=0.0, description="Typical overnight move")
    max_acceptable_gap: float = Field(0.15, ge=0.0, description="Max gap to trade after")
    gap_recovery_hours: int = Field(48, ge=0, description="Hours to wait after major gap")


class UnityCorrelations(BaseModel):
    """Unity market correlations."""

    qqq_correlation: float = Field(0.65, ge=-1.0, le=1.0, description="QQQ correlation")
    vix_sensitivity: float = Field(1.5, ge=0.0, description="VIX beta")
    sector_etfs: List[str] = Field(["GAMR", "METV", "XLC"], description="Related sector ETFs")


class UnityConfig(BaseModel):
    """Unity-specific configuration."""

    ticker: str = Field("U", description="Stock ticker")
    company_name: str = Field("Unity Software Inc.", description="Company name")
    volatility: UnityVolatility = Field(default_factory=UnityVolatility)
    earnings: UnityEarnings = Field(default_factory=UnityEarnings)
    gap_behavior: UnityGapBehavior = Field(default_factory=UnityGapBehavior)
    correlations: UnityCorrelations = Field(default_factory=UnityCorrelations)


class VolatilityFactors(BaseModel):
    """Volatility-based position sizing factors."""

    low: float = Field(1.20, ge=0.0, description="Factor for low volatility")
    normal: float = Field(1.00, ge=0.0, description="Factor for normal volatility")
    high: float = Field(0.70, ge=0.0, description="Factor for high volatility")
    extreme: float = Field(0.50, ge=0.0, description="Factor for extreme volatility")


class VolatilityThresholds(BaseModel):
    """Volatility thresholds for position sizing."""

    low: float = Field(0.40, ge=0.0, description="Low volatility threshold")
    normal: float = Field(0.60, ge=0.0, description="Normal volatility threshold")
    high: float = Field(0.80, ge=0.0, description="High volatility threshold")
    extreme: float = Field(1.00, ge=0.0, description="Extreme volatility threshold")


class RegimeParams(BaseModel):
    """Parameters for a specific market regime."""

    put_delta: float = Field(0.30, ge=0.0, le=1.0, description="Target delta for puts")
    target_dte: int = Field(35, ge=1, le=365, description="Days to expiration")
    roll_profit_target: float = Field(0.50, ge=0.0, le=1.0, description="Roll at profit %")
    position_size_factor: float = Field(1.0, ge=0.0, description="Position size multiplier")


class AdaptiveStopConditions(BaseModel):
    """Stop conditions for adaptive system."""

    max_volatility: float = Field(1.0, ge=0.0, description="Stop above this volatility")
    max_drawdown: float = Field(0.20, ge=0.0, le=1.0, description="Stop at this drawdown")
    min_days_to_earnings: int = Field(7, ge=0, description="Skip if earnings within days")


class OutcomeTracking(BaseModel):
    """Outcome tracking configuration."""

    enabled: bool = Field(True, description="Enable outcome tracking")
    database_path: str = Field("~/.wheel_trading/wheel_outcomes.db", description="Database path")
    retention_days: int = Field(365, ge=1, description="Keep outcomes for days")


class AdaptiveConfig(BaseModel):
    """Adaptive system configuration."""

    regime_persistence_days: int = Field(3, ge=1, description="Days to confirm regime change")
    base_position_pct: float = Field(0.20, ge=0.0, le=1.0, description="Base position size %")
    max_position_pct: float = Field(0.25, ge=0.0, le=1.0, description="Max position size %")

    volatility_factors: VolatilityFactors = Field(default_factory=VolatilityFactors)
    volatility_thresholds: VolatilityThresholds = Field(default_factory=VolatilityThresholds)

    regime_params: Dict[str, RegimeParams] = Field(
        default_factory=lambda: {
            "normal": RegimeParams(),
            "volatile": RegimeParams(
                put_delta=0.25, target_dte = config.trading.target_dte, roll_profit_target=0.25, position_size_factor=0.7
            ),
            "stressed": RegimeParams(
                put_delta=0.20, target_dte = config.trading.target_dte, roll_profit_target=0.25, position_size_factor=0.5
            ),
            "low_volatility": RegimeParams(put_delta=0.35, target_dte = config.trading.target_dte, position_size_factor=1.2),
        }
    )

    stop_conditions: AdaptiveStopConditions = Field(default_factory=AdaptiveStopConditions)
    outcome_tracking: OutcomeTracking = Field(default_factory=OutcomeTracking)


class MLFeatures(BaseModel):
    """ML feature flags."""

    use_iv_rank: bool = Field(True, description="Use IV rank feature")
    use_iv_skew: bool = Field(True, description="Use IV skew feature")
    use_realized_vol: bool = Field(True, description="Use realized volatility feature")
    use_macro_factors: bool = Field(False, description="Use macro factors")
    use_sentiment: bool = Field(False, description="Use sentiment analysis")


class ModelConfig(BaseModel):
    """Individual model configuration."""

    type: str = Field(..., description="Model type")
    retrain_days: Optional[int] = Field(None, ge=1, description="Retrain frequency in days")
    min_samples: Optional[int] = Field(None, ge=1, description="Minimum samples for training")
    lookback_days: Optional[int] = Field(None, ge=1, description="Lookback period in days")


class MLModels(BaseModel):
    """ML models configuration."""

    probability_model: ModelConfig = Field(
        default=ModelConfig(type="gradient_boost", retrain_days=30, min_samples=1000)
    )
    volatility_model: ModelConfig = Field(default=ModelConfig(type="garch", lookback_days=252))


class MLConfig(BaseModel):
    """Machine learning configuration."""

    enabled: bool = Field(False, description="Enable ML enhancement")
    model_path: Path = Field(Path("./models/"), description="Model storage path")
    features: MLFeatures = Field(default_factory=MLFeatures)
    models: MLModels = Field(default_factory=MLModels)

    @field_validator("model_path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        # Convert string to Path if needed
        if isinstance(v, str):
            v = Path(v)
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        "INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR)$",
        description="Log level",
    )
    format: str = Field("json", pattern="^(json|text)$", description="Log format")
    file: Path = Field(Path("./logs/wheel_trading.log"), description="Log file path")
    rotation: str = Field("daily", description="Log rotation")
    retention_days: int = Field(30, ge=1, description="Log retention in days")


class AlertsConfig(BaseModel):
    """Alert thresholds configuration."""

    margin_warning_percent: float = Field(
        0.40, ge=0.0, le=1.0, description="Margin utilization warning threshold"
    )
    delta_warning: float = Field(0.40, ge=0.0, le=1.0, description="Delta warning threshold")
    loss_warning_percent: float = Field(
        0.10, ge=0.0, le=1.0, description="Unrealized loss warning threshold"
    )


class PerformanceConfig(BaseModel):
    """Performance tracking configuration."""

    track_decisions: bool = Field(True, description="Track all decisions")
    track_parameters: bool = Field(True, description="Track parameter usage")
    export_format: str = Field("csv", pattern="^(csv|json)$", description="Export format")
    export_path: Path = Field(Path("./data/performance/"), description="Export path")


class DatentoFilters(BaseModel):
    """Databento data filtering parameters."""

    moneyness_range: float = Field(0.20, ge=0.0, le=1.0, description="Range around spot price")
    max_expirations: int = Field(3, ge=1, description="Number of expirations to keep")
    min_volume: int = Field(0, ge=0, description="Minimum volume filter")


class DatentoStorage(BaseModel):
    """Databento storage configuration."""

    local_retention_days: int = Field(30, ge=1, description="Days to keep locally")
    compression: bool = Field(True, description="Enable compression")
    partitioning: str = Field("daily", description="Data partitioning scheme")


class DatentoRateLimits(BaseModel):
    """Databento rate limits (provider-specific)."""

    max_concurrent_live: int = Field(10, ge=1, description="Max concurrent live connections")
    max_historical_rps: int = Field(100, ge=1, description="Max requests per second")
    max_symbols_per_request: int = Field(2000, ge=1, description="Max symbols per request")
    max_file_size_gb: int = Field(2, ge=1, description="Max file size in GB")


class DatentoLoader(BaseModel):
    """Databento data loader configuration."""

    max_workers: int = Field(8, ge=1, description="Parallel processing workers")
    chunk_size: int = Field(250, ge=1, description="Symbols per chunk")
    max_requests_per_second: int = Field(100, ge=1, description="Rate limit")
    retry_delays: List[int] = Field([1, 2, 5, 10], description="Retry delays in seconds")
    required_days: int = Field(750, ge=1, description="Required historical days")
    minimum_days: int = Field(500, ge=1, description="Minimum acceptable days")


class DatentoConfig(BaseModel):
    """Databento configuration."""

    filters: DatentoFilters = Field(default_factory=DatentoFilters)
    storage: DatentoStorage = Field(default_factory=DatentoStorage)
    rate_limits: DatentoRateLimits = Field(default_factory=DatentoRateLimits)
    loader: DatentoLoader = Field(default_factory=DatentoLoader)


class IVSurfaceConfig(BaseModel):
    """IV surface configuration."""

    lookback_days: int = Field(252, ge=1, description="Historical days for IV calculation")
    min_history: int = Field(30, ge=1, description="Minimum days required")
    default_iv_rank: float = Field(50.0, ge=0.0, le=100.0, description="Default IV rank")
    default_half_life: float = Field(30.0, ge=1.0, description="Default decay rate")


class SeasonalityConfig(BaseModel):
    """Seasonality analysis configuration."""

    min_samples: int = Field(10, ge=1, description="Minimum samples per period")
    min_years: int = Field(2, ge=1, description="Minimum years of data")


class PerformanceTrackerConfig(BaseModel):
    """Performance tracker configuration."""

    database_path: str = Field("~/.wheel_trading/performance.db", description="Database path")
    track_all_decisions: bool = Field(True, description="Track all decisions")


class DynamicOptimizationConfig(BaseModel):
    """Dynamic optimization configuration."""

    cvar_penalty: float = Field(0.20, ge=0.0, description="CVaR penalty in objective")
    base_kelly: float = Field(0.50, ge=0.0, le=1.0, description="Half-Kelly base")


class AnalyticsConfig(BaseModel):
    """Analytics configuration."""

    iv_surface: IVSurfaceConfig = Field(default_factory=IVSurfaceConfig)
    seasonality: SeasonalityConfig = Field(default_factory=SeasonalityConfig)
    performance_tracker: PerformanceTrackerConfig = Field(default_factory=PerformanceTrackerConfig)
    dynamic_optimization: DynamicOptimizationConfig = Field(
        default_factory=DynamicOptimizationConfig
    )


class APIOperationConfig(BaseModel):
    """API operation parameters."""

    max_concurrent_puts: int = Field(3, ge=1, description="Max concurrent put positions")
    max_position_pct: float = Field(0.12, ge=0.0, le=1.0, description="Max % per leg")
    min_confidence: float = Field(0.75, ge=0.0, le=1.0, description="Min confidence for trades")
    max_decision_time: float = Field(0.2, ge=0.0, description="Max decision time in seconds")
    max_bid_ask_spread: float = Field(0.50, ge=0.0, description="Maximum spread allowed")
    min_volume: int = Field(10, ge=0, description="Minimum volume")
    min_open_interest: int = Field(100, ge=0, description="Minimum open interest")


class OperationsConfig(BaseModel):
    """Operational settings configuration."""

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    api: APIOperationConfig = Field(default_factory=APIOperationConfig)


class SlippageConfig(BaseModel):
    """Slippage model configuration."""

    type: str = Field("fixed", pattern="^(fixed|proportional)$", description="Slippage type")
    value: float = Field(0.05, ge=0.0, description="Slippage value")


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    start_date: str = Field("2023-01-01", description="Start date")
    end_date: Optional[str] = Field(None, description="End date")
    initial_capital: float = Field(100000, ge=0.0, description="Initial capital")
    commission_per_contract: float = Field(0.65, ge=0.0, description="Commission per contract")
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                datetime.fromisoformat(v)
            except ValueError:
                raise ValueError(f"Invalid date format: {v}, expected YYYY-MM-DD")
        return v


class PerformanceSLA(BaseModel):
    """Performance SLA thresholds."""

    black_scholes_ms: int = Field(50, ge=1, description="Black-Scholes calculation time")
    greeks_ms: int = Field(100, ge=1, description="Greeks calculation time")
    risk_metrics_ms: int = Field(200, ge=1, description="Risk metrics calculation time")
    decision_ms: int = Field(500, ge=1, description="Total decision time")
    api_call_ms: int = Field(5000, ge=1, description="External API call time")
    event_analysis_ms: int = Field(30, ge=1, description="Event analysis time")
    iv_calculation_ms: int = Field(10, ge=1, description="IV calculation time")


class PerformanceMonitoring(BaseModel):
    """Performance monitoring configuration."""

    metrics_retention_days: int = Field(90, ge=1, description="Keep metrics for days")
    alert_threshold_pct: int = Field(150, ge=100, description="Alert if >X% of SLA")
    sample_rate: float = Field(0.1, ge=0.0, le=1.0, description="Sample rate for operations")


class SystemPerformance(BaseModel):
    """System performance targets."""

    max_decision_time_ms: int = Field(200, ge=1, description="Max decision time in ms")
    max_memory_mb: int = Field(512, ge=1, description="Max memory usage in MB")
    sla: PerformanceSLA = Field(default_factory=PerformanceSLA)
    monitoring: PerformanceMonitoring = Field(default_factory=PerformanceMonitoring)


class SystemFeatures(BaseModel):
    """System feature flags."""

    enable_diagnostics: bool = Field(True, description="Enable diagnostics")
    enable_self_tuning: bool = Field(False, description="Enable self-tuning")
    enable_config_tracking: bool = Field(True, description="Enable config tracking")
    enable_auto_recovery: bool = Field(True, description="Enable auto-recovery")


class DatabasePaths(BaseModel):
    """Database storage paths."""

    performance: str = Field("~/.wheel_trading/performance.db", description="Performance DB")
    schwab_data: str = Field("~/.wheel_trading/schwab_data.db", description="Schwab data DB")
    cache: str = Field(config.storage.database_path, description="Cache DB")


class StorageConfig(BaseModel):
    """Storage configuration."""

    databases: DatabasePaths = Field(default_factory=DatabasePaths)
    backup_interval_hours: int = Field(24, ge=1, description="Backup interval in hours")


class SystemConfig(BaseModel):
    """System configuration."""

    performance: SystemPerformance = Field(default_factory=SystemPerformance)
    health_checks: List[str] = Field(
        [
            "math_library_sanity",
            "configuration_validity",
            "type_consistency",
            "data_connectivity",
            "risk_limits_valid",
        ],
        description="Health checks to run",
    )
    features: SystemFeatures = Field(default_factory=SystemFeatures)
    storage: StorageConfig = Field(default_factory=StorageConfig)


class TrackingConfig(BaseModel):
    """Configuration tracking settings."""

    track_usage: bool = Field(True, description="Track parameter usage")
    suggest_tuning: bool = Field(True, description="Suggest parameter adjustments")
    warn_unused: bool = Field(True, description="Warn about unused parameters")
    auto_disable_broken: bool = Field(True, description="Auto-disable broken features")


class MetadataConfig(BaseModel):
    """Configuration metadata."""

    version: str = Field("1.0.0", description="Config version")
    last_updated: str = Field(..., description="Last update date")
    environment: str = Field("development", description="Environment")
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


class AuthConfig(BaseModel):
    """Authentication configuration."""

    client_id: Optional[SecretStr] = Field(None, description="OAuth client ID")
    client_secret: Optional[SecretStr] = Field(None, description="OAuth client secret")
    redirect_uri: str = Field("http://localhost:8182/callback", description="OAuth callback URL")
    auth_url: str = Field(
        "https://api.schwabapi.com/v1/oauth/authorize", description="Authorization endpoint"
    )
    token_url: str = Field("https://api.schwabapi.com/v1/oauth/token", description="Token endpoint")
    scope: str = Field("AccountsRead MarketDataRead", description="OAuth scopes to request")

    # Token management
    auto_refresh: bool = Field(True, description="Enable automatic token refresh")
    token_refresh_buffer_minutes: int = Field(
        5, ge=1, le=60, description="Minutes before expiry to refresh token"
    )

    # Cache settings
    enable_cache: bool = Field(True, description="Enable API response caching")
    cache_ttl_seconds: int = Field(3600, ge=0, description="Default cache TTL in seconds")
    cache_max_size_mb: int = Field(100, ge=1, description="Maximum cache size in MB")

    # Rate limiting
    rate_limit_rps: float = Field(10.0, gt=0, description="Requests per second limit")
    rate_limit_burst: int = Field(20, ge=1, description="Burst capacity for rate limiter")
    enable_circuit_breaker: bool = Field(True, description="Enable circuit breaker pattern")
    circuit_breaker_threshold: int = Field(5, ge=1, description="Failures before circuit opens")
    circuit_breaker_recovery_seconds: int = Field(
        60, ge=1, description="Recovery timeout in seconds"
    )

    # Retry settings
    max_retry_attempts: int = Field(3, ge=1, le=10, description="Maximum retry attempts")
    retry_backoff_factor: float = Field(2.0, ge=1.0, description="Exponential backoff factor")

    @field_validator("client_id", "client_secret")
    @classmethod
    def validate_credentials(cls, v: Optional[SecretStr]) -> Optional[SecretStr]:
        if v and len(v.get_secret_value()) < 10:
            raise ValueError("Credential appears too short to be valid")
        return v

    @model_validator(mode="after")
    def validate_auth_config(self) -> "AuthConfig":
        """Validate authentication configuration consistency."""
        # If credentials are provided, they should both be provided
        has_id = self.client_id is not None
        has_secret = self.client_secret is not None

        if has_id != has_secret:
            raise ValueError("Both client_id and client_secret must be provided together")

        return self


class OptimizationBounds(BaseModel):
    """Optimization parameter bounds."""

    delta_min: float = Field(0.10, ge=0.0, le=1.0, description="Minimum delta")
    delta_max: float = Field(0.40, ge=0.0, le=1.0, description="Maximum delta")
    dte_min: int = Field(21, ge=1, description="Minimum DTE")
    dte_max: int = Field(49, ge=1, description="Maximum DTE")
    kelly_min: float = Field(0.10, ge=0.0, le=1.0, description="Minimum Kelly")
    kelly_max: float = Field(0.50, ge=0.0, le=1.0, description="Maximum Kelly")


class OptimizationConfig(BaseModel):
    """Dynamic optimization configuration."""

    enabled: bool = Field(True, description="Enable dynamic optimization")
    mode: str = Field("dynamic", pattern="^(dynamic|tiered)$", description="Optimization mode")
    min_confidence: float = Field(
        0.60, ge=0.0, le=1.0, description="Min confidence for optimization"
    )
    volatility_lookback: int = Field(20, ge=1, description="Volatility lookback days")
    min_history_days: int = Field(250, ge=1, description="History required for optimization")
    bounds: OptimizationBounds = Field(default_factory=OptimizationBounds)


class WheelConfig(BaseModel):
    """Root configuration schema for the wheel trading system."""

    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    databento: DatentoConfig = Field(default_factory=DatentoConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    unity: UnityConfig = Field(default_factory=UnityConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    operations: OperationsConfig = Field(default_factory=OperationsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    metadata: MetadataConfig = Field(...)

    @model_validator(mode="after")
    def validate_global_consistency(self) -> "WheelConfig":
        """Validate cross-section consistency."""
        # Ensure strategy delta target is within risk limits
        if self.strategy.delta_target > self.strategy.max_delta_short_put:
            raise ValueError("strategy.delta_target cannot exceed max_delta_short_put")

        # Ensure Kelly fraction in risk config matches limits
        if self.risk.kelly_fraction > 1.0:
            raise ValueError("Half-Kelly (0.5) is recommended, Full-Kelly (1.0) is maximum")

        # Validate Unity volatility average is within typical range
        unity_vol = self.unity.volatility
        if not (unity_vol.typical_range[0] <= unity_vol.average <= unity_vol.typical_range[1]):
            raise ValueError("Unity average volatility must be within typical range")

        return self


def load_config(config_path: Union[str, Path]) -> WheelConfig:
    """Load and validate configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Add current timestamp if not in metadata
    if "metadata" not in config_dict:
        config_dict["metadata"] = {}
    if "last_updated" not in config_dict["metadata"]:
        config_dict["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    return WheelConfig(**config_dict)


def validate_config_health(config: WheelConfig) -> Dict[str, Union[bool, str]]:
    """
    Perform health checks on configuration.
    Returns dict with check results and recommendations.
    """
    health = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": [],
    }

    # Check risk parameter relationships
    if config.risk.max_position_size > 0.25:
        health["warnings"].append("max_position_size > 25% may lead to concentrated risk")

    if config.risk.kelly_fraction > 0.5:
        health["warnings"].append("kelly_fraction > 0.5 (Half-Kelly) increases risk significantly")

    # Check strategy parameters
    if config.strategy.delta_target > 0.35:
        health["warnings"].append("delta_target > 0.35 increases assignment probability")

    if config.strategy.days_to_expiry_target < 30:
        health["warnings"].append("days_to_expiry_target < 30 may not capture enough premium")

    # Check API operation thresholds
    if hasattr(config, "api") and hasattr(config.api, "operation"):
        if config.api.operation.min_open_interest < 100:
            health["recommendations"].append(
                "Consider increasing min_open_interest to 100+ for better liquidity"
            )

    # Check ML configuration
    if config.ml.enabled and not config.ml.model_path.exists():
        health["errors"].append(f"ML enabled but model_path does not exist: {config.ml.model_path}")
        health["valid"] = False

    # Check operational settings
    if config.operations.logging.retention_days > 90:
        health["recommendations"].append("Log retention > 90 days may use significant disk space")

    # Add summary
    health["summary"] = (
        f"Configuration health: {len(health['errors'])} errors, "
        f"{len(health['warnings'])} warnings, "
        f"{len(health['recommendations'])} recommendations"
    )

    return health

def get_config():
    """Get configuration - simple version to avoid circular import."""
    from .loader import ConfigurationLoader
    loader = ConfigurationLoader()
    return loader.config
