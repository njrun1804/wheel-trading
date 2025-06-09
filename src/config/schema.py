"""
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


class RollTriggers(BaseModel):
    """Roll trigger configuration."""

    profit_target_percent: float = Field(
        0.50, ge=0.0, le=1.0, description="Roll when profit target reached (0-1)"
    )
    delta_breach_threshold: float = Field(
        0.45, ge=0.0, le=1.0, description="Roll when delta breached"
    )
    dte_threshold: int = Field(21, ge=0, le=365, description="Roll when DTE below threshold")


class StrategyConfig(BaseModel):
    """Strategy parameters configuration."""

    delta_target: float = Field(0.30, ge=0.0, le=1.0, description="Target delta for short puts")
    days_to_expiry_target: int = Field(45, ge=1, le=365, description="Target days to expiry")
    min_days_to_expiry: int = Field(
        21, ge=0, le=365, description="Minimum DTE before considering roll"
    )
    max_delta_short_put: float = Field(
        0.35, ge=0.0, le=1.0, description="Maximum delta for short puts"
    )
    strike_intervals: List[float] = Field(
        [1.0, 2.5, 5.0], description="Strike selection intervals"
    )
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
    max_contracts_per_trade: int = Field(10, ge=1, description="Max contracts per trade")
    max_notional_percent: float = Field(
        0.25, ge=0.0, le=1.0, description="Max notional as percent of portfolio"
    )

    @model_validator(mode="after")
    def validate_risk_hierarchy(self) -> "RiskLimits":
        if self.max_cvar_95 < self.max_var_95:
            raise ValueError("max_cvar_95 must be >= max_var_95")
        return self


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_size: float = Field(
        0.20, ge=0.0, le=1.0, description="Max position size as fraction of portfolio"
    )
    max_margin_percent: float = Field(
        0.50, ge=0.0, le=1.0, description="Maximum margin utilization"
    )
    max_drawdown_percent: float = Field(
        0.20, ge=0.0, le=1.0, description="Maximum drawdown before halting"
    )
    cvar_percentile: float = Field(
        0.95, ge=0.0, le=1.0, description="CVaR percentile for risk calculations"
    )
    objective_risk_weight: float = Field(
        0.20, ge=0.0, description="Risk weight in objective function"
    )
    kelly_fraction: float = Field(
        0.50, ge=0.0, le=1.0, description="Kelly criterion fraction"
    )
    limits: RiskLimits = Field(default_factory=RiskLimits)

    @model_validator(mode="after")
    def validate_risk_consistency(self) -> "RiskConfig":
        if self.max_position_size > self.limits.max_notional_percent:
            raise ValueError("max_position_size cannot exceed max_notional_percent")
        if self.kelly_fraction > 1.0:
            raise ValueError("kelly_fraction should not exceed 1.0 (full Kelly)")
        return self


class CacheTTL(BaseModel):
    """Cache TTL configuration."""

    options_chain: int = Field(300, ge=0, description="Options chain cache TTL in seconds")
    stock_quotes: int = Field(60, ge=0, description="Stock quotes cache TTL in seconds")
    volatility: int = Field(900, ge=0, description="Volatility cache TTL in seconds")
    greeks: int = Field(300, ge=0, description="Greeks cache TTL in seconds")


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


class DataQuality(BaseModel):
    """Data quality configuration."""

    stale_data_seconds: int = Field(30, ge=0, description="Stale data threshold in seconds")
    min_confidence_score: float = Field(
        0.80, ge=0.0, le=1.0, description="Minimum confidence score"
    )
    min_bid_ask_spread: float = Field(0.10, ge=0.0, description="Minimum bid-ask spread")
    min_open_interest: int = Field(50, ge=0, description="Minimum open interest")
    min_volume: int = Field(10, ge=0, description="Minimum volume")


class DataConfig(BaseModel):
    """Market data configuration."""

    cache_ttl: CacheTTL = Field(default_factory=CacheTTL)
    api_timeouts: APITimeouts = Field(default_factory=APITimeouts)
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


class TradingConfig(BaseModel):
    """Trading environment configuration."""

    mode: str = Field(
        "paper",
        pattern="^(live|paper|backtest)$",
        description="Trading mode",
    )
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    market_hours: MarketHours = Field(default_factory=MarketHours)


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


class UnityConfig(BaseModel):
    """Unity-specific configuration."""

    ticker: str = Field("U", description="Stock ticker")
    company_name: str = Field("Unity Software Inc.", description="Company name")
    volatility: UnityVolatility = Field(default_factory=UnityVolatility)


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
    volatility_model: ModelConfig = Field(
        default=ModelConfig(type="garch", lookback_days=252)
    )


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
    export_format: str = Field(
        "csv", pattern="^(csv|json)$", description="Export format"
    )
    export_path: Path = Field(Path("./data/performance/"), description="Export path")


class OperationsConfig(BaseModel):
    """Operational settings configuration."""

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


class SlippageConfig(BaseModel):
    """Slippage model configuration."""

    type: str = Field(
        "fixed", pattern="^(fixed|proportional)$", description="Slippage type"
    )
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


class SystemPerformance(BaseModel):
    """System performance targets."""

    max_decision_time_ms: int = Field(200, ge=1, description="Max decision time in ms")
    max_memory_mb: int = Field(512, ge=1, description="Max memory usage in MB")


class SystemFeatures(BaseModel):
    """System feature flags."""

    enable_diagnostics: bool = Field(True, description="Enable diagnostics")
    enable_self_tuning: bool = Field(False, description="Enable self-tuning")
    enable_config_tracking: bool = Field(True, description="Enable config tracking")
    enable_auto_recovery: bool = Field(True, description="Enable auto-recovery")


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
        "https://api.schwabapi.com/v1/oauth/authorize",
        description="Authorization endpoint"
    )
    token_url: str = Field(
        "https://api.schwabapi.com/v1/oauth/token",
        description="Token endpoint"
    )
    scope: str = Field(
        "AccountsRead MarketDataRead",
        description="OAuth scopes to request"
    )
    
    # Token management
    auto_refresh: bool = Field(True, description="Enable automatic token refresh")
    token_refresh_buffer_minutes: int = Field(
        5, ge=1, le=60, description="Minutes before expiry to refresh token"
    )
    
    # Cache settings
    enable_cache: bool = Field(True, description="Enable API response caching")
    cache_ttl_seconds: int = Field(
        3600, ge=0, description="Default cache TTL in seconds"
    )
    cache_max_size_mb: int = Field(
        100, ge=1, description="Maximum cache size in MB"
    )
    
    # Rate limiting
    rate_limit_rps: float = Field(
        10.0, gt=0, description="Requests per second limit"
    )
    rate_limit_burst: int = Field(
        20, ge=1, description="Burst capacity for rate limiter"
    )
    enable_circuit_breaker: bool = Field(
        True, description="Enable circuit breaker pattern"
    )
    circuit_breaker_threshold: int = Field(
        5, ge=1, description="Failures before circuit opens"
    )
    circuit_breaker_recovery_seconds: int = Field(
        60, ge=1, description="Recovery timeout in seconds"
    )
    
    # Retry settings
    max_retry_attempts: int = Field(
        3, ge=1, le=10, description="Maximum retry attempts"
    )
    retry_backoff_factor: float = Field(
        2.0, ge=1.0, description="Exponential backoff factor"
    )
    
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


class WheelConfig(BaseModel):
    """Root configuration schema for the wheel trading system."""

    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    unity: UnityConfig = Field(default_factory=UnityConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
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
        health["warnings"].append(
            "max_position_size > 25% may lead to concentrated risk"
        )
    
    if config.risk.kelly_fraction > 0.5:
        health["warnings"].append(
            "kelly_fraction > 0.5 (Half-Kelly) increases risk significantly"
        )
    
    # Check strategy parameters
    if config.strategy.delta_target > 0.35:
        health["warnings"].append(
            "delta_target > 0.35 increases assignment probability"
        )
    
    if config.strategy.days_to_expiry_target < 30:
        health["warnings"].append(
            "days_to_expiry_target < 30 may not capture enough premium"
        )
    
    # Check data quality thresholds
    if config.data.quality.min_open_interest < 100:
        health["recommendations"].append(
            "Consider increasing min_open_interest to 100+ for better liquidity"
        )
    
    # Check ML configuration
    if config.ml.enabled and not config.ml.model_path.exists():
        health["errors"].append(
            f"ML enabled but model_path does not exist: {config.ml.model_path}"
        )
        health["valid"] = False
    
    # Check operational settings
    if config.operations.logging.retention_days > 90:
        health["recommendations"].append(
            "Log retention > 90 days may use significant disk space"
        )
    
    # Add summary
    health["summary"] = (
        f"Configuration health: {len(health['errors'])} errors, "
        f"{len(health['warnings'])} warnings, "
        f"{len(health['recommendations'])} recommendations"
    )
    
    return health