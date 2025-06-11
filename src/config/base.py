"""Configuration management for wheel trading application."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Trading
    broker_api_key: SecretStr | None = Field(
        None,
        description="Broker API key",
        alias="BROKER_API_KEY",
    )
    broker_api_secret: SecretStr | None = Field(
        None,
        description="Broker API secret",
        alias="BROKER_API_SECRET",
    )
    trading_mode: Literal["live", "paper", "backtest"] = Field(
        default="paper",
        description="Trading mode",
        alias="TRADING_MODE",
    )

    # Strategy
    wheel_delta_target: float = Field(
        default=0.3,
        description="Target delta for wheel options",
        alias="WHEEL_DELTA_TARGET",
    )
    days_to_expiry_target: int = Field(
        default=45,
        description="Target days to expiry for options",
        alias="DAYS_TO_EXPIRY_TARGET",
    )
    max_position_size: float = Field(
        default=0.2,
        description="Maximum position size as fraction of portfolio",
        alias="MAX_POSITION_SIZE",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
        alias="LOG_LEVEL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()
