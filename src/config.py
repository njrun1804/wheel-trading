"""Configuration management for wheel trading application."""

from typing import Literal, Optional

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
    broker_api_key: Optional[SecretStr] = Field(
        None,
        description="Broker API key",
        alias="BROKER_API_KEY",
    )
    broker_api_secret: Optional[SecretStr] = Field(
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

    # Google Cloud
    google_cloud_project: Optional[str] = Field(
        None,
        description="Google Cloud project ID",
        alias="GOOGLE_CLOUD_PROJECT",
    )
    google_application_credentials: Optional[str] = Field(
        None,
        description="Path to Google Cloud credentials",
        alias="GOOGLE_APPLICATION_CREDENTIALS",
    )


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
