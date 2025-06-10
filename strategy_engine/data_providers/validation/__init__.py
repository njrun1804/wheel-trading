"""Data validation module to ensure only real market data is used."""

from .live_data_validator import LiveDataValidator, validate_market_data

__all__ = ["LiveDataValidator", "validate_market_data"]
