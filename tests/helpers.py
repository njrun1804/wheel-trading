"""Test helper functions for checking secret availability."""

from src.unity_wheel.secrets.integration import get_secret_manager


def has_fred_api_key() -> bool:
    """Check if FRED API key is available in SecretManager."""
    try:
        manager = get_secret_manager()
        manager.get_secret("ofred_api_key", prompt_if_missing=False)
        return True
    except Exception:
        return False


def has_databento_api_key() -> bool:
    """Check if Databento API key is available in SecretManager."""
    try:
        manager = get_secret_manager()
        manager.get_secret("databento_api_key", prompt_if_missing=False)
        return True
    except Exception:
        return False


def has_schwab_credentials() -> bool:
    """Check if Schwab credentials are available in SecretManager."""
    try:
        manager = get_secret_manager()
        manager.get_credentials("schwab", prompt_if_missing=False)
        return True
    except Exception:
        return False