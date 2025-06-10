"""Integration module to update existing code to use SecretManager."""

import os
from functools import lru_cache
from typing import Any, Dict, Optional

from ..utils import get_logger
from .manager import SecretManager

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_secret_manager() -> SecretManager:
    """Get singleton instance of SecretManager.

    Caches the instance to avoid multiple initializations.
    """
    return SecretManager()


def get_schwab_credentials() -> Dict[str, str]:
    """Get Schwab credentials from SecretManager.

    Returns:
        Dictionary with 'client_id' and 'client_secret'

    Raises:
        SecretNotFoundError: If credentials not configured
    """
    manager = get_secret_manager()
    return manager.get_credentials("schwab")


def get_databento_api_key() -> str:
    """Get Databento API key from SecretManager.

    Returns:
        Databento API key

    Raises:
        SecretNotFoundError: If API key not configured
    """
    manager = get_secret_manager()
    return manager.get_secret("databento_api_key")


def get_fred_api_key() -> str:
    """Get FRED API key from SecretManager.

    Returns:
        FRED API key

    Raises:
        SecretNotFoundError: If API key not configured
    """
    manager = get_secret_manager()
    return manager.get_secret("ofred_api_key")


def migrate_env_to_secrets() -> None:
    """Migrate existing environment variables to SecretManager.

    This helper function checks for existing environment variables
    and migrates them to the SecretManager for better security.
    """
    manager = get_secret_manager()
    migrations = [
        # Schwab credentials
        ("WHEEL_AUTH__CLIENT_ID", "schwab_client_id"),
        ("WHEEL_AUTH__CLIENT_SECRET", "schwab_client_secret"),
        ("SCHWAB_APP_KEY", "schwab_client_id"),  # Legacy
        ("SCHWAB_SECRET", "schwab_client_secret"),  # Legacy
        # Databento
        ("DATABENTO_API_KEY", "databento_api_key"),
        # FRED
        ("FRED_API_KEY", "ofred_api_key"),
        ("OFRED_API_KEY", "ofred_api_key"),
    ]

    migrated = 0
    for env_var, secret_id in migrations:
        value = os.environ.get(env_var)
        if value:
            try:
                # Check if already exists
                existing = manager.backend.get_secret(secret_id)
                if not existing:
                    manager.backend.set_secret(secret_id, value)
                    logger.info(f"Migrated {env_var} to secret {secret_id}")
                    migrated += 1
            except Exception as e:
                logger.warning(f"Failed to migrate {env_var}: {e}")

    if migrated > 0:
        logger.info(f"Migrated {migrated} environment variables to SecretManager")
        print(f"\n✓ Migrated {migrated} credentials from environment to SecretManager")
        print("You can now remove these from your environment variables.")


class SecretInjector:
    """Context manager for temporarily injecting secrets into environment.

    Useful for libraries that only read from environment variables.
    """

    def __init__(self, service: Optional[str] = None, secrets: Optional[Dict[str, str]] = None):
        """Initialize secret injector.

        Args:
            service: Service name to inject all credentials for
            secrets: Specific secrets to inject (key -> env_var mapping)
        """
        self.service = service
        self.secrets = secrets or {}
        self.original_env = {}
        self.injected_vars = set()

    def __enter__(self):
        """Inject secrets into environment."""
        manager = get_secret_manager()

        # Handle service credentials
        if self.service:
            if self.service == "schwab":
                creds = manager.get_credentials("schwab")
                self.secrets.update(
                    {
                        "WHEEL_AUTH__CLIENT_ID": creds["client_id"],
                        "WHEEL_AUTH__CLIENT_SECRET": creds["client_secret"],
                    }
                )
            elif self.service == "databento":
                self.secrets["DATABENTO_API_KEY"] = manager.get_secret("databento_api_key")
            elif self.service == "fred":
                self.secrets["FRED_API_KEY"] = manager.get_secret("ofred_api_key")

        # Inject all secrets
        for env_var, value in self.secrets.items():
            # Save original value if exists
            if env_var in os.environ:
                self.original_env[env_var] = os.environ[env_var]

            os.environ[env_var] = value
            self.injected_vars.add(env_var)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original environment."""
        # Remove injected variables
        for env_var in self.injected_vars:
            if env_var in self.original_env:
                # Restore original
                os.environ[env_var] = self.original_env[env_var]
            else:
                # Remove if didn't exist before
                os.environ.pop(env_var, None)
