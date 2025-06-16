"""Integration module to update existing code to use SecretManager."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


import os
from functools import lru_cache

from unity_wheel.utils import get_logger

from .manager import SecretManager

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_secret_manager() -> SecretManager:
    """Return a cached ``SecretManager`` instance."""
    return SecretManager()


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
            except (ValueError, KeyError, AttributeError) as e:
                logger.warning(f"Failed to migrate {env_var}: {e}")

    if migrated > 0:
        logger.info(f"Migrated {migrated} environment variables to SecretManager")
        logger.info(
            "\n✓ Migrated {migrated} credentials from environment to SecretManager"
        )
        logger.info("You can now remove these from your environment variables.")


class SecretInjector:
    """Context manager for temporarily injecting secrets into environment.

    Useful for libraries that only read from environment variables.
    """

    def __init__(
        self, service: str | None = None, secrets: dict[str, str] | None = None
    ):
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
            if self.service == "databento":
                self.secrets["DATABENTO_API_KEY"] = manager.get_secret(
                    "databento_api_key"
                )
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
