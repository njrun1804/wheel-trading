"""Unified secret management for local and Google Cloud environments."""

import base64
import getpass
import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .exceptions import SecretConfigError, SecretNotFoundError, SecretProviderError

logger = logging.getLogger(__name__)


class SecretProvider(str, Enum):
    """Available secret providers."""

    LOCAL = "local"
    GCP = "gcp"
    ENVIRONMENT = "environment"


class BaseSecretBackend(ABC):
    """Abstract base class for secret backends."""

    @abstractmethod
    def get_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve a secret by ID."""

    @abstractmethod
    def set_secret(self, secret_id: str, value: str) -> None:
        """Store a secret."""

    @abstractmethod
    def delete_secret(self, secret_id: str) -> None:
        """Delete a secret."""

    @abstractmethod
    def list_secrets(self) -> list[str]:
        """List all available secret IDs."""


class LocalSecretBackend(BaseSecretBackend):
    """Local encrypted secret storage."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize local secret backend.

        Args:
            storage_path: Path to store encrypted secrets. Defaults to ~/.wheel_trading/secrets/
        """
        self.storage_path = storage_path or Path.home() / ".wheel_trading" / "secrets"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.secrets_file = self.storage_path / "secrets.enc"
        self._fernet = self._get_or_create_cipher()

    def _get_or_create_cipher(self) -> Fernet:
        """Get or create encryption cipher using machine-specific key."""
        key_file = self.storage_path / ".key"

        if key_file.exists():
            with open(key_file, "rb") as f:
                key = f.read()
        else:
            # Generate machine-specific key
            machine_id = f"{os.getuid()}:{os.uname().nodename}".encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"wheel_trading_secrets",
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(machine_id))

            # Save key with restricted permissions
            key_file.touch(mode=0o600)
            with open(key_file, "wb") as f:
                f.write(key)

        return Fernet(key)

    def _load_secrets(self) -> Dict[str, str]:
        """Load and decrypt secrets from disk."""
        if not self.secrets_file.exists():
            return {}

        try:
            with open(self.secrets_file, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = self._fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            raise SecretProviderError(f"Failed to load local secrets: {e}")

    def _save_secrets(self, secrets: Dict[str, str]) -> None:
        """Encrypt and save secrets to disk."""
        try:
            json_data = json.dumps(secrets).encode()
            encrypted_data = self._fernet.encrypt(json_data)

            # Save with restricted permissions
            self.secrets_file.touch(mode=0o600)
            with open(self.secrets_file, "wb") as f:
                f.write(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise SecretProviderError(f"Failed to save local secrets: {e}")

    def get_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve a secret by ID."""
        secrets = self._load_secrets()
        return secrets.get(secret_id)

    def set_secret(self, secret_id: str, value: str) -> None:
        """Store a secret."""
        secrets = self._load_secrets()
        secrets[secret_id] = value
        self._save_secrets(secrets)

    def delete_secret(self, secret_id: str) -> None:
        """Delete a secret."""
        secrets = self._load_secrets()
        if secret_id in secrets:
            del secrets[secret_id]
            self._save_secrets(secrets)

    def list_secrets(self) -> list[str]:
        """List all available secret IDs."""
        return list(self._load_secrets().keys())


class GCPSecretBackend(BaseSecretBackend):
    """Google Cloud Secret Manager backend."""

    def __init__(self, project_id: Optional[str] = None):
        """Initialize GCP Secret Manager backend.

        Args:
            project_id: GCP project ID. If not provided, will try to detect from environment.
        """
        try:
            from google.api_core import exceptions as gcp_exceptions
            from google.cloud import secretmanager
        except ImportError:
            raise SecretConfigError(
                "Google Cloud Secret Manager client not installed. "
                "Run: pip install google-cloud-secret-manager"
            )

        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID")
        if not self.project_id:
            raise SecretConfigError(
                "GCP project ID not provided. Set GCP_PROJECT_ID environment variable "
                "or pass project_id parameter."
            )

        self.client = secretmanager.SecretManagerServiceClient()
        self._gcp_exceptions = gcp_exceptions

    def _get_secret_path(self, secret_id: str) -> str:
        """Get full secret resource path."""
        return f"projects/{self.project_id}/secrets/{secret_id}/versions/latest"

    def _get_parent_path(self) -> str:
        """Get parent path for listing secrets."""
        return f"projects/{self.project_id}"

    def get_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve a secret by ID."""
        try:
            response = self.client.access_secret_version(
                request={"name": self._get_secret_path(secret_id)}
            )
            return response.payload.data.decode("UTF-8")
        except self._gcp_exceptions.NotFound:
            return None
        except Exception as e:
            logger.error(f"Failed to get secret from GCP: {e}")
            raise SecretProviderError(f"Failed to get secret from GCP: {e}")

    def set_secret(self, secret_id: str, value: str) -> None:
        """Store a secret."""
        try:
            # Try to create the secret
            try:
                self.client.create_secret(
                    request={
                        "parent": self._get_parent_path(),
                        "secret_id": secret_id,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )
            except self._gcp_exceptions.AlreadyExists:
                pass  # Secret already exists, we'll add a new version

            # Add secret version
            parent = f"projects/{self.project_id}/secrets/{secret_id}"
            self.client.add_secret_version(
                request={
                    "parent": parent,
                    "payload": {"data": value.encode("UTF-8")},
                }
            )
        except Exception as e:
            logger.error(f"Failed to set secret in GCP: {e}")
            raise SecretProviderError(f"Failed to set secret in GCP: {e}")

    def delete_secret(self, secret_id: str) -> None:
        """Delete a secret."""
        try:
            secret_name = f"projects/{self.project_id}/secrets/{secret_id}"
            self.client.delete_secret(request={"name": secret_name})
        except self._gcp_exceptions.NotFound:
            pass  # Already deleted
        except Exception as e:
            logger.error(f"Failed to delete secret from GCP: {e}")
            raise SecretProviderError(f"Failed to delete secret from GCP: {e}")

    def list_secrets(self) -> list[str]:
        """List all available secret IDs."""
        try:
            secrets = []
            for secret in self.client.list_secrets(request={"parent": self._get_parent_path()}):
                # Extract secret ID from full resource name
                secret_id = secret.name.split("/")[-1]
                secrets.append(secret_id)
            return secrets
        except Exception as e:
            logger.error(f"Failed to list secrets from GCP: {e}")
            raise SecretProviderError(f"Failed to list secrets from GCP: {e}")


class EnvironmentSecretBackend(BaseSecretBackend):
    """Environment variable secret backend (read-only)."""

    def __init__(self, prefix: str = "WHEEL_"):
        """Initialize environment backend.

        Args:
            prefix: Prefix for environment variables to consider as secrets.
        """
        self.prefix = prefix

    def get_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve a secret by ID from environment variables."""
        env_var = f"{self.prefix}{secret_id.upper()}"
        return os.environ.get(env_var)

    def set_secret(self, secret_id: str, value: str) -> None:
        """Not supported for environment backend."""
        raise SecretProviderError("Cannot set secrets in environment backend")

    def delete_secret(self, secret_id: str) -> None:
        """Not supported for environment backend."""
        raise SecretProviderError("Cannot delete secrets from environment backend")

    def list_secrets(self) -> list[str]:
        """List all available secret IDs from environment."""
        secrets = []
        for key in os.environ:
            if key.startswith(self.prefix):
                secret_id = key[len(self.prefix) :].lower()
                secrets.append(secret_id)
        return secrets


class SecretManager:
    """Unified secret manager supporting multiple backends."""

    # Known credential sets
    CREDENTIAL_SETS = {
        "schwab": {
            "client_id": "Schwab OAuth Client ID",
            "client_secret": "Schwab OAuth Client Secret",
        },
        "databento": {
            "api_key": "Databento API Key",
        },
        "ofred": {
            "api_key": "FRED (Federal Reserve Economic Data) API Key",
        },
    }

    def __init__(self, provider: Optional[Union[SecretProvider, str]] = None, **kwargs: Any):
        """Initialize secret manager.

        Args:
            provider: Secret provider to use. Defaults to auto-detection.
            **kwargs: Additional arguments passed to the backend.
        """
        if provider is None:
            provider = self._detect_provider()
        elif isinstance(provider, str):
            provider = SecretProvider(provider)

        self.provider = provider
        self.backend = self._create_backend(provider, **kwargs)
        logger.info(f"Initialized SecretManager with {provider.value} provider")

    def _detect_provider(self) -> SecretProvider:
        """Auto-detect the best provider based on environment."""
        # Check if running in GCP
        if os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                # Try to import and use GCP
                import google.cloud.secretmanager

                return SecretProvider.GCP
            except ImportError:
                logger.warning(
                    "GCP environment detected but google-cloud-secret-manager not installed"
                )

        # Default to local
        return SecretProvider.LOCAL

    def _create_backend(self, provider: SecretProvider, **kwargs: Any) -> BaseSecretBackend:
        """Create the appropriate backend for the provider."""
        if provider == SecretProvider.LOCAL:
            return LocalSecretBackend(**kwargs)
        elif provider == SecretProvider.GCP:
            return GCPSecretBackend(**kwargs)
        elif provider == SecretProvider.ENVIRONMENT:
            return EnvironmentSecretBackend(**kwargs)
        else:
            raise SecretConfigError(f"Unknown provider: {provider}")

    def get_secret(self, secret_id: str, prompt_if_missing: bool = True) -> str:
        """Get a secret, optionally prompting if not found.

        Args:
            secret_id: Secret identifier (e.g., "schwab_client_id")
            prompt_if_missing: Whether to prompt user if secret not found

        Returns:
            Secret value

        Raises:
            SecretNotFoundError: If secret not found and prompt_if_missing is False
        """
        # Try primary backend
        value = self.backend.get_secret(secret_id)

        # Fallback to environment if not found
        if value is None and self.provider != SecretProvider.ENVIRONMENT:
            env_backend = EnvironmentSecretBackend()
            value = env_backend.get_secret(secret_id)

        # Prompt if still not found
        if value is None and prompt_if_missing:
            value = self._prompt_for_secret(secret_id)
            if value:
                # Save to backend (if writable)
                try:
                    self.backend.set_secret(secret_id, value)
                except SecretProviderError:
                    logger.warning(f"Could not save {secret_id} to {self.provider.value} backend")

        if value is None:
            raise SecretNotFoundError(f"Secret '{secret_id}' not found")

        return value

    def _prompt_for_secret(self, secret_id: str) -> Optional[str]:
        """Prompt user for a secret value."""
        # Check if this is a known credential
        for service, creds in self.CREDENTIAL_SETS.items():
            for cred_type, description in creds.items():
                if secret_id == f"{service}_{cred_type}":
                    prompt = f"Enter {description}: "
                    if "secret" in cred_type or "key" in cred_type:
                        return getpass.getpass(prompt)
                    else:
                        return input(prompt)

        # Generic prompt
        if "secret" in secret_id.lower() or "key" in secret_id.lower():
            return getpass.getpass(f"Enter value for {secret_id}: ")
        else:
            return input(f"Enter value for {secret_id}: ")

    def get_credentials(self, service: str, prompt_if_missing: bool = True) -> Dict[str, str]:
        """Get all credentials for a service.

        Args:
            service: Service name (e.g., "schwab", "databento", "ofred")
            prompt_if_missing: Whether to prompt for missing credentials

        Returns:
            Dictionary of credential name to value

        Raises:
            SecretConfigError: If service is unknown
            SecretNotFoundError: If any credential is missing and prompt_if_missing is False
        """
        if service not in self.CREDENTIAL_SETS:
            raise SecretConfigError(f"Unknown service: {service}")

        credentials = {}
        for cred_type in self.CREDENTIAL_SETS[service]:
            secret_id = f"{service}_{cred_type}"
            credentials[cred_type] = self.get_secret(secret_id, prompt_if_missing)

        return credentials

    def set_credentials(self, service: str, **credentials: str) -> None:
        """Set credentials for a service.

        Args:
            service: Service name
            **credentials: Credential values (e.g., client_id="...", client_secret="...")
        """
        if service not in self.CREDENTIAL_SETS:
            raise SecretConfigError(f"Unknown service: {service}")

        for cred_type, value in credentials.items():
            if cred_type not in self.CREDENTIAL_SETS[service]:
                raise SecretConfigError(f"Unknown credential type '{cred_type}' for {service}")

            secret_id = f"{service}_{cred_type}"
            self.backend.set_secret(secret_id, value)

    def setup_all_credentials(self) -> None:
        """Interactive setup for all required credentials."""
        print("\n=== Unity Wheel Trading Bot - Credential Setup ===\n")
        print(f"Using {self.provider.value} provider for secret storage.\n")

        for service, creds in self.CREDENTIAL_SETS.items():
            print(f"\n--- {service.upper()} Credentials ---")

            # Check if already configured
            try:
                existing = self.get_credentials(service, prompt_if_missing=False)
                print(f"✓ {service} credentials already configured")
                update = input("Update? (y/N): ").lower().strip() == "y"
                if not update:
                    continue
            except SecretNotFoundError:
                pass

            # Collect credentials
            new_creds = {}
            for cred_type, description in creds.items():
                prompt = f"{description}: "
                if "secret" in cred_type or "key" in cred_type:
                    value = getpass.getpass(prompt)
                else:
                    value = input(prompt)
                new_creds[cred_type] = value

            # Save credentials
            self.set_credentials(service, **new_creds)
            print(f"✓ {service} credentials saved")

        print("\n✓ All credentials configured successfully!")
        print(f"\nCredentials are stored in: {self.provider.value}")
        if self.provider == SecretProvider.LOCAL:
            print(f"Location: ~/.wheel_trading/secrets/")

    def list_configured_services(self) -> Dict[str, bool]:
        """List which services have been configured.

        Returns:
            Dictionary mapping service name to whether it's fully configured
        """
        configured = {}
        for service in self.CREDENTIAL_SETS:
            try:
                self.get_credentials(service, prompt_if_missing=False)
                configured[service] = True
            except SecretNotFoundError:
                configured[service] = False
        return configured
