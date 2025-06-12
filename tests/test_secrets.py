"""Tests for secret management module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from unity_wheel.secrets import SecretManager, SecretProvider
from unity_wheel.secrets.exceptions import (
    SecretConfigError,
    SecretNotFoundError,
    SecretProviderError,
)
from unity_wheel.secrets.integration import (
    SecretInjector,
    get_databento_api_key,
    get_fred_api_key,
    migrate_env_to_secrets,
)
from unity_wheel.secrets.manager import EnvironmentSecretBackend, LocalSecretBackend


class TestLocalSecretBackend:
    """Test local encrypted secret storage."""

    def test_init_creates_directory(self):
        """Test that initialization creates storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test_secrets"
            backend = LocalSecretBackend(storage_path)

            assert storage_path.exists()
            assert storage_path.is_dir()
            assert (storage_path / ".key").exists()

    def test_set_get_secret(self):
        """Test setting and getting secrets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalSecretBackend(Path(tmpdir))

            # Set secret
            backend.set_secret("test_key", "test_value")

            # Get secret
            value = backend.get_secret("test_key")
            assert value == "test_value"

            # Get non-existent secret
            assert backend.get_secret("non_existent") is None

    def test_delete_secret(self):
        """Test deleting secrets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalSecretBackend(Path(tmpdir))

            # Set and delete
            backend.set_secret("test_key", "test_value")
            backend.delete_secret("test_key")

            # Verify deleted
            assert backend.get_secret("test_key") is None

    def test_list_secrets(self):
        """Test listing secrets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalSecretBackend(Path(tmpdir))

            # Initially empty
            assert backend.list_secrets() == []

            # Add secrets
            backend.set_secret("key1", "value1")
            backend.set_secret("key2", "value2")

            # List should contain both
            secrets = backend.list_secrets()
            assert len(secrets) == 2
            assert "key1" in secrets
            assert "key2" in secrets

    def test_encryption(self):
        """Test that secrets are actually encrypted on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalSecretBackend(Path(tmpdir))

            # Set a secret
            secret_value = "super_secret_password_123"
            backend.set_secret("password", secret_value)

            # Read raw file content
            secrets_file = Path(tmpdir) / "secrets.enc"
            with open(secrets_file, "rb") as f:
                raw_content = f.read()

            # Ensure secret value is not in raw content
            assert secret_value.encode() not in raw_content
            assert b"password" not in raw_content  # Key should also be encrypted


class TestEnvironmentSecretBackend:
    """Test environment variable secret backend."""

    def test_get_secret_from_env(self):
        """Test getting secrets from environment variables."""
        backend = EnvironmentSecretBackend(prefix="TEST_")

        # Set environment variable
        os.environ["TEST_API_KEY"] = "test_value_123"

        try:
            # Get secret (lowercase key)
            value = backend.get_secret("api_key")
            assert value == "test_value_123"

            # Non-existent
            assert backend.get_secret("non_existent") is None
        finally:
            # Cleanup
            del os.environ["TEST_API_KEY"]

    def test_list_secrets_from_env(self):
        """Test listing secrets from environment."""
        backend = EnvironmentSecretBackend(prefix="TEST_")

        # Set multiple env vars
        os.environ["TEST_KEY1"] = "value1"
        os.environ["TEST_KEY2"] = "value2"
        os.environ["OTHER_KEY"] = "value3"  # Different prefix

        try:
            secrets = backend.list_secrets()
            assert len(secrets) == 2
            assert "key1" in secrets
            assert "key2" in secrets
            assert "other_key" not in secrets
        finally:
            # Cleanup
            for key in ["TEST_KEY1", "TEST_KEY2", "OTHER_KEY"]:
                os.environ.pop(key, None)

    def test_readonly_operations(self):
        """Test that write operations raise errors."""
        backend = EnvironmentSecretBackend()

        with pytest.raises(SecretProviderError):
            backend.set_secret("key", "value")

        with pytest.raises(SecretProviderError):
            backend.delete_secret("key")


class TestSecretManager:
    """Test the main SecretManager class."""

    def test_auto_detect_provider(self):
        """Test provider auto-detection."""
        # Without GCP env vars, should use local
        manager = SecretManager()
        assert manager.provider == SecretProvider.LOCAL

    def test_get_secret_with_prompt(self):
        """Test getting secret with prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecretManager(provider=SecretProvider.LOCAL, storage_path=Path(tmpdir))

            # Mock input to simulate user entry
            with patch("getpass.getpass", return_value="user_entered_secret"):
                value = manager.get_secret("test_api_key", prompt_if_missing=True)
                assert value == "user_entered_secret"

                # Should be saved
                assert manager.backend.get_secret("test_api_key") == "user_entered_secret"

    def test_get_credentials(self):
        """Test getting service credentials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecretManager(provider=SecretProvider.LOCAL, storage_path=Path(tmpdir))

            # Set up Schwab credentials
            manager.backend.set_secret("schwab_client_id", "test_id")
            manager.backend.set_secret("schwab_client_secret", "test_secret")

            # Get credentials
            creds = manager.get_credentials("schwab", prompt_if_missing=False)
            assert creds["client_id"] == "test_id"
            assert creds["client_secret"] == "test_secret"

    def test_set_credentials(self):
        """Test setting service credentials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecretManager(provider=SecretProvider.LOCAL, storage_path=Path(tmpdir))

            # Set credentials
            manager.set_credentials("databento", api_key="test_databento_key")

            # Verify stored
            assert manager.backend.get_secret("databento_api_key") == "test_databento_key"

    def test_list_configured_services(self):
        """Test listing configured services."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecretManager(provider=SecretProvider.LOCAL, storage_path=Path(tmpdir))

            # Initially none configured
            configured = manager.list_configured_services()
            assert configured == {"schwab": False, "databento": False, "ofred": False}

            # Configure Schwab
            manager.set_credentials("schwab", client_id="id", client_secret="secret")

            # Now Schwab should be configured
            configured = manager.list_configured_services()
            assert configured["schwab"] is True
            assert configured["databento"] is False


class TestSecretInjector:
    """Test the SecretInjector context manager."""

    def test_inject_service_credentials(self):
        """Test injecting service credentials into environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up manager with test credentials
            manager = SecretManager(provider=SecretProvider.LOCAL, storage_path=Path(tmpdir))
            manager.set_credentials("schwab", client_id="test_id", client_secret="test_secret")

            # Mock get_secret_manager to return our test manager
            with patch(
                "src.unity_wheel.secrets.integration.get_secret_manager", return_value=manager
            ):
                # Test injection
                assert "WHEEL_AUTH__CLIENT_ID" not in os.environ

                with SecretInjector(service="schwab"):
                    # Should be injected
                    assert os.environ["WHEEL_AUTH__CLIENT_ID"] == "test_id"
                    assert os.environ["WHEEL_AUTH__CLIENT_SECRET"] == "test_secret"

                # Should be removed after context
                assert "WHEEL_AUTH__CLIENT_ID" not in os.environ
                assert "WHEEL_AUTH__CLIENT_SECRET" not in os.environ

    def test_restore_original_env(self):
        """Test that original environment is restored."""
        # Set original value
        os.environ["TEST_VAR"] = "original_value"

        try:
            with SecretInjector(secrets={"TEST_VAR": "injected_value"}):
                assert os.environ["TEST_VAR"] == "injected_value"

            # Should be restored
            assert os.environ["TEST_VAR"] == "original_value"
        finally:
            # Cleanup
            del os.environ["TEST_VAR"]


class TestIntegrationFunctions:
    """Test integration helper functions."""

    def test_migrate_env_to_secrets(self):
        """Test migrating environment variables to secrets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecretManager(provider=SecretProvider.LOCAL, storage_path=Path(tmpdir))

            # Set environment variables
            os.environ["DATABENTO_API_KEY"] = "env_databento_key"

            try:
                with patch(
                    "src.unity_wheel.secrets.integration.get_secret_manager", return_value=manager
                ):
                    # Run migration
                    migrate_env_to_secrets()

                    # Verify migrated
                    assert manager.backend.get_secret("databento_api_key") == "env_databento_key"
            finally:
                # Cleanup
                del os.environ["DATABENTO_API_KEY"]


@pytest.mark.skipif(
    "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ, reason="GCP credentials not configured"
)
class TestGCPSecretBackend:
    """Test Google Cloud Secret Manager backend (requires GCP setup)."""

    def test_gcp_backend_operations(self):
        """Test basic GCP backend operations."""
        # This would require actual GCP setup
        # Placeholder for GCP-specific tests
        pass
