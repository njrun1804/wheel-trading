"""Exceptions for secret management."""


class SecretError(Exception):
    """Base exception for all secret-related errors."""


class SecretNotFoundError(SecretError):
    """Raised when a requested secret is not found."""


class SecretConfigError(SecretError):
    """Raised when secret configuration is invalid."""


class SecretProviderError(SecretError):
    """Raised when a secret provider operation fails."""