"""
Authentication exceptions with self-recovery hints.
"""


class AuthError(Exception):
    """Base authentication exception."""

    def __init__(
        self,
        message: str,
        recovery_action: str | None = None,
        is_recoverable: bool = True,
    ):
        super().__init__(message)
        self.recovery_action = recovery_action
        self.is_recoverable = is_recoverable


class TokenExpiredError(AuthError):
    """Token has expired and needs refresh."""

    def __init__(self, message: str = "Access token expired"):
        super().__init__(
            message,
            recovery_action="Refreshing token automatically",
            is_recoverable=True,
        )


class RateLimitError(AuthError):
    """API rate limit exceeded."""

    def __init__(
        self, retry_after: int | None = None, message: str = "Rate limit exceeded"
    ):
        self.retry_after = retry_after
        recovery = (
            f"Retry after {retry_after}s"
            if retry_after
            else "Applying exponential backoff"
        )
        super().__init__(message, recovery_action=recovery, is_recoverable=True)


class InvalidCredentialsError(AuthError):
    """Invalid or missing credentials."""

    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(
            message,
            recovery_action="Please run initial setup with valid credentials",
            is_recoverable=False,
        )


class NetworkError(AuthError):
    """Network connectivity issues."""

    def __init__(self, message: str = "Network error"):
        super().__init__(
            message, recovery_action="Falling back to cached data", is_recoverable=True
        )
