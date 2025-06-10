class SchwabError(Exception):
    """Base exception for all Schwab client errors."""

    pass


class SchwabAuthError(SchwabError):
    """Authentication or authorization error."""

    pass


class SchwabDataError(SchwabError):
    """Data validation or parsing error."""

    pass


class SchwabNetworkError(SchwabError):
    """Network or connectivity error."""

    pass


class SchwabRateLimitError(SchwabError):
    """Rate limit exceeded error."""

    pass
