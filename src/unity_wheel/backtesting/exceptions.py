class InsufficientDataError(Exception):
    """Raised when there is not enough historical data for backtesting."""

    def __init__(self, message: str):
        super().__init__(message)
