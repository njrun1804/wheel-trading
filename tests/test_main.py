"""Tests for the main module."""

import logging
from unittest.mock import patch

from src.main import main


def test_main_runs_without_error(caplog):
    """Test that main function runs without errors."""
    with caplog.at_level(logging.INFO):
        main()

    assert "Starting Wheel Trading Application" in caplog.text
    assert "Application started successfully" in caplog.text


def test_main_logs_environment():
    """Test that main logs environment information."""
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: {
            "NODE_ENV": "test",
            "GOOGLE_CLOUD_PROJECT": "test-project",
        }.get(key, default)

        # Capture logs
        with patch("src.main.logger") as mock_logger:
            main()

            # Verify environment was logged
            calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("Environment: test" in call for call in calls)
            assert any("Google Cloud Project: test-project" in call for call in calls)
