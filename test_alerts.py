#!/usr/bin/env python3
"""Test alert channels."""

import os

from dotenv import load_dotenv

load_dotenv(".env.monitoring")

from src.unity_wheel.monitoring.scripts.daily_health_check import DailyHealthCheck

# Create test alert
checker = DailyHealthCheck()
checker.alerts.append(
    {
        "level": "INFO",
        "message": "This is a test alert - monitoring is working!",
        "metric": "test",
        "value": 1.0,
    }
)

# Send alerts
checker.send_alerts()
print("\n✅ Test alerts sent - check your Slack/email")
