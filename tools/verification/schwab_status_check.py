#!/usr/bin/env python3
"""Check if Schwab's OAuth service is operational."""

import asyncio
import json

import aiohttp


async def check_schwab_status():
    """Check various Schwab API endpoints."""
    print("\n=== Schwab API Status Check ===\n")

    endpoints = [
        ("OAuth Authorization", "https://api.schwabapi.com/v1/oauth/authorize"),
        ("OAuth Token", "https://api.schwabapi.com/v1/oauth/token"),
        ("API Base", "https://api.schwabapi.com/"),
    ]

    async with aiohttp.ClientSession() as session:
        for name, url in endpoints:
            try:
                async with session.get(url, timeout=5) as response:
                    print(f"{name}: {response.status} - ", end="")
                    if response.status < 500:
                        print("✅ Operational")
                    else:
                        print("❌ Server Error")
            except asyncio.TimeoutError:
                print(f"{name}: ⏱️  Timeout")
            except Exception as e:
                print(f"{name}: ❌ {type(e).__name__}")

    print("\n=== What This Means ===")
    print("- 200-499 responses = Service is up")
    print("- 500+ or timeouts = Service issues")
    print("- 'invalid_client' with working endpoints = Credential problem")


asyncio.run(check_schwab_status())
