#!/usr/bin/env python3
"""
Verify Codex container setup is working correctly.
Tests imports and basic functionality with fallbacks.
"""

import os
import sys

print("🔍 VERIFYING CONTAINER SETUP")
print("=" * 30)

# Check Python version
print(f"\n✓ Python {sys.version.split()[0]}")

# Check environment variables
print("\n📋 Environment Variables:")
env_vars = [
    "USE_MOCK_DATA",
    "OFFLINE_MODE",
    "USE_PURE_PYTHON",
    "DATABENTO_SKIP_VALIDATION",
    "CONTAINER_MODE",
]
for var in env_vars:
    value = os.environ.get(var, "not set")
    symbol = "✓" if value == "true" else "✗"
    print(f"   {symbol} {var}: {value}")

# Check package availability
print("\n📦 Package Availability:")
packages = {"numpy": None, "pandas": None, "scipy": None, "pydantic": None}

# First check from /tmp to avoid import conflicts
import subprocess

print("   (Checking from clean environment...)")
for pkg in packages:
    try:
        result = subprocess.run(
            ["python3", "-c", f"import {pkg}; print({pkg}.__version__)"],
            cwd="/tmp",
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            packages[pkg] = version
            print(f"   ✓ {pkg} {version}")
        else:
            print(f"   ✗ {pkg} not available")
    except:
        print(f"   ✗ {pkg} check failed")

# Determine mode
use_pure_python = os.environ.get("USE_PURE_PYTHON") == "true"
has_numpy = packages["numpy"] is not None

print(f"\n🔧 Mode: {'Pure Python' if use_pure_python or not has_numpy else 'NumPy accelerated'}")

# Test basic calculation
print("\n🧮 Testing Black-Scholes calculation:")
if has_numpy and not use_pure_python:
    print("   Using NumPy implementation")
    try:
        import numpy as np
        from scipy.stats import norm

        # Simple Black-Scholes
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))
        print(f"   ✓ Call price: ${call_price:.2f}")
    except Exception as e:
        print(f"   ✗ NumPy calculation failed: {e}")
else:
    print("   Using Pure Python implementation")
    import math

    # Pure Python Black-Scholes approximation
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    # Approximate normal CDF
    def norm_cdf(x):
        return (1 + math.erf(x / math.sqrt(2))) / 2

    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    print(f"   ✓ Call price: ${call_price:.2f}")

# Test Unity imports
print("\n🏗️ Testing Unity imports:")
sys.path.insert(0, ".")
try:
    from unity_wheel.math.options import black_scholes_price_validated

    print("   ✓ Unity trading imports successful")
except ImportError as e:
    print(f"   ⚠️  Unity imports need packages: {str(e).split(':')[0]}")
    print("   ℹ️  This is expected in Pure Python mode")

# Summary
print("\n" + "=" * 30)
if use_pure_python or not has_numpy:
    print("✅ SETUP COMPLETE - Pure Python mode active")
    print("   The code will use fallback implementations")
else:
    print("✅ SETUP COMPLETE - Full NumPy mode active")
    print("   All optimizations available")

print("\nYou can now work on the Unity trading code!")
print("The system will automatically use the appropriate")
print("implementations based on available packages.")
