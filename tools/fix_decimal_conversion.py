#!/usr/bin/env python3
"""Fix decimal conversion in backtester."""

# Read the current backtester
with open("src/unity_wheel/backtesting/wheel_backtester.py", "r") as f:
    content = f.read()

# Add conversion after dataframe creation
old_code = """                df = pd.DataFrame(
                    result,
                    columns=[
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "returns",
                        "volatility_20d",
                        "volatility_250d",
                        "risk_free_rate",
                    ],
                )
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                return df"""

new_code = """                df = pd.DataFrame(
                    result,
                    columns=[
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "returns",
                        "volatility_20d",
                        "volatility_250d",
                        "risk_free_rate",
                    ],
                )

                # Convert decimal types to float for calculations
                numeric_cols = ["open", "high", "low", "close", "volume", "returns",
                               "volatility_20d", "volatility_250d", "risk_free_rate"]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                return df"""

# Replace the code
content = content.replace(old_code, new_code)

# Write back
with open("src/unity_wheel/backtesting/wheel_backtester.py", "w") as f:
    f.write(content)

print("✅ Fixed decimal to float conversion in backtester")
print("✅ All numeric data properly converted for calculations")
