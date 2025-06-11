#!/usr/bin/env python3
"""
Investigate and fix Greek calculation issues identified in the database validation.

Focus on the put delta and theta calculation problems.
"""


import duckdb

from src.unity_wheel.math.options import calculate_all_greeks


def investigate_greek_issues():
    """Investigate the specific Greek calculation issues."""

    # Connect to database
    conn = duckdb.connect("data/wheel_trading_master.duckdb", read_only=True)

    print("Investigating Greek calculation issues...")
    print("=" * 60)

    # Get the problematic options
    problematic_options = conn.execute(
        """
        SELECT
            o.option_symbol, o.underlying, o.strike, o.expiration, o.option_type,
            o.premium, o.spot_price, o.delta, o.gamma, o.theta, o.vega,
            o.implied_volatility, o.dte, o.date
        FROM options o
        WHERE o.delta IS NOT NULL
            AND o.implied_volatility > 0
            AND o.expiration > o.date
            AND (
                (o.option_type = 'P' AND (o.delta > 0 OR o.delta < -1)) OR
                (o.option_type = 'C' AND (o.delta < 0 OR o.delta > 1)) OR
                (o.theta > 0 AND o.dte > 1)
            )
        ORDER BY ABS(o.delta) DESC
        LIMIT 20
    """
    ).fetchall()

    print(f"Found {len(problematic_options)} problematic options:")
    print()

    # Risk-free rate assumption (could be improved by getting from FRED data)
    risk_free_rate = 0.05  # 5% assumption

    for i, option in enumerate(problematic_options[:10], 1):
        (
            symbol,
            underlying,
            strike,
            expiration,
            option_type,
            premium,
            spot_price,
            delta,
            gamma,
            theta,
            vega,
            iv,
            dte,
            date,
        ) = option

        print(f"{i}. {symbol} ({option_type.upper()})")
        print(f"   Strike: ${strike}, Spot: ${spot_price:.2f}, DTE: {dte}")
        print(f"   Stored Delta: {delta:.6f}")
        print(f"   Stored Theta: {theta:.6f}")
        print(f"   IV: {iv:.4f}")

        # Calculate what the Greeks should be
        try:
            time_to_exp = dte / 365.0

            # Calculate theoretical Greeks using our library
            # Convert single-letter type to full word
            full_option_type = "put" if option_type == "P" else "call"
            theoretical_greeks, confidence = calculate_all_greeks(
                S=float(spot_price),
                K=float(strike),
                T=time_to_exp,
                r=risk_free_rate,
                sigma=float(iv),
                option_type=full_option_type,
            )

            print(f"   Theoretical Delta: {theoretical_greeks['delta']:.6f}")
            print(f"   Theoretical Theta: {theoretical_greeks['theta']:.6f}")
            print(f"   Confidence: {confidence:.2f}")

            # Check if the issue is a sign convention problem
            if option_type == "P":  # Put
                expected_delta_range = (-1, 0)
                if delta > 0:
                    print("   ❌ Put delta is positive (should be negative)")
                if not (-1 <= delta <= 0):
                    print(f"   ❌ Put delta out of valid range [-1, 0]: {delta}")
            else:  # Call (C)
                expected_delta_range = (0, 1)
                if delta < 0:
                    print("   ❌ Call delta is negative (should be positive)")
                if not (0 <= delta <= 1):
                    print(f"   ❌ Call delta out of valid range [0, 1]: {delta}")

            if theta > 0 and dte > 1:
                print("   ❌ Theta is positive (should typically be negative for time decay)")

        except Exception as e:
            print(f"   ❌ Error calculating theoretical Greeks: {e}")

        print()

    # Summary statistics
    print("\nSummary of Greek calculation issues:")
    print("-" * 40)

    # Count different types of issues
    put_delta_issues = conn.execute(
        """
        SELECT COUNT(*) FROM options
        WHERE option_type = 'P' AND delta > 0
    """
    ).fetchone()[0]

    call_delta_issues = conn.execute(
        """
        SELECT COUNT(*) FROM options
        WHERE option_type = 'C' AND delta < 0
    """
    ).fetchone()[0]

    positive_theta_issues = conn.execute(
        """
        SELECT COUNT(*) FROM options
        WHERE theta > 0 AND dte > 1
    """
    ).fetchone()[0]

    print(f"Put options with positive delta: {put_delta_issues}")
    print(f"Call options with negative delta: {call_delta_issues}")
    print(f"Options with positive theta (DTE > 1): {positive_theta_issues}")

    # Check if it's a systematic issue
    print("\nChecking for systematic patterns...")

    # Check if all puts have wrong sign
    put_delta_stats = conn.execute(
        """
        SELECT
            COUNT(*) as total_puts,
            COUNT(CASE WHEN delta > 0 THEN 1 END) as positive_deltas,
            AVG(delta) as avg_delta,
            MIN(delta) as min_delta,
            MAX(delta) as max_delta
        FROM options
        WHERE option_type = 'P' AND delta IS NOT NULL
    """
    ).fetchone()

    print("Put delta statistics:")
    print(f"  Total puts: {put_delta_stats[0]}")
    pos_pct = (put_delta_stats[1] / put_delta_stats[0] * 100) if put_delta_stats[0] > 0 else 0
    print(f"  Positive deltas: {put_delta_stats[1]} ({pos_pct:.1f}%)")
    avg_delta = put_delta_stats[2] if put_delta_stats[2] is not None else 0
    min_delta = put_delta_stats[3] if put_delta_stats[3] is not None else 0
    max_delta = put_delta_stats[4] if put_delta_stats[4] is not None else 0
    print(f"  Average delta: {avg_delta:.4f}")
    print(f"  Range: [{min_delta:.4f}, {max_delta:.4f}]")

    # Check call delta stats
    call_delta_stats = conn.execute(
        """
        SELECT
            COUNT(*) as total_calls,
            COUNT(CASE WHEN delta < 0 THEN 1 END) as negative_deltas,
            AVG(delta) as avg_delta,
            MIN(delta) as min_delta,
            MAX(delta) as max_delta
        FROM options
        WHERE option_type = 'C' AND delta IS NOT NULL
    """
    ).fetchone()

    print("\nCall delta statistics:")
    print(f"  Total calls: {call_delta_stats[0]}")
    neg_pct = (call_delta_stats[1] / call_delta_stats[0] * 100) if call_delta_stats[0] > 0 else 0
    print(f"  Negative deltas: {call_delta_stats[1]} ({neg_pct:.1f}%)")
    avg_call_delta = call_delta_stats[2] if call_delta_stats[2] is not None else 0
    min_call_delta = call_delta_stats[3] if call_delta_stats[3] is not None else 0
    max_call_delta = call_delta_stats[4] if call_delta_stats[4] is not None else 0
    print(f"  Average delta: {avg_call_delta:.4f}")
    print(f"  Range: [{min_call_delta:.4f}, {max_call_delta:.4f}]")

    conn.close()

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)

    if put_delta_stats[1] > put_delta_stats[0] * 0.8:  # More than 80% have wrong sign
        print("1. 🔧 CRITICAL: Put delta sign convention is incorrect")
        print("   - Most/all put deltas are positive when they should be negative")
        print("   - This suggests a systematic error in the Greek calculation")
        print("   - Need to fix the delta calculation for puts")

    if positive_theta_issues > 0:
        print("2. ⚠️  WARNING: Some options have positive theta")
        print("   - This could indicate calculation errors or data issues")
        print("   - Theta should typically be negative (time decay)")
        print("   - Check if this is due to dividends or other factors")

    print("\n3. 💡 SUGGESTED FIXES:")
    print("   a) Review and recalculate all option Greeks")
    print("   b) Ensure proper sign conventions:")
    print("      - Call delta: 0 to 1")
    print("      - Put delta: -1 to 0")
    print("      - Theta: typically negative")
    print("   c) Validate against known option pricing formulas")
    print("   d) Consider implementing data quality checks in the pipeline")


if __name__ == "__main__":
    investigate_greek_issues()
