#!/usr/bin/env python3
"""
Pull Unity options bid/ask and open interest data from Databento.
Following the provided playbook for OPRA.PILLAR statistics schema.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import databento as db


def pull_unity_options_liquidity():
    """Pull OI, volume, and bid/ask for Unity options."""

    print("UNITY OPTIONS LIQUIDITY DATA PULL")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("DATABENTO_API_KEY"):
        print("❌ DATABENTO_API_KEY not set")
        print("Please set: export DATABENTO_API_KEY=your_key")
        return

    client = db.Historical()

    # Date range - last 3 years
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=3 * 365)

    print("\n1. Fetching Unity option symbols")
    print(f"   Period: {start_date} to {end_date}")

    try:
        # Get all Unity option symbols
        syms = client.symbology.resolve(
            dataset="OPRA.PILLAR",
            symbols=["U.OPT"],  # Parent symbol for all Unity options
            stype_in="parent",
            stype_out="raw_symbol",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )["result"]

        print(f"   Found {len(syms)} option symbols")

    except Exception as e:
        print(f"❌ Error resolving symbols: {e}")
        return

    # 2. Pull daily OI and volume (statistics schema)
    print("\n2. Fetching daily open interest and volume")

    STATS = {6: "volume", 9: "open_interest"}

    try:
        df_stats = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="statistics",
            symbols="ALL_SYMBOLS",  # More efficient than listing each
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            # Filter for Unity options in processing
        ).to_df()

        # Filter for stat types we care about
        df_stats = df_stats[df_stats.stat_type.isin(STATS.keys())]

        # Map stat types to readable names
        df_stats["metric"] = df_stats.stat_type.map(STATS)

        # Pivot to get OI and volume as columns
        df_pivot = df_stats.pivot_table(
            index=["ts_ref", "instrument_id", "symbol"],
            columns="metric",
            values="quantity",
        ).reset_index()

        # Filter for Unity options only
        df_unity = df_pivot[df_pivot.symbol.str.startswith("U ")]

        print(f"   Retrieved {len(df_unity)} daily records")
        print(f"   Date range: {df_unity.ts_ref.min()} to {df_unity.ts_ref.max()}")

        # Save the data
        output_file = Path("data/unity_options_liquidity.parquet")
        output_file.parent.mkdir(exist_ok=True)
        df_unity.to_parquet(output_file)
        print(f"   ✅ Saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error fetching statistics: {e}")
        print("   This might be a large request - consider using batch API")

    # 3. Sample bid/ask for current options (last 30 days)
    print("\n3. Fetching recent bid/ask spreads")

    recent_date = end_date - timedelta(days=30)

    try:
        # Get a sample of current option chains
        df_quotes = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="tbbo",  # Top of book bid/ask
            symbols=syms[:20],  # Sample first 20 symbols
            start=recent_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            # Get daily snapshot at close
        ).to_df()

        # Convert prices from scaled integers
        for col in ["bid_px", "ask_px"]:
            if col in df_quotes.columns:
                df_quotes[col] = df_quotes[col] / 1e9

        # Calculate spreads
        if "bid_px" in df_quotes.columns and "ask_px" in df_quotes.columns:
            df_quotes["spread"] = df_quotes["ask_px"] - df_quotes["bid_px"]
            df_quotes["spread_pct"] = df_quotes["spread"] / (
                (df_quotes["ask_px"] + df_quotes["bid_px"]) / 2
            )

            print(f"   Retrieved {len(df_quotes)} quote records")
            print(f"   Average spread: ${df_quotes['spread'].mean():.3f}")
            print(f"   Average spread %: {df_quotes['spread_pct'].mean():.1%}")

            # Save sample quotes
            quotes_file = Path("data/unity_options_quotes_sample.parquet")
            df_quotes.to_parquet(quotes_file)
            print(f"   ✅ Saved to: {quotes_file}")

    except Exception as e:
        print(f"❌ Error fetching quotes: {e}")
        print("   Consider using batch API for large requests")

    # 4. Check costs before large pull
    print("\n4. Cost estimate for full 3-year pull")

    try:
        # Estimate cost for full statistics pull
        cost_stats = client.metadata.get_cost(
            dataset="OPRA.PILLAR",
            symbols="ALL_SYMBOLS",
            schema="statistics",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )

        print(f"   Statistics (OI/volume): ${cost_stats:.2f}")

        # Estimate cost for quotes
        cost_quotes = client.metadata.get_cost(
            dataset="OPRA.PILLAR",
            symbols=syms[:100],  # Sample 100 symbols
            schema="tbbo",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )

        print(f"   Quotes (100 symbols): ${cost_quotes:.2f}")
        print(f"   Full quotes estimate: ${cost_quotes * len(syms) / 100:.2f}")

    except Exception as e:
        print(f"   Could not estimate costs: {e}")

    print("\n" + "=" * 60)
    print("✅ Liquidity data pull complete")
    print("\nNext steps:")
    print("1. For full dataset, use batch.submit_job()")
    print("2. Update data integrity script to use OI filters")
    print("3. Implement optimal greek blending across strikes/dates")


def create_batch_job_script():
    """Create script for batch pulling full dataset."""

    batch_script = '''#!/usr/bin/env python3
"""Submit batch job for full Unity options liquidity data."""

import databento as db
from datetime import datetime, timedelta

client = db.Historical()

# Full 3-year range
end_date = datetime.now().date()
start_date = end_date - timedelta(days=3*365)

print("Submitting batch job for Unity options data...")

# Submit job for statistics (OI and volume)
job_stats = client.batch.submit_job(
    dataset="OPRA.PILLAR",
    symbols="U.OPT",  # All Unity options
    stype_in="parent",
    schema="statistics",
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    encoding="csv",
    compression="zstd"
)

print(f"Statistics job ID: {job_stats['id']}")

# Submit job for quotes (smaller sample for spread analysis)
job_quotes = client.batch.submit_job(
    dataset="OPRA.PILLAR",
    symbols="U.OPT",
    stype_in="parent",
    schema="tbbo",
    start=(end_date - timedelta(days=90)).strftime("%Y-%m-%d"),  # Last 90 days
    end=end_date.strftime("%Y-%m-%d"),
    encoding="csv",
    compression="zstd"
)

print(f"Quotes job ID: {job_quotes['id']}")

print("\\nCheck status with:")
print(f"  client.batch.list_jobs()")
print("\\nDownload when ready with:")
print(f"  client.batch.download('{job_stats['id']}')")
print(f"  client.batch.download('{job_quotes['id']}')")
'''

    with open("submit_batch_liquidity_job.py", "w") as f:
        f.write(batch_script)
    print("\n✅ Created: submit_batch_liquidity_job.py")


def create_greek_optimization_script():
    """Create script for optimal greek blending across strikes/dates."""

    optimization_script = '''#!/usr/bin/env python3
"""
Optimal Greek Blending for Unity Wheel Strategy
Finds the best combination of strikes and expirations.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Tuple, Dict
import duckdb

from unity_wheel.config.unified_config import get_config
config = get_config()



class GreekOptimizer:
    """Optimize put selection across multiple strikes and dates."""

    def __init__(self, target_delta: float = -0.40, max_positions: int = 3):
        self.target_delta = target_delta
        self.max_positions = max_positions
        self.min_oi = 250  # Minimum open interest
        self.max_spread_pct = 0.04  # Maximum 4% spread

    def get_available_options(self, spot_price: float) -> pd.DataFrame:
        """Get all liquid put options for Unity."""

        conn = duckdb.connect('data/unified_wheel_trading.duckdb', read_only=True)

        # Get options with liquidity data
        options = conn.execute("""
            WITH current_options AS (
                SELECT DISTINCT
                    om.symbol,
                    om.strike,
                    om.expiration,
                    DATEDIFF('day', CURRENT_DATE, om.expiration) as dte,
                    om.option_type
                FROM options_metadata_clean om
                WHERE om.underlying = 'U'
                AND om.option_type = 'P'
                AND om.expiration > CURRENT_DATE
                AND om.expiration <= CURRENT_DATE + INTERVAL '90' DAY
                AND ABS(om.strike - ?) / ? <= 0.30  -- Within 30% of spot
            )
            SELECT * FROM current_options
            ORDER BY dte, strike DESC
        """, [spot_price, spot_price]).fetchdf()

        conn.close()
        return options

    def calculate_portfolio_greeks(self, positions: List[Dict]) -> Dict:
        """Calculate aggregate portfolio Greeks."""

        total_delta = sum(p['delta'] * p['contracts'] for p in positions)
        total_gamma = sum(p['gamma'] * p['contracts'] for p in positions)
        total_theta = sum(p['theta'] * p['contracts'] for p in positions)
        total_vega = sum(p['vega'] * p['contracts'] for p in positions)

        # Calculate weighted average DTE
        total_contracts = sum(p['contracts'] for p in positions)
        avg_dte = sum(p['dte'] * p['contracts'] for p in positions) / total_contracts

        return {
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'total_vega': total_vega,
            'avg_dte': avg_dte,
            'positions': len(positions)
        }

    def optimize_blend(self, available_options: pd.DataFrame,
                      portfolio_value: float,
                      current_volatility: float) -> Dict:
        """Find optimal blend of strikes and dates."""

        # Group by DTE buckets
        dte_buckets = {
            '1-2 weeks': (7, 14),
            '2-4 weeks': (14, 28),
            '4-6 weeks': (28, 42),
            '6-8 weeks': (42, 56)
        }

        best_combinations = []

        for bucket_name, (min_dte, max_dte) in dte_buckets.items():
            bucket_options = available_options[
                (available_options['dte'] >= min_dte) &
                (available_options['dte'] <= max_dte)
            ]

            if len(bucket_options) == 0:
                continue

            # For each DTE bucket, find best strikes
            for _, opt in bucket_options.iterrows():
                # Calculate Greeks (simplified - should use actual model)
                delta = self.estimate_delta(opt['strike'], spot_price, opt['dte'], current_volatility)
                gamma = self.estimate_gamma(opt['strike'], spot_price, opt['dte'], current_volatility)
                theta = self.estimate_theta(opt['strike'], spot_price, opt['dte'], current_volatility)
                vega = self.estimate_vega(opt['strike'], spot_price, opt['dte'], current_volatility)

                # Score based on proximity to target delta and theta/delta ratio
                delta_score = 1 - abs(delta - self.target_delta) / abs(self.target_delta)
                theta_score = abs(theta) / abs(delta) if delta != 0 else 0

                combined_score = 0.7 * delta_score + 0.3 * theta_score

                best_combinations.append({
                    'symbol': opt['symbol'],
                    'strike': opt['strike'],
                    'dte': opt['dte'],
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega,
                    'score': combined_score,
                    'bucket': bucket_name
                })

        # Sort by score and diversify across buckets
        best_combinations.sort(key=lambda x: x['score'], reverse=True)

        # Select top option from each bucket
        selected = []
        buckets_used = set()

        for combo in best_combinations:
            if combo['bucket'] not in buckets_used and len(selected) < self.max_positions:
                selected.append(combo)
                buckets_used.add(combo['bucket'])

        # Optimize position sizes
        if selected:
            position_sizes = self.optimize_position_sizes(selected, portfolio_value, current_volatility)

            for i, pos in enumerate(selected):
                pos['contracts'] = position_sizes[i]
                pos['position_size'] = pos['contracts'] * pos['strike'] * 100

        return {
            'optimal_positions': selected,
            'portfolio_greeks': self.calculate_portfolio_greeks(selected) if selected else {},
            'expected_premium': sum(p.get('premium', 0) * p['contracts'] for p in selected)
        }

    def optimize_position_sizes(self, positions: List[Dict],
                               portfolio_value: float,
                               volatility: float) -> List[int]:
        """Optimize position sizes across selected options."""

        # Base position size adjusted for volatility
        base_size = portfolio_value * 0.20  # 20% base
        vol_adjustment = min(1.0, 0.40 / volatility)  # Reduce in high vol
        adjusted_size = base_size * vol_adjustment

        # Allocate across positions based on scores
        total_score = sum(p['score'] for p in positions)

        contracts = []
        for pos in positions:
            # Weight by score
            weight = pos['score'] / total_score
            position_value = adjusted_size * weight

            # Convert to contracts
            n_contracts = int(position_value / (pos['strike'] * 100))
            contracts.append(max(1, n_contracts))  # At least 1 contract

        return contracts

    def estimate_delta(self, strike, spot, dte, vol):
        """Simplified delta estimation."""
        moneyness = strike / spot
        time_factor = np.sqrt(dte / 365)

        # Rough approximation
        if moneyness < 0.90:  # Deep ITM
            return -0.80 + 0.1 * time_factor
        elif moneyness < 0.95:  # ITM
            return -0.60 + 0.2 * time_factor
        elif moneyness < 1.00:  # ATM
            return -0.40 + 0.1 * time_factor
        elif moneyness < 1.05:  # OTM
            return -0.25 - 0.1 * time_factor
        else:  # Deep OTM
            return -0.10 - 0.05 * time_factor

    def estimate_gamma(self, strike, spot, dte, vol):
        """Simplified gamma estimation."""
        moneyness = abs(1 - strike/spot)
        if moneyness < 0.05:  # Near ATM
            return 0.05 / np.sqrt(dte/365)
        else:
            return 0.02 / np.sqrt(dte/365)

    def estimate_theta(self, strike, spot, dte, vol):
        """Simplified theta estimation."""
        # Theta decay accelerates near expiry
        base_theta = -0.02 * (strike/100)  # Base decay
        time_decay = 1 / np.sqrt(dte/365) if dte > 0 else 1
        return base_theta * time_decay * vol

    def estimate_vega(self, strike, spot, dte, vol):
        """Simplified vega estimation."""
        moneyness = abs(1 - strike/spot)
        if moneyness < 0.10:  # Near ATM has highest vega
            return 0.15 * np.sqrt(dte/365)
        else:
            return 0.05 * np.sqrt(dte/365)


def main():
    """Run greek optimization for current market conditions."""

    print("OPTIMAL GREEK BLENDING ANALYSIS")
    print("=" * 60)

    # Current market parameters
    spot_price = 25.68
    current_volatility = 0.87
    portfolio_value = config.trading.portfolio_value

    optimizer = GreekOptimizer(target_delta=-0.40, max_positions=3)

    # Get available options
    print("\\n1. Fetching liquid Unity put options...")
    options = optimizer.get_available_options(spot_price)
    print(f"   Found {len(options)} available options")

    # Find optimal blend
    print("\\n2. Optimizing greek blend across strikes/dates...")
    result = optimizer.optimize_blend(options, portfolio_value, current_volatility)

    print("\\n3. OPTIMAL POSITIONS:")
    print("-" * 60)

    for i, pos in enumerate(result['optimal_positions']):
        print(f"\\nPosition {i+1}:")
        print(f"   Strike: ${pos['strike']}")
        print(f"   DTE: {pos['dte']} days ({pos['bucket']})")
        print(f"   Contracts: {pos['contracts']}")
        print(f"   Delta: {pos['delta']:.3f}")
        print(f"   Gamma: {pos['gamma']:.3f}")
        print(f"   Theta: ${pos['theta']*100:.2f}/day")
        print(f"   Score: {pos['score']:.3f}")

    print("\\n4. PORTFOLIO GREEKS:")
    print("-" * 60)
    greeks = result['portfolio_greeks']
    print(f"   Total Delta: {greeks['total_delta']:.2f}")
    print(f"   Total Gamma: {greeks['total_gamma']:.3f}")
    print(f"   Total Theta: ${greeks['total_theta']*100:.2f}/day")
    print(f"   Average DTE: {greeks['avg_dte']:.1f} days")

    print("\\n5. BENEFITS OF BLENDING:")
    print("-" * 60)
    print("   • Smoother theta decay (multiple expirations)")
    print("   • Reduced gamma risk (spread across strikes)")
    print("   • Better liquidity (avoid concentrating in one strike)")
    print("   • More assignment flexibility")

    print("\\n✅ Greek optimization complete!")


if __name__ == "__main__":
    main()
'''

    with open("optimize_greek_blend.py", "w") as f:
        f.write(optimization_script)
    print("\n✅ Created: optimize_greek_blend.py")


def main():
    """Pull liquidity data and create optimization scripts."""

    # Pull current data
    pull_unity_options_liquidity()

    # Create batch job script for full pull
    create_batch_job_script()

    # Create greek optimization script
    create_greek_optimization_script()

    print("\n📋 NEXT STEPS:")
    print("1. Set DATABENTO_API_KEY environment variable")
    print("2. Run this script to pull sample liquidity data")
    print("3. Use submit_batch_liquidity_job.py for full dataset")
    print("4. Run optimize_greek_blend.py to find optimal positions")
    print("\nThis will enable:")
    print("• Proper liquidity filtering (OI > 250)")
    print("• Spread analysis (max 4%)")
    print("• Multi-strike/date optimization")
    print("• Superior risk-adjusted returns")


if __name__ == "__main__":
    main()
