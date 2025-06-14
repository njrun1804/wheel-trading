#!/usr/bin/env python3
"""Fix the diagnose mode in run.py"""

import os
import re
from pathlib import Path

def fix_diagnose_mode():
    """Fix the diagnose functionality in run.py"""
    run_file = Path("src/unity_wheel/cli/run.py")
    
    if not run_file.exists():
        # Try alternate location
        run_file = Path("run.py")
    
    if not run_file.exists():
        print("❌ Cannot find run.py")
        return False
    
    print(f"Fixing diagnose mode in {run_file}...")
    
    # Read the file
    with open(run_file, 'r') as f:
        content = f.read()
    
    # Check if diagnose function exists
    if "def run_diagnostics" not in content:
        # Add diagnose function before main
        diagnose_code = '''
def run_diagnostics():
    """Run system diagnostics."""
    from rich.console import Console
    from rich.table import Table
    import duckdb
    
    console = Console()
    console.print("\\n[bold blue]Unity Wheel Trading System Diagnostics[/bold blue]\\n")
    
    results = {}
    
    # 1. Check database
    try:
        conn = duckdb.connect("data/wheel_trading_optimized.duckdb", read_only=True)
        
        # Get data counts
        stock_count = conn.execute(
            "SELECT COUNT(*) FROM market.price_data WHERE symbol='U'"
        ).fetchone()[0]
        
        options_count = conn.execute(
            "SELECT COUNT(*) FROM options.contracts WHERE symbol='U'"
        ).fetchone()[0]
        
        ml_count = conn.execute(
            "SELECT COUNT(*) FROM analytics.ml_features WHERE symbol='U'"
        ).fetchone()[0]
        
        conn.close()
        
        results["Database"] = f"✅ Connected ({stock_count} stocks, {options_count} options, {ml_count} ML records)"
    except Exception as e:
        results["Database"] = f"❌ Error: {str(e)[:50]}"
    
    # 2. Check API configuration
    try:
        from ..secrets.manager import SecretManager
        secrets = SecretManager()
        
        databento_key = secrets.get_secret("databento_api_key")
        fred_key = secrets.get_secret("ofred_api_key") or secrets.get_secret("fred_api_key")
        
        if databento_key and not databento_key.startswith("your_"):
            results["Databento API"] = "✅ Configured"
        else:
            results["Databento API"] = "❌ Not configured"
            
        if fred_key and not fred_key.startswith("your_"):
            results["FRED API"] = "✅ Configured"
        else:
            results["FRED API"] = "❌ Not configured"
    except Exception as e:
        results["APIs"] = f"❌ Error: {str(e)[:50]}"
    
    # 3. Check configuration
    try:
        from ..config.loader import get_config
        config = get_config()
        results["Configuration"] = "✅ Loaded"
    except Exception as e:
        results["Configuration"] = f"❌ Error: {str(e)[:50]}"
    
    # 4. Check math libraries
    try:
        import numpy as np
        import scipy
        from ..math.options import black_scholes_price
        
        # Test calculation
        price = black_scholes_price(100, 100, 0.05, 0.25, 30/365, "call")
        if price > 0:
            results["Math Libraries"] = "✅ Working"
        else:
            results["Math Libraries"] = "❌ Calculation error"
    except Exception as e:
        results["Math Libraries"] = f"❌ Error: {str(e)[:50]}"
    
    # 5. Display results
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    for component, status in results.items():
        table.add_row(component, status)
    
    console.print(table)
    
    # Overall status
    if all("✅" in status for status in results.values()):
        console.print("\\n[green]✅ All systems operational![/green]")
        return 0
    else:
        console.print("\\n[red]❌ Some components need attention[/red]")
        return 1
'''
        
        # Insert before main function
        main_pos = content.find("@click.command()")
        if main_pos > 0:
            content = content[:main_pos] + diagnose_code + "\n\n" + content[main_pos:]
        else:
            # Append if can't find right position
            content = content + "\n\n" + diagnose_code
    
    # Fix the main function to call diagnostics
    if "if diagnose:" in content:
        # Replace the diagnose section
        content = re.sub(
            r'if diagnose:.*?(?=\n    \w|\n@|\Z)',
            '''if diagnose:
        return run_diagnostics()
    ''',
            content,
            flags=re.DOTALL
        )
    
    # Write back
    with open(run_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed diagnose mode")
    return True

if __name__ == "__main__":
    fix_diagnose_mode()