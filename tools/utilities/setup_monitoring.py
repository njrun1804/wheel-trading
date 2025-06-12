#!/usr/bin/env python3
"""
Operational Monitoring Setup
Configures automated monitoring, alerts, and position tracking.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def setup_daily_monitor():
    """Create comprehensive daily monitoring script."""

    print("1. CREATING DAILY MONITOR SCRIPT")
    print("-" * 50)

    monitor_script = """#!/usr/bin/env python3
\"\"\"
Daily Health Check and Alert System
Runs at 4:10 PM ET on trading days.
\"\"\"

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import duckdb
import numpy as np
import requests
from src.unity_wheel.utils import is_trading_day


class DailyHealthCheck:
    def __init__(self):
        self.alerts = []
        self.metrics = {}
        self.db_path = Path("data/unified_wheel_trading.duckdb")

    def check_volatility(self):
        \"\"\"Check current volatility against thresholds.\"\"\"
        conn = duckdb.connect(str(self.db_path), read_only=True)

        current_vol = conn.execute(\"\"\"
            SELECT volatility_20d
            FROM backtest_features_clean
            WHERE symbol = config.trading.symbol
            ORDER BY date DESC
            LIMIT 1
        \"\"\").fetchone()[0]

        self.metrics['volatility'] = current_vol

        if current_vol > 1.20:
            self.alerts.append({
                'level': 'CRITICAL',
                'message': f'Volatility {current_vol:.1%} exceeds 120% - STOP TRADING',
                'metric': 'volatility',
                'value': current_vol
            })
        elif current_vol > 1.00:
            self.alerts.append({
                'level': 'WARNING',
                'message': f'Volatility {current_vol:.1%} exceeds 100% - reduce positions',
                'metric': 'volatility',
                'value': current_vol
            })

        conn.close()

    def check_volume_zscore(self):
        \"\"\"Check volume z-score for regime changes.\"\"\"
        conn = duckdb.connect(str(self.db_path), read_only=True)

        volume_stats = conn.execute(\"\"\"
            WITH volume_data AS (
                SELECT
                    volume,
                    AVG(volume) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as avg_vol,
                    STDDEV(volume) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as std_vol
                FROM market_data_clean
                WHERE symbol = config.trading.symbol
                AND data_type = 'stock'
                ORDER BY date DESC
                LIMIT 1
            )
            SELECT
                volume,
                (volume - avg_vol) / NULLIF(std_vol, 0) as z_score
            FROM volume_data
        \"\"\").fetchone()

        if volume_stats:
            current_vol, z_score = volume_stats
            self.metrics['volume_zscore'] = z_score or 0

            if abs(z_score or 0) > 2.0:
                self.alerts.append({
                    'level': 'WARNING',
                    'message': f'Volume spike detected: Z-score {z_score:.1f}',
                    'metric': 'volume_zscore',
                    'value': z_score
                })

        conn.close()

    def check_earnings_proximity(self):
        \"\"\"Check days until next earnings.\"\"\"
        # Unity earnings dates (approximate - should be updated quarterly)
        earnings_dates = [
            datetime(2025, 8, 7),  # Q2 2025
            datetime(2025, 11, 6), # Q3 2025
        ]

        today = datetime.now()
        days_to_earnings = 999

        for earnings_date in earnings_dates:
            if earnings_date > today:
                days_to_earnings = (earnings_date - today).days
                break

        self.metrics['days_to_earnings'] = days_to_earnings

        if days_to_earnings <= 7:
            self.alerts.append({
                'level': 'CRITICAL',
                'message': f'Earnings in {days_to_earnings} days - NO NEW POSITIONS',
                'metric': 'days_to_earnings',
                'value': days_to_earnings
            })
        elif days_to_earnings <= 14:
            self.alerts.append({
                'level': 'WARNING',
                'message': f'Earnings in {days_to_earnings} days - consider reducing',
                'metric': 'days_to_earnings',
                'value': days_to_earnings
            })

    def check_positions(self):
        \"\"\"Check open positions from tracking file.\"\"\"
        positions_file = Path("my_positions.yaml")

        if positions_file.exists():
            with open(positions_file) as f:
                positions_data = yaml.safe_load(f) or {}

            open_puts = len([p for p in positions_data.get('positions', [])
                           if p.get('status') == 'open' and p.get('type') == 'PUT'])

            self.metrics['open_positions'] = open_puts

            if open_puts > 3:
                self.alerts.append({
                    'level': 'WARNING',
                    'message': f'{open_puts} open puts exceeds limit of 3',
                    'metric': 'open_positions',
                    'value': open_puts
                })
        else:
            self.metrics['open_positions'] = 0

    def check_market_conditions(self):
        \"\"\"Check if market is open and other conditions.\"\"\"
        if not is_trading_day(datetime.now()):
            self.metrics['market_open'] = False
            return

        self.metrics['market_open'] = True

        # Additional checks can be added here
        # - Check S&P 500 level
        # - Check VIX
        # - Check Unity price vs moving averages

    def send_alerts(self):
        \"\"\"Send alerts via configured channels.\"\"\"
        if not self.alerts:
            return

        # Slack webhook
        slack_url = os.environ.get('SLACK_WEBHOOK_URL')
        if slack_url:
            self.send_slack_alert(slack_url)

        # Email (via sendmail or SMTP)
        email_to = os.environ.get('EMAIL_ALERTS_TO')
        if email_to:
            self.send_email_alert(email_to)

        # Always log to file
        self.log_alerts()

    def send_slack_alert(self, webhook_url):
        \"\"\"Send alert to Slack.\"\"\"
        color_map = {
            'CRITICAL': '#ff0000',
            'WARNING': '#ff9900',
            'INFO': '#00ff00'
        }

        attachments = []
        for alert in self.alerts:
            attachments.append({
                'color': color_map.get(alert['level'], '#808080'),
                'title': f"{alert['level']}: {alert['metric']}",
                'text': alert['message'],
                'ts': int(datetime.now().timestamp())
            })

        payload = {
            'text': f"Unity Wheel Trading - Daily Health Check",
            'attachments': attachments
        }

        try:
            response = requests.post(webhook_url, json=payload)
            if response.status_code == 200:
                print("✅ Slack alert sent")
        except Exception as e:
            print(f"❌ Slack alert failed: {e}")

    def send_email_alert(self, email_to):
        \"\"\"Send email alert.\"\"\"
        subject = f"Unity Trading Alert - {datetime.now().strftime('%Y-%m-%d')}"

        body = "Unity Wheel Trading - Daily Health Check\\n\\n"
        body += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"

        for alert in self.alerts:
            body += f"[{alert['level']}] {alert['message']}\\n"

        body += f"\\nCurrent Metrics:\\n"
        for metric, value in self.metrics.items():
            body += f"  {metric}: {value}\\n"

        # Use sendmail command (works on most Unix systems)
        try:
            cmd = f'echo "{body}" | mail -s "{subject}" {email_to}'
            subprocess.run(cmd, shell=True, check=True)
            print("✅ Email alert sent")
        except Exception as e:
            print(f"❌ Email alert failed: {e}")

    def log_alerts(self):
        \"\"\"Log alerts to file.\"\"\"
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / "health_checks.jsonl"

        with open(log_file, 'a') as f:
            for alert in self.alerts:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'level': alert['level'],
                    'metric': alert['metric'],
                    'value': alert['value'],
                    'message': alert['message']
                }
                f.write(json.dumps(log_entry) + '\\n')

    def generate_report(self):
        \"\"\"Generate daily report.\"\"\"
        print("=" * 60)
        print(f"DAILY HEALTH CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        print("\\nCurrent Metrics:")
        for metric, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")

        if self.alerts:
            print(f"\\n🚨 ALERTS ({len(self.alerts)}):")
            for alert in self.alerts:
                print(f"  [{alert['level']}] {alert['message']}")
        else:
            print("\\n✅ All systems normal")

        print("\\n" + "=" * 60)

    def run(self):
        \"\"\"Run all health checks.\"\"\"
        self.check_volatility()
        self.check_volume_zscore()
        self.check_earnings_proximity()
        self.check_positions()
        self.check_market_conditions()

        self.generate_report()
        self.send_alerts()

        # Return exit code based on alerts
        if any(a['level'] == 'CRITICAL' for a in self.alerts):
            return 2  # Critical
        elif self.alerts:
            return 1  # Warning
        return 0  # OK


if __name__ == "__main__":
    checker = DailyHealthCheck()
    sys.exit(checker.run())
"""

    # Create monitoring directory
    monitor_dir = Path("src/unity_wheel/monitoring/scripts")
    monitor_dir.mkdir(parents=True, exist_ok=True)

    monitor_file = monitor_dir / "daily_health_check.py"
    with open(monitor_file, "w") as f:
        f.write(monitor_script)

    # Make executable
    os.chmod(monitor_file, 0o755)

    print(f"   ✅ Created: {monitor_file}")
    return monitor_file


def setup_position_tracking():
    """Create position tracking file template."""

    print("\n\n2. CREATING POSITION TRACKING")
    print("-" * 50)

    positions_template = {
        "last_updated": datetime.now().isoformat(),
        "account_value": 100000,
        "cash_available": 85000,
        "positions": [
            {
                "id": "example_001",
                "type": "PUT",
                "symbol": "U",
                "strike": 23.0,
                "expiration": "2025-07-18",
                "contracts": 5,
                "entry_date": "2025-06-11",
                "entry_price": 0.85,
                "current_price": 0.65,
                "status": "open",
                "dte_at_entry": 37,
                "delta_at_entry": -0.40,
                "notes": "High vol entry",
            }
        ],
        "closed_positions": [],
        "performance": {
            "total_premium_collected": 0,
            "total_assignments": 0,
            "win_rate": 0,
            "current_month_pnl": 0,
        },
    }

    positions_file = Path("my_positions.yaml")

    if not positions_file.exists():
        with open(positions_file, "w") as f:
            yaml.dump(positions_template, f, default_flow_style=False)
        print(f"   ✅ Created: {positions_file}")
    else:
        print(f"   ℹ️  Exists: {positions_file}")

    # Create position update script
    update_script = """#!/usr/bin/env python3
\"\"\"Update position tracking with current prices.\"\"\"

import yaml
from datetime import datetime
from pathlib import Path

def update_positions():
    positions_file = Path("my_positions.yaml")

    with open(positions_file) as f:
        data = yaml.safe_load(f)

    # Update timestamp
    data['last_updated'] = datetime.now().isoformat()

    # TODO: Add logic to update current prices from market data
    # TODO: Calculate current P&L
    # TODO: Check for positions near assignment

    with open(positions_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ Positions updated at {datetime.now()}")

if __name__ == "__main__":
    update_positions()
"""

    update_file = Path("update_positions.py")
    with open(update_file, "w") as f:
        f.write(update_script)
    os.chmod(update_file, 0o755)

    print(f"   ✅ Created: {update_file}")


def setup_crontab():
    """Generate crontab entries for automated monitoring."""

    print("\n\n3. CRONTAB CONFIGURATION")
    print("-" * 50)

    cwd = Path.cwd()
    python_path = sys.executable

    crontab_entries = f"""# Unity Wheel Trading - Automated Monitoring
# Add these to your crontab with: crontab -e

# Daily health check at 4:10 PM ET on weekdays
10 16 * * 1-5 cd {cwd} && {python_path} src/unity_wheel/monitoring/scripts/daily_health_check.py >> logs/health_check.log 2>&1

# Update positions every hour during market hours (9:30 AM - 4:00 PM ET)
30 9-15 * * 1-5 cd {cwd} && {python_path} update_positions.py >> logs/position_updates.log 2>&1

# Weekly performance report on Friday at 4:30 PM ET
30 16 * * 5 cd {cwd} && {python_path} src/unity_wheel/monitoring/scripts/weekly_report.py >> logs/weekly_reports.log 2>&1

# Data quality check every morning at 8:00 AM ET
0 8 * * 1-5 cd {cwd} && {python_path} src/unity_wheel/monitoring/scripts/data_quality_monitor.py >> logs/data_quality.log 2>&1
"""

    crontab_file = Path("crontab_entries.txt")
    with open(crontab_file, "w") as f:
        f.write(crontab_entries)

    print(f"   ✅ Created: {crontab_file}")
    print("\n   To install:")
    print("   $ crontab -e")
    print(f"   # Then paste contents of {crontab_file}")

    # Check current crontab
    try:
        current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if "daily_health_check" in current.stdout:
            print("\n   ℹ️  Monitoring already in crontab")
        else:
            print("\n   ⚠️  Monitoring NOT in crontab - please install")
    except:
        print("\n   ⚠️  Could not check crontab")


def setup_alert_channels():
    """Configure alert notification channels."""

    print("\n\n4. ALERT CHANNEL SETUP")
    print("-" * 50)

    env_file = Path(".env.monitoring")

    env_template = """# Unity Wheel Trading - Monitoring Configuration

# Slack Webhook (get from https://api.slack.com/apps)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Email alerts
EMAIL_ALERTS_TO=your-email@example.com
EMAIL_ALERTS_FROM=unity-trading@localhost

# Monitoring thresholds
VOLATILITY_CRITICAL=1.20
VOLATILITY_WARNING=1.00
VOLUME_ZSCORE_THRESHOLD=2.0
MAX_CONCURRENT_PUTS=3
EARNINGS_BLACKOUT_DAYS=7

# Position limits by volatility regime
POSITION_LIMIT_LOW_VOL=0.25
POSITION_LIMIT_MED_VOL=0.20
POSITION_LIMIT_HIGH_VOL=0.10
POSITION_LIMIT_EXTREME_VOL=0.05
"""

    with open(env_file, "w") as f:
        f.write(env_template)

    print(f"   ✅ Created: {env_file}")
    print("\n   Configure alerts by editing this file with your:")
    print("   • Slack webhook URL")
    print("   • Email address")
    print("   • Custom thresholds")

    # Create alert test script
    test_script = """#!/usr/bin/env python3
\"\"\"Test alert channels.\"\"\"

import os
from dotenv import load_dotenv
load_dotenv('.env.monitoring')

from src.unity_wheel.monitoring.scripts.daily_health_check import DailyHealthCheck

# Create test alert
checker = DailyHealthCheck()
checker.alerts.append({
    'level': 'INFO',
    'message': 'This is a test alert - monitoring is working!',
    'metric': 'test',
    'value': 1.0
})

# Send alerts
checker.send_alerts()
print("\\n✅ Test alerts sent - check your Slack/email")
"""

    test_file = Path("test_alerts.py")
    with open(test_file, "w") as f:
        f.write(test_script)
    os.chmod(test_file, 0o755)

    print(f"   ✅ Created: {test_file}")
    print("\n   Test alerts with:")
    print(f"   $ python {test_file}")


def create_monitoring_dashboard():
    """Create simple monitoring dashboard script."""

    print("\n\n5. MONITORING DASHBOARD")
    print("-" * 50)

    dashboard_script = """#!/usr/bin/env python3
\"\"\"Simple monitoring dashboard - run in terminal.\"\"\"

import time
import os
from datetime import datetime
from pathlib import Path
import duckdb
import yaml

from unity_wheel.config.unified_config import get_config
config = get_config()


def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def get_current_metrics():
    metrics = {}

    # Get market data
    db_path = Path("data/unified_wheel_trading.duckdb")
    if db_path.exists():
        conn = duckdb.connect(str(db_path), read_only=True)

        # Current volatility
        vol = conn.execute(\"\"\"
            SELECT volatility_20d, stock_price
            FROM backtest_features_clean
            WHERE symbol = config.trading.symbol
            ORDER BY date DESC
            LIMIT 1
        \"\"\").fetchone()

        if vol:
            metrics['volatility'] = vol[0]
            metrics['unity_price'] = vol[1]

        conn.close()

    # Get positions
    positions_file = Path("my_positions.yaml")
    if positions_file.exists():
        with open(positions_file) as f:
            data = yaml.safe_load(f) or {}

        open_positions = [p for p in data.get('positions', [])
                         if p.get('status') == 'open']
        metrics['open_puts'] = len(open_positions)
        metrics['cash_available'] = data.get('cash_available', 0)

    return metrics

def display_dashboard():
    while True:
        clear_screen()
        metrics = get_current_metrics()

        print("=" * 60)
        print(f"UNITY WHEEL TRADING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        print(f"\\nMARKET DATA:")
        print(f"  Unity Price:  ${metrics.get('unity_price', 0):.2f}")
        print(f"  Volatility:   {metrics.get('volatility', 0):.1%}")

        # Volatility indicator
        vol = metrics.get('volatility', 0)
        if vol > 1.20:
            vol_status = "🔴 EXTREME - STOP TRADING"
        elif vol > 1.00:
            vol_status = "🟠 VERY HIGH - REDUCE SIZE"
        elif vol > 0.80:
            vol_status = "🟡 HIGH - BE CAUTIOUS"
        elif vol > 0.60:
            vol_status = "🟢 ELEVATED - NORMAL OPS"
        else:
            vol_status = "🟢 LOW - INCREASE SIZE"
        print(f"  Status:       {vol_status}")

        print(f"\\nPOSITION STATUS:")
        print(f"  Open Puts:    {metrics.get('open_puts', 0)}/3")
        print(f"  Cash Free:    ${metrics.get('cash_available', 0):,.0f}")

        print(f"\\nRECOMMENDED PARAMETERS:")
        if vol > 0.80:
            print(f"  Delta:        0.40")
            print(f"  DTE:          21-30")
            print(f"  Position:     10%")
        else:
            print(f"  Delta:        0.35")
            print(f"  DTE:          30-45")
            print(f"  Position:     15-20%")

        print("\\n" + "=" * 60)
        print("Press Ctrl+C to exit | Refreshing in 30s...")

        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\\n\\nDashboard stopped.")
            break

if __name__ == "__main__":
    display_dashboard()
"""

    dashboard_file = Path("monitor_dashboard.py")
    with open(dashboard_file, "w") as f:
        f.write(dashboard_script)
    os.chmod(dashboard_file, 0o755)

    print(f"   ✅ Created: {dashboard_file}")
    print("\n   Run dashboard with:")
    print(f"   $ python {dashboard_file}")


def main():
    """Set up complete monitoring infrastructure."""

    print("OPERATIONAL MONITORING SETUP")
    print("=" * 60)
    print("Setting up automated monitoring and alerts...\n")

    # 1. Create daily monitor script
    monitor_file = setup_daily_monitor()

    # 2. Set up position tracking
    setup_position_tracking()

    # 3. Generate crontab config
    setup_crontab()

    # 4. Configure alert channels
    setup_alert_channels()

    # 5. Create dashboard
    create_monitoring_dashboard()

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    print(f"\n   ✅ Created: {log_dir}/")

    print("\n" + "=" * 60)
    print("✅ MONITORING SETUP COMPLETE!")

    print("\n📋 NEXT STEPS:")
    print("1. Configure alerts:")
    print("   $ vi .env.monitoring")
    print("\n2. Test alerts:")
    print("   $ python test_alerts.py")
    print("\n3. Install crontab:")
    print("   $ crontab -e")
    print("   # Paste contents from crontab_entries.txt")
    print("\n4. Test health check:")
    print(f"   $ python {monitor_file}")
    print("\n5. Run dashboard:")
    print("   $ python monitor_dashboard.py")

    print("\n⚠️  IMPORTANT:")
    print("Configure Slack webhook or email alerts before going live!")


if __name__ == "__main__":
    main()
