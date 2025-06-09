#!/bin/bash
# DEPRECATED: This continuous monitoring script has been replaced by on-demand health checks
#
# The wheel trading bot now uses a pull-when-asked architecture with no continuous
# background processes. For system health checks, use:
#
#   ./scripts/health_check.sh
#
# For automated monitoring, you can schedule health_check.sh via cron if needed:
#
#   # Run health check every hour
#   0 * * * * /path/to/wheel-trading/scripts/health_check.sh
#
# For trading recommendations, run:
#
#   python run_on_demand.py --portfolio 100000

echo "=== NOTICE: Architecture Change ==="
echo ""
echo "The continuous monitoring script has been deprecated in favor of"
echo "on-demand health checks that align with the pull-when-asked architecture."
echo ""
echo "Please use:"
echo "  ./scripts/health_check.sh     - For system health checks"
echo "  python run_on_demand.py       - For trading recommendations"
echo ""
echo "No background processes or continuous monitoring needed!"
echo ""

# Run the health check instead
exec "$(dirname "$0")/health_check.sh" "$@"
