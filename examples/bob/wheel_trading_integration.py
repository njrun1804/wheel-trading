#!/usr/bin/env python3
"""
BOB integration examples for wheel trading strategy.

Demonstrates how to use BOB for:
- Position analysis and optimization
- Trade signal generation
- Risk management
- Performance analytics
- Real-time monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from bolt.trading_system_integration import TradingBoltIntegration
from unity_wheel.strategy.wheel import WheelStrategy
from unity_wheel.models.position import Position
from unity_wheel.risk.analytics import RiskAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def analyze_wheel_positions():
    """Analyze current wheel trading positions."""
    # Initialize trading-aware BOB
    trading_bob = TradingBoltIntegration(
        num_agents=8,
        trading_config={
            'database': 'data/wheel_trading_master.duckdb',
            'symbol': 'U',
            'strategy': 'wheel',
            'risk_limits': {
                'max_position': 100000,
                'max_delta': 0.30,
                'min_confidence': 0.30
            }
        }
    )
    
    try:
        await trading_bob.initialize()
        logger.info("Trading BOB initialized")
        
        # Get current positions
        positions = await trading_bob.get_current_positions()
        logger.info(f"Found {len(positions)} active positions")
        
        # Analyze positions with BOB
        analysis_query = f"""
        Analyze these wheel trading positions for Unity stock:
        - Calculate current Greeks (delta, gamma, theta, vega)
        - Identify risk concentrations
        - Evaluate profit/loss scenarios
        - Suggest position adjustments if needed
        - Consider current market conditions
        
        Positions: {positions}
        """
        
        result = await trading_bob.execute_query(
            analysis_query,
            context={
                "include_stress_tests": True,
                "scenarios": [
                    {"name": "10% down move", "underlying_change": -0.10},
                    {"name": "IV spike", "iv_change": 0.20},
                    {"name": "Time decay", "days_forward": 7}
                ]
            }
        )
        
        # Process analysis results
        logger.info("\n=== Position Analysis Results ===")
        
        for task_result in result['results']:
            if task_result['task'] == 'greek_calculation':
                greeks = task_result['result']
                logger.info(f"Portfolio Greeks:")
                logger.info(f"  Delta: {greeks['total_delta']:.3f}")
                logger.info(f"  Gamma: {greeks['total_gamma']:.3f}")
                logger.info(f"  Theta: ${greeks['total_theta']:.2f}/day")
                logger.info(f"  Vega: ${greeks['total_vega']:.2f}")
                
            elif task_result['task'] == 'risk_analysis':
                risks = task_result['result']
                logger.info(f"\nRisk Concentrations:")
                for risk in risks['concentrations']:
                    logger.info(f"  - {risk['type']}: {risk['description']}")
                    
            elif task_result['task'] == 'scenario_analysis':
                scenarios = task_result['result']
                logger.info(f"\nScenario Analysis:")
                for scenario in scenarios:
                    logger.info(f"  {scenario['name']}: ${scenario['pnl']:,.2f}")
                    
            elif task_result['task'] == 'position_recommendations':
                recommendations = task_result['result']
                logger.info(f"\nRecommendations:")
                for rec in recommendations:
                    logger.info(f"  - {rec['action']}: {rec['reason']}")
        
        return result
        
    finally:
        await trading_bob.shutdown()


async def generate_wheel_signals():
    """Generate trading signals for wheel strategy."""
    trading_bob = TradingBoltIntegration(
        num_agents=8,
        trading_config={
            'database': 'data/wheel_trading_master.duckdb',
            'symbol': 'U',
            'strategy': 'wheel'
        }
    )
    
    try:
        await trading_bob.initialize()
        
        # Signal generation query
        signal_query = """
        Generate wheel trading signals for Unity (U) stock:
        1. Identify optimal put strikes for selling (cash-secured puts)
        2. Identify optimal call strikes for covered calls
        3. Consider current volatility and market conditions
        4. Target 30-45 DTE options
        5. Minimum premium of $0.50
        6. Maximum delta of 0.35
        7. Calculate expected returns and win probability
        """
        
        # Get market snapshot for context
        market_data = await trading_bob.get_market_snapshot("U")
        
        result = await trading_bob.execute_query(
            signal_query,
            context={
                "market_data": market_data,
                "current_price": market_data['last_price'],
                "implied_volatility": market_data['iv_30d'],
                "existing_positions": await trading_bob.get_current_positions()
            }
        )
        
        # Process signals
        logger.info("\n=== Wheel Trading Signals ===")
        logger.info(f"Current Unity Price: ${market_data['last_price']:.2f}")
        logger.info(f"30-day IV: {market_data['iv_30d']:.1%}\n")
        
        signals = result.get_result('trade_signals', [])
        
        # Cash-secured puts
        puts = [s for s in signals if s['type'] == 'cash_secured_put']
        logger.info("Cash-Secured Put Opportunities:")
        for put in sorted(puts, key=lambda x: x['expected_return'], reverse=True)[:5]:
            logger.info(f"  Strike ${put['strike']}, {put['dte']}d DTE")
            logger.info(f"    Premium: ${put['premium']:.2f}")
            logger.info(f"    Delta: {put['delta']:.3f}")
            logger.info(f"    Annual Return: {put['annualized_return']:.1%}")
            logger.info(f"    Win Probability: {put['win_probability']:.1%}")
            logger.info(f"    Confidence: {put['confidence']:.1%}\n")
        
        # Covered calls
        calls = [s for s in signals if s['type'] == 'covered_call']
        logger.info("\nCovered Call Opportunities:")
        for call in sorted(calls, key=lambda x: x['expected_return'], reverse=True)[:5]:
            logger.info(f"  Strike ${call['strike']}, {call['dte']}d DTE")
            logger.info(f"    Premium: ${call['premium']:.2f}")
            logger.info(f"    Delta: {call['delta']:.3f}")
            logger.info(f"    Annual Return: {call['annualized_return']:.1%}")
            logger.info(f"    Win Probability: {call['win_probability']:.1%}")
            logger.info(f"    Confidence: {call['confidence']:.1%}\n")
        
        return signals
        
    finally:
        await trading_bob.shutdown()


async def optimize_strategy_parameters():
    """Optimize wheel strategy parameters using historical data."""
    trading_bob = TradingBoltIntegration(
        num_agents=8,
        trading_config={
            'database': 'data/wheel_trading_master.duckdb',
            'symbol': 'U',
            'strategy': 'wheel'
        }
    )
    
    try:
        await trading_bob.initialize()
        
        # Current strategy parameters
        current_params = {
            "target_delta": 0.30,
            "target_dte": 45,
            "min_premium": 0.50,
            "profit_target": 0.50,  # 50% of max profit
            "stop_loss": 2.00,      # 200% of credit received
            "assignment_threshold": 0.70
        }
        
        # Optimization query
        optimization_query = f"""
        Optimize wheel strategy parameters for Unity stock:
        - Use 2 years of historical data
        - Optimize for Sharpe ratio while maintaining win rate > 70%
        - Consider different market regimes (trending, sideways, volatile)
        - Test parameter sensitivity
        - Provide confidence intervals
        
        Current parameters: {current_params}
        """
        
        result = await trading_bob.execute_query(
            optimization_query,
            context={
                "backtest_start": "2022-01-01",
                "backtest_end": "2024-01-01",
                "optimization_method": "bayesian",
                "cross_validation_folds": 5,
                "walk_forward_windows": 4
            }
        )
        
        # Process optimization results
        logger.info("\n=== Strategy Optimization Results ===")
        
        optimized = result.get_result('optimized_parameters', {})
        logger.info("\nOptimized Parameters:")
        for param, value in optimized.items():
            old_value = current_params.get(param, 'N/A')
            logger.info(f"  {param}: {old_value} → {value}")
        
        # Performance comparison
        performance = result.get_result('performance_comparison', {})
        logger.info("\nPerformance Improvement:")
        logger.info(f"  Sharpe Ratio: {performance['old_sharpe']:.2f} → {performance['new_sharpe']:.2f}")
        logger.info(f"  Win Rate: {performance['old_win_rate']:.1%} → {performance['new_win_rate']:.1%}")
        logger.info(f"  Avg Return: {performance['old_return']:.1%} → {performance['new_return']:.1%}")
        logger.info(f"  Max Drawdown: {performance['old_drawdown']:.1%} → {performance['new_drawdown']:.1%}")
        
        # Regime analysis
        regimes = result.get_result('regime_performance', [])
        logger.info("\nPerformance by Market Regime:")
        for regime in regimes:
            logger.info(f"  {regime['name']}:")
            logger.info(f"    Sharpe: {regime['sharpe']:.2f}")
            logger.info(f"    Win Rate: {regime['win_rate']:.1%}")
        
        return optimized
        
    finally:
        await trading_bob.shutdown()


async def monitor_real_time_risk():
    """Monitor real-time risk metrics for wheel positions."""
    trading_bob = TradingBoltIntegration(
        num_agents=8,
        trading_config={
            'database': 'data/wheel_trading_master.duckdb',
            'symbol': 'U',
            'strategy': 'wheel'
        }
    )
    
    try:
        await trading_bob.initialize()
        
        # Configure monitoring
        monitor_config = {
            "update_interval": 5.0,  # 5 seconds
            "alert_thresholds": {
                "delta": 0.35,
                "loss_percent": 0.10,
                "margin_usage": 0.80,
                "correlation": 0.90
            },
            "metrics": [
                "portfolio_greeks",
                "position_pnl",
                "margin_usage",
                "stress_scenarios"
            ]
        }
        
        # Start monitoring
        logger.info("Starting real-time risk monitoring...")
        
        # Monitor for 30 seconds (in production, this would run continuously)
        end_time = datetime.now() + timedelta(seconds=30)
        
        while datetime.now() < end_time:
            # Get current risk metrics
            risk_query = """
            Calculate real-time risk metrics for all wheel positions:
            - Current Greeks and P&L
            - Margin usage and buying power
            - Correlation risk
            - Potential adjustments needed
            """
            
            result = await trading_bob.execute_query(
                risk_query,
                context={
                    "real_time": True,
                    "include_alerts": True,
                    "thresholds": monitor_config["alert_thresholds"]
                }
            )
            
            # Display metrics
            metrics = result.get_result('risk_metrics', {})
            alerts = result.get_result('alerts', [])
            
            logger.info(f"\n[{datetime.now().strftime('%H:%M:%S')}] Risk Monitor Update")
            logger.info(f"  Portfolio Delta: {metrics.get('total_delta', 0):.3f}")
            logger.info(f"  P&L: ${metrics.get('unrealized_pnl', 0):,.2f}")
            logger.info(f"  Margin Usage: {metrics.get('margin_usage_percent', 0):.1%}")
            
            # Handle alerts
            if alerts:
                logger.warning("  ALERTS:")
                for alert in alerts:
                    logger.warning(f"    - {alert['type']}: {alert['message']}")
                    
                    # Get adjustment recommendations
                    if alert['severity'] == 'high':
                        adjustment_query = f"Recommend immediate adjustment for alert: {alert}"
                        adj_result = await trading_bob.execute_query(adjustment_query)
                        logger.info(f"    Recommendation: {adj_result.get_result('recommendation')}")
            
            # Wait for next update
            await asyncio.sleep(monitor_config["update_interval"])
        
        logger.info("\nRisk monitoring completed")
        
    finally:
        await trading_bob.shutdown()


async def analyze_historical_performance():
    """Analyze historical wheel trading performance."""
    trading_bob = TradingBoltIntegration(
        num_agents=8,
        trading_config={
            'database': 'data/wheel_trading_master.duckdb',
            'symbol': 'U',
            'strategy': 'wheel'
        }
    )
    
    try:
        await trading_bob.initialize()
        
        # Get historical trades
        trades = await trading_bob.get_historical_trades(
            start_date="2023-01-01",
            end_date="2024-01-01",
            strategy="wheel"
        )
        
        logger.info(f"Analyzing {len(trades)} historical wheel trades...")
        
        # Comprehensive analysis query
        analysis_query = f"""
        Perform comprehensive analysis of {len(trades)} wheel trades:
        1. Calculate overall performance metrics (return, Sharpe, win rate)
        2. Analyze performance by market conditions
        3. Identify best and worst trades
        4. Find optimal entry/exit patterns
        5. Suggest improvements based on patterns
        
        Focus on Unity (U) stock wheel strategy
        """
        
        result = await trading_bob.execute_query(
            analysis_query,
            context={
                "trades": trades,
                "analysis_dimensions": [
                    "time_of_day",
                    "day_of_week",
                    "volatility_regime",
                    "strike_selection",
                    "dte_analysis",
                    "assignment_patterns"
                ]
            }
        )
        
        # Display analysis results
        logger.info("\n=== Historical Performance Analysis ===")
        
        # Overall metrics
        overall = result.get_result('overall_metrics', {})
        logger.info("\nOverall Performance:")
        logger.info(f"  Total Return: {overall.get('total_return', 0):.1%}")
        logger.info(f"  Sharpe Ratio: {overall.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Win Rate: {overall.get('win_rate', 0):.1%}")
        logger.info(f"  Avg Win: ${overall.get('avg_win', 0):.2f}")
        logger.info(f"  Avg Loss: ${overall.get('avg_loss', 0):.2f}")
        logger.info(f"  Max Drawdown: {overall.get('max_drawdown', 0):.1%}")
        
        # Pattern analysis
        patterns = result.get_result('patterns', [])
        logger.info("\nKey Patterns Identified:")
        for pattern in patterns[:5]:  # Top 5 patterns
            logger.info(f"  - {pattern['description']}")
            logger.info(f"    Impact: {pattern['performance_impact']}")
            logger.info(f"    Frequency: {pattern['frequency']}")
        
        # Recommendations
        recommendations = result.get_result('recommendations', [])
        logger.info("\nImprovement Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            logger.info(f"  {i}. {rec['recommendation']}")
            logger.info(f"     Expected Impact: {rec['expected_improvement']}")
        
        return result
        
    finally:
        await trading_bob.shutdown()


async def create_trading_dashboard():
    """Create a comprehensive trading dashboard using BOB."""
    trading_bob = TradingBoltIntegration(
        num_agents=8,
        trading_config={
            'database': 'data/wheel_trading_master.duckdb',
            'symbol': 'U',
            'strategy': 'wheel'
        }
    )
    
    try:
        await trading_bob.initialize()
        
        # Dashboard query
        dashboard_query = """
        Create comprehensive wheel trading dashboard data:
        1. Current positions summary with Greeks
        2. P&L analysis (realized and unrealized)
        3. Risk metrics and alerts
        4. Performance trends (30d, 90d, YTD)
        5. Upcoming expirations and actions needed
        6. Market conditions summary
        """
        
        result = await trading_bob.execute_query(dashboard_query)
        
        # Display dashboard
        logger.info("\n" + "="*60)
        logger.info("         WHEEL TRADING DASHBOARD - UNITY (U)")
        logger.info("="*60)
        
        dashboard = result.get_result('dashboard_data', {})
        
        # Positions summary
        positions = dashboard.get('positions_summary', {})
        logger.info("\nPOSITIONS SUMMARY")
        logger.info(f"  Active Positions: {positions.get('count', 0)}")
        logger.info(f"  Total Delta: {positions.get('total_delta', 0):.3f}")
        logger.info(f"  Total Theta: ${positions.get('total_theta', 0):.2f}/day")
        
        # P&L summary
        pnl = dashboard.get('pnl_summary', {})
        logger.info("\nP&L SUMMARY")
        logger.info(f"  Unrealized P&L: ${pnl.get('unrealized', 0):,.2f}")
        logger.info(f"  Realized P&L (MTD): ${pnl.get('realized_mtd', 0):,.2f}")
        logger.info(f"  Realized P&L (YTD): ${pnl.get('realized_ytd', 0):,.2f}")
        
        # Risk alerts
        alerts = dashboard.get('risk_alerts', [])
        if alerts:
            logger.info("\nRISK ALERTS")
            for alert in alerts:
                logger.info(f"  ⚠️  {alert['message']}")
        else:
            logger.info("\nRISK ALERTS: None ✓")
        
        # Performance trends
        performance = dashboard.get('performance_trends', {})
        logger.info("\nPERFORMANCE TRENDS")
        logger.info(f"  30-day Return: {performance.get('return_30d', 0):.1%}")
        logger.info(f"  90-day Return: {performance.get('return_90d', 0):.1%}")
        logger.info(f"  YTD Return: {performance.get('return_ytd', 0):.1%}")
        logger.info(f"  Current Win Rate: {performance.get('win_rate', 0):.1%}")
        
        # Upcoming actions
        actions = dashboard.get('upcoming_actions', [])
        logger.info("\nUPCOMING ACTIONS")
        for action in actions[:5]:  # Show next 5 actions
            logger.info(f"  {action['date']}: {action['description']}")
        
        logger.info("\n" + "="*60)
        
        return dashboard
        
    finally:
        await trading_bob.shutdown()


async def main():
    """Run all wheel trading examples."""
    examples = [
        ("Position Analysis", analyze_wheel_positions),
        ("Signal Generation", generate_wheel_signals),
        ("Strategy Optimization", optimize_strategy_parameters),
        ("Real-Time Risk Monitoring", monitor_real_time_risk),
        ("Historical Performance", analyze_historical_performance),
        ("Trading Dashboard", create_trading_dashboard)
    ]
    
    for name, example_func in examples:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*70}")
        
        try:
            await example_func()
            logger.info(f"\n✓ {name} completed successfully")
        except Exception as e:
            logger.error(f"\n✗ {name} failed: {e}")
        
        # Pause between examples
        await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())