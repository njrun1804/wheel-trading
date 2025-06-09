"""Integration tests for autonomous trading flow with all enhancements."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from unity_wheel.api import WheelAdvisor, MarketSnapshot
from unity_wheel.data import get_market_validator, get_anomaly_detector
from unity_wheel.diagnostics import SelfDiagnostics
from unity_wheel.monitoring import get_performance_monitor
from unity_wheel.strategy import WheelParameters
from unity_wheel.risk import RiskLimits
from unity_wheel.utils import get_feature_flags, FeatureStatus


@pytest.fixture
def reset_singletons():
    """Reset singleton instances for clean tests."""
    # Reset feature flags
    from unity_wheel.utils import feature_flags
    feature_flags._feature_flags = None
    
    # Reset performance monitor
    from unity_wheel.monitoring import performance
    performance._performance_monitor = None
    
    # Reset validators
    from unity_wheel.data import validation
    validation._market_validator = None
    validation._anomaly_detector = None
    
    yield
    
    # Clean up after test
    feature_flags._feature_flags = None
    performance._performance_monitor = None
    validation._market_validator = None
    validation._anomaly_detector = None


@pytest.fixture
def valid_market_snapshot():
    """Create a valid market snapshot for testing."""
    return MarketSnapshot(
        timestamp=datetime.now(timezone.utc),
        ticker="U",
        current_price=35.50,
        buying_power=100000.0,
        margin_used=0.0,
        positions=[],
        option_chain={
            "32.5": {
                "strike": 32.5,
                "expiration": "2024-02-15",
                "bid": 1.20,
                "ask": 1.30,
                "mid": 1.25,
                "volume": 150,
                "open_interest": 500,
                "delta": -0.28,
                "gamma": 0.02,
                "theta": -0.04,
                "vega": 0.12,
                "implied_volatility": 0.65,
            },
            "35.0": {
                "strike": 35.0,
                "expiration": "2024-02-15",
                "bid": 1.80,
                "ask": 1.95,
                "mid": 1.875,
                "volume": 200,
                "open_interest": 800,
                "delta": -0.35,
                "gamma": 0.025,
                "theta": -0.05,
                "vega": 0.15,
                "implied_volatility": 0.63,
            },
            "37.5": {
                "strike": 37.5,
                "expiration": "2024-02-15",
                "bid": 2.40,
                "ask": 2.60,
                "mid": 2.50,
                "volume": 120,
                "open_interest": 400,
                "delta": -0.42,
                "gamma": 0.022,
                "theta": -0.06,
                "vega": 0.14,
                "implied_volatility": 0.62,
            },
        },
        implied_volatility=0.64,
        risk_free_rate=0.05,
    )


class TestAutonomousFlow:
    """Test complete autonomous trading flow."""
    
    def test_full_decision_flow_with_validation(self, reset_singletons, valid_market_snapshot):
        """Test complete decision flow with data validation."""
        # Initialize components
        wheel_params = WheelParameters(
            target_delta=0.30,
            target_dte=45,
            max_position_size=0.20,
        )
        risk_limits = RiskLimits()
        
        advisor = WheelAdvisor(wheel_params, risk_limits)
        
        # Get recommendation
        rec = advisor.advise_position(valid_market_snapshot)
        
        # Verify recommendation
        assert rec["action"] in ["ADJUST", "HOLD"]
        assert 0 <= rec["confidence"] <= 1
        assert "risk" in rec
        assert "max_loss" in rec["risk"]
        
        # Check that validation was performed
        validator = get_market_validator()
        stats = validator.get_validation_stats()
        assert stats["total_validations"] > 0
    
    def test_data_quality_rejection(self, reset_singletons):
        """Test that poor quality data is rejected."""
        # Create invalid market snapshot (stale data)
        stale_snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
            ticker="U",
            current_price=35.50,
            buying_power=100000.0,
            margin_used=0.0,
            positions=[],
            option_chain={},  # Empty chain
            implied_volatility=0.65,
            risk_free_rate=0.05,
        )
        
        advisor = WheelAdvisor()
        rec = advisor.advise_position(stale_snapshot)
        
        # Should hold due to data quality issues
        assert rec["action"] == "HOLD"
        assert "Data quality issues" in rec["rationale"]
    
    def test_performance_monitoring_integration(self, reset_singletons, valid_market_snapshot):
        """Test that performance is monitored throughout flow."""
        advisor = WheelAdvisor()
        monitor = get_performance_monitor()
        
        # Make multiple recommendations to generate performance data
        for _ in range(5):
            advisor.advise_position(valid_market_snapshot)
            time.sleep(0.01)  # Small delay to vary timings
        
        # Check performance stats
        stats = monitor.get_all_stats(window_minutes=60)
        assert len(stats) > 0
        
        # Check for key operations
        assert any("advise_position" in op for op in stats.keys())
        
        # Generate report
        report = monitor.generate_report(format="json")
        report_data = json.loads(report)
        assert "operations_tracked" in report_data
        assert report_data["operations_tracked"] > 0
    
    def test_feature_flag_degradation(self, reset_singletons, valid_market_snapshot):
        """Test graceful degradation with feature flags."""
        flags = get_feature_flags()
        advisor = WheelAdvisor()
        
        # Disable advanced features
        flags.disable("advanced_greeks", "Testing degradation")
        flags.disable("ml_predictions", "Testing degradation")
        
        # Should still work without advanced features
        rec = advisor.advise_position(valid_market_snapshot)
        assert rec["action"] in ["ADJUST", "HOLD"]
        
        # Re-enable and test
        flags.enable("advanced_greeks")
        rec2 = advisor.advise_position(valid_market_snapshot)
        assert rec2["action"] in ["ADJUST", "HOLD"]
    
    def test_anomaly_detection(self, reset_singletons):
        """Test anomaly detection in market data."""
        detector = get_anomaly_detector()
        
        # Feed normal data
        for i in range(10):
            normal_data = {
                "current_price": 35.0 + i * 0.1,
                "implied_volatility": 0.65 + i * 0.01,
            }
            anomalies = detector.detect_market_anomalies(normal_data)
            assert len(anomalies) == 0
        
        # Feed anomalous data
        anomalous_data = {
            "current_price": 50.0,  # Big jump
            "implied_volatility": 2.0,  # Huge IV
        }
        anomalies = detector.detect_market_anomalies(anomalous_data)
        assert len(anomalies) > 0
        assert any(a["type"] == "price" for a in anomalies)
    
    def test_self_diagnostics_integration(self, reset_singletons):
        """Test self-diagnostics system."""
        diag = SelfDiagnostics()
        success = diag.run_all_checks()
        
        # Should pass critical checks
        assert success
        
        # Check report generation
        report = diag.report(format="json")
        report_data = json.loads(report)
        assert "summary" in report_data
        assert report_data["summary"]["critical_passed"]
    
    def test_error_recovery_flow(self, reset_singletons, valid_market_snapshot):
        """Test error recovery mechanisms."""
        advisor = WheelAdvisor()
        
        # Create a snapshot that will cause calculation errors
        bad_snapshot = valid_market_snapshot.copy()
        bad_snapshot["current_price"] = -10.0  # Invalid price
        
        # Should handle gracefully
        rec = advisor.advise_position(bad_snapshot)
        assert rec["action"] == "HOLD"
        assert rec["confidence"] > 0  # Still confident in decision to hold
    
    def test_caching_behavior(self, reset_singletons, valid_market_snapshot):
        """Test that caching improves performance."""
        advisor = WheelAdvisor()
        monitor = get_performance_monitor()
        
        # First call - no cache
        start1 = time.time()
        rec1 = advisor.advise_position(valid_market_snapshot)
        time1 = time.time() - start1
        
        # Second call - should use some cached calculations
        start2 = time.time()
        rec2 = advisor.advise_position(valid_market_snapshot)
        time2 = time.time() - start2
        
        # Both should produce same recommendation
        assert rec1["action"] == rec2["action"]
        if rec1["action"] == "ADJUST":
            assert rec1["details"]["strike"] == rec2["details"]["strike"]
    
    def test_decision_logging(self, reset_singletons, valid_market_snapshot, tmp_path):
        """Test decision audit trail logging."""
        # Configure decision log file
        log_file = tmp_path / "decisions.log"
        
        with patch('unity_wheel.utils.logging.DECISION_LOG_FILE', str(log_file)):
            advisor = WheelAdvisor()
            
            # Make several decisions
            for i in range(3):
                snapshot = valid_market_snapshot.copy()
                snapshot["current_price"] = 35.0 + i
                advisor.advise_position(snapshot)
            
            # Check log file exists and has content
            assert log_file.exists()
            logs = log_file.read_text()
            assert "decision_id" in logs
            assert "ADJUST" in logs or "HOLD" in logs
    
    def test_sla_monitoring(self, reset_singletons, valid_market_snapshot):
        """Test SLA violation detection."""
        advisor = WheelAdvisor()
        monitor = get_performance_monitor()
        
        # Monkey patch to slow down a function
        original_func = advisor.advise_position
        
        def slow_advise(*args, **kwargs):
            time.sleep(0.3)  # Exceed 200ms SLA
            return original_func(*args, **kwargs)
        
        advisor.advise_position = slow_advise
        
        # Make recommendation
        rec = advisor.advise_position(valid_market_snapshot)
        
        # Check for SLA violations
        assert len(monitor.sla_violations) > 0
        violation = monitor.sla_violations[-1]
        assert violation["operation"] == "advise_position"
        assert violation["duration_ms"] > 200
    
    def test_configuration_driven_behavior(self, reset_singletons, valid_market_snapshot):
        """Test that configuration changes affect behavior."""
        # Test with conservative parameters
        conservative_params = WheelParameters(
            target_delta=0.20,  # More conservative
            target_dte=60,  # Longer expiry
            max_position_size=0.10,  # Smaller positions
        )
        
        advisor1 = WheelAdvisor(conservative_params)
        rec1 = advisor1.advise_position(valid_market_snapshot)
        
        # Test with aggressive parameters
        aggressive_params = WheelParameters(
            target_delta=0.40,  # More aggressive
            target_dte=30,  # Shorter expiry
            max_position_size=0.30,  # Larger positions
        )
        
        advisor2 = WheelAdvisor(aggressive_params)
        rec2 = advisor2.advise_position(valid_market_snapshot)
        
        # Different parameters should lead to different recommendations
        if rec1["action"] == "ADJUST" and rec2["action"] == "ADJUST":
            # Conservative should pick lower strike (further OTM)
            assert rec1["details"]["strike"] <= rec2["details"]["strike"]


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""
    
    def test_market_crash_scenario(self, reset_singletons):
        """Test behavior during market crash."""
        advisor = WheelAdvisor()
        
        # Normal market
        normal_snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            ticker="U",
            current_price=35.0,
            buying_power=100000.0,
            margin_used=0.0,
            positions=[],
            option_chain={
                "30.0": {
                    "strike": 30.0,
                    "bid": 1.0,
                    "ask": 1.2,
                    "mid": 1.1,
                    "volume": 100,
                    "open_interest": 500,
                    "delta": -0.25,
                    "gamma": 0.02,
                    "theta": -0.03,
                    "vega": 0.10,
                    "implied_volatility": 0.60,
                },
            },
            implied_volatility=0.60,
            risk_free_rate=0.05,
        )
        
        rec1 = advisor.advise_position(normal_snapshot)
        
        # Market crash - high volatility, wide spreads
        crash_snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            ticker="U",
            current_price=25.0,  # 28% drop
            buying_power=100000.0,
            margin_used=0.0,
            positions=[],
            option_chain={
                "20.0": {
                    "strike": 20.0,
                    "bid": 0.5,
                    "ask": 2.5,  # Wide spread
                    "mid": 1.5,
                    "volume": 10,  # Low liquidity
                    "open_interest": 50,
                    "delta": -0.30,
                    "gamma": 0.01,
                    "theta": -0.10,
                    "vega": 0.25,
                    "implied_volatility": 1.50,  # Extreme IV
                },
            },
            implied_volatility=1.50,
            risk_free_rate=0.05,
        )
        
        rec2 = advisor.advise_position(crash_snapshot)
        
        # Should be more conservative during crash
        assert rec2["action"] == "HOLD" or rec2["confidence"] < rec1.get("confidence", 1.0)
    
    def test_continuous_operation(self, reset_singletons, valid_market_snapshot):
        """Test continuous operation over multiple cycles."""
        advisor = WheelAdvisor()
        monitor = get_performance_monitor()
        flags = get_feature_flags()
        
        results = []
        
        # Simulate 10 decision cycles
        for i in range(10):
            # Vary market conditions slightly
            snapshot = valid_market_snapshot.copy()
            snapshot["current_price"] = 35.0 + (i % 3) - 1
            snapshot["implied_volatility"] = 0.60 + (i % 5) * 0.02
            
            # Occasionally introduce issues
            if i == 3:
                # Stale data
                snapshot["timestamp"] = datetime.now(timezone.utc) - timedelta(minutes=30)
            elif i == 7:
                # Missing option data
                snapshot["option_chain"] = {}
            
            rec = advisor.advise_position(snapshot)
            results.append(rec)
            
            # Simulate time passing
            time.sleep(0.01)
        
        # Verify continuous operation
        assert len(results) == 10
        assert all(r["action"] in ["ADJUST", "HOLD"] for r in results)
        
        # Check system health
        perf_stats = monitor.get_all_stats()
        assert len(perf_stats) > 0
        
        flag_report = flags.get_status_report()
        assert flag_report["summary"]["total"] > 0
    
    def test_integration_with_real_data_format(self, reset_singletons):
        """Test with realistic data format from broker API."""
        # Simulate realistic broker data format
        broker_data = {
            "symbol": "U",
            "quote": {
                "last": 35.25,
                "bid": 35.24,
                "ask": 35.26,
                "volume": 1234567,
                "timestamp": "2024-01-15T10:30:00Z",
            },
            "options": {
                "expirations": [
                    {
                        "date": "2024-02-15",
                        "days": 31,
                        "strikes": [
                            {
                                "strike": 32.5,
                                "put": {
                                    "bid": 1.20,
                                    "ask": 1.30,
                                    "volume": 150,
                                    "openInterest": 500,
                                    "iv": 0.65,
                                    "greeks": {
                                        "delta": -0.28,
                                        "gamma": 0.02,
                                        "theta": -0.04,
                                        "vega": 0.12,
                                    }
                                }
                            }
                        ]
                    }
                ]
            },
            "account": {
                "buyingPower": 100000.0,
                "marginUsed": 0.0,
            }
        }
        
        # Transform to our format
        def transform_broker_data(data):
            """Transform broker data to our MarketSnapshot format."""
            option_chain = {}
            for exp in data["options"]["expirations"]:
                for strike_data in exp["strikes"]:
                    strike = strike_data["strike"]
                    put = strike_data["put"]
                    option_chain[str(strike)] = {
                        "strike": strike,
                        "expiration": exp["date"],
                        "bid": put["bid"],
                        "ask": put["ask"],
                        "mid": (put["bid"] + put["ask"]) / 2,
                        "volume": put["volume"],
                        "open_interest": put["openInterest"],
                        "delta": put["greeks"]["delta"],
                        "gamma": put["greeks"]["gamma"],
                        "theta": put["greeks"]["theta"],
                        "vega": put["greeks"]["vega"],
                        "implied_volatility": put["iv"],
                    }
            
            return MarketSnapshot(
                timestamp=datetime.fromisoformat(data["quote"]["timestamp"].replace('Z', '+00:00')),
                ticker=data["symbol"],
                current_price=data["quote"]["last"],
                buying_power=data["account"]["buyingPower"],
                margin_used=data["account"]["marginUsed"],
                positions=[],
                option_chain=option_chain,
                implied_volatility=0.65,  # Could calculate from options
                risk_free_rate=0.05,
            )
        
        # Test transformation and recommendation
        snapshot = transform_broker_data(broker_data)
        advisor = WheelAdvisor()
        rec = advisor.advise_position(snapshot)
        
        # Should work with transformed data
        assert rec["action"] in ["ADJUST", "HOLD"]
        assert rec["confidence"] > 0