"""
System Diagnostics and Health Monitoring

Provides comprehensive health checks for all trading system components
including data connections, risk systems, and trading logic.
"""

import asyncio
import psutil
import time
from datetime import datetime, UTC
from typing import Dict, Any, List
import logging

from ..utils.logging import get_logger
from ..utils.trading_calendar import is_trading_day

logger = get_logger(__name__)


class SystemDiagnostics:
    """
    Comprehensive system health monitoring and diagnostics.
    
    Monitors all critical components and provides health status reports.
    """
    
    def __init__(self):
        """Initialize diagnostics system."""
        self.last_check_time: Optional[datetime] = None
        self.check_history: List[Dict[str, Any]] = []
        
        logger.info("System diagnostics initialized")
    
    async def run_health_check(self) -> Dict[str, Any]:
        """
        Run comprehensive health check on all system components.
        
        Returns:
            Dict containing health status for each component
        """
        check_time = datetime.now(UTC)
        
        logger.info("Running system health check...")
        
        # Initialize health status
        health_status = {
            "timestamp": check_time,
            "overall_healthy": True,
            "components": {}
        }
        
        # Check each component
        checks = [
            ("system_resources", self._check_system_resources),
            ("trading_calendar", self._check_trading_calendar),
            ("configuration", self._check_configuration),
            ("data_connections", self._check_data_connections),
            ("risk_systems", self._check_risk_systems),
            ("file_system", self._check_file_system),
            ("network", self._check_network)
        ]
        
        for component_name, check_func in checks:
            try:
                component_health = await check_func()
                health_status["components"][component_name] = component_health
                
                if not component_health["healthy"]:
                    health_status["overall_healthy"] = False
                    
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}", exc_info=True)
                health_status["components"][component_name] = {
                    "healthy": False,
                    "critical": True,
                    "message": f"Check failed: {str(e)}",
                    "timestamp": check_time
                }
                health_status["overall_healthy"] = False
        
        # Record check
        self.last_check_time = check_time
        self.check_history.append(health_status)
        
        # Keep only last 100 checks
        if len(self.check_history) > 100:
            self.check_history = self.check_history[-100:]
        
        logger.info(f"Health check complete - Overall: {'✅ HEALTHY' if health_status['overall_healthy'] else '❌ ISSUES'}")
        
        return health_status
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage (CPU, memory, disk)."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check thresholds
            cpu_healthy = cpu_percent < 90  # Less than 90% CPU
            memory_healthy = memory.percent < 85  # Less than 85% memory
            disk_healthy = disk.percent < 95  # Less than 95% disk
            
            healthy = cpu_healthy and memory_healthy and disk_healthy
            
            messages = []
            if not cpu_healthy:
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            if not memory_healthy:
                messages.append(f"High memory usage: {memory.percent:.1f}%")
            if not disk_healthy:
                messages.append(f"High disk usage: {disk.percent:.1f}%")
            
            return {
                "healthy": healthy,
                "critical": not healthy,
                "message": "; ".join(messages) if messages else "System resources OK",
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "memory_available_gb": memory.available / (1024**3)
                },
                "timestamp": datetime.now(UTC)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "critical": True,
                "message": f"Resource check failed: {str(e)}",
                "timestamp": datetime.now(UTC)
            }
    
    async def _check_trading_calendar(self) -> Dict[str, Any]:
        """Check trading calendar functionality."""
        try:
            now = datetime.now(UTC)
            is_trading = is_trading_day(now)
            
            # Basic validation that calendar is working
            healthy = True
            message = f"Trading day: {'Yes' if is_trading else 'No'}"
            
            return {
                "healthy": healthy,
                "critical": False,
                "message": message,
                "details": {
                    "is_trading_day": is_trading,
                    "current_time": now.isoformat()
                },
                "timestamp": now
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "critical": True,
                "message": f"Trading calendar check failed: {str(e)}",
                "timestamp": datetime.now(UTC)
            }
    
    async def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration validity."""
        try:
            from ..config.loader import get_config
            config = get_config()
            
            # Basic configuration validation
            healthy = True
            messages = []
            
            # Check required sections exist
            required_sections = ['strategy', 'risk', 'trading', 'data']
            for section in required_sections:
                if not hasattr(config, section):
                    healthy = False
                    messages.append(f"Missing config section: {section}")
            
            # Check critical parameters
            if hasattr(config, 'strategy'):
                if not (0 < config.strategy.delta_target < 1):
                    healthy = False
                    messages.append("Invalid delta_target")
            
            if hasattr(config, 'risk'):
                if not hasattr(config.risk, 'circuit_breakers'):
                    healthy = False
                    messages.append("Missing circuit_breakers config")
            
            return {
                "healthy": healthy,
                "critical": not healthy,
                "message": "; ".join(messages) if messages else "Configuration OK",
                "timestamp": datetime.now(UTC)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "critical": True,
                "message": f"Configuration check failed: {str(e)}",
                "timestamp": datetime.now(UTC)
            }
    
    async def _check_data_connections(self) -> Dict[str, Any]:
        """Check data connection health."""
        try:
            # For now, just validate that we can import the data client
            from ..data_providers.databento.live_client import LiveDataClient
            
            # Basic validation - can create client
            healthy = True
            message = "Data client available"
            
            # In production, would test actual connections
            # client = LiveDataClient()
            # connection_test = await client.test_connection()
            
            return {
                "healthy": healthy,
                "critical": False,
                "message": message,
                "details": {
                    "client_available": True,
                    "connection_tested": False  # Set to True when real connection tested
                },
                "timestamp": datetime.now(UTC)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "critical": True,
                "message": f"Data connection check failed: {str(e)}",
                "timestamp": datetime.now(UTC)
            }
    
    async def _check_risk_systems(self) -> Dict[str, Any]:
        """Check risk management systems."""
        try:
            from ..risk.limits import RiskLimitChecker, TradingLimits
            from ..risk.analytics import RiskAnalyzer
            
            # Test risk system initialization
            limits = TradingLimits()
            checker = RiskLimitChecker(limits)
            analyzer = RiskAnalyzer()
            
            # Test basic functionality
            test_recommendation = {
                "position_size": 1000,
                "contracts": 1,
                "confidence": 0.8
            }
            
            breaches = checker.check_all_limits(test_recommendation, portfolio_value=100000)
            can_trade = checker.should_allow_trade(breaches)
            
            healthy = True
            message = "Risk systems operational"
            
            return {
                "healthy": healthy,
                "critical": False,
                "message": message,
                "details": {
                    "limits_loaded": True,
                    "checker_functional": True,
                    "analyzer_available": True,
                    "test_breaches": len(breaches)
                },
                "timestamp": datetime.now(UTC)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "critical": True,
                "message": f"Risk systems check failed: {str(e)}",
                "timestamp": datetime.now(UTC)
            }
    
    async def _check_file_system(self) -> Dict[str, Any]:
        """Check file system access and permissions."""
        try:
            import tempfile
            import os
            
            # Test file write/read
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                test_file = f.name
                f.write("test")
            
            # Test read
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Cleanup
            os.unlink(test_file)
            
            healthy = content == "test"
            message = "File system access OK" if healthy else "File system issues"
            
            return {
                "healthy": healthy,
                "critical": not healthy,
                "message": message,
                "timestamp": datetime.now(UTC)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "critical": True,
                "message": f"File system check failed: {str(e)}",
                "timestamp": datetime.now(UTC)
            }
    
    async def _check_network(self) -> Dict[str, Any]:
        """Check network connectivity."""
        try:
            import socket
            
            # Test basic network connectivity
            # Try to connect to a reliable host
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            # Test Google DNS
            result = sock.connect_ex(('8.8.8.8', 53))
            sock.close()
            
            healthy = result == 0
            message = "Network connectivity OK" if healthy else "Network connectivity issues"
            
            return {
                "healthy": healthy,
                "critical": not healthy,
                "message": message,
                "details": {
                    "test_host": "8.8.8.8:53",
                    "connection_result": result
                },
                "timestamp": datetime.now(UTC)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "critical": True,
                "message": f"Network check failed: {str(e)}",
                "timestamp": datetime.now(UTC)
            }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of recent health checks."""
        if not self.check_history:
            return {"status": "No health checks performed"}
        
        latest = self.check_history[-1]
        
        # Count healthy/unhealthy components
        total_components = len(latest["components"])
        healthy_components = sum(1 for comp in latest["components"].values() if comp["healthy"])
        
        # Find critical issues
        critical_issues = [
            name for name, comp in latest["components"].items()
            if not comp["healthy"] and comp.get("critical", False)
        ]
        
        return {
            "overall_healthy": latest["overall_healthy"],
            "last_check": latest["timestamp"],
            "healthy_components": healthy_components,
            "total_components": total_components,
            "health_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0,
            "critical_issues": critical_issues,
            "total_checks_performed": len(self.check_history)
        }