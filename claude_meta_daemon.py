#!/usr/bin/env python3
"""
Claude-Meta Integration Daemon
Production daemon that runs continuously to monitor Claude thought streams
and evolve the meta system based on real-time insights
"""

import asyncio
import signal
import sys
import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import daemon
import lockfile

from claude_stream_integration import ClaudeThoughtStreamIntegration
from meta_claude_integration_hooks import MetaClaudeIntegrationManager
from meta_prime import MetaPrime
from meta_daemon import MetaDaemon


@dataclass
class DaemonConfig:
    """Configuration for Claude-Meta daemon"""
    pid_file: str = "/tmp/claude_meta_daemon.pid"
    log_file: str = "/tmp/claude_meta_daemon.log"
    working_directory: str = "."
    api_key: Optional[str] = None
    thinking_budget: int = 16000
    monitoring_interval: int = 5
    evolution_threshold: int = 10  # Trigger evolution after N insights
    max_concurrent_requests: int = 3


class ClaudeMetaDaemon:
    """Production daemon for Claude-Meta integration"""
    
    def __init__(self, config: DaemonConfig):
        self.config = config
        self.running = False
        self.integration_manager: Optional[MetaClaudeIntegrationManager] = None
        self.meta_prime = MetaPrime()
        
        # Performance tracking
        self.start_time = time.time()
        self.requests_processed = 0
        self.insights_generated = 0
        self.evolutions_triggered = 0
        self.last_health_check = time.time()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGUSR1, self._status_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def _status_handler(self, signum, frame):
        """Handle status request signal (SIGUSR1)"""
        self._log_status()
        
    def _log_status(self):
        """Log current daemon status"""
        uptime = time.time() - self.start_time
        status = {
            "uptime_hours": uptime / 3600,
            "requests_processed": self.requests_processed,
            "insights_generated": self.insights_generated,
            "evolutions_triggered": self.evolutions_triggered,
            "last_health_check": self.last_health_check,
            "integration_active": self.integration_manager is not None
        }
        self.logger.info(f"Daemon Status: {json.dumps(status, indent=2)}")
        
        # Record status in meta system
        self.meta_prime.observe("daemon_status_check", status)
    
    async def start_daemon(self):
        """Start the daemon"""
        self.logger.info("Starting Claude-Meta Integration Daemon")
        self.running = True
        
        try:
            # Initialize integration manager
            self.integration_manager = MetaClaudeIntegrationManager()
            
            # Start integrated monitoring
            await self.integration_manager.start_integrated_monitoring(self.config.api_key)
            
            self.logger.info("Claude-Meta integration started successfully")
            
            # Start daemon tasks
            async with asyncio.TaskGroup() as tg:
                # Core monitoring loop
                tg.create_task(self._monitoring_loop())
                
                # Health check loop
                tg.create_task(self._health_check_loop())
                
                # Evolution management loop
                tg.create_task(self._evolution_management_loop())
                
                # Performance monitoring loop
                tg.create_task(self._performance_monitoring_loop())
                
        except Exception as e:
            self.logger.error(f"Failed to start daemon: {e}")
            raise
            
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("Starting main monitoring loop")
        
        while self.running:
            try:
                # Check for new Claude requests to process
                # In production, this would integrate with your actual Claude usage
                await self._check_for_claude_activity()
                
                # Process any pending insights
                await self._process_pending_insights()
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _check_for_claude_activity(self):
        """Check for Claude activity and process if needed"""
        # This would be extended to monitor actual Claude usage
        # For now, we simulate periodic activity checks
        
        if self.integration_manager and self.integration_manager.claude_integration:
            monitor = self.integration_manager.claude_monitor
            
            # Get current analytics
            analytics = monitor.get_monitoring_analytics()
            
            # Update our counters
            self.requests_processed = analytics.get("total_requests", 0)
            
            # Log activity if significant
            if analytics.get("total_thinking_tokens", 0) > 0:
                self.logger.debug(f"Claude activity: {analytics['total_thinking_tokens']} thinking tokens processed")
    
    async def _process_pending_insights(self):
        """Process any pending insights from Claude"""
        if not self.integration_manager:
            return
            
        # Get recent insights
        recent_insights = self.integration_manager.claude_insights[-5:]  # Last 5 insights
        
        for insight in recent_insights:
            if insight.insight_id not in getattr(self, '_processed_insights', set()):
                await self._handle_insight(insight)
                
                # Track processed insights
                if not hasattr(self, '_processed_insights'):
                    self._processed_insights = set()
                self._processed_insights.add(insight.insight_id)
    
    async def _handle_insight(self, insight):
        """Handle a specific insight"""
        self.insights_generated += 1
        
        self.logger.info(f"Processing insight: {insight.insight_type} (confidence: {insight.confidence:.2f})")
        
        # Record insight processing
        self.meta_prime.observe("daemon_insight_processed", {
            "insight_id": insight.insight_id,
            "insight_type": insight.insight_type,
            "confidence": insight.confidence,
            "daemon_uptime": time.time() - self.start_time
        })
        
        # Check if we should trigger evolution
        if self.config.evolution_threshold > 0 and self.insights_generated % self.config.evolution_threshold == 0:
            await self._trigger_evolution()
    
    async def _trigger_evolution(self):
        """Trigger meta system evolution"""
        self.evolutions_triggered += 1
        
        self.logger.info(f"Triggering meta system evolution #{self.evolutions_triggered}")
        
        try:
            # Use integration manager to trigger insight-driven evolution
            if self.integration_manager:
                # This would trigger the evolution process
                evolution_success = await self._execute_evolution()
                
                if evolution_success:
                    self.logger.info("Evolution completed successfully")
                else:
                    self.logger.warning("Evolution failed or was skipped")
                    
        except Exception as e:
            self.logger.error(f"Error triggering evolution: {e}")
    
    async def _execute_evolution(self):
        """Execute actual evolution based on insights"""
        # This is where the meta system would actually evolve
        # For now, we'll simulate the process
        
        recent_insights = self.integration_manager.claude_insights[-self.config.evolution_threshold:]
        
        if not recent_insights:
            return False
            
        # Analyze insight patterns to determine evolution type
        insight_types = [i.insight_type for i in recent_insights]
        avg_confidence = sum(i.confidence for i in recent_insights) / len(recent_insights)
        
        if avg_confidence > 0.8:
            evolution_type = max(set(insight_types), key=insight_types.count)
            
            # Record evolution
            self.meta_prime.observe("daemon_evolution_triggered", {
                "evolution_number": self.evolutions_triggered,
                "evolution_type": evolution_type,
                "insights_analyzed": len(recent_insights),
                "avg_confidence": avg_confidence,
                "trigger": "insight_threshold_reached"
            })
            
            return True
        
        return False
    
    async def _health_check_loop(self):
        """Health check loop"""
        while self.running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(60)  # Health check every minute
                
            except Exception as e:
                self.logger.error(f"Error in health check: {e}")
                await asyncio.sleep(30)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        self.last_health_check = time.time()
        
        health_status = {
            "daemon_healthy": True,
            "integration_active": self.integration_manager is not None,
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "memory_usage": self._get_memory_usage(),
            "recent_activity": self.requests_processed > 0
        }
        
        # Check integration health
        if self.integration_manager:
            integration_status = self.integration_manager.get_integration_status()
            health_status.update({
                "claude_integration": integration_status["claude_integration_active"],
                "insights_generated": integration_status["insights_generated"],
                "patterns_cached": integration_status["pattern_cache_size"]
            })
        
        # Log health status
        if health_status["daemon_healthy"]:
            self.logger.debug(f"Health check passed: {health_status}")
        else:
            self.logger.warning(f"Health check issues: {health_status}")
        
        # Record in meta system
        self.meta_prime.observe("daemon_health_check", health_status)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.running:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _collect_performance_metrics(self):
        """Collect and log performance metrics"""
        metrics = {
            "requests_processed": self.requests_processed,
            "insights_generated": self.insights_generated,
            "evolutions_triggered": self.evolutions_triggered,
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "avg_insights_per_hour": self.insights_generated / max((time.time() - self.start_time) / 3600, 0.1)
        }
        
        if self.integration_manager and self.integration_manager.claude_integration:
            claude_metrics = self.integration_manager.claude_monitor.get_monitoring_analytics()
            metrics.update({
                "claude_requests": claude_metrics.get("total_requests", 0),
                "thinking_tokens": claude_metrics.get("total_thinking_tokens", 0),
                "patterns_detected": claude_metrics.get("total_patterns_detected", 0)
            })
        
        self.logger.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")
        
        # Record in meta system
        self.meta_prime.observe("daemon_performance_metrics", metrics)
    
    def stop_daemon(self):
        """Stop the daemon gracefully"""
        self.logger.info("Stopping Claude-Meta daemon...")
        self.running = False
        
        if self.integration_manager:
            self.integration_manager.stop_monitoring()
        
        # Record shutdown
        shutdown_metrics = {
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "total_requests": self.requests_processed,
            "total_insights": self.insights_generated,
            "total_evolutions": self.evolutions_triggered
        }
        
        self.meta_prime.observe("daemon_shutdown", shutdown_metrics)
        self.logger.info(f"Daemon stopped. Final metrics: {shutdown_metrics}")


def create_daemon_config() -> DaemonConfig:
    """Create daemon configuration from environment variables"""
    return DaemonConfig(
        pid_file=os.getenv("CLAUDE_META_PID_FILE", "/tmp/claude_meta_daemon.pid"),
        log_file=os.getenv("CLAUDE_META_LOG_FILE", "/tmp/claude_meta_daemon.log"),
        working_directory=os.getenv("CLAUDE_META_WORK_DIR", "."),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        thinking_budget=int(os.getenv("CLAUDE_THINKING_BUDGET", "16000")),
        monitoring_interval=int(os.getenv("CLAUDE_META_MONITOR_INTERVAL", "5")),
        evolution_threshold=int(os.getenv("CLAUDE_META_EVOLUTION_THRESHOLD", "10"))
    )


async def run_daemon_async(config: DaemonConfig):
    """Run the daemon in async mode"""
    daemon_instance = ClaudeMetaDaemon(config)
    
    try:
        await daemon_instance.start_daemon()
    except KeyboardInterrupt:
        daemon_instance.logger.info("Received interrupt, shutting down...")
    finally:
        daemon_instance.stop_daemon()


def start_daemon(foreground: bool = False):
    """Start the Claude-Meta daemon"""
    config = create_daemon_config()
    
    if not config.api_key:
        print("❌ ANTHROPIC_API_KEY environment variable required")
        sys.exit(1)
    
    if foreground:
        # Run in foreground
        print(f"🚀 Starting Claude-Meta daemon in foreground...")
        print(f"📁 Working directory: {config.working_directory}")
        print(f"📝 Log file: {config.log_file}")
        print(f"🔧 PID file: {config.pid_file}")
        
        asyncio.run(run_daemon_async(config))
    else:
        # Run as daemon
        print(f"🚀 Starting Claude-Meta daemon...")
        print(f"📁 Working directory: {config.working_directory}")
        print(f"📝 Log file: {config.log_file}")
        print(f"🔧 PID file: {config.pid_file}")
        
        with daemon.DaemonContext(
            working_directory=config.working_directory,
            pidfile=lockfile.FileLock(config.pid_file),
            stdout=open(config.log_file, 'w+'),
            stderr=open(config.log_file, 'w+')
        ):
            asyncio.run(run_daemon_async(config))


def stop_daemon():
    """Stop the running daemon"""
    config = create_daemon_config()
    
    try:
        with open(config.pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        os.kill(pid, signal.SIGTERM)
        print(f"✅ Daemon stop signal sent to PID {pid}")
        
    except FileNotFoundError:
        print("❌ Daemon PID file not found - daemon may not be running")
    except ProcessLookupError:
        print("❌ Daemon process not found")
        # Clean up stale PID file
        try:
            os.remove(config.pid_file)
        except FileNotFoundError:
            pass


def status_daemon():
    """Get daemon status"""
    config = create_daemon_config()
    
    try:
        with open(config.pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Send status signal
        os.kill(pid, signal.SIGUSR1)
        print(f"✅ Status request sent to daemon PID {pid}")
        print(f"📝 Check log file for details: {config.log_file}")
        
    except FileNotFoundError:
        print("❌ Daemon PID file not found - daemon is not running")
    except ProcessLookupError:
        print("❌ Daemon process not found")


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude-Meta Integration Daemon")
    parser.add_argument("action", choices=["start", "stop", "status", "foreground"],
                       help="Daemon action")
    
    args = parser.parse_args()
    
    if args.action == "start":
        start_daemon(foreground=False)
    elif args.action == "foreground":
        start_daemon(foreground=True)
    elif args.action == "stop":
        stop_daemon()
    elif args.action == "status":
        status_daemon()


if __name__ == "__main__":
    main()