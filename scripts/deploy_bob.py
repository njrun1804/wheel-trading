#!/usr/bin/env python3
"""
BOB (Bolt Orchestrator Bootstrap) Deployment Script
Main deployment and management script for BOB system
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from bob.config.config_loader import ConfigLoader
from bob.core.health_checker import HealthChecker
from src.unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)


class BOBDeployment:
    """Manages BOB deployment and lifecycle"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.project_root = Path(__file__).resolve().parent.parent
        self.bob_root = self.project_root / "bob"
        self.config_loader = ConfigLoader(environment)
        self.config = self.config_loader.load_config()
        self.health_checker = HealthChecker(self.config)
        
    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """Check system prerequisites"""
        issues = []
        
        # Python version
        if sys.version_info < (3, 9):
            issues.append("Python 3.9+ required")
            
        # Check for required directories
        required_dirs = [
            self.bob_root,
            self.bob_root / "config",
            self.bob_root / "logs",
            self.bob_root / "cache"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.info(f"Creating directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
                
        # Check for Einstein integration
        einstein_path = self.project_root / "einstein"
        if not einstein_path.exists():
            issues.append("Einstein not found - BOB requires Einstein for search")
            
        # Check for Bolt integration
        bolt_path = self.project_root / "bolt"
        if not bolt_path.exists():
            issues.append("Bolt not found - BOB requires Bolt for orchestration")
            
        # Check database
        db_path = Path(self.config.get("database", {}).get("path", "data/wheel_trading_master.duckdb"))
        if not db_path.exists():
            logger.warning(f"Database not found at {db_path} - will be created")
            
        return len(issues) == 0, issues
        
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        logger.info("Installing BOB dependencies...")
        
        try:
            # Install base requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(self.bob_root / "requirements.txt")
            ], check=True)
            
            # Install development dependencies if in dev mode
            if self.environment == "development":
                dev_requirements = self.bob_root / "requirements-dev.txt"
                if dev_requirements.exists():
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "-r", 
                        str(dev_requirements)
                    ], check=True)
                    
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
            
    def configure_environment(self) -> bool:
        """Configure environment-specific settings"""
        logger.info(f"Configuring {self.environment} environment...")
        
        # Set environment variables
        env_vars = self.config.get("environment_variables", {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            
        # Configure logging
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        os.environ["LOG_LEVEL"] = log_level
        
        # Configure resource limits
        resources = self.config.get("resources", {})
        if "ulimit" in resources:
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_NOFILE, 
                                 (resources["ulimit"], resources["ulimit"]))
            except Exception as e:
                logger.warning(f"Failed to set ulimit: {e}")
                
        return True
        
    def migrate_from_legacy(self) -> bool:
        """Migrate from Einstein/Bolt to BOB"""
        logger.info("Checking for legacy system migration...")
        
        # Check for existing Einstein/Bolt processes
        legacy_processes = []
        
        try:
            # Check Einstein
            result = subprocess.run(["pgrep", "-f", "einstein"], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                legacy_processes.append("Einstein")
                
            # Check Bolt
            result = subprocess.run(["pgrep", "-f", "bolt"], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                legacy_processes.append("Bolt")
                
        except Exception:
            pass  # pgrep might not be available
            
        if legacy_processes:
            logger.warning(f"Found running legacy processes: {legacy_processes}")
            logger.info("BOB will integrate with existing systems")
            
        # Copy configuration if needed
        legacy_config = self.project_root / "config.yaml"
        if legacy_config.exists() and not (self.bob_root / "config" / "migrated.yaml").exists():
            import shutil
            shutil.copy2(legacy_config, self.bob_root / "config" / "migrated.yaml")
            logger.info("Migrated legacy configuration")
            
        return True
        
    def start_services(self) -> bool:
        """Start BOB services"""
        logger.info("Starting BOB services...")
        
        services = self.config.get("services", {})
        
        for service_name, service_config in services.items():
            if not service_config.get("enabled", True):
                continue
                
            logger.info(f"Starting {service_name}...")
            
            # Build command
            cmd = service_config.get("command", [])
            if not cmd:
                logger.warning(f"No command defined for {service_name}")
                continue
                
            # Start service
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_root),
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait for startup
                time.sleep(service_config.get("startup_delay", 2))
                
                # Check if still running
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    logger.error(f"Service {service_name} failed to start")
                    logger.error(f"stdout: {stdout.decode()}")
                    logger.error(f"stderr: {stderr.decode()}")
                    return False
                    
                logger.info(f"Started {service_name} (PID: {process.pid})")
                
                # Save PID for management
                pid_file = self.bob_root / "logs" / f"{service_name}.pid"
                pid_file.write_text(str(process.pid))
                
            except Exception as e:
                logger.error(f"Failed to start {service_name}: {e}")
                return False
                
        return True
        
    def verify_deployment(self) -> bool:
        """Verify deployment health"""
        logger.info("Verifying deployment...")
        
        # Run health checks
        health_status = self.health_checker.check_all()
        
        if not health_status["healthy"]:
            logger.error("Deployment verification failed:")
            for check, result in health_status["checks"].items():
                if not result["passed"]:
                    logger.error(f"  {check}: {result.get('error', 'Failed')}")
            return False
            
        logger.info("All health checks passed")
        
        # Run integration tests if in dev mode
        if self.environment == "development":
            logger.info("Running integration tests...")
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/test_bob_integration.py", "-v"
            ], capture_output=True)
            
            if result.returncode != 0:
                logger.warning("Some integration tests failed")
                
        return True
        
    def deploy(self) -> bool:
        """Full deployment process"""
        logger.info(f"Starting BOB deployment for {self.environment} environment")
        
        # Check prerequisites
        ready, issues = self.check_prerequisites()
        if not ready:
            logger.error("Prerequisites not met:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
            
        # Install dependencies
        if not self.install_dependencies():
            return False
            
        # Configure environment
        if not self.configure_environment():
            return False
            
        # Migrate from legacy if needed
        if not self.migrate_from_legacy():
            return False
            
        # Start services
        if not self.start_services():
            return False
            
        # Verify deployment
        if not self.verify_deployment():
            return False
            
        logger.info("BOB deployment completed successfully!")
        
        # Print access information
        self.print_access_info()
        
        return True
        
    def stop_services(self) -> bool:
        """Stop all BOB services"""
        logger.info("Stopping BOB services...")
        
        pid_files = list((self.bob_root / "logs").glob("*.pid"))
        
        for pid_file in pid_files:
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 15)  # SIGTERM
                logger.info(f"Stopped service {pid_file.stem}")
                pid_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to stop {pid_file.stem}: {e}")
                
        return True
        
    def print_access_info(self):
        """Print access information"""
        info = self.config.get("access_info", {})
        
        print("\n" + "="*60)
        print("BOB Deployment Successful!")
        print("="*60)
        print(f"Environment: {self.environment}")
        print(f"API Endpoint: {info.get('api_endpoint', 'http://localhost:8000')}")
        print(f"Web UI: {info.get('web_ui', 'http://localhost:3000')}")
        print(f"Health Check: {info.get('health_endpoint', 'http://localhost:8000/health')}")
        print("\nQuick Commands:")
        print("  python scripts/deploy_bob.py status    # Check status")
        print("  python scripts/deploy_bob.py stop      # Stop services")
        print("  python scripts/deploy_bob.py logs      # View logs")
        print("="*60 + "\n")
        

def main():
    parser = argparse.ArgumentParser(description="BOB Deployment Manager")
    parser.add_argument("command", choices=["deploy", "stop", "status", "logs", "restart"],
                       help="Deployment command")
    parser.add_argument("-e", "--environment", default="development",
                       choices=["development", "staging", "production"],
                       help="Deployment environment")
    parser.add_argument("--force", action="store_true",
                       help="Force deployment even with failures")
    
    args = parser.parse_args()
    
    deployment = BOBDeployment(args.environment)
    
    if args.command == "deploy":
        success = deployment.deploy()
        sys.exit(0 if success else 1)
        
    elif args.command == "stop":
        success = deployment.stop_services()
        sys.exit(0 if success else 1)
        
    elif args.command == "status":
        health_status = deployment.health_checker.check_all()
        print(json.dumps(health_status, indent=2))
        sys.exit(0 if health_status["healthy"] else 1)
        
    elif args.command == "logs":
        log_dir = deployment.bob_root / "logs"
        subprocess.run(["tail", "-f"] + list(map(str, log_dir.glob("*.log"))))
        
    elif args.command == "restart":
        deployment.stop_services()
        time.sleep(2)
        success = deployment.deploy()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()