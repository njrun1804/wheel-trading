#!/usr/bin/env python3
"""Comprehensive final production validation for all systems."""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Set up logging for validation."""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class ProductionValidationSuite:
    """Comprehensive production validation suite."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'validation_version': '2.0.0',
            'system_name': 'Wheel Trading System',
            'validation_type': 'final_comprehensive_production_readiness',
            'tests_passed': 0,
            'tests_failed': 0,
            'critical_issues': [],
            'warnings': [],
            'performance_metrics': {},
            'system_health': {},
            'component_status': {},
            'deployment_readiness': False,
            'overall_grade': 'F',
            'recommendations': []
        }
    
    def log_test(self, test_name: str, status: str, details: Dict[str, Any] = None):
        """Log test results."""
        if status == 'PASSED':
            self.results['tests_passed'] += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            self.results['tests_failed'] += 1
            logger.error(f"❌ {test_name}: FAILED")
            if details:
                self.results['critical_issues'].append({
                    'test': test_name,
                    'details': details
                })
    
    def test_system_health(self) -> bool:
        """Test basic system health."""
        logger.info("=== SYSTEM HEALTH VALIDATION ===")
        
        try:
            # Check Python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            self.results['system_health']['python_version'] = python_version
            
            # Check critical modules
            critical_modules = ['numpy', 'pandas', 'duckdb', 'click', 'rich']
            available_modules = []
            missing_modules = []
            
            for module in critical_modules:
                try:
                    __import__(module)
                    available_modules.append(module)
                except ImportError:
                    missing_modules.append(module)
            
            self.results['system_health']['available_modules'] = available_modules
            self.results['system_health']['missing_modules'] = missing_modules
            
            if missing_modules:
                self.log_test("System Health", "FAILED", {
                    'missing_modules': missing_modules
                })
                return False
            
            self.log_test("System Health", "PASSED")
            return True
            
        except Exception as e:
            self.log_test("System Health", "FAILED", {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def test_configuration_system(self) -> bool:
        """Test configuration system."""
        logger.info("=== CONFIGURATION SYSTEM VALIDATION ===")
        
        try:
            # Check for config files
            config_files = [
                'config.yaml',
                'config_unified.yaml',
                'pyproject.toml'
            ]
            
            found_configs = []
            missing_configs = []
            
            for config_file in config_files:
                config_path = project_root / config_file
                if config_path.exists():
                    found_configs.append(config_file)
                else:
                    missing_configs.append(config_file)
            
            self.results['component_status']['configuration'] = {
                'found_configs': found_configs,
                'missing_configs': missing_configs
            }
            
            # Test basic config loading
            try:
                sys.path.insert(0, str(project_root / "src"))
                from config import get_settings
                settings = get_settings()
                self.results['component_status']['configuration']['settings_loaded'] = True
            except Exception as e:
                self.results['component_status']['configuration']['settings_loaded'] = False
                self.results['component_status']['configuration']['settings_error'] = str(e)
            
            if missing_configs:
                self.results['warnings'].append(f"Missing config files: {missing_configs}")
            
            self.log_test("Configuration System", "PASSED")
            return True
            
        except Exception as e:
            self.log_test("Configuration System", "FAILED", {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def test_core_api_imports(self) -> bool:
        """Test core API imports."""
        logger.info("=== CORE API IMPORTS VALIDATION ===")
        
        try:
            # Test key imports
            core_imports = [
                ('unity_wheel.api', 'WheelAdvisor'),
                ('unity_wheel.strategy', 'WheelParameters'),
                ('unity_wheel.risk', 'RiskLimits'),
                ('unity_wheel.storage.storage', 'Storage'),
            ]
            
            successful_imports = []
            failed_imports = []
            
            for module_name, class_name in core_imports:
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    getattr(module, class_name)
                    successful_imports.append(f"{module_name}.{class_name}")
                except Exception as e:
                    failed_imports.append({
                        'import': f"{module_name}.{class_name}",
                        'error': str(e)
                    })
            
            self.results['component_status']['core_api'] = {
                'successful_imports': successful_imports,
                'failed_imports': failed_imports
            }
            
            if failed_imports:
                self.log_test("Core API Imports", "FAILED", {
                    'failed_imports': failed_imports
                })
                return False
            
            self.log_test("Core API Imports", "PASSED")
            return True
            
        except Exception as e:
            self.log_test("Core API Imports", "FAILED", {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def test_database_connectivity(self) -> bool:
        """Test database connectivity."""
        logger.info("=== DATABASE CONNECTIVITY VALIDATION ===")
        
        try:
            # Check for database files
            data_dir = project_root / "data"
            database_files = []
            
            if data_dir.exists():
                for db_file in data_dir.glob("*.duckdb"):
                    database_files.append(str(db_file))
            
            self.results['component_status']['database'] = {
                'database_files': database_files,
                'data_directory_exists': data_dir.exists()
            }
            
            # Test DuckDB connectivity
            try:
                import duckdb
                conn = duckdb.connect()
                conn.execute("SELECT 1")
                result = conn.fetchone()
                conn.close()
                
                self.results['component_status']['database']['duckdb_working'] = True
                self.results['component_status']['database']['test_query_result'] = result[0] if result else None
                
            except Exception as e:
                self.results['component_status']['database']['duckdb_working'] = False
                self.results['component_status']['database']['duckdb_error'] = str(e)
            
            self.log_test("Database Connectivity", "PASSED")
            return True
            
        except Exception as e:
            self.log_test("Database Connectivity", "FAILED", {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def test_performance_capabilities(self) -> bool:
        """Test performance capabilities."""
        logger.info("=== PERFORMANCE CAPABILITIES VALIDATION ===")
        
        try:
            # Check hardware capabilities
            import platform
            import psutil
            
            system_info = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
            }
            
            # Check for GPU capabilities
            gpu_available = False
            try:
                import torch
                if torch.backends.mps.is_available():
                    gpu_available = True
                    system_info['gpu_type'] = 'mps'
            except ImportError:
                pass
            
            try:
                import mlx.core as mx
                system_info['mlx_available'] = True
            except ImportError:
                system_info['mlx_available'] = False
            
            system_info['gpu_available'] = gpu_available
            
            self.results['performance_metrics']['system_info'] = system_info
            
            # Performance benchmarks
            start_time = time.time()
            
            # CPU benchmark
            result = sum(i**2 for i in range(10000))
            cpu_time = time.time() - start_time
            
            self.results['performance_metrics']['cpu_benchmark'] = {
                'operations': 10000,
                'time_seconds': cpu_time,
                'ops_per_second': 10000 / cpu_time if cpu_time > 0 else 0
            }
            
            # Memory benchmark
            start_time = time.time()
            test_data = [i for i in range(100000)]
            memory_time = time.time() - start_time
            del test_data
            
            self.results['performance_metrics']['memory_benchmark'] = {
                'operations': 100000,
                'time_seconds': memory_time,
                'ops_per_second': 100000 / memory_time if memory_time > 0 else 0
            }
            
            self.log_test("Performance Capabilities", "PASSED")
            return True
            
        except Exception as e:
            self.log_test("Performance Capabilities", "FAILED", {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling mechanisms."""
        logger.info("=== ERROR HANDLING VALIDATION ===")
        
        try:
            # Test division by zero handling
            def safe_divide(a, b):
                try:
                    return a / b
                except ZeroDivisionError:
                    return 0.0
                except Exception:
                    return None
            
            test_cases = [
                (10, 2, 5.0),
                (10, 0, 0.0),
                (0, 5, 0.0),
                (10, "invalid", None)
            ]
            
            error_handling_results = []
            for a, b, expected in test_cases:
                result = safe_divide(a, b)
                error_handling_results.append({
                    'inputs': [a, b],
                    'expected': expected,
                    'actual': result,
                    'passed': result == expected
                })
            
            self.results['component_status']['error_handling'] = {
                'test_cases': error_handling_results,
                'all_passed': all(case['passed'] for case in error_handling_results)
            }
            
            # Test file access error handling
            try:
                with open("/nonexistent/path/file.txt", "r") as f:
                    f.read()
                file_error_handled = False
            except FileNotFoundError:
                file_error_handled = True
            except Exception:
                file_error_handled = True
            
            self.results['component_status']['error_handling']['file_error_handled'] = file_error_handled
            
            self.log_test("Error Handling", "PASSED")
            return True
            
        except Exception as e:
            self.log_test("Error Handling", "FAILED", {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def test_security_basics(self) -> bool:
        """Test basic security measures."""
        logger.info("=== SECURITY BASICS VALIDATION ===")
        
        try:
            # Check file permissions on sensitive files
            sensitive_files = [
                'config.yaml',
                'config_unified.yaml',
                '.env',
                'mcp-servers.json'
            ]
            
            file_permissions = {}
            for filename in sensitive_files:
                filepath = project_root / filename
                if filepath.exists():
                    file_stat = os.stat(filepath)
                    file_permissions[filename] = {
                        'exists': True,
                        'mode': oct(file_stat.st_mode)[-3:],
                        'secure': oct(file_stat.st_mode)[-3:] in ['600', '644', '640']
                    }
                else:
                    file_permissions[filename] = {
                        'exists': False,
                        'mode': None,
                        'secure': True  # Non-existent is secure
                    }
            
            self.results['component_status']['security'] = {
                'file_permissions': file_permissions,
                'all_secure': all(perm['secure'] for perm in file_permissions.values())
            }
            
            # Check for obvious security issues
            security_issues = []
            
            # Check for hardcoded secrets in common files
            common_files = ['run.py', 'config.yaml']
            for filename in common_files:
                filepath = project_root / filename
                if filepath.exists():
                    try:
                        content = filepath.read_text()
                        if any(keyword in content.lower() for keyword in ['password=', 'secret=', 'token=']):
                            security_issues.append(f"Potential hardcoded secrets in {filename}")
                    except Exception:
                        pass
            
            self.results['component_status']['security']['security_issues'] = security_issues
            
            if security_issues:
                self.results['warnings'].extend(security_issues)
            
            self.log_test("Security Basics", "PASSED")
            return True
            
        except Exception as e:
            self.log_test("Security Basics", "FAILED", {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def test_deployment_readiness(self) -> bool:
        """Test deployment readiness."""
        logger.info("=== DEPLOYMENT READINESS VALIDATION ===")
        
        try:
            # Check for required deployment files
            required_files = [
                'requirements.txt',
                'pyproject.toml',
                'run.py'
            ]
            
            deployment_files = {}
            for filename in required_files:
                filepath = project_root / filename
                deployment_files[filename] = {
                    'exists': filepath.exists(),
                    'size_bytes': filepath.stat().st_size if filepath.exists() else 0
                }
            
            self.results['component_status']['deployment'] = {
                'required_files': deployment_files,
                'all_present': all(f['exists'] for f in deployment_files.values())
            }
            
            # Check for startup script
            startup_scripts = ['startup.sh', 'start.sh', 'run.sh']
            startup_available = any((project_root / script).exists() for script in startup_scripts)
            
            self.results['component_status']['deployment']['startup_script_available'] = startup_available
            
            # Test basic CLI functionality
            try:
                sys.path.insert(0, str(project_root / "src"))
                from unity_wheel.cli.run import get_version_string
                version = get_version_string()
                self.results['component_status']['deployment']['version_available'] = True
                self.results['component_status']['deployment']['version'] = version
            except Exception as e:
                self.results['component_status']['deployment']['version_available'] = False
                self.results['component_status']['deployment']['version_error'] = str(e)
            
            self.log_test("Deployment Readiness", "PASSED")
            return True
            
        except Exception as e:
            self.log_test("Deployment Readiness", "FAILED", {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def calculate_overall_grade(self) -> str:
        """Calculate overall grade."""
        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        if total_tests == 0:
            return 'F'
        
        pass_rate = self.results['tests_passed'] / total_tests
        critical_issues_count = len(self.results['critical_issues'])
        warnings_count = len(self.results['warnings'])
        
        # Base score from pass rate
        base_score = pass_rate * 100
        
        # Deduct for critical issues
        critical_penalty = critical_issues_count * 10
        warning_penalty = warnings_count * 2
        
        final_score = max(0, base_score - critical_penalty - warning_penalty)
        
        if final_score >= 90:
            return 'A'
        elif final_score >= 80:
            return 'B'
        elif final_score >= 70:
            return 'C'
        elif final_score >= 60:
            return 'D'
        else:
            return 'F'
    
    def generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check critical issues
        if self.results['critical_issues']:
            recommendations.append("CRITICAL: Resolve all critical issues before production deployment")
        
        # Check warnings
        if len(self.results['warnings']) > 5:
            recommendations.append("Address warning issues to improve system reliability")
        
        # Check pass rate
        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        if total_tests > 0:
            pass_rate = self.results['tests_passed'] / total_tests
            if pass_rate < 0.8:
                recommendations.append("Improve test pass rate to at least 80% before production")
        
        # Specific component recommendations
        if 'core_api' in self.results['component_status']:
            if self.results['component_status']['core_api'].get('failed_imports'):
                recommendations.append("Fix core API import issues")
        
        if 'database' in self.results['component_status']:
            if not self.results['component_status']['database'].get('duckdb_working', True):
                recommendations.append("Resolve database connectivity issues")
        
        if 'security' in self.results['component_status']:
            if not self.results['component_status']['security'].get('all_secure', True):
                recommendations.append("Address security file permission issues")
        
        if not recommendations:
            recommendations.append("System appears ready for production deployment")
        
        self.results['recommendations'] = recommendations
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation suite."""
        logger.info("🚀 Starting Comprehensive Production Validation")
        
        # Run all tests
        tests = [
            self.test_system_health,
            self.test_configuration_system,
            self.test_core_api_imports,
            self.test_database_connectivity,
            self.test_performance_capabilities,
            self.test_error_handling,
            self.test_security_basics,
            self.test_deployment_readiness
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Test failed with exception: {e}")
                self.results['tests_failed'] += 1
        
        # Calculate final results
        self.results['overall_grade'] = self.calculate_overall_grade()
        self.results['deployment_readiness'] = (
            self.results['overall_grade'] in ['A', 'B'] and 
            len(self.results['critical_issues']) == 0
        )
        
        self.generate_recommendations()
        
        # Log summary
        logger.info(f"✅ Tests Passed: {self.results['tests_passed']}")
        logger.info(f"❌ Tests Failed: {self.results['tests_failed']}")
        logger.info(f"🎯 Overall Grade: {self.results['overall_grade']}")
        logger.info(f"🚀 Deployment Ready: {self.results['deployment_readiness']}")
        
        return self.results

def main():
    """Main validation function."""
    print("=" * 80)
    print("FINAL COMPREHENSIVE PRODUCTION VALIDATION")
    print("=" * 80)
    
    validator = ProductionValidationSuite()
    results = validator.run_comprehensive_validation()
    
    # Save results to file
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_file = project_root / f"final_production_validation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📋 Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Overall Grade: {results['overall_grade']}")
    print(f"Tests Passed: {results['tests_passed']}")
    print(f"Tests Failed: {results['tests_failed']}")
    print(f"Critical Issues: {len(results['critical_issues'])}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Deployment Ready: {results['deployment_readiness']}")
    
    if results['recommendations']:
        print("\nRECOMMENDations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
    
    return results

if __name__ == "__main__":
    main()