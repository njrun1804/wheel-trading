#!/usr/bin/env python3
"""
Comprehensive System Health Verification During Deployment
Monitors all critical system components and provides detailed status reporting.
"""

import asyncio
import json
import logging
import os
import psutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_health_verification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemHealthVerifier:
    """Comprehensive system health verification during deployment."""
    
    def __init__(self):
        self.verification_start_time = time.time()
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'verification_results': {},
            'performance_metrics': {},
            'issues_found': [],
            'recommendations': []
        }
        
    async def run_comprehensive_verification(self):
        """Execute all verification steps."""
        logger.info("🔍 Starting comprehensive system health verification...")
        
        verification_steps = [
            ("1. System Startup Monitoring", self._verify_startup_sequence),
            ("2. Hardware Acceleration Check", self._verify_hardware_acceleration),
            ("3. Core Functionality Test", self._verify_core_functionality),
            ("4. Configuration Validation", self._verify_configuration),
            ("5. API Endpoints Check", self._verify_api_endpoints),
            ("6. Resource Usage Monitor", self._verify_resource_usage),
            ("7. Error Handling Test", self._verify_error_handling),
            ("8. Performance Benchmarks", self._verify_performance_benchmarks)
        ]
        
        for step_name, step_func in verification_steps:
            logger.info(f"📋 {step_name}")
            try:
                result = await step_func()
                self.report['verification_results'][step_name] = result
                if result.get('status') == 'PASS':
                    logger.info(f"✅ {step_name}: PASSED")
                else:
                    logger.warning(f"⚠️  {step_name}: ISSUES FOUND")
                    if result.get('issues'):
                        self.report['issues_found'].extend(result['issues'])
            except Exception as e:
                logger.error(f"❌ {step_name}: ERROR - {str(e)}")
                self.report['verification_results'][step_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                self.report['issues_found'].append(f"{step_name}: {str(e)}")
        
        # Generate final report
        await self._generate_final_report()
        
    async def _verify_startup_sequence(self) -> Dict[str, Any]:
        """Monitor system startup sequence and initialization times."""
        result = {
            'status': 'PASS',
            'startup_metrics': {},
            'issues': []
        }
        
        try:
            # Check startup script exists
            startup_script = Path('startup.sh')
            if not startup_script.exists():
                result['issues'].append("startup.sh not found")
                result['status'] = 'FAIL'
                return result
            
            # Time startup sequence
            start_time = time.time()
            process = await asyncio.create_subprocess_exec(
                'python3', 'clean_startup.py',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            startup_time = time.time() - start_time
            
            result['startup_metrics'] = {
                'startup_time_seconds': startup_time,
                'exit_code': process.returncode,
                'stdout_lines': len(stdout.decode().split('\n')),
                'stderr_lines': len(stderr.decode().split('\n'))
            }
            
            if process.returncode != 0:
                result['issues'].append(f"Startup failed with exit code {process.returncode}")
                result['status'] = 'FAIL'
            elif startup_time > 30:
                result['issues'].append(f"Startup took {startup_time:.2f}s (>30s threshold)")
                result['status'] = 'WARN'
                
        except Exception as e:
            result['issues'].append(f"Startup verification failed: {str(e)}")
            result['status'] = 'ERROR'
            
        return result
    
    async def _verify_hardware_acceleration(self) -> Dict[str, Any]:
        """Verify hardware acceleration is active (M4 Pro, Metal GPU, ANE)."""
        result = {
            'status': 'PASS',
            'hardware_info': {},
            'issues': []
        }
        
        try:
            # Check CPU info
            cpu_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
            result['hardware_info']['cpu'] = cpu_info
            
            # Check memory
            memory = psutil.virtual_memory()
            result['hardware_info']['memory'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'percent_used': memory.percent
            }
            
            # Check for Metal GPU support
            try:
                import subprocess
                metal_check = subprocess.run(
                    ['python3', '-c', 'import mlx.core as mx; print(mx.metal.get_active_memory() / 1024**3)'],
                    capture_output=True, text=True, timeout=10
                )
                if metal_check.returncode == 0:
                    result['hardware_info']['metal_gpu'] = {
                        'available': True,
                        'active_memory_gb': float(metal_check.stdout.strip())
                    }
                else:
                    result['issues'].append("Metal GPU not accessible")
                    result['status'] = 'WARN'
            except Exception as e:
                result['issues'].append(f"Metal GPU check failed: {str(e)}")
                result['status'] = 'WARN'
            
            # Check for M4 Pro specific features
            if cpu_info['cpu_count'] < 12:
                result['issues'].append(f"Expected 12 cores for M4 Pro, got {cpu_info['cpu_count']}")
                result['status'] = 'WARN'
                
            if result['hardware_info']['memory']['total_gb'] < 20:
                result['issues'].append(f"Expected >20GB RAM for M4 Pro, got {result['hardware_info']['memory']['total_gb']}GB")
                result['status'] = 'WARN'
            
        except Exception as e:
            result['issues'].append(f"Hardware acceleration check failed: {str(e)}")
            result['status'] = 'ERROR'
            
        return result
    
    async def _verify_core_functionality(self) -> Dict[str, Any]:
        """Test core functionality: search, analysis, GPU acceleration."""
        result = {
            'status': 'PASS',
            'functionality_tests': {},
            'issues': []
        }
        
        try:
            # Test accelerated tools if available
            tests = [
                ('ripgrep_turbo', self._test_ripgrep_turbo),
                ('dependency_graph', self._test_dependency_graph),
                ('python_analysis', self._test_python_analysis),
                ('duckdb_turbo', self._test_duckdb_turbo)
            ]
            
            for test_name, test_func in tests:
                try:
                    test_result = await test_func()
                    result['functionality_tests'][test_name] = test_result
                    if not test_result.get('success', False):
                        result['issues'].append(f"{test_name} test failed")
                        result['status'] = 'WARN'
                except Exception as e:
                    result['functionality_tests'][test_name] = {
                        'success': False,
                        'error': str(e)
                    }
                    result['issues'].append(f"{test_name} test error: {str(e)}")
                    result['status'] = 'WARN'
            
        except Exception as e:
            result['issues'].append(f"Core functionality test failed: {str(e)}")
            result['status'] = 'ERROR'
            
        return result
    
    async def _test_ripgrep_turbo(self) -> Dict[str, Any]:
        """Test ripgrep turbo functionality."""
        try:
            # Test basic ripgrep search
            process = await asyncio.create_subprocess_exec(
                'python3', '-c', 
                'from unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo; '
                'import asyncio; '
                'async def test(): '
                '  rg = get_ripgrep_turbo(); '
                '  results = await rg.parallel_search(["def"], "src"); '
                '  print(len(results)); '
                'asyncio.run(test())',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                result_count = int(stdout.decode().strip())
                return {
                    'success': True,
                    'result_count': result_count,
                    'performance': 'good' if result_count > 0 else 'no_results'
                }
            else:
                return {
                    'success': False,
                    'error': stderr.decode()
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_dependency_graph(self) -> Dict[str, Any]:
        """Test dependency graph functionality."""
        try:
            # Test dependency graph building
            process = await asyncio.create_subprocess_exec(
                'python3', '-c',
                'from unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph; '
                'import asyncio; '
                'async def test(): '
                '  graph = get_dependency_graph(); '
                '  await graph.build_graph(); '
                '  print("success"); '
                'asyncio.run(test())',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'output': stdout.decode().strip(),
                'error': stderr.decode() if process.returncode != 0 else None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_python_analysis(self) -> Dict[str, Any]:
        """Test Python analysis functionality."""
        try:
            # Test Python analyzer
            process = await asyncio.create_subprocess_exec(
                'python3', '-c',
                'from unity_wheel.accelerated_tools.python_analysis_turbo import get_python_analyzer; '
                'import asyncio; '
                'async def test(): '
                '  analyzer = get_python_analyzer(); '
                '  analysis = await analyzer.analyze_directory("src"); '
                '  print(len(analysis) if analysis else 0); '
                'asyncio.run(test())',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                analysis_count = int(stdout.decode().strip())
                return {
                    'success': True,
                    'analysis_count': analysis_count
                }
            else:
                return {
                    'success': False,
                    'error': stderr.decode()
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_duckdb_turbo(self) -> Dict[str, Any]:
        """Test DuckDB turbo functionality."""
        try:
            # Test DuckDB connection
            process = await asyncio.create_subprocess_exec(
                'python3', '-c',
                'from unity_wheel.accelerated_tools.duckdb_turbo import get_duckdb_turbo; '
                'import asyncio; '
                'async def test(): '
                '  db = get_duckdb_turbo("data/wheel_trading_master.duckdb"); '
                '  result = await db.execute("SELECT 1 as test"); '
                '  print("success"); '
                'asyncio.run(test())',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'output': stdout.decode().strip(),
                'error': stderr.decode() if process.returncode != 0 else None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _verify_configuration(self) -> Dict[str, Any]:
        """Validate configuration loading and environment variables."""
        result = {
            'status': 'PASS',
            'config_checks': {},
            'issues': []
        }
        
        try:
            # Check main config files
            config_files = ['config.yaml', 'config_unified.yaml', 'pyproject.toml']
            for config_file in config_files:
                config_path = Path(config_file)
                result['config_checks'][config_file] = {
                    'exists': config_path.exists(),
                    'size_bytes': config_path.stat().st_size if config_path.exists() else 0
                }
                if not config_path.exists():
                    result['issues'].append(f"Missing config file: {config_file}")
                    result['status'] = 'WARN'
            
            # Check environment variables
            env_vars = ['CLAUDE_API_KEY', 'DATABENTO_API_KEY', 'TD_AMERITRADE_API_KEY']
            for env_var in env_vars:
                if env_var in os.environ:
                    result['config_checks'][f'env_{env_var}'] = 'present'
                else:
                    result['config_checks'][f'env_{env_var}'] = 'missing'
                    result['issues'].append(f"Missing environment variable: {env_var}")
                    result['status'] = 'WARN'
            
            # Test config loading
            try:
                process = await asyncio.create_subprocess_exec(
                    'python3', '-c',
                    'from src.config.loader import load_config; '
                    'config = load_config(); '
                    'print("Config loaded successfully")',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                result['config_checks']['config_loading'] = {
                    'success': process.returncode == 0,
                    'output': stdout.decode(),
                    'error': stderr.decode() if process.returncode != 0 else None
                }
                
                if process.returncode != 0:
                    result['issues'].append("Config loading failed")
                    result['status'] = 'FAIL'
                    
            except Exception as e:
                result['issues'].append(f"Config loading test failed: {str(e)}")
                result['status'] = 'ERROR'
            
        except Exception as e:
            result['issues'].append(f"Configuration validation failed: {str(e)}")
            result['status'] = 'ERROR'
            
        return result
    
    async def _verify_api_endpoints(self) -> Dict[str, Any]:
        """Check API endpoints and integration points."""
        result = {
            'status': 'PASS',
            'api_checks': {},
            'issues': []
        }
        
        try:
            # Test main API advisor
            try:
                process = await asyncio.create_subprocess_exec(
                    'python3', '-c',
                    'from src.unity_wheel.api.advisor import get_trading_recommendation; '
                    'print("API import successful")',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                result['api_checks']['advisor_import'] = {
                    'success': process.returncode == 0,
                    'output': stdout.decode(),
                    'error': stderr.decode() if process.returncode != 0 else None
                }
                
                if process.returncode != 0:
                    result['issues'].append("API advisor import failed")
                    result['status'] = 'FAIL'
                    
            except Exception as e:
                result['issues'].append(f"API advisor test failed: {str(e)}")
                result['status'] = 'ERROR'
            
            # Test run.py entry point
            try:
                process = await asyncio.create_subprocess_exec(
                    'python3', 'run.py', '--help',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                result['api_checks']['run_py'] = {
                    'success': process.returncode == 0,
                    'help_output_length': len(stdout.decode())
                }
                
                if process.returncode != 0:
                    result['issues'].append("run.py entry point failed")
                    result['status'] = 'FAIL'
                    
            except Exception as e:
                result['issues'].append(f"run.py test failed: {str(e)}")
                result['status'] = 'ERROR'
            
        except Exception as e:
            result['issues'].append(f"API endpoint verification failed: {str(e)}")
            result['status'] = 'ERROR'
            
        return result
    
    async def _verify_resource_usage(self) -> Dict[str, Any]:
        """Monitor resource usage (CPU, memory, GPU)."""
        result = {
            'status': 'PASS',
            'resource_metrics': {},
            'issues': []
        }
        
        try:
            # Get current resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            result['resource_metrics'] = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'disk_percent': round((disk.used / disk.total) * 100, 2)
            }
            
            # Check resource thresholds
            if cpu_percent > 90:
                result['issues'].append(f"High CPU usage: {cpu_percent}%")
                result['status'] = 'WARN'
            
            if memory.percent > 90:
                result['issues'].append(f"High memory usage: {memory.percent}%")
                result['status'] = 'WARN'
            
            if disk.free < 5 * (1024**3):  # Less than 5GB free
                result['issues'].append(f"Low disk space: {round(disk.free / (1024**3), 2)}GB free")
                result['status'] = 'WARN'
            
            # Check process counts
            process_count = len(psutil.pids())
            result['resource_metrics']['process_count'] = process_count
            
            if process_count > 500:
                result['issues'].append(f"High process count: {process_count}")
                result['status'] = 'WARN'
            
        except Exception as e:
            result['issues'].append(f"Resource usage monitoring failed: {str(e)}")
            result['status'] = 'ERROR'
            
        return result
    
    async def _verify_error_handling(self) -> Dict[str, Any]:
        """Verify error handling and recovery systems."""
        result = {
            'status': 'PASS',
            'error_handling_tests': {},
            'issues': []
        }
        
        try:
            # Test error recovery utilities
            try:
                process = await asyncio.create_subprocess_exec(
                    'python3', '-c',
                    'from src.unity_wheel.utils.recovery import RecoveryManager; '
                    'rm = RecoveryManager(); '
                    'print("Recovery manager initialized")',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                result['error_handling_tests']['recovery_manager'] = {
                    'success': process.returncode == 0,
                    'output': stdout.decode(),
                    'error': stderr.decode() if process.returncode != 0 else None
                }
                
                if process.returncode != 0:
                    result['issues'].append("Recovery manager initialization failed")
                    result['status'] = 'WARN'
                    
            except Exception as e:
                result['issues'].append(f"Recovery manager test failed: {str(e)}")
                result['status'] = 'WARN'
            
            # Test logging system
            try:
                log_dir = Path('logs')
                result['error_handling_tests']['logging'] = {
                    'log_directory_exists': log_dir.exists(),
                    'log_files_count': len(list(log_dir.glob('*.log'))) if log_dir.exists() else 0
                }
                
                if not log_dir.exists():
                    result['issues'].append("Log directory does not exist")
                    result['status'] = 'WARN'
                    
            except Exception as e:
                result['issues'].append(f"Logging system test failed: {str(e)}")
                result['status'] = 'WARN'
            
        except Exception as e:
            result['issues'].append(f"Error handling verification failed: {str(e)}")
            result['status'] = 'ERROR'
            
        return result
    
    async def _verify_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks meet production targets."""
        result = {
            'status': 'PASS',
            'performance_tests': {},
            'issues': []
        }
        
        try:
            # Test search performance
            search_start = time.time()
            try:
                process = await asyncio.create_subprocess_exec(
                    'find', 'src', '-name', '*.py',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                search_time = time.time() - search_start
                
                result['performance_tests']['file_search'] = {
                    'time_seconds': search_time,
                    'files_found': len(stdout.decode().split('\n')) - 1,
                    'meets_target': search_time < 1.0  # Target: <1s
                }
                
                if search_time > 1.0:
                    result['issues'].append(f"File search slow: {search_time:.2f}s (target: <1s)")
                    result['status'] = 'WARN'
                    
            except Exception as e:
                result['issues'].append(f"File search performance test failed: {str(e)}")
                result['status'] = 'WARN'
            
            # Test Python import performance
            import_start = time.time()
            try:
                process = await asyncio.create_subprocess_exec(
                    'python3', '-c',
                    'import sys; '
                    'import src.unity_wheel.api.advisor; '
                    'print("Import successful")',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                import_time = time.time() - import_start
                
                result['performance_tests']['python_imports'] = {
                    'time_seconds': import_time,
                    'success': process.returncode == 0,
                    'meets_target': import_time < 5.0  # Target: <5s
                }
                
                if import_time > 5.0:
                    result['issues'].append(f"Python imports slow: {import_time:.2f}s (target: <5s)")
                    result['status'] = 'WARN'
                    
            except Exception as e:
                result['issues'].append(f"Python import performance test failed: {str(e)}")
                result['status'] = 'WARN'
            
        except Exception as e:
            result['issues'].append(f"Performance benchmark verification failed: {str(e)}")
            result['status'] = 'ERROR'
            
        return result
    
    async def _generate_final_report(self):
        """Generate final comprehensive report."""
        total_time = time.time() - self.verification_start_time
        
        # Count results
        passed = sum(1 for r in self.report['verification_results'].values() 
                    if r.get('status') == 'PASS')
        warned = sum(1 for r in self.report['verification_results'].values() 
                    if r.get('status') == 'WARN')
        failed = sum(1 for r in self.report['verification_results'].values() 
                    if r.get('status') in ['FAIL', 'ERROR'])
        
        # Generate recommendations
        if self.report['issues_found']:
            self.report['recommendations'] = self._generate_recommendations()
        
        # Add summary
        self.report['summary'] = {
            'total_verification_time_seconds': total_time,
            'tests_passed': passed,
            'tests_warned': warned,
            'tests_failed': failed,
            'total_issues': len(self.report['issues_found']),
            'overall_status': 'CRITICAL' if failed > 0 else 'WARNING' if warned > 0 else 'HEALTHY'
        }
        
        # Save report
        report_file = f"system_health_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        # Log summary
        logger.info("=" * 80)
        logger.info("🏁 SYSTEM HEALTH VERIFICATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Overall Status: {self.report['summary']['overall_status']}")
        logger.info(f"⏱️  Total Time: {total_time:.2f}s")
        logger.info(f"✅ Tests Passed: {passed}")
        logger.info(f"⚠️  Tests Warned: {warned}")
        logger.info(f"❌ Tests Failed: {failed}")
        logger.info(f"🔍 Total Issues: {len(self.report['issues_found'])}")
        logger.info(f"📄 Full Report: {report_file}")
        
        if self.report['issues_found']:
            logger.info("\n🚨 ISSUES FOUND:")
            for issue in self.report['issues_found']:
                logger.warning(f"  • {issue}")
        
        if self.report['recommendations']:
            logger.info("\n💡 RECOMMENDATIONS:")
            for rec in self.report['recommendations']:
                logger.info(f"  • {rec}")
        
        logger.info("=" * 80)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on issues found."""
        recommendations = []
        issues = ' '.join(self.report['issues_found']).lower()
        
        if 'startup' in issues:
            recommendations.append("Review startup sequence and dependencies")
        
        if 'memory' in issues or 'cpu' in issues:
            recommendations.append("Optimize resource usage and check for memory leaks")
        
        if 'config' in issues:
            recommendations.append("Verify configuration files and environment variables")
        
        if 'api' in issues:
            recommendations.append("Check API endpoints and integration points")
        
        if 'metal' in issues or 'gpu' in issues:
            recommendations.append("Verify Metal GPU drivers and MLX installation")
        
        if 'performance' in issues:
            recommendations.append("Profile performance bottlenecks and optimize critical paths")
        
        if not recommendations:
            recommendations.append("Monitor system closely and review logs for any anomalies")
        
        return recommendations

async def main():
    """Main entry point for system health verification."""
    verifier = SystemHealthVerifier()
    await verifier.run_comprehensive_verification()

if __name__ == "__main__":
    asyncio.run(main())