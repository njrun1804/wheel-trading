#!/usr/bin/env python3
"""
Basic System Health Check During Deployment
Uses only built-in Python modules and available system tools.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def log_info(message):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def log_error(message):
    """Log error with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] ERROR: {message}")

def log_warning(message):
    """Log warning with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] WARNING: {message}")

def run_command(cmd, timeout=30):
    """Run a command with timeout."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f'Command timed out after {timeout}s'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def check_startup_sequence():
    """Check system startup sequence and timing."""
    log_info("🚀 Checking startup sequence...")
    
    results = {}
    
    # Check if startup.sh exists
    startup_path = Path('startup.sh')
    results['startup_script_exists'] = startup_path.exists()
    
    if startup_path.exists():
        log_info("✅ startup.sh found")
    else:
        log_error("❌ startup.sh not found")
        return results
    
    # Check clean_startup.py
    clean_startup_path = Path('clean_startup.py')
    results['clean_startup_exists'] = clean_startup_path.exists()
    
    if clean_startup_path.exists():
        log_info("✅ clean_startup.py found")
        
        # Time the startup
        log_info("⏱️  Testing startup sequence...")
        start_time = time.time()
        cmd_result = run_command('python3 clean_startup.py', timeout=60)
        startup_time = time.time() - start_time
        
        results['startup_test'] = {
            'success': cmd_result['success'],
            'time_seconds': startup_time,
            'returncode': cmd_result.get('returncode'),
            'output_lines': len(cmd_result.get('stdout', '').split('\n'))
        }
        
        if cmd_result['success']:
            log_info(f"✅ Startup completed in {startup_time:.2f}s")
        else:
            log_error(f"❌ Startup failed in {startup_time:.2f}s")
    else:
        log_error("❌ clean_startup.py not found")
    
    return results

def check_hardware_acceleration():
    """Check hardware acceleration capabilities."""
    log_info("🔧 Checking hardware acceleration...")
    
    results = {}
    
    # Check system info
    system_info = run_command('system_profiler SPHardwareDataType')
    if system_info['success']:
        output = system_info['stdout']
        results['system_info_available'] = True
        results['is_apple_silicon'] = 'Apple' in output and ('M1' in output or 'M2' in output or 'M3' in output or 'M4' in output)
        
        if results['is_apple_silicon']:
            log_info("✅ Apple Silicon detected")
        else:
            log_warning("⚠️  Apple Silicon not detected")
    else:
        results['system_info_available'] = False
        log_warning("⚠️  Could not get system info")
    
    # Check CPU cores
    cpu_info = run_command('sysctl -n hw.ncpu')
    if cpu_info['success']:
        cpu_count = int(cpu_info['stdout'].strip())
        results['cpu_cores'] = cpu_count
        
        if cpu_count >= 12:
            log_info(f"✅ Sufficient CPU cores: {cpu_count}")
        else:
            log_warning(f"⚠️  Limited CPU cores: {cpu_count} (expected ≥12 for M4 Pro)")
    else:
        log_warning("⚠️  Could not get CPU count")
    
    # Check memory
    memory_info = run_command('sysctl -n hw.memsize')
    if memory_info['success']:
        memory_bytes = int(memory_info['stdout'].strip())
        memory_gb = memory_bytes / (1024**3)
        results['memory_gb'] = round(memory_gb, 2)
        
        if memory_gb >= 20:
            log_info(f"✅ Sufficient memory: {memory_gb:.1f}GB")
        else:
            log_warning(f"⚠️  Limited memory: {memory_gb:.1f}GB (expected ≥20GB for M4 Pro)")
    else:
        log_warning("⚠️  Could not get memory info")
    
    # Check Metal GPU support
    metal_check = run_command('python3 -c "import mlx.core as mx; print(mx.metal.device_info())"')
    results['metal_available'] = metal_check['success']
    
    if metal_check['success']:
        log_info("✅ Metal GPU support available")
        results['metal_info'] = metal_check['stdout'].strip()
    else:
        log_warning("⚠️  Metal GPU support not available")
    
    return results

def check_core_functionality():
    """Test core system functionality."""
    log_info("⚙️  Checking core functionality...")
    
    results = {}
    
    # Test basic Python imports
    basic_imports = [
        'numpy',
        'pandas', 
        'duckdb',
        'asyncio'
    ]
    
    results['imports'] = {}
    for module in basic_imports:
        import_test = run_command(f'python3 -c "import {module}; print(\'{module} OK\')"')
        results['imports'][module] = import_test['success']
        
        if import_test['success']:
            log_info(f"✅ {module} import OK")
        else:
            log_error(f"❌ {module} import failed")
    
    # Test file system access
    test_dirs = ['src', 'data', 'config']
    results['directories'] = {}
    
    for test_dir in test_dirs:
        dir_path = Path(test_dir)
        results['directories'][test_dir] = dir_path.exists()
        
        if dir_path.exists():
            log_info(f"✅ {test_dir}/ directory exists")
        else:
            log_warning(f"⚠️  {test_dir}/ directory missing")
    
    # Test database access
    db_path = Path('data/wheel_trading_master.duckdb')
    results['database_exists'] = db_path.exists()
    
    if db_path.exists():
        log_info("✅ Main database file exists")
        
        # Test database connection
        db_test = run_command('python3 -c "import duckdb; conn = duckdb.connect(\'data/wheel_trading_master.duckdb\'); print(\'DB OK\')"')
        results['database_connection'] = db_test['success']
        
        if db_test['success']:
            log_info("✅ Database connection OK")
        else:
            log_error("❌ Database connection failed")
    else:
        log_warning("⚠️  Main database file missing")
    
    return results

def check_configuration():
    """Check configuration files and environment."""
    log_info("📋 Checking configuration...")
    
    results = {}
    
    # Check config files
    config_files = [
        'config.yaml',
        'config_unified.yaml', 
        'pyproject.toml',
        'requirements.txt'
    ]
    
    results['config_files'] = {}
    for config_file in config_files:
        config_path = Path(config_file)
        results['config_files'][config_file] = {
            'exists': config_path.exists(),
            'size': config_path.stat().st_size if config_path.exists() else 0
        }
        
        if config_path.exists():
            log_info(f"✅ {config_file} exists ({config_path.stat().st_size} bytes)")
        else:
            log_warning(f"⚠️  {config_file} missing")
    
    # Check environment variables
    env_vars = [
        'CLAUDE_API_KEY',
        'DATABENTO_API_KEY', 
        'TD_AMERITRADE_API_KEY',
        'PATH',
        'PYTHONPATH'
    ]
    
    results['environment'] = {}
    for env_var in env_vars:
        has_var = env_var in os.environ
        results['environment'][env_var] = has_var
        
        if has_var:
            if 'API_KEY' in env_var:
                log_info(f"✅ {env_var} is set")
            else:
                log_info(f"✅ {env_var} = {os.environ[env_var][:50]}{'...' if len(os.environ[env_var]) > 50 else ''}")
        else:
            log_warning(f"⚠️  {env_var} not set")
    
    return results

def check_api_integration():
    """Check API endpoints and integration."""
    log_info("🌐 Checking API integration...")
    
    results = {}
    
    # Test run.py help
    help_test = run_command('python3 run.py --help')
    results['run_py_help'] = help_test['success']
    
    if help_test['success']:
        log_info("✅ run.py help command works")
    else:
        log_error("❌ run.py help command failed")
    
    # Test core API imports
    api_import_test = run_command('python3 -c "from src.unity_wheel.api.advisor import get_trading_recommendation; print(\'API import OK\')"')
    results['api_import'] = api_import_test['success']
    
    if api_import_test['success']:
        log_info("✅ API imports work")
    else:
        log_error("❌ API imports failed")
    
    return results

def check_resource_usage():
    """Check current resource usage."""
    log_info("📊 Checking resource usage...")
    
    results = {}
    
    # Check disk space
    disk_info = run_command('df -h .')
    if disk_info['success']:
        results['disk_info'] = disk_info['stdout']
        log_info("✅ Disk space information available")
    else:
        log_warning("⚠️  Could not get disk space info")
    
    # Check running processes
    process_info = run_command('ps aux | wc -l')
    if process_info['success']:
        process_count = int(process_info['stdout'].strip()) - 1  # Subtract header
        results['process_count'] = process_count
        log_info(f"✅ Process count: {process_count}")
    else:
        log_warning("⚠️  Could not get process count")
    
    # Check load average
    load_info = run_command('uptime')
    if load_info['success']:
        results['load_info'] = load_info['stdout'].strip()
        log_info("✅ System load information available")
    else:
        log_warning("⚠️  Could not get load information")
    
    return results

def check_performance():
    """Run basic performance tests."""
    log_info("🏃 Running performance tests...")
    
    results = {}
    
    # Test file search performance
    log_info("Testing file search performance...")
    start_time = time.time()
    search_result = run_command('find src -name "*.py" | wc -l')
    search_time = time.time() - start_time
    
    if search_result['success']:
        file_count = int(search_result['stdout'].strip())
        results['file_search'] = {
            'time_seconds': search_time,
            'files_found': file_count,
            'performance': 'good' if search_time < 1.0 else 'slow'
        }
        log_info(f"✅ File search: {file_count} files in {search_time:.2f}s")
    else:
        log_warning("⚠️  File search test failed")
    
    # Test Python startup performance
    log_info("Testing Python import performance...")
    start_time = time.time()
    import_result = run_command('python3 -c "import sys; print(len(sys.modules))"')
    import_time = time.time() - start_time
    
    if import_result['success']:
        module_count = int(import_result['stdout'].strip())
        results['python_startup'] = {
            'time_seconds': import_time,
            'modules_loaded': module_count,
            'performance': 'good' if import_time < 2.0 else 'slow'
        }
        log_info(f"✅ Python startup: {module_count} modules in {import_time:.2f}s")
    else:
        log_warning("⚠️  Python startup test failed")
    
    return results

def generate_report(all_results):
    """Generate final health report."""
    log_info("📋 Generating health report...")
    
    # Calculate overall health score
    total_checks = 0
    passed_checks = 0
    issues = []
    
    for category, results in all_results.items():
        if isinstance(results, dict):
            for key, value in results.items():
                total_checks += 1
                if isinstance(value, bool) and value:
                    passed_checks += 1
                elif isinstance(value, dict) and value.get('success', False):
                    passed_checks += 1
                elif not value:
                    issues.append(f"{category}.{key}")
    
    health_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    # Determine overall status
    if health_score >= 90:
        overall_status = "HEALTHY"
    elif health_score >= 70:
        overall_status = "WARNING"  
    else:
        overall_status = "CRITICAL"
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': overall_status,
        'health_score': round(health_score, 1),
        'total_checks': total_checks,
        'passed_checks': passed_checks,
        'failed_checks': total_checks - passed_checks,
        'issues': issues,
        'detailed_results': all_results
    }
    
    # Save report
    report_file = f"health_check_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Log summary
    print("\n" + "="*80)
    print("🏁 SYSTEM HEALTH CHECK COMPLETE")
    print("="*80)
    print(f"📊 Overall Status: {overall_status}")
    print(f"💯 Health Score: {health_score:.1f}%")
    print(f"✅ Passed: {passed_checks}/{total_checks}")
    print(f"❌ Failed: {total_checks - passed_checks}")
    print(f"📄 Full Report: {report_file}")
    
    if issues:
        print(f"\n🚨 ISSUES FOUND ({len(issues)}):")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  • {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more (see full report)")
    
    print("="*80)
    
    return report

def main():
    """Main health check execution."""
    print("🔍 Starting Basic System Health Check...")
    print("="*80)
    
    start_time = time.time()
    
    # Run all health checks
    health_checks = [
        ("Startup Sequence", check_startup_sequence),
        ("Hardware Acceleration", check_hardware_acceleration), 
        ("Core Functionality", check_core_functionality),
        ("Configuration", check_configuration),
        ("API Integration", check_api_integration),
        ("Resource Usage", check_resource_usage),
        ("Performance", check_performance)
    ]
    
    all_results = {}
    
    for check_name, check_func in health_checks:
        print(f"\n📋 {check_name}")
        print("-" * 40)
        try:
            results = check_func()
            all_results[check_name.lower().replace(' ', '_')] = results
        except Exception as e:
            log_error(f"{check_name} failed: {str(e)}")
            all_results[check_name.lower().replace(' ', '_')] = {'error': str(e)}
    
    # Generate final report
    total_time = time.time() - start_time
    all_results['execution_time_seconds'] = total_time
    
    report = generate_report(all_results)
    
    return report

if __name__ == "__main__":
    main()