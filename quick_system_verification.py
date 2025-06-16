#!/usr/bin/env python3
"""
Quick System Verification for Deployment Health Check
Focused on critical deployment validation points.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def run_cmd(cmd, timeout=30):
    """Run command safely with timeout."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode, 
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip()
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def check_system_startup():
    """Verify system startup and initialization."""
    print("🚀 1. SYSTEM STARTUP VERIFICATION")
    print("-" * 50)
    
    results = {}
    
    # Check startup script
    startup_exists = Path('startup.sh').exists()
    clean_startup_exists = Path('clean_startup.py').exists()
    
    print(f"   startup.sh exists: {'✅' if startup_exists else '❌'}")
    print(f"   clean_startup.py exists: {'✅' if clean_startup_exists else '❌'}")
    
    if clean_startup_exists:
        print("   Testing startup sequence...")
        start_time = time.time()
        startup_result = run_cmd('python3 clean_startup.py', timeout=60)
        startup_time = time.time() - start_time
        
        results['startup'] = {
            'success': startup_result['success'],
            'time_seconds': round(startup_time, 2),
            'returncode': startup_result.get('returncode', -1)
        }
        
        if startup_result['success']:
            print(f"   ✅ Startup completed in {startup_time:.2f}s")
        else:
            print(f"   ❌ Startup failed in {startup_time:.2f}s")
            print(f"      Error: {startup_result.get('stderr', 'Unknown')}")
    
    return results

def check_hardware():
    """Verify hardware acceleration status."""
    print("\n🔧 2. HARDWARE ACCELERATION VERIFICATION")
    print("-" * 50)
    
    results = {}
    
    # Check CPU cores
    cpu_result = run_cmd('sysctl -n hw.ncpu')
    if cpu_result['success']:
        cpu_count = int(cpu_result['stdout'])
        results['cpu_cores'] = cpu_count
        print(f"   CPU cores: {cpu_count} {'✅' if cpu_count >= 12 else '⚠️'}")
    
    # Check memory
    mem_result = run_cmd('sysctl -n hw.memsize')
    if mem_result['success']:
        memory_gb = int(mem_result['stdout']) / (1024**3)
        results['memory_gb'] = round(memory_gb, 1)
        print(f"   Memory: {memory_gb:.1f}GB {'✅' if memory_gb >= 20 else '⚠️'}")
    
    # Check Apple Silicon
    system_result = run_cmd('system_profiler SPHardwareDataType | grep "Chip"')
    if system_result['success']:
        chip_info = system_result['stdout']
        is_apple_silicon = any(x in chip_info for x in ['M1', 'M2', 'M3', 'M4'])
        results['apple_silicon'] = is_apple_silicon
        print(f"   Apple Silicon: {'✅' if is_apple_silicon else '❌'}")
        print(f"      {chip_info}")
    
    # Check Metal GPU
    metal_result = run_cmd('python3 -c "import mlx.core as mx; print(f\\"Metal GPU: {mx.metal.device_info()}\\")"')
    results['metal_gpu'] = metal_result['success']
    print(f"   Metal GPU: {'✅' if metal_result['success'] else '❌'}")
    if metal_result['success']:
        print(f"      {metal_result['stdout']}")
    
    return results

def check_core_functionality():
    """Test core system functionality."""
    print("\n⚙️ 3. CORE FUNCTIONALITY VERIFICATION")
    print("-" * 50)
    
    results = {}
    
    # Test critical imports
    critical_imports = ['numpy', 'pandas', 'duckdb', 'mlx']
    results['imports'] = {}
    
    for module in critical_imports:
        import_result = run_cmd(f'python3 -c "import {module}; print(\\"{module} OK\\")"')
        results['imports'][module] = import_result['success']
        print(f"   {module}: {'✅' if import_result['success'] else '❌'}")
    
    # Test project structure
    required_dirs = ['src', 'data', 'config']
    results['directories'] = {}
    
    for directory in required_dirs:
        exists = Path(directory).exists()
        results['directories'][directory] = exists
        print(f"   {directory}/: {'✅' if exists else '❌'}")
    
    # Test database
    db_path = Path('data/wheel_trading_master.duckdb')
    db_exists = db_path.exists()
    results['database_exists'] = db_exists
    print(f"   Database file: {'✅' if db_exists else '❌'}")
    
    if db_exists:
        db_test = run_cmd('python3 -c "import duckdb; conn = duckdb.connect(\\"data/wheel_trading_master.duckdb\\"); print(\\"DB connection OK\\")"')
        results['database_connection'] = db_test['success']
        print(f"   Database connection: {'✅' if db_test['success'] else '❌'}")
    
    return results

def check_configuration():
    """Validate configuration and environment."""
    print("\n📋 4. CONFIGURATION VALIDATION")
    print("-" * 50)
    
    results = {}
    
    # Check config files
    config_files = ['config.yaml', 'config_unified.yaml', 'pyproject.toml']
    results['config_files'] = {}
    
    for config_file in config_files:
        exists = Path(config_file).exists()
        results['config_files'][config_file] = exists
        print(f"   {config_file}: {'✅' if exists else '❌'}")
    
    # Check environment variables
    env_vars = ['CLAUDE_API_KEY', 'DATABENTO_API_KEY']
    results['environment'] = {}
    
    for env_var in env_vars:
        has_var = env_var in os.environ
        results['environment'][env_var] = has_var
        print(f"   {env_var}: {'✅' if has_var else '⚠️'}")
    
    return results

def check_api_endpoints():
    """Check API endpoints and integration."""
    print("\n🌐 5. API INTEGRATION VERIFICATION")  
    print("-" * 50)
    
    results = {}
    
    # Test run.py
    help_result = run_cmd('python3 run.py --help')
    results['run_py'] = help_result['success']
    print(f"   run.py --help: {'✅' if help_result['success'] else '❌'}")
    
    # Test API imports
    api_test = run_cmd('python3 -c "from src.unity_wheel.api.advisor import get_trading_recommendation; print(\\"API import OK\\")"')
    results['api_import'] = api_test['success']
    print(f"   API imports: {'✅' if api_test['success'] else '❌'}")
    
    return results

def check_resources():
    """Monitor resource usage."""
    print("\n📊 6. RESOURCE USAGE MONITORING")
    print("-" * 50)
    
    results = {}
    
    # Disk space
    disk_result = run_cmd('df -h . | tail -1')
    if disk_result['success']:
        disk_info = disk_result['stdout']
        results['disk_info'] = disk_info
        print(f"   Disk space: {disk_info}")
    
    # Process count
    proc_result = run_cmd('ps aux | wc -l')
    if proc_result['success']:
        proc_count = int(proc_result['stdout']) - 1
        results['process_count'] = proc_count
        print(f"   Running processes: {proc_count}")
    
    # Load average
    load_result = run_cmd('uptime')
    if load_result['success']:
        results['load_average'] = load_result['stdout']
        print(f"   System load: {load_result['stdout']}")
    
    return results

def check_performance():
    """Test performance benchmarks."""
    print("\n🏃 8. PERFORMANCE BENCHMARKS")
    print("-" * 50)
    
    results = {}
    
    # File search performance
    start_time = time.time()
    search_result = run_cmd('find src -name "*.py" | wc -l')
    search_time = time.time() - start_time
    
    if search_result['success']:
        file_count = int(search_result['stdout'])
        results['file_search'] = {
            'time_seconds': round(search_time, 3),
            'files_found': file_count,
            'performance': 'good' if search_time < 1.0 else 'slow'
        }
        print(f"   File search: {file_count} files in {search_time:.3f}s {'✅' if search_time < 1.0 else '⚠️'}")
    
    # Python import performance  
    start_time = time.time()
    import_result = run_cmd('python3 -c "import numpy, pandas; print(\\"imports OK\\")"')
    import_time = time.time() - start_time
    
    if import_result['success']:
        results['python_imports'] = {
            'time_seconds': round(import_time, 3),
            'performance': 'good' if import_time < 2.0 else 'slow'
        }
        print(f"   Python imports: {import_time:.3f}s {'✅' if import_time < 2.0 else '⚠️'}")
    
    return results

def generate_deployment_report(all_results):
    """Generate deployment health report."""
    print("\n" + "="*80)
    print("🏁 DEPLOYMENT HEALTH VERIFICATION COMPLETE")  
    print("="*80)
    
    # Calculate health metrics
    total_tests = 0
    passed_tests = 0
    critical_failures = []
    warnings = []
    
    for section, results in all_results.items():
        if isinstance(results, dict):
            for test, result in results.items():
                total_tests += 1
                if isinstance(result, bool) and result:
                    passed_tests += 1
                elif isinstance(result, dict) and result.get('success', False):
                    passed_tests += 1
                else:
                    if section in ['startup', 'hardware', 'core_functionality']:
                        critical_failures.append(f"{section}.{test}")
                    else:
                        warnings.append(f"{section}.{test}")
    
    health_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Determine deployment readiness
    if len(critical_failures) == 0 and health_score >= 90:
        deployment_status = "READY"
    elif len(critical_failures) == 0 and health_score >= 70:
        deployment_status = "READY_WITH_WARNINGS"
    else:
        deployment_status = "NOT_READY"
    
    # Create final report
    deployment_report = {
        'timestamp': datetime.now().isoformat(),
        'deployment_status': deployment_status,
        'health_score': round(health_score, 1),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'critical_failures': critical_failures,
        'warnings': warnings,
        'detailed_results': all_results
    }
    
    # Save report
    report_file = f"deployment_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(deployment_report, f, indent=2)
    
    # Print summary
    print(f"📊 Deployment Status: {deployment_status}")
    print(f"💯 Health Score: {health_score:.1f}%")
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    
    if critical_failures:
        print(f"\n🚨 CRITICAL ISSUES ({len(critical_failures)}):")
        for failure in critical_failures:
            print(f"   • {failure}")
        print("\n❌ DEPLOYMENT NOT RECOMMENDED")
    elif warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings[:5]:
            print(f"   • {warning}")
        if len(warnings) > 5:
            print(f"   ... and {len(warnings) - 5} more")
        print("\n⚠️  DEPLOYMENT READY WITH WARNINGS")
    else:
        print("\n✅ DEPLOYMENT READY - All systems operational")
    
    print(f"\n📄 Full Report: {report_file}")
    print("="*80)
    
    return deployment_report

def main():
    """Execute comprehensive deployment verification."""
    print("🔍 DEPLOYMENT HEALTH VERIFICATION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # Run all verification steps
    verification_steps = [
        ("startup", check_system_startup),
        ("hardware", check_hardware),
        ("core_functionality", check_core_functionality), 
        ("configuration", check_configuration),
        ("api_endpoints", check_api_endpoints),
        ("resources", check_resources),
        ("performance", check_performance)
    ]
    
    all_results = {}
    
    for step_name, step_func in verification_steps:
        try:
            results = step_func()
            all_results[step_name] = results
        except Exception as e:
            print(f"   ❌ Error in {step_name}: {str(e)}")
            all_results[step_name] = {'error': str(e)}
    
    # Add execution time
    total_time = time.time() - start_time
    all_results['execution_time_seconds'] = round(total_time, 2)
    
    # Generate deployment report
    deployment_report = generate_deployment_report(all_results)
    
    # Return appropriate exit code
    if deployment_report['deployment_status'] == 'NOT_READY':
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())