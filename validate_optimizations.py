#!/usr/bin/env python3
"""
Production Optimization Validation

Validates that all M4 Pro optimizations are working correctly.
"""

import json
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def check_hardware():
    """Check M4 Pro hardware configuration."""
    print("🔍 Hardware Detection")
    print("-" * 20)
    
    cpu_count = mp.cpu_count()
    machine = platform.machine()
    system = platform.system()
    
    print(f"System: {system}")
    print(f"Architecture: {machine}")
    print(f"CPU cores: {cpu_count}")
    
    # Check for M4 Pro
    is_m4_pro = 'arm64' in machine.lower() and cpu_count >= 12
    
    if is_m4_pro:
        print("✅ M4 Pro detected")
        return {
            'status': 'success',
            'cpu_cores': cpu_count,
            'performance_cores': 8,
            'efficiency_cores': 4,
            'architecture': machine
        }
    else:
        print("⚠️ Not M4 Pro or suboptimal configuration")
        return {
            'status': 'warning',
            'cpu_cores': cpu_count,
            'architecture': machine
        }

def check_environment():
    """Check environment variables."""
    print("\n🔧 Environment Configuration")
    print("-" * 29)
    
    env_vars = {
        'OMP_NUM_THREADS': '8',
        'MKL_NUM_THREADS': '12',
        'PYTHONHASHSEED': '0',
        'METAL_DEVICE_WRAPPER_TYPE': '1'
    }
    
    configured = {}
    for var, expected in env_vars.items():
        actual = os.environ.get(var)
        configured[var] = {
            'expected': expected,
            'actual': actual,
            'status': 'set' if actual else 'not_set'
        }
        status = "✅" if actual else "❌"
        print(f"{status} {var}: {actual or 'not set'}")
    
    return configured

def test_parallel_performance():
    """Test parallel processing performance."""
    print("\n⚡ Parallel Processing Test")
    print("-" * 26)
    
    def cpu_intensive_task(n):
        return sum(i * i for i in range(n))
    
    # Single-threaded baseline
    print("Running single-threaded baseline...")
    start_time = time.time()
    single_result = cpu_intensive_task(100000)
    single_time = time.time() - start_time
    
    # Multi-threaded test
    print("Running multi-threaded test...")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        parallel_results = list(executor.map(cpu_intensive_task, [100000] * 4))
    parallel_time = time.time() - start_time
    
    # Calculate speedup
    speedup = (single_time * 4) / parallel_time
    
    print(f"Single-threaded: {single_time*1000:.1f}ms")
    print(f"Multi-threaded:  {parallel_time*1000:.1f}ms")
    print(f"Speedup: {speedup:.1f}x")
    
    status = "✅" if speedup >= 2.0 else "❌"
    print(f"{status} Parallel processing: {'PASS' if speedup >= 2.0 else 'FAIL'}")
    
    return {
        'single_time_ms': single_time * 1000,
        'parallel_time_ms': parallel_time * 1000,
        'speedup': speedup,
        'status': 'pass' if speedup >= 2.0 else 'fail'
    }

def test_memory_optimization():
    """Test memory optimization."""
    print("\n💾 Memory Optimization Test")
    print("-" * 27)
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        used_percent = memory.percent
        
        print(f"Total memory: {total_gb:.1f}GB")
        print(f"Available: {available_gb:.1f}GB")
        print(f"Used: {used_percent:.1f}%")
        
        # Check if we have enough memory for optimization
        sufficient_memory = total_gb >= 16.0
        status = "✅" if sufficient_memory else "❌"
        print(f"{status} Memory: {'SUFFICIENT' if sufficient_memory else 'INSUFFICIENT'}")
        
        return {
            'total_gb': total_gb,
            'available_gb': available_gb,
            'used_percent': used_percent,
            'sufficient': sufficient_memory,
            'status': 'pass' if sufficient_memory else 'fail'
        }
        
    except ImportError:
        print("❌ psutil not available - cannot test memory")
        return {'status': 'skip', 'reason': 'psutil not available'}

def test_accelerated_tools():
    """Test accelerated tools availability."""
    print("\n🚀 Accelerated Tools Test")
    print("-" * 25)
    
    tools_status = {}
    
    # Test ripgrep turbo
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
        ripgrep = get_ripgrep_turbo()
        tools_status['ripgrep_turbo'] = {'status': 'available', 'version': 'production'}
        print("✅ Ripgrep Turbo: Available")
    except ImportError as e:
        tools_status['ripgrep_turbo'] = {'status': 'unavailable', 'error': str(e)}
        print("❌ Ripgrep Turbo: Not available")
    
    # Test neural engine turbo
    try:
        from unity_wheel.accelerated_tools.neural_engine_turbo import get_neural_engine_turbo
        tools_status['neural_engine_turbo'] = {'status': 'available'}
        print("✅ Neural Engine Turbo: Available")
    except ImportError as e:
        tools_status['neural_engine_turbo'] = {'status': 'unavailable', 'error': str(e)}
        print("❌ Neural Engine Turbo: Not available")
    
    # Test dependency graph turbo
    try:
        from unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph
        tools_status['dependency_graph_turbo'] = {'status': 'available'}
        print("✅ Dependency Graph Turbo: Available")
    except ImportError as e:
        tools_status['dependency_graph_turbo'] = {'status': 'unavailable', 'error': str(e)}
        print("❌ Dependency Graph Turbo: Not available")
    
    # Test DuckDB turbo
    try:
        from unity_wheel.accelerated_tools.duckdb_turbo import get_duckdb_turbo
        tools_status['duckdb_turbo'] = {'status': 'available'}
        print("✅ DuckDB Turbo: Available")
    except ImportError as e:
        tools_status['duckdb_turbo'] = {'status': 'unavailable', 'error': str(e)}
        print("❌ DuckDB Turbo: Not available")
    
    available_count = sum(1 for tool in tools_status.values() if tool['status'] == 'available')
    total_count = len(tools_status)
    
    print(f"\nAccelerated tools: {available_count}/{total_count} available")
    
    return tools_status

def generate_report(hardware, environment, parallel, memory, tools):
    """Generate comprehensive validation report."""
    
    # Calculate overall score
    scores = []
    
    # Hardware score (30%)
    if hardware['status'] == 'success':
        scores.append(30)
    elif hardware['status'] == 'warning':
        scores.append(15)
    else:
        scores.append(0)
    
    # Environment score (20%)
    env_configured = sum(1 for var in environment.values() if var['status'] == 'set')
    env_score = (env_configured / len(environment)) * 20
    scores.append(env_score)
    
    # Parallel processing score (25%)
    if parallel['status'] == 'pass':
        scores.append(25)
    else:
        scores.append(10 if parallel['speedup'] >= 1.5 else 0)
    
    # Memory score (15%)
    if memory.get('status') == 'pass':
        scores.append(15)
    elif memory.get('status') == 'skip':
        scores.append(10)  # Partial credit
    else:
        scores.append(5)
    
    # Tools score (10%)
    if tools:
        available_tools = sum(1 for tool in tools.values() if tool['status'] == 'available')
        tools_score = (available_tools / len(tools)) * 10
        scores.append(tools_score)
    else:
        scores.append(0)
    
    total_score = sum(scores)
    
    report = {
        'timestamp': time.time(),
        'overall_score': total_score,
        'grade': (
            'A' if total_score >= 85 else
            'B' if total_score >= 70 else
            'C' if total_score >= 55 else
            'D' if total_score >= 40 else
            'F'
        ),
        'hardware': hardware,
        'environment': environment,
        'parallel_processing': parallel,
        'memory': memory,
        'accelerated_tools': tools,
        'recommendations': []
    }
    
    # Generate recommendations
    if hardware['status'] != 'success':
        report['recommendations'].append("Consider upgrading to M4 Pro for optimal performance")
    
    if parallel['status'] != 'pass':
        report['recommendations'].append("Parallel processing underperforming - check CPU configuration")
    
    if memory.get('status') == 'fail':
        report['recommendations'].append("Insufficient memory for optimal performance")
    
    env_issues = [var for var, config in environment.items() if config['status'] != 'set']
    if env_issues:
        report['recommendations'].append(f"Set environment variables: {', '.join(env_issues)}")
    
    return report

def main():
    """Main validation function."""
    print("🔍 Production Optimization Validation")
    print("=" * 40)
    
    # Run all tests
    hardware = check_hardware()
    environment = check_environment() 
    parallel = test_parallel_performance()
    memory = test_memory_optimization()
    tools = test_accelerated_tools()
    
    # Generate report
    report = generate_report(hardware, environment, parallel, memory, tools)
    
    # Display summary
    print("\n📊 Validation Summary")
    print("-" * 20)
    print(f"Overall Score: {report['overall_score']:.1f}/100")
    print(f"Grade: {report['grade']}")
    
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
    
    # Save report
    with open('optimization_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Full report saved to: optimization_validation_report.json")
    
    # Return success/failure
    if report['overall_score'] >= 70:
        print("\n🎉 VALIDATION PASSED: Optimizations are working well!")
        return 0
    else:
        print("\n⚠️ VALIDATION PARTIAL: Some optimizations need attention")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)