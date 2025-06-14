"""Run all Jarvis2 tests and create a fix report."""
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# Test suites to run
TEST_SUITES = [
    ('device_routing', 'tests/test_device_routing.py'),
    ('process_isolation', 'tests/test_process_isolation.py'),
    ('memory_management', 'tests/test_memory_management.py'),
    ('performance', 'tests/test_performance_benchmarks.py'),
    ('mcts_correctness', 'tests/test_mcts_correctness.py'),
    ('integration', 'tests/test_jarvis2_integration.py')
]

def run_test_suite(name, test_file):
    """Run a test suite and capture results."""
    print(f"\n{'='*60}")
    print(f"Running {name} tests...")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, '-m', 'pytest',
        test_file,
        '-v',
        '-n0',  # No parallelization
        '--tb=short',
        '--timeout=60',  # 60 second timeout per test
        '-W', 'ignore::DeprecationWarning',
        '--json-report',
        f'--json-report-file=test_results_{name}.json'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results
    passed = failed = 0
    if result.returncode == 0:
        print(f"✅ All {name} tests passed!")
        # Count from output
        for line in result.stdout.split('\n'):
            if 'passed' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed':
                        try:
                            passed = int(parts[i-1])
                        except:
                            pass
    else:
        print(f"❌ Some {name} tests failed")
        print("\nSTDOUT:")
        print(result.stdout[-1000:])  # Last 1000 chars
        print("\nSTDERR:")
        print(result.stderr[-1000:])
        
    return {
        'name': name,
        'passed': result.returncode == 0,
        'output': result.stdout,
        'errors': result.stderr
    }

def generate_fix_report(results):
    """Generate a report of what needs fixing."""
    report = [
        "# Jarvis2 Test Results and Fix Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Summary\n"
    ]
    
    total_passed = sum(1 for r in results if r['passed'])
    total_failed = len(results) - total_passed
    
    report.append(f"- Total test suites: {len(results)}")
    report.append(f"- Passed: {total_passed}")
    report.append(f"- Failed: {total_failed}")
    
    report.append("\n## Test Suite Results\n")
    
    for result in results:
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        report.append(f"\n### {result['name']} - {status}")
        
        if not result['passed']:
            # Extract failure info
            report.append("\n**Failures:**")
            
            # Look for assertion errors
            for line in result['output'].split('\n'):
                if 'FAILED' in line or 'AssertionError' in line:
                    report.append(f"- {line.strip()}")
                    
            # Add stderr if present
            if result['errors']:
                report.append("\n**Errors:**")
                report.append("```")
                report.append(result['errors'][-500:])  # Last 500 chars
                report.append("```")
    
    report.append("\n## Fix Strategy\n")
    
    # Analyze common issues
    if any('MPS' in r['output'] or 'Metal' in r['output'] for r in results if not r['passed']):
        report.append("\n### PyTorch MPS Issues")
        report.append("- Use spawn method for multiprocessing (already configured)")
        report.append("- Ensure PyTorch only imported in worker processes")
        report.append("- Consider using MLX instead of PyTorch where possible")
        
    if any('timeout' in r['output'].lower() for r in results if not r['passed']):
        report.append("\n### Timeout Issues")
        report.append("- Increase timeouts for spawn-based process creation")
        report.append("- Reduce test data sizes for faster execution")
        report.append("- Use CPU-only PyTorch for unit tests")
        
    if any('memory' in r['output'].lower() for r in results if not r['passed']):
        report.append("\n### Memory Issues")
        report.append("- Ensure proper cleanup of shared memory")
        report.append("- Monitor Metal memory limits (18GB)")
        report.append("- Use smaller buffer pools for tests")
    
    report.append("\n## Next Steps\n")
    report.append("1. Fix failing tests one by one")
    report.append("2. Re-run to verify fixes")
    report.append("3. Update expectations for M4 Pro performance")
    report.append("4. Consider CPU-only testing for CI")
    
    return '\n'.join(report)

def main():
    """Run all tests and generate report."""
    print("Jarvis2 Comprehensive Test Runner")
    print("="*60)
    
    # Change to jarvis2 directory
    jarvis_dir = Path(__file__).parent
    import os
    os.chdir(jarvis_dir)
    
    results = []
    
    # Run each test suite
    for name, test_file in TEST_SUITES:
        if Path(test_file).exists():
            result = run_test_suite(name, test_file)
            results.append(result)
        else:
            print(f"⚠️  {test_file} not found")
            
    # Generate report
    report = generate_fix_report(results)
    
    # Save report
    report_file = Path('TEST_RESULTS_REPORT.md')
    report_file.write_text(report)
    print(f"\n📄 Report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for result in results:
        status = "✅" if result['passed'] else "❌"
        print(f"{status} {result['name']}")
        
    # Return exit code
    return 0 if all(r['passed'] for r in results) else 1

if __name__ == "__main__":
    sys.exit(main())