#!/usr/bin/env python3

"""
Enhanced Node.js Memory Setup Validation for M4 Pro
Comprehensive validation script to ensure optimal configuration
Prevents RangeError: Invalid string length errors
"""

import os
import sys
import json
import subprocess
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class MemorySetupValidator:
    def __init__(self):
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'validations': [],
            'summary': {
                'total_checks': 0,
                'passed': 0,
                'warnings': 0,
                'failed': 0,
                'critical_issues': 0
            },
            'recommendations': []
        }
        
        self.critical_issues = []
        self.warnings = []
        
        # Colors for terminal output
        self.colors = {
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'end': '\033[0m'
        }

    def colorize(self, text: str, color: str) -> str:
        """Add color to terminal output"""
        return f"{self.colors.get(color, '')}{text}{self.colors['end']}"

    def add_validation(self, category: str, name: str, status: str, details: str, value: Optional[str] = None):
        """Add a validation result"""
        validation = {
            'category': category,
            'name': name,
            'status': status,  # 'pass', 'warning', 'fail', 'critical'
            'details': details,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        self.validation_results['validations'].append(validation)
        self.validation_results['summary']['total_checks'] += 1
        
        if status == 'pass':
            self.validation_results['summary']['passed'] += 1
            icon = '✅'
            color = 'green'
        elif status == 'warning':
            self.validation_results['summary']['warnings'] += 1
            self.warnings.append(validation)
            icon = '⚠️'
            color = 'yellow'
        elif status == 'fail':
            self.validation_results['summary']['failed'] += 1
            icon = '❌'
            color = 'red'
        elif status == 'critical':
            self.validation_results['summary']['critical_issues'] += 1
            self.critical_issues.append(validation)
            icon = '🚨'
            color = 'red'
        else:
            icon = 'ℹ️'
            color = 'blue'
        
        print(f"{icon} {self.colorize(f'[{category}]', 'cyan')} {name}: {self.colorize(details, color)}")
        if value:
            print(f"   Value: {self.colorize(value, 'white')}")

    def run_command(self, command: str, shell: bool = True) -> Tuple[str, str, int]:
        """Run a shell command and return stdout, stderr, return code"""
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", 1
        except Exception as e:
            return "", str(e), 1

    def get_system_info(self):
        """Gather comprehensive system information"""
        print(f"{self.colorize('🔍 Gathering System Information...', 'blue')}")
        
        info = {}
        
        # macOS version
        stdout, _, _ = self.run_command("sw_vers -productVersion")
        info['macos_version'] = stdout
        
        # Hardware info
        stdout, _, _ = self.run_command("sysctl -n hw.model")
        info['hardware_model'] = stdout
        
        stdout, _, _ = self.run_command("sysctl -n hw.ncpu")
        info['cpu_cores'] = int(stdout) if stdout.isdigit() else 0
        
        stdout, _, _ = self.run_command("sysctl -n hw.memsize")
        if stdout.isdigit():
            info['total_memory_gb'] = round(int(stdout) / 1024 / 1024 / 1024, 1)
        
        # Node.js version
        stdout, _, code = self.run_command("node --version")
        if code == 0:
            info['node_version'] = stdout
        else:
            info['node_version'] = 'Not installed'
        
        # Shell info
        info['shell'] = os.environ.get('SHELL', 'unknown')
        
        self.validation_results['system_info'] = info
        
        print(f"   macOS: {info.get('macos_version', 'unknown')}")
        print(f"   Hardware: {info.get('hardware_model', 'unknown')}")
        print(f"   CPU cores: {info.get('cpu_cores', 'unknown')}")
        print(f"   Memory: {info.get('total_memory_gb', 'unknown')}GB")
        print(f"   Node.js: {info.get('node_version', 'unknown')}")
        print()

    def validate_environment_variables(self):
        """Validate Node.js environment variables"""
        print(f"{self.colorize('🌐 Validating Environment Variables...', 'blue')}")
        
        # NODE_OPTIONS
        node_options = os.environ.get('NODE_OPTIONS')
        if node_options:
            # Check max-old-space-size
            heap_match = re.search(r'--max-old-space-size=(\d+)', node_options)
            if heap_match:
                heap_size = int(heap_match.group(1))
                if heap_size >= 20480:
                    self.add_validation('Environment', 'Heap Size', 'pass', 
                                     f'Optimal heap size: {heap_size}MB', node_options)
                elif heap_size >= 18432:
                    self.add_validation('Environment', 'Heap Size', 'warning', 
                                     f'Good heap size: {heap_size}MB (recommended: 20480MB)', node_options)
                else:
                    self.add_validation('Environment', 'Heap Size', 'fail', 
                                     f'Insufficient heap size: {heap_size}MB', node_options)
            else:
                self.add_validation('Environment', 'Heap Size', 'fail', 
                                 'max-old-space-size not found in NODE_OPTIONS', node_options)
            
            # Check semi-space size
            if '--max-semi-space-size=1024' in node_options:
                self.add_validation('Environment', 'Semi-space Size', 'pass', 
                                 'Optimal semi-space size: 1024MB')
            elif '--max-semi-space-size=512' in node_options:
                self.add_validation('Environment', 'Semi-space Size', 'warning', 
                                 'Semi-space size: 512MB (recommended: 1024MB)')
            else:
                self.add_validation('Environment', 'Semi-space Size', 'warning', 
                                 'Semi-space size not optimized')
            
            # Check expose-gc
            if '--expose-gc' in node_options:
                self.add_validation('Environment', 'Manual GC', 'pass', 
                                 'Manual garbage collection enabled')
            else:
                self.add_validation('Environment', 'Manual GC', 'warning', 
                                 'Manual GC not enabled (recommended: --expose-gc)')
            
            # Check v8-pool-size
            if '--v8-pool-size=12' in node_options:
                self.add_validation('Environment', 'V8 Pool Size', 'pass', 
                                 'V8 pool optimized for M4 Pro (12 threads)')
            else:
                self.add_validation('Environment', 'V8 Pool Size', 'warning', 
                                 'V8 pool size not optimized for M4 Pro')
        else:
            self.add_validation('Environment', 'NODE_OPTIONS', 'critical', 
                             'NODE_OPTIONS not set - critical for memory optimization')
        
        # UV_THREADPOOL_SIZE
        uv_threads = os.environ.get('UV_THREADPOOL_SIZE')
        if uv_threads == '12':
            self.add_validation('Environment', 'Thread Pool Size', 'pass', 
                             'Thread pool optimized for M4 Pro (12 threads)', uv_threads)
        elif uv_threads:
            self.add_validation('Environment', 'Thread Pool Size', 'warning', 
                             f'Thread pool size: {uv_threads} (recommended: 12)', uv_threads)
        else:
            self.add_validation('Environment', 'Thread Pool Size', 'warning', 
                             'UV_THREADPOOL_SIZE not set (recommended: 12)')
        
        # Memory allocator settings
        malloc_arena = os.environ.get('MALLOC_ARENA_MAX')
        if malloc_arena == '4':
            self.add_validation('Environment', 'Memory Allocator', 'pass', 
                             'Memory allocator optimized', malloc_arena)
        else:
            self.add_validation('Environment', 'Memory Allocator', 'warning', 
                             'MALLOC_ARENA_MAX not optimized (recommended: 4)')

    def validate_system_limits(self):
        """Validate system resource limits"""
        print(f"{self.colorize('🔧 Validating System Limits...', 'blue')}")
        
        # File descriptor limit
        stdout, _, _ = self.run_command("ulimit -n")
        if stdout.isdigit():
            fd_limit = int(stdout)
            if fd_limit >= 32768:
                self.add_validation('System Limits', 'File Descriptors', 'pass', 
                                 f'Optimal file descriptor limit: {fd_limit}')
            elif fd_limit >= 16384:
                self.add_validation('System Limits', 'File Descriptors', 'warning', 
                                 f'Good file descriptor limit: {fd_limit} (recommended: 32768)')
            else:
                self.add_validation('System Limits', 'File Descriptors', 'fail', 
                                 f'Low file descriptor limit: {fd_limit}')
        
        # Process limit
        stdout, _, _ = self.run_command("ulimit -u")
        if stdout.isdigit():
            proc_limit = int(stdout)
            if proc_limit >= 8192:
                self.add_validation('System Limits', 'Process Limit', 'pass', 
                                 f'Optimal process limit: {proc_limit}')
            elif proc_limit >= 4096:
                self.add_validation('System Limits', 'Process Limit', 'warning', 
                                 f'Good process limit: {proc_limit} (recommended: 8192)')
            else:
                self.add_validation('System Limits', 'Process Limit', 'fail', 
                                 f'Low process limit: {proc_limit}')
        
        # Stack size
        stdout, _, _ = self.run_command("ulimit -s")
        if stdout.isdigit():
            stack_size = int(stdout)
            if stack_size >= 65536:
                self.add_validation('System Limits', 'Stack Size', 'pass', 
                                 f'Optimal stack size: {stack_size}KB')
            elif stack_size >= 32768:
                self.add_validation('System Limits', 'Stack Size', 'warning', 
                                 f'Good stack size: {stack_size}KB (recommended: 65536KB)')
            else:
                self.add_validation('System Limits', 'Stack Size', 'warning', 
                                 f'Default stack size: {stack_size}KB')

    def validate_launchd_configuration(self):
        """Validate LaunchAgent configuration"""
        print(f"{self.colorize('🚀 Validating LaunchAgent Configuration...', 'blue')}")
        
        launchd_path = Path.home() / "Library/LaunchAgents/com.nodejs.memory-limits.plist"
        
        if launchd_path.exists():
            self.add_validation('LaunchAgent', 'Plist File', 'pass', 
                             'LaunchAgent plist file exists')
            
            # Check if it's loaded
            stdout, _, code = self.run_command(
                f"launchctl list | grep com.nodejs.memory-limits"
            )
            if code == 0:
                self.add_validation('LaunchAgent', 'Service Status', 'pass', 
                                 'LaunchAgent is loaded and active')
            else:
                self.add_validation('LaunchAgent', 'Service Status', 'warning', 
                                 'LaunchAgent plist exists but may not be loaded')
        else:
            self.add_validation('LaunchAgent', 'Configuration', 'warning', 
                             'LaunchAgent not configured for persistent settings')

    def validate_node_functionality(self):
        """Validate Node.js functionality with current configuration"""
        print(f"{self.colorize('🔬 Validating Node.js Functionality...', 'blue')}")
        
        # Check if Node.js is available
        stdout, stderr, code = self.run_command("node --version")
        if code != 0:
            self.add_validation('Node.js', 'Installation', 'critical', 
                             'Node.js not available', stderr)
            return
        
        # Test heap limit
        heap_test = '''
        const v8 = require('v8');
        const stats = v8.getHeapStatistics();
        console.log(JSON.stringify({
            heapLimit: Math.round(stats.heap_size_limit / 1024 / 1024),
            heapUsed: Math.round(stats.used_heap_size / 1024 / 1024),
            gcAvailable: typeof global.gc !== 'undefined'
        }));
        '''
        
        stdout, stderr, code = self.run_command(f"node -e '{heap_test}'")
        if code == 0:
            try:
                heap_info = json.loads(stdout)
                heap_limit = heap_info['heapLimit']
                
                if heap_limit >= 20000:
                    self.add_validation('Node.js', 'Heap Configuration', 'pass', 
                                     f'Optimal heap limit: {heap_limit}MB')
                elif heap_limit >= 18000:
                    self.add_validation('Node.js', 'Heap Configuration', 'warning', 
                                     f'Good heap limit: {heap_limit}MB (target: 20GB)')
                else:
                    self.add_validation('Node.js', 'Heap Configuration', 'fail', 
                                     f'Insufficient heap limit: {heap_limit}MB')
                
                if heap_info['gcAvailable']:
                    self.add_validation('Node.js', 'Manual GC', 'pass', 
                                     'Manual garbage collection available')
                else:
                    self.add_validation('Node.js', 'Manual GC', 'warning', 
                                     'Manual GC not available (need --expose-gc)')
                    
            except json.JSONDecodeError:
                self.add_validation('Node.js', 'Heap Test', 'fail', 
                                 'Could not parse heap statistics', stdout)
        else:
            self.add_validation('Node.js', 'Heap Test', 'fail', 
                             'Could not retrieve heap statistics', stderr)

    def validate_string_allocation(self):
        """Test string allocation capabilities"""
        print(f"{self.colorize('📝 Testing String Allocation...', 'blue')}")
        
        # Test small string allocation
        small_test = '''
        try {
            const testString = 'x'.repeat(100 * 1024 * 1024); // 100MB
            console.log('SUCCESS:100MB');
        } catch (error) {
            console.log('ERROR:' + error.message);
        }
        '''
        
        stdout, stderr, code = self.run_command(f"node -e '{small_test}'")
        if 'SUCCESS:100MB' in stdout:
            self.add_validation('String Allocation', '100MB Test', 'pass', 
                             'Successfully allocated 100MB string')
            
            # Test larger string
            large_test = '''
            try {
                const testString = 'x'.repeat(1000 * 1024 * 1024); // 1GB
                console.log('SUCCESS:1000MB');
            } catch (error) {
                console.log('ERROR:' + error.message);
            }
            '''
            
            stdout, stderr, code = self.run_command(f"node -e '{large_test}'")
            if 'SUCCESS:1000MB' in stdout:
                self.add_validation('String Allocation', '1GB Test', 'pass', 
                                 'Successfully allocated 1GB string')
            elif 'Invalid string length' in stdout:
                self.add_validation('String Allocation', '1GB Test', 'warning', 
                                 'String length limit reached at <1GB')
            else:
                self.add_validation('String Allocation', '1GB Test', 'fail', 
                                 'Unexpected error in 1GB string test')
        else:
            self.add_validation('String Allocation', '100MB Test', 'fail', 
                             'Failed to allocate 100MB string - configuration issue')

    def validate_file_configurations(self):
        """Validate shell configuration files"""
        print(f"{self.colorize('📁 Validating Configuration Files...', 'blue')}")
        
        # Check .zshenv
        zshenv_path = Path.home() / ".zshenv"
        if zshenv_path.exists():
            content = zshenv_path.read_text()
            if 'NODE_OPTIONS' in content and 'max-old-space-size' in content:
                self.add_validation('Config Files', '.zshenv', 'pass', 
                                 'Node.js configuration found in .zshenv')
            else:
                self.add_validation('Config Files', '.zshenv', 'warning', 
                                 '.zshenv exists but missing Node.js config')
        else:
            self.add_validation('Config Files', '.zshenv', 'warning', 
                             '.zshenv not found')
        
        # Check .bashrc if it exists
        bashrc_path = Path.home() / ".bashrc"
        if bashrc_path.exists():
            content = bashrc_path.read_text()
            if 'NODE_OPTIONS' in content:
                self.add_validation('Config Files', '.bashrc', 'pass', 
                                 'Node.js configuration found in .bashrc')
            else:
                self.add_validation('Config Files', '.bashrc', 'warning', 
                                 '.bashrc exists but missing Node.js config')

    def validate_system_configuration(self):
        """Validate system-level configuration"""
        print(f"{self.colorize('⚙️ Validating System Configuration...', 'blue')}")
        
        # Check /etc/launchd.conf
        launchd_conf = Path("/etc/launchd.conf")
        if launchd_conf.exists():
            try:
                content = launchd_conf.read_text()
                if 'maxfiles' in content:
                    self.add_validation('System Config', '/etc/launchd.conf', 'pass', 
                                     'System limits configured')
                else:
                    self.add_validation('System Config', '/etc/launchd.conf', 'warning', 
                                     'launchd.conf exists but no limit configuration')
            except PermissionError:
                self.add_validation('System Config', '/etc/launchd.conf', 'warning', 
                                 'Cannot read /etc/launchd.conf (permission denied)')
        else:
            self.add_validation('System Config', '/etc/launchd.conf', 'warning', 
                             '/etc/launchd.conf not found')
        
        # Check sysctl.conf
        sysctl_conf = Path("/etc/sysctl.conf")
        if sysctl_conf.exists():
            try:
                content = sysctl_conf.read_text()
                if 'kern.maxfiles' in content:
                    self.add_validation('System Config', '/etc/sysctl.conf', 'pass', 
                                     'Kernel limits configured')
                else:
                    self.add_validation('System Config', '/etc/sysctl.conf', 'warning', 
                                     'sysctl.conf exists but no kernel limits')
            except PermissionError:
                self.add_validation('System Config', '/etc/sysctl.conf', 'warning', 
                                 'Cannot read /etc/sysctl.conf (permission denied)')
        else:
            self.add_validation('System Config', '/etc/sysctl.conf', 'warning', 
                             '/etc/sysctl.conf not configured')

    def generate_recommendations(self):
        """Generate specific recommendations based on validation results"""
        recommendations = []
        
        # Critical issues first
        if self.critical_issues:
            recommendations.append("🚨 CRITICAL: Address critical issues immediately:")
            for issue in self.critical_issues:
                recommendations.append(f"   - {issue['name']}: {issue['details']}")
        
        # Environment variable recommendations
        node_options = os.environ.get('NODE_OPTIONS')
        if not node_options:
            recommendations.append("Run: ./scripts/configure-nodejs-memory.sh to set up environment variables")
        elif 'max-old-space-size=20480' not in node_options:
            recommendations.append("Update NODE_OPTIONS to include --max-old-space-size=20480")
        
        # System limit recommendations
        stdout, _, _ = self.run_command("ulimit -n")
        if stdout.isdigit() and int(stdout) < 32768:
            recommendations.append("Increase file descriptor limit: ulimit -n 32768")
        
        # LaunchAgent recommendation
        launchd_path = Path.home() / "Library/LaunchAgents/com.nodejs.memory-limits.plist"
        if not launchd_path.exists():
            recommendations.append("Configure persistent settings with LaunchAgent")
        
        # Testing recommendations
        if any(v['category'] == 'String Allocation' and v['status'] in ['fail', 'warning'] 
               for v in self.validation_results['validations']):
            recommendations.append("Run comprehensive tests: ./scripts/test-memory-config.js")
        
        # General recommendations
        if self.validation_results['summary']['warnings'] > 0:
            recommendations.append("Review warning items for optimal performance")
        
        if self.validation_results['summary']['failed'] > 0:
            recommendations.append("Fix failed validations before production use")
        
        self.validation_results['recommendations'] = recommendations

    def run_validation(self):
        """Run complete validation suite"""
        print(f"{self.colorize('🔍 Enhanced Node.js Memory Setup Validation', 'bold')}")
        print(f"{self.colorize('Optimized for M4 Pro with 24GB unified memory', 'cyan')}")
        print()
        
        try:
            self.get_system_info()
            self.validate_environment_variables()
            self.validate_system_limits()
            self.validate_launchd_configuration()
            self.validate_node_functionality()
            self.validate_string_allocation()
            self.validate_file_configurations()
            self.validate_system_configuration()
            
            self.generate_recommendations()
            
            return self.generate_report()
            
        except KeyboardInterrupt:
            print(f"\n{self.colorize('❌ Validation interrupted by user', 'red')}")
            return None
        except Exception as e:
            print(f"\n{self.colorize(f'❌ Validation failed: {str(e)}', 'red')}")
            return None

    def generate_report(self):
        """Generate and save validation report"""
        print(f"\n{self.colorize('📋 Validation Summary', 'bold')}")
        print("=" * 50)
        
        # Summary statistics
        summary = self.validation_results['summary']
        total = summary['total_checks']
        passed = summary['passed']
        warnings = summary['warnings']
        failed = summary['failed']
        critical = summary['critical_issues']
        
        print(f"Total checks: {total}")
        print(f"{self.colorize(f'✅ Passed: {passed}', 'green')}")
        if warnings > 0:
            print(f"{self.colorize(f'⚠️  Warnings: {warnings}', 'yellow')}")
        if failed > 0:
            print(f"{self.colorize(f'❌ Failed: {failed}', 'red')}")
        if critical > 0:
            print(f"{self.colorize(f'🚨 Critical: {critical}', 'red')}")
        
        # Overall status
        if critical > 0:
            status = "CRITICAL ISSUES FOUND"
            color = 'red'
        elif failed > 0:
            status = "CONFIGURATION ISSUES FOUND"
            color = 'red'
        elif warnings > 0:
            status = "CONFIGURATION WARNINGS"
            color = 'yellow'
        else:
            status = "CONFIGURATION OPTIMAL"
            color = 'green'
        
        print(f"\nOverall Status: {self.colorize(status, color)}")
        
        # Recommendations
        if self.validation_results['recommendations']:
            print(f"\n{self.colorize('🎯 Recommendations:', 'blue')}")
            for i, rec in enumerate(self.validation_results['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(__file__).parent / f"memory-validation-report-{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Exit code based on results
        if critical > 0 or failed > 0:
            return 1
        else:
            return 0

def main():
    """Main entry point"""
    validator = MemorySetupValidator()
    exit_code = validator.run_validation()
    
    if exit_code is not None:
        sys.exit(exit_code)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()