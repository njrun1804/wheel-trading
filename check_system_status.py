#!/usr/bin/env python3
"""
System Status Check - Quick health check for all components
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any

def check_meta_daemon() -> Dict[str, Any]:
    """Check meta daemon status"""
    
    status = {
        'name': 'Meta Daemon',
        'running': False,
        'healthy': False,
        'pid': None,
        'errors': []
    }
    
    # Check PID file
    pid_file = Path("meta_daemon.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            
            # Check if process exists
            try:
                import os
                os.kill(pid, 0)
                status['running'] = True
                status['pid'] = pid
            except OSError:
                status['errors'].append("PID file exists but process not running")
                
        except Exception as e:
            status['errors'].append(f"Error reading PID file: {e}")
    else:
        status['errors'].append("No PID file found")
    
    # Check log file for errors
    log_file = Path("meta_daemon.log")
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-20:]  # Last 20 lines
                
            recent_errors = [line for line in lines if 'ERROR' in line]
            if recent_errors:
                status['errors'].extend([f"Recent error: {line.split(' - ')[-1].strip()}" for line in recent_errors[-3:]])
            else:
                if status['running']:
                    status['healthy'] = True
                    
        except Exception as e:
            status['errors'].append(f"Error reading log file: {e}")
    
    return status

def check_jarvis() -> Dict[str, Any]:
    """Check Jarvis status"""
    
    status = {
        'name': 'Jarvis2',
        'running': False,
        'healthy': False,
        'pid': None,
        'errors': []
    }
    
    try:
        # Check for jarvis processes
        result = subprocess.run(['pgrep', '-f', 'jarvis'], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            if pids and pids[0]:
                status['running'] = True
                status['pid'] = int(pids[0])
                status['healthy'] = True  # Assume healthy if running
        else:
            status['errors'].append("No Jarvis processes found")
            
    except Exception as e:
        status['errors'].append(f"Error checking Jarvis: {e}")
    
    return status

def check_trading_system() -> Dict[str, Any]:
    """Check trading system status"""
    
    status = {
        'name': 'Trading System',
        'running': False,
        'healthy': False,
        'pid': None,
        'errors': []
    }
    
    try:
        # Test basic import
        result = subprocess.run(['python', '-c', 'import sys; sys.path.append("."); from unity_wheel.api.advisor import WheelAdvisor; print("OK")'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and 'OK' in result.stdout:
            status['running'] = True
            status['healthy'] = True
        else:
            status['errors'].append(f"Import test failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        status['errors'].append("Trading system import timed out")
    except Exception as e:
        status['errors'].append(f"Error testing trading system: {e}")
    
    return status

def check_database() -> Dict[str, Any]:
    """Check database status"""
    
    status = {
        'name': 'Database',
        'running': False,
        'healthy': False,
        'pid': None,
        'errors': []
    }
    
    # Check for database files
    db_files = [
        "data/wheel_trading_master.duckdb",
        "data/wheel_trading_optimized.duckdb",
        "meta_evolution.db"
    ]
    
    found_dbs = []
    for db_file in db_files:
        if Path(db_file).exists():
            found_dbs.append(db_file)
    
    if found_dbs:
        status['running'] = True
        status['healthy'] = True
        status['errors'].append(f"Found databases: {', '.join(found_dbs)}")
    else:
        status['errors'].append("No database files found")
    
    return status

def check_orchestrator() -> Dict[str, Any]:
    """Check orchestrator status"""
    
    status = {
        'name': 'Orchestrator',
        'running': False,
        'healthy': False,
        'pid': None,
        'errors': []
    }
    
    # Check if orchestrator files exist
    orchestrator_files = [
        "orchestrate.py",
        "orchestrate_turbo.py",
        "jarvis2/core/orchestrator.py"
    ]
    
    found_files = []
    for orch_file in orchestrator_files:
        if Path(orch_file).exists():
            found_files.append(orch_file)
    
    if found_files:
        try:
            # Test basic orchestrator functionality
            result = subprocess.run(['python', found_files[0], '--help'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                status['running'] = True
                status['healthy'] = True
            else:
                status['errors'].append(f"Orchestrator test failed: {result.stderr[:100]}")
                
        except subprocess.TimeoutExpired:
            status['errors'].append("Orchestrator test timed out")
        except Exception as e:
            status['errors'].append(f"Error testing orchestrator: {e}")
    else:
        status['errors'].append("No orchestrator files found")
    
    return status

def main():
    """Main status check function"""
    
    print("🔍 Unity Wheel Trading - System Status Check")
    print("=" * 50)
    
    # Check all components
    components = [
        check_meta_daemon(),
        check_jarvis(),
        check_trading_system(),
        check_database(),
        check_orchestrator()
    ]
    
    # Print status for each component
    for component in components:
        name = component['name']
        running = component['running']
        healthy = component['healthy']
        pid = component['pid']
        errors = component['errors']
        
        # Status emoji
        if running and healthy:
            status_emoji = "✅"
            status_text = "HEALTHY"
        elif running:
            status_emoji = "⚠️"
            status_text = "RUNNING (ERRORS)"
        else:
            status_emoji = "❌"
            status_text = "STOPPED"
        
        print(f"\n{status_emoji} {name}: {status_text}")
        
        if pid:
            print(f"   PID: {pid}")
        
        if errors:
            print("   Issues:")
            for error in errors[-3:]:  # Show last 3 errors
                print(f"     • {error}")
    
    # Overall system status
    print("\n" + "=" * 50)
    
    running_count = sum(1 for c in components if c['running'])
    healthy_count = sum(1 for c in components if c['healthy'])
    
    print(f"📊 Overall Status:")
    print(f"   Components running: {running_count}/{len(components)}")
    print(f"   Components healthy: {healthy_count}/{len(components)}")
    
    if healthy_count == len(components):
        print("🎉 All systems operational!")
    elif running_count >= 3:
        print("⚠️  System partially operational")
    else:
        print("❌ System needs attention")
    
    # Quick commands
    print(f"\n📋 Quick Commands:")
    print("   python start_everything.py        - Start all components")
    print("   python run.py                     - Trading recommendation")
    print("   python run.py --diagnose          - System diagnostics")
    print("   tail -f meta_daemon.log           - Monitor meta daemon")
    print("   python unified_meta_system.py     - Manual meta system")
    
    # Save status to file
    status_data = {
        'timestamp': time.time(),
        'components': components,
        'summary': {
            'running_count': running_count,
            'healthy_count': healthy_count,
            'total_components': len(components)
        }
    }
    
    with open('system_status.json', 'w') as f:
        json.dump(status_data, f, indent=2)
    
    print(f"\n💾 Status saved to: system_status.json")

if __name__ == "__main__":
    main()