#!/usr/bin/env python3
"""
Complete System Startup - Start Meta System + Jarvis + Verification
Ensures all components are running and healthy at system startup
"""

import subprocess
import time
import sys
import os
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional

class SystemStarter:
    """Manages startup of all meta system and Jarvis components"""
    
    def __init__(self):
        self.components = {
            'meta_daemon': {'pid': None, 'status': 'stopped', 'health': 'unknown'},
            'jarvis2': {'pid': None, 'status': 'stopped', 'health': 'unknown'}, 
            'unified_meta': {'pid': None, 'status': 'stopped', 'health': 'unknown'},
            'orchestrator': {'pid': None, 'status': 'stopped', 'health': 'unknown'}
        }
        
    def check_existing_processes(self):
        """Check if components are already running"""
        
        print("🔍 Checking for existing processes...")
        
        # Check meta daemon PID file
        pid_file = Path("meta_daemon.pid")
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                if self._is_process_running(pid):
                    self.components['meta_daemon']['pid'] = pid
                    self.components['meta_daemon']['status'] = 'running'
                    print(f"✅ Meta Daemon already running (PID: {pid})")
                else:
                    print("⚠️  Meta Daemon PID file exists but process not running")
                    pid_file.unlink()
            except Exception as e:
                print(f"❌ Error checking meta daemon: {e}")
                
        # Check for other processes
        try:
            result = subprocess.run(['pgrep', '-f', 'jarvis2'], capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                if pids and pids[0]:
                    self.components['jarvis2']['pid'] = int(pids[0])
                    self.components['jarvis2']['status'] = 'running'
                    print(f"✅ Jarvis2 already running (PID: {pids[0]})")
        except Exception:
            pass
            
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running"""
        try:
            os.kill(pid, 0)  # Doesn't actually kill, just checks if process exists
            return True
        except OSError:
            return False
            
    def start_meta_daemon(self) -> bool:
        """Start the meta daemon if not running"""
        
        if self.components['meta_daemon']['status'] == 'running':
            print("Meta Daemon already running")
            return True
            
        print("🚀 Starting Meta Daemon...")
        
        try:
            # Start meta daemon in background
            process = subprocess.Popen([
                'python', 'meta_daemon.py', '--background'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Give it time to start
            time.sleep(3)
            
            # Check if it's running
            if process.poll() is None:  # Still running
                self.components['meta_daemon']['pid'] = process.pid
                self.components['meta_daemon']['status'] = 'running'
                print(f"✅ Meta Daemon started (PID: {process.pid})")
                return True
            else:
                print("❌ Meta Daemon failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting Meta Daemon: {e}")
            return False
            
    def start_jarvis2(self) -> bool:
        """Start Jarvis2 if not running"""
        
        if self.components['jarvis2']['status'] == 'running':
            print("Jarvis2 already running")
            return True
            
        print("🚀 Starting Jarvis2...")
        
        jarvis_files = [
            'jarvis2/core/jarvis2.py',
            'jarvis2_complete.py',
            'jarvis2_core.py'
        ]
        
        for jarvis_file in jarvis_files:
            if Path(jarvis_file).exists():
                try:
                    process = subprocess.Popen([
                        'python', jarvis_file, '--daemon'
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    time.sleep(2)
                    
                    if process.poll() is None:
                        self.components['jarvis2']['pid'] = process.pid
                        self.components['jarvis2']['status'] = 'running'
                        print(f"✅ Jarvis2 started (PID: {process.pid})")
                        return True
                        
                except Exception as e:
                    print(f"❌ Error starting Jarvis2 from {jarvis_file}: {e}")
                    continue
                    
        print("❌ Could not start Jarvis2")
        return False
        
    def start_unified_meta_system(self) -> bool:
        """Start the unified meta system for continuous monitoring"""
        
        print("🚀 Starting Unified Meta System...")
        
        try:
            process = subprocess.Popen([
                'python', '-c', 
                '''
import asyncio
from unified_meta_system import UnifiedMetaSystem

async def run_continuous():
    with UnifiedMetaSystem() as meta:
        await meta.run_complete_loop()

asyncio.run(run_continuous())
                '''
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            time.sleep(2)
            
            if process.poll() is None:
                self.components['unified_meta']['pid'] = process.pid
                self.components['unified_meta']['status'] = 'running'
                print(f"✅ Unified Meta System started (PID: {process.pid})")
                return True
            else:
                print("❌ Unified Meta System failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting Unified Meta System: {e}")
            return False
            
    def check_health(self):
        """Check health of all running components"""
        
        print("\n🏥 Health Check...")
        
        for component, info in self.components.items():
            if info['status'] == 'running' and info['pid']:
                if self._is_process_running(info['pid']):
                    info['health'] = 'healthy'
                    print(f"✅ {component}: Healthy (PID: {info['pid']})")
                else:
                    info['health'] = 'failed'
                    info['status'] = 'stopped'
                    print(f"❌ {component}: Process died (PID: {info['pid']})")
            else:
                print(f"⚠️  {component}: Not running")
                
    def verify_functionality(self):
        """Verify that components are actually working"""
        
        print("\n🧪 Functionality Tests...")
        
        # Test meta system database only if not disabled
        import os
        if os.environ.get('DISABLE_META_AUTOSTART') != '1':
            try:
                from meta_prime import MetaPrime
                meta = MetaPrime()
                meta.observe('startup_test', {'test': True})
                print("✅ Meta system database: Working")
            except Exception as e:
                print(f"❌ Meta system database: {e}")
        else:
            print("⚠️  Meta system database: Disabled (DISABLE_META_AUTOSTART=1)")
            
        # Test trading system
        try:
            result = subprocess.run(['python', 'run.py', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Trading system: Working")
            else:
                print("❌ Trading system: Failed")
        except Exception as e:
            print(f"❌ Trading system: {e}")
            
        # Test orchestrator
        try:
            if Path('orchestrate.py').exists():
                result = subprocess.run(['python', 'orchestrate.py', '--help'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("✅ Orchestrator: Working")
                else:
                    print("❌ Orchestrator: Failed")
        except Exception as e:
            print(f"❌ Orchestrator: {e}")
            
    def create_status_file(self):
        """Create a status file for monitoring"""
        
        status = {
            'startup_time': time.time(),
            'components': self.components,
            'system_status': 'running' if any(c['status'] == 'running' for c in self.components.values()) else 'stopped'
        }
        
        with open('system_status.json', 'w') as f:
            import json
            json.dump(status, f, indent=2)
            
        print("\n📊 Status file created: system_status.json")
        
    def show_summary(self):
        """Show startup summary"""
        
        print("\n" + "="*50)
        print("🚀 SYSTEM STARTUP SUMMARY")
        print("="*50)
        
        running_count = sum(1 for c in self.components.values() if c['status'] == 'running')
        healthy_count = sum(1 for c in self.components.values() if c['health'] == 'healthy')
        
        print(f"Components running: {running_count}/{len(self.components)}")
        print(f"Components healthy: {healthy_count}/{len(self.components)}")
        
        for component, info in self.components.items():
            status_emoji = "✅" if info['status'] == 'running' else "❌"
            health_emoji = "💚" if info['health'] == 'healthy' else "❤️" if info['health'] == 'failed' else "💛"
            print(f"{status_emoji} {health_emoji} {component}: {info['status']} ({info['health']})")
            
        print("\n📋 Available Commands:")
        print("  python run.py                 - Trading recommendations")
        print("  python run.py --diagnose      - System diagnostics")
        print("  ./orchestrate '<command>'     - AI orchestrator")
        print("  python unified_meta_system.py - Manual meta system")
        print("  python check_system_status.py - Check status")
        
        print("\n🔍 Monitoring:")
        print("  tail -f meta_daemon.log       - Meta daemon logs")
        print("  cat system_status.json        - System status")
        print("  ps aux | grep python          - Running processes")
        
def main():
    """Main startup function"""
    
    print("🚀 Unity Wheel Trading - Complete System Startup")
    print("="*55)
    
    starter = SystemStarter()
    
    try:
        # Step 1: Check existing processes
        starter.check_existing_processes()
        
        # Step 2: Start components
        print("\n🚀 Starting Components...")
        starter.start_meta_daemon()
        starter.start_jarvis2()
        
        # Optional: Start unified meta system
        if '--with-unified-meta' in sys.argv:
            starter.start_unified_meta_system()
            
        # Step 3: Health check
        starter.check_health()
        
        # Step 4: Functionality tests
        starter.verify_functionality()
        
        # Step 5: Create status file
        starter.create_status_file()
        
        # Step 6: Show summary
        starter.show_summary()
        
        print("\n🎉 System startup complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Startup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        sys.exit(1)

# ALL AUTO-SPAWN COMPLETELY DISABLED FOR EINSTEIN TESTING
print("🔪 ALL auto-spawn systems DISABLED for clean Einstein testing")
exit(0)

# if __name__ == "__main__":
#     main()