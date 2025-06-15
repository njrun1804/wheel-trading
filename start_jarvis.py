#!/usr/bin/env python3
"""
Start Jarvis2 as a background daemon
"""

import subprocess
import sys
import time
from pathlib import Path

def start_jarvis_daemon():
    """Start Jarvis2 as a background daemon"""
    
    print("🚀 Starting Jarvis2 daemon...")
    
    # Start jarvis2_complete.py in background
    try:
        process = subprocess.Popen([
            'python', 'jarvis2_complete.py', '--daemon'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(2)
        
        # Check if it's still running
        if process.poll() is None:
            print(f"✅ Jarvis2 started successfully (PID: {process.pid})")
            
            # Save PID for monitoring
            with open('jarvis2.pid', 'w') as f:
                f.write(str(process.pid))
                
            return True
        else:
            print("❌ Jarvis2 failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Jarvis2: {e}")
        return False

if __name__ == "__main__":
    start_jarvis_daemon()