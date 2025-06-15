#!/usr/bin/env python3
"""
Claude CLI Stream Monitor
Monitors Claude Code CLI output for thinking deltas - zero cost approach
"""

import json
import re
import subprocess
import threading
import time
import queue
import os
import shutil
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from meta_prime import MetaPrime
from claude_cli_reasoning_capture import ClaudeCLIReasoningCapture


@dataclass
class CLIThinkingDelta:
    """Thinking delta captured from CLI stream"""
    timestamp: float
    delta_type: str
    content: str
    partial: bool = False
    session_id: str = ""


class ClaudeCLIStreamMonitor:
    """Monitors Claude CLI output for thinking deltas"""
    
    def __init__(self):
        self.meta_prime = MetaPrime()
        self.cli_capture = ClaudeCLIReasoningCapture()
        
        # CLI monitoring state
        self.monitoring = False
        self.process = None
        self.event_queue = queue.Queue()
        self.thinking_deltas = []
        
        # Regex patterns for parsing CLI output
        self.think_re = re.compile(r'^data: (\{.*?\})$')
        self.ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        
        # Find Claude CLI
        self.claude_cli = shutil.which("claude") or "claude"
        
        print("🎯 Claude CLI Stream Monitor initialized")
        print(f"🔧 Using Claude CLI: {self.claude_cli}")
    
    def start_monitoring(self, prompt: str = "Monitor thinking", model: str = "claude-3-5-haiku"):
        """Start monitoring Claude CLI for thinking deltas"""
        
        if self.monitoring:
            print("⚠️ Already monitoring")
            return
        
        print(f"🔄 Starting CLI monitoring with prompt: '{prompt}'")
        
        try:
            # Spawn Claude CLI with stream-json output
            self.process = subprocess.Popen([
                self.claude_cli, "chat",
                "--output-format", "stream-json",
                "--thinking", "enabled", 
                "--model", model
            ], 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
            )
            
            # Start monitoring threads
            self.monitoring = True
            
            # Thread to read CLI output
            threading.Thread(
                target=self._stream_reader, 
                args=(self.process.stdout,), 
                daemon=True
            ).start()
            
            # Thread to process events
            threading.Thread(
                target=self._event_processor, 
                daemon=True
            ).start()
            
            # Send the prompt
            self.process.stdin.write(f"{prompt}\n")
            self.process.stdin.flush()
            
            print("✅ CLI monitoring started")
            
        except Exception as e:
            print(f"❌ Failed to start CLI monitoring: {e}")
            self.monitoring = False
    
    def _stream_reader(self, stdout):
        """Read and parse CLI output stream"""
        
        buffer = ""
        
        for line in stdout:
            try:
                # Clean ANSI escape codes
                clean_line = self.ansi_escape.sub('', line)
                
                # Look for thinking delta events
                match = self.think_re.match(clean_line.strip())
                if match:
                    try:
                        event_data = json.loads(match.group(1))
                        self.event_queue.put(event_data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON decode error: {e}")
                        continue
                
                # Also capture regular output for context
                elif clean_line.strip():
                    # This is regular Claude output
                    self.event_queue.put({
                        "type": "content",
                        "content": clean_line.strip(),
                        "timestamp": time.time()
                    })
                    
            except Exception as e:
                print(f"⚠️ Stream reader error: {e}")
                continue
    
    def _event_processor(self):
        """Process events from the CLI stream"""
        
        session_id = f"cli_monitor_{int(time.time())}"
        
        while self.monitoring:
            try:
                # Get event with timeout
                event = self.event_queue.get(timeout=1.0)
                
                if event.get("type") == "thinking_delta":
                    # Process thinking delta
                    self._handle_thinking_delta(event, session_id)
                    
                elif event.get("type") == "content":
                    # Process regular content
                    self._handle_content(event, session_id)
                    
                elif event.get("type") == "message_start":
                    print("📝 Claude message started")
                    
                elif event.get("type") == "message_stop":
                    print("✅ Claude message completed")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Event processor error: {e}")
                continue
    
    def _handle_thinking_delta(self, event: Dict[str, Any], session_id: str):
        """Handle thinking delta from CLI stream"""
        
        thinking_content = event.get("delta", {}).get("text", "") or event.get("text", "")
        
        if thinking_content:
            # Create thinking delta
            delta = CLIThinkingDelta(
                timestamp=time.time(),
                delta_type="thinking",
                content=thinking_content,
                partial=event.get("partial", False),
                session_id=session_id
            )
            
            self.thinking_deltas.append(delta)
            
            # Record in meta system
            self.meta_prime.observe("cli_thinking_delta_captured", {
                "session_id": session_id,
                "content_length": len(thinking_content),
                "partial": delta.partial,
                "total_deltas": len(self.thinking_deltas)
            })
            
            # Integrate with existing CLI capture
            self.cli_capture.capture_claude_analysis(
                thinking_content, 
                {"source": "cli_stream", "delta_type": "thinking"}
            )
            
            print(f"🧠 Thinking delta: {thinking_content[:50]}...")
    
    def _handle_content(self, event: Dict[str, Any], session_id: str):
        """Handle regular content from CLI stream"""
        
        content = event.get("content", "")
        
        if content and len(content.strip()) > 10:
            # This is substantive Claude output
            self.cli_capture.capture_claude_analysis(
                content,
                {"source": "cli_stream", "delta_type": "content"}
            )
    
    def stop_monitoring(self):
        """Stop CLI monitoring"""
        
        if not self.monitoring:
            return
        
        print("🛑 Stopping CLI monitoring...")
        
        self.monitoring = False
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                print(f"⚠️ Error stopping process: {e}")
        
        # Save captured session
        if self.thinking_deltas:
            session_file = self.cli_capture.save_session_data()
            print(f"💾 Captured session saved: {session_file}")
        
        print(f"✅ Monitoring stopped - captured {len(self.thinking_deltas)} thinking deltas")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        
        return {
            "monitoring_active": self.monitoring,
            "thinking_deltas_captured": len(self.thinking_deltas),
            "cli_events_captured": len(self.cli_capture.captured_events),
            "session_active": self.process is not None and self.process.poll() is None
        }


def test_cli_monitoring():
    """Test CLI monitoring functionality"""
    
    print("🧪 TESTING CLAUDE CLI STREAM MONITORING")
    print("=" * 60)
    
    monitor = ClaudeCLIStreamMonitor()
    
    # Test CLI availability
    try:
        result = subprocess.run([monitor.claude_cli, "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Claude CLI available: {result.stdout.strip()}")
        else:
            print(f"❌ Claude CLI error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Cannot find Claude CLI: {e}")
        return False
    
    # Start monitoring with a test prompt
    monitor.start_monitoring("Think about the number 42 briefly, then say hello", "claude-3-5-haiku")
    
    # Monitor for 15 seconds
    print("⏰ Monitoring for 15 seconds...")
    time.sleep(15)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Show results
    stats = monitor.get_monitoring_stats()
    print(f"\n📊 MONITORING RESULTS:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Show captured thinking deltas
    if monitor.thinking_deltas:
        print(f"\n🧠 CAPTURED THINKING DELTAS:")
        for i, delta in enumerate(monitor.thinking_deltas[:3], 1):
            print(f"   {i}. {delta.content[:80]}...")
    
    success = len(monitor.thinking_deltas) > 0
    print(f"\n🎯 Test result: {'SUCCESS' if success else 'NO_DELTAS'}")
    
    return success


# Production CLI monitoring daemon
class ProductionCLIMonitor:
    """Production daemon for continuous CLI monitoring"""
    
    def __init__(self):
        self.monitor = ClaudeCLIStreamMonitor()
        self.running = False
        
    def start_daemon(self):
        """Start production monitoring daemon"""
        
        print("🚀 Starting production CLI monitoring daemon...")
        
        self.running = True
        
        # Monitor in a loop, restarting on failures
        while self.running:
            try:
                # Start monitoring
                self.monitor.start_monitoring(
                    "Monitor this session for learning", 
                    "claude-3-5-haiku"
                )
                
                # Keep monitoring while active
                while self.running and self.monitor.monitoring:
                    time.sleep(10)
                    
                    # Check if process died
                    if (self.monitor.process and 
                        self.monitor.process.poll() is not None):
                        print("⚠️ CLI process died, restarting...")
                        break
                
                # Stop current monitoring
                self.monitor.stop_monitoring()
                
                if self.running:
                    print("🔄 Restarting monitoring in 5 seconds...")
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                print("\n👋 Shutting down daemon...")
                self.running = False
            except Exception as e:
                print(f"❌ Daemon error: {e}")
                time.sleep(10)  # Wait before retry
        
        print("✅ Daemon stopped")
    
    def stop_daemon(self):
        """Stop the daemon"""
        self.running = False
        if self.monitor:
            self.monitor.stop_monitoring()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "daemon":
        # Run as daemon
        daemon = ProductionCLIMonitor()
        daemon.start_daemon()
    else:
        # Run test
        test_cli_monitoring()