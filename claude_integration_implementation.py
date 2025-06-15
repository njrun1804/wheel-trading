#!/usr/bin/env python3
"""
Claude Integration Implementation - Hybrid Approach
Combines File System Bridge + Process Detection for immediate implementation
"""

import asyncio
import os
import time
import json
import psutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from meta_prime import MetaPrime
from meta_coordinator import MetaCoordinator
from meta_reality_bridge import MetaRealityBridge


@dataclass
class ClaudeSession:
    """Represents an active Claude Code session"""
    process_id: int
    start_time: float
    working_directory: str
    files_accessed: List[str]
    commands_executed: List[str]
    thinking_patterns: List[Dict[str, Any]]
    user_interactions: List[Dict[str, Any]]


@dataclass 
class ClaudeThoughtPattern:
    """Detected pattern in Claude's behavior"""
    pattern_type: str  # 'file_sequence', 'search_pattern', 'edit_style', 'problem_solving'
    confidence: float
    evidence: List[str]
    prediction: str
    context: Dict[str, Any]


class ClaudeProcessMonitor:
    """Monitors Claude Code processes and behavior patterns"""
    
    def __init__(self):
        self.meta_prime = MetaPrime()
        self.active_sessions: Dict[int, ClaudeSession] = {}
        self.claude_processes = []
        self.monitoring_active = False
        
        print("🤖 Claude Process Monitor initialized")
        
    async def start_claude_monitoring(self):
        """Start monitoring Claude Code processes"""
        
        self.monitoring_active = True
        
        print("🔍 Starting Claude Code process detection...")
        
        while self.monitoring_active:
            try:
                # Detect Claude Code processes
                claude_processes = self._detect_claude_processes()
                
                for proc in claude_processes:
                    if proc.pid not in self.active_sessions:
                        await self._start_session_monitoring(proc)
                
                # Clean up terminated sessions
                terminated_pids = []
                for pid in self.active_sessions:
                    if not psutil.pid_exists(pid):
                        await self._end_session_monitoring(pid)
                        terminated_pids.append(pid)
                
                for pid in terminated_pids:
                    del self.active_sessions[pid]
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                print(f"Error in Claude monitoring: {e}")
                await asyncio.sleep(5)
    
    def _detect_claude_processes(self) -> List[psutil.Process]:
        """Detect running Claude Code processes"""
        
        claude_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
            try:
                proc_info = proc.info
                
                # Look for Claude-related process names
                if proc_info['name'] and any(claude_indicator in proc_info['name'].lower() 
                                           for claude_indicator in ['claude', 'anthropic']):
                    claude_processes.append(proc)
                
                # Look for Claude in command line
                elif proc_info['cmdline'] and any('claude' in cmd.lower() 
                                                for cmd in proc_info['cmdline']):
                    claude_processes.append(proc)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return claude_processes
    
    async def _start_session_monitoring(self, process: psutil.Process):
        """Start monitoring a new Claude session"""
        
        try:
            session = ClaudeSession(
                process_id=process.pid,
                start_time=time.time(),
                working_directory=process.cwd() if hasattr(process, 'cwd') else os.getcwd(),
                files_accessed=[],
                commands_executed=[], 
                thinking_patterns=[],
                user_interactions=[]
            )
            
            self.active_sessions[process.pid] = session
            
            self.meta_prime.observe("claude_session_started", {
                "pid": process.pid,
                "working_directory": session.working_directory,
                "process_name": process.name(),
                "start_time": session.start_time
            })
            
            print(f"🔍 Started monitoring Claude session: PID {process.pid}")
            
        except Exception as e:
            print(f"Error starting session monitoring: {e}")
    
    async def _end_session_monitoring(self, pid: int):
        """End monitoring for a terminated session"""
        
        if pid in self.active_sessions:
            session = self.active_sessions[pid]
            duration = time.time() - session.start_time
            
            self.meta_prime.observe("claude_session_ended", {
                "pid": pid,
                "duration_seconds": duration,
                "files_accessed_count": len(session.files_accessed),
                "commands_executed_count": len(session.commands_executed),
                "patterns_detected": len(session.thinking_patterns)
            })
            
            print(f"📊 Claude session ended: PID {pid}, Duration: {duration:.1f}s")


class ClaudeFileSystemMonitor(FileSystemEventHandler):
    """Enhanced file system monitoring focused on Claude's behavior"""
    
    def __init__(self, claude_monitor: ClaudeProcessMonitor):
        self.claude_monitor = claude_monitor
        self.meta_prime = MetaPrime()
        self.file_access_patterns = {}
        self.edit_sequences = []
        
    def on_modified(self, event):
        """Track file modifications that might be from Claude"""
        
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Only track relevant file types
        if not self._is_relevant_file(file_path):
            return
            
        # Try to correlate with active Claude sessions
        claude_session = self._get_likely_claude_session(file_path)
        
        if claude_session:
            asyncio.create_task(self._analyze_claude_file_modification(file_path, claude_session))
    
    def _is_relevant_file(self, file_path: Path) -> bool:
        """Check if file is relevant for Claude monitoring"""
        
        relevant_extensions = {'.py', '.js', '.ts', '.md', '.txt', '.json', '.yaml', '.yml'}
        return file_path.suffix.lower() in relevant_extensions
    
    def _get_likely_claude_session(self, file_path: Path) -> Optional[ClaudeSession]:
        """Try to identify which Claude session modified this file"""
        
        # Simple heuristic: file is in working directory of a Claude session
        for session in self.claude_monitor.active_sessions.values():
            if str(file_path).startswith(session.working_directory):
                return session
                
        return None
    
    async def _analyze_claude_file_modification(self, file_path: Path, session: ClaudeSession):
        """Analyze Claude's file modification for thinking patterns"""
        
        try:
            # Read file content to analyze changes
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Detect thinking patterns
            patterns = await self._detect_thinking_patterns(content, file_path)
            
            for pattern in patterns:
                session.thinking_patterns.append({
                    "timestamp": time.time(),
                    "file": str(file_path),
                    "pattern": pattern
                })
                
                self.meta_prime.observe("claude_thinking_pattern", {
                    "session_pid": session.process_id,
                    "file_path": str(file_path),
                    "pattern_type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "evidence": pattern.evidence[:3],  # First 3 pieces of evidence
                    "prediction": pattern.prediction
                })
            
            # Track file in session
            if str(file_path) not in session.files_accessed:
                session.files_accessed.append(str(file_path))
                
        except Exception as e:
            print(f"Error analyzing Claude file modification: {e}")
    
    async def _detect_thinking_patterns(self, content: str, file_path: Path) -> List[ClaudeThoughtPattern]:
        """Detect Claude's thinking patterns from file content"""
        
        patterns = []
        
        # Pattern 1: TODO/FIXME comments indicate planning
        if any(indicator in content for indicator in ['TODO', 'FIXME', 'NOTE', 'HACK']):
            todo_lines = [line for line in content.split('\n') if any(indicator in line for indicator in ['TODO', 'FIXME', 'NOTE'])]
            
            patterns.append(ClaudeThoughtPattern(
                pattern_type="planning_comments",
                confidence=0.8,
                evidence=todo_lines[:3],
                prediction="Claude is planning future improvements",
                context={"file": str(file_path), "comment_count": len(todo_lines)}
            ))
        
        # Pattern 2: Function/class structure indicates problem decomposition  
        if content.count('def ') > 2 or content.count('class ') > 1:
            patterns.append(ClaudeThoughtPattern(
                pattern_type="problem_decomposition",
                confidence=0.7,
                evidence=[f"Functions: {content.count('def ')}", f"Classes: {content.count('class ')}"],
                prediction="Claude is breaking down problem into components",
                context={"file": str(file_path), "structure_complexity": "high"}
            ))
        
        # Pattern 3: Import statements indicate dependency thinking
        import_lines = [line for line in content.split('\n') if line.strip().startswith(('import ', 'from '))]
        if len(import_lines) > 5:
            patterns.append(ClaudeThoughtPattern(
                pattern_type="dependency_analysis", 
                confidence=0.6,
                evidence=import_lines[:3],
                prediction="Claude is considering external dependencies",
                context={"file": str(file_path), "import_count": len(import_lines)}
            ))
        
        return patterns


class ClaudeIntegrationSystem:
    """Main Claude integration system combining all monitoring approaches"""
    
    def __init__(self):
        self.meta_prime = MetaPrime()
        self.meta_coordinator = MetaCoordinator()
        self.process_monitor = ClaudeProcessMonitor()
        self.file_observer = Observer()
        self.file_handler = ClaudeFileSystemMonitor(self.process_monitor)
        
        print("🚀 Claude Integration System initialized")
        print("📊 Combining process monitoring + file system analysis")
        
    async def start_integration(self):
        """Start comprehensive Claude integration monitoring"""
        
        print("🔄 Starting Claude integration monitoring...")
        
        # Start file system monitoring
        watch_path = Path.cwd()
        self.file_observer.schedule(self.file_handler, str(watch_path), recursive=True)
        self.file_observer.start()
        
        self.meta_prime.observe("claude_integration_started", {
            "integration_type": "hybrid_process_file_monitoring",
            "watch_path": str(watch_path),
            "monitoring_capabilities": [
                "process_detection",
                "file_modification_tracking", 
                "thinking_pattern_analysis",
                "session_correlation"
            ]
        })
        
        # Start process monitoring (runs continuously)
        await self.process_monitor.start_claude_monitoring()
    
    async def get_claude_insights(self) -> Dict[str, Any]:
        """Get current insights about Claude's behavior"""
        
        insights = {
            "active_sessions": len(self.process_monitor.active_sessions),
            "total_patterns_detected": 0,
            "recent_thinking_patterns": [],
            "file_access_summary": {}
        }
        
        for session in self.process_monitor.active_sessions.values():
            insights["total_patterns_detected"] += len(session.thinking_patterns)
            
            # Get recent patterns
            recent_patterns = session.thinking_patterns[-5:]  # Last 5 patterns
            for pattern in recent_patterns:
                insights["recent_thinking_patterns"].append({
                    "type": pattern["pattern"]["pattern_type"],
                    "confidence": pattern["pattern"]["confidence"],
                    "prediction": pattern["pattern"]["prediction"]
                })
        
        return insights
    
    def stop_integration(self):
        """Stop Claude integration monitoring"""
        
        self.process_monitor.monitoring_active = False
        self.file_observer.stop()
        
        print("🛑 Claude integration monitoring stopped")


async def main():
    """Main integration function"""
    
    print("🎯 CLAUDE INTEGRATION - HYBRID IMPLEMENTATION")
    print("=" * 60)
    print("Strategy: Process Monitoring + Enhanced File System Analysis")
    print("Advantage: Immediate implementation with existing capabilities")
    print()
    
    integration_system = ClaudeIntegrationSystem()
    
    try:
        # Start integration monitoring
        await integration_system.start_integration()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping Claude integration...")
        integration_system.stop_integration()


if __name__ == "__main__":
    asyncio.run(main())