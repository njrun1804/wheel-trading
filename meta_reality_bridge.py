"""
Meta Reality Bridge - Connects Meta System to Real Development Workflow
Based on Jarvis2's event-driven integration strategy with M4 parallel optimization

This bridge enables the meta system to observe and learn from actual development work
on the Unity Wheel trading codebase, not just its own self-evolution.
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import sqlite3

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

from meta_prime import MetaPrime
from meta_coordinator import MetaCoordinator
from meta_config import get_meta_config


@dataclass
class RealityEvent:
    """Represents a real development workflow event"""
    timestamp: float
    event_type: str  # 'file_edit', 'search_pattern', 'test_run', 'build'
    target_path: str
    details: Dict[str, Any]
    claude_context: Optional[str] = None


class RealCodebaseHandler(FileSystemEventHandler):
    """Watches the real trading codebase for changes"""
    
    def __init__(self, bridge: 'MetaRealityBridge'):
        self.bridge = bridge
        self.last_events = {}  # Debounce rapid fire events
        
    def on_modified(self, event):
        """Real file modified - learn from Claude's editing patterns"""
        if not event.is_directory and self._should_observe(event.src_path):
            self._debounced_handle(event.src_path, 'modified')
    
    def on_created(self, event):
        """New file created - observe Claude's creation patterns"""
        if not event.is_directory and self._should_observe(event.src_path):
            self._debounced_handle(event.src_path, 'created')
    
    def _should_observe(self, file_path: str) -> bool:
        """Check if we should observe this file change"""
        path = Path(file_path)
        
        # Watch trading codebase
        if 'src/unity_wheel' in str(path):
            return True
            
        # Watch config changes
        if path.name in ['config.yaml', 'pyproject.toml', 'requirements.txt']:
            return True
            
        # Watch Python files in root (like run.py)
        if path.suffix == '.py' and path.parent.name in ['wheel-trading', '.']:
            return True
            
        return False
    
    def _debounced_handle(self, file_path: str, change_type: str):
        """Handle file changes with debouncing to avoid spam"""
        now = time.time()
        key = f"{file_path}_{change_type}"
        
        # Debounce - only process if >configured time since last event
        if key in self.last_events and now - self.last_events[key] < self.bridge.config.timing.file_change_debounce_seconds:
            return
            
        self.last_events[key] = now
        
        # Async handle
        asyncio.create_task(self.bridge.handle_real_code_change(
            Path(file_path), change_type
        ))


class MetaRealityBridge:
    """Bridges the meta system with real development workflow"""
    
    def __init__(self):
        # Core components
        self.config = get_meta_config()
        self.meta_prime = MetaPrime()
        self.meta_coordinator = MetaCoordinator()
        
        # M4 optimization - parallel processing
        self.p_core_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="M4_P")
        self.e_core_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="M4_E")
        
        # Reality observation
        self.reality_observer = Observer()
        self.reality_handler = RealCodebaseHandler(self)
        
        # Learning database
        self.reality_db = sqlite3.connect('meta_reality_learning.db')
        self._init_reality_schema()
        
        # Pattern tracking
        self.claude_patterns = {
            'frequent_files': {},
            'editing_sequences': [],
            'performance_concerns': [],
            'common_searches': []
        }
        
        print("🌉 Meta Reality Bridge initialized")
        print("🎯 Bridging meta system with real development workflow")
        print(f"⚡ M4 optimization: {self.p_core_executor._max_workers} P-cores, {self.e_core_executor._max_workers} E-cores")
        
    def _init_reality_schema(self):
        """Initialize reality learning database"""
        
        self.reality_db.execute("""
            CREATE TABLE IF NOT EXISTS reality_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                target_path TEXT NOT NULL,
                details_json TEXT NOT NULL,
                claude_context TEXT,
                learned_pattern TEXT
            )
        """)
        
        self.reality_db.execute("""
            CREATE TABLE IF NOT EXISTS claude_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5
            )
        """)
        
        self.reality_db.commit()
    
    def start_reality_observation(self):
        """Start observing real development workflow - synchronous setup"""
        
        # Set up file watching for trading codebase
        trading_src = Path("src/unity_wheel")
        if trading_src.exists():
            self.reality_observer.schedule(
                self.reality_handler,
                path=str(trading_src),
                recursive=True
            )
            print(f"👁️ Watching real codebase: {trading_src}")
        
        # Watch root directory for config/script changes
        self.reality_observer.schedule(
            self.reality_handler,
            path=".",
            recursive=False
        )
        
        self.reality_observer.start()
        print("🚀 Reality observation started")
        
        # Record bridge activation
        self.meta_prime.observe("reality_bridge_activated", {
            "timestamp": time.time(),
            "watching_paths": ["src/unity_wheel", "."],
            "m4_cores_allocated": {
                "p_cores": 8,
                "e_cores": 4
            }
        })
    
    async def handle_real_code_change(self, file_path: Path, change_type: str):
        """Handle real codebase changes with parallel M4 processing"""
        
        start_time = time.time()
        
        # Use P-cores for intensive analysis
        analysis_future = self.p_core_executor.submit(
            self._analyze_code_change, file_path, change_type
        )
        
        # Use E-cores for background pattern learning  
        learning_future = self.e_core_executor.submit(
            self._learn_from_change, file_path, change_type
        )
        
        try:
            # Get analysis results with async wait
            await asyncio.sleep(0.001)  # Yield control to event loop
            analysis = analysis_future.result(timeout=self.config.timing.reality_bridge_timeout_seconds)
            pattern_update = learning_future.result(timeout=self.config.timing.reality_bridge_learning_timeout_seconds)
            
            # Create reality event
            reality_event = RealityEvent(
                timestamp=time.time(),
                event_type=f"real_{change_type}",
                target_path=str(file_path),
                details=analysis,
                claude_context=self._infer_claude_context(file_path, analysis)
            )
            
            # Record event
            self._record_reality_event(reality_event)
            
            # Feed back to meta system
            self._feed_to_meta_system(reality_event)
            
            processing_time = (time.time() - start_time) * 1000
            print(f"🔄 Real change processed: {file_path.name} ({processing_time:.1f}ms)")
            
        except Exception as e:
            print(f"❌ Error processing real change {file_path}: {e}")
    
    def _analyze_code_change(self, file_path: Path, change_type: str) -> Dict[str, Any]:
        """Analyze what Claude changed (P-core intensive)"""
        
        analysis = {
            "file_type": file_path.suffix,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "change_type": change_type,
            "timestamp": time.time()
        }
        
        if file_path.exists() and file_path.suffix == '.py':
            try:
                content = file_path.read_text()
                
                # Analyze code characteristics
                analysis.update({
                    "line_count": len(content.split('\n')),
                    "has_async": 'async def' in content,
                    "has_classes": 'class ' in content,
                    "has_imports": 'import ' in content or 'from ' in content,
                    "function_count": content.count('def '),
                    "comment_density": content.count('#') / max(len(content.split('\n')), 1)
                })
                
                # Check for performance-related changes
                if any(perf_keyword in content.lower() for perf_keyword in 
                       ['parallel', 'async', 'cache', 'optimize', 'performance']):
                    analysis["performance_related"] = True
                
                # Check for trading domain content
                if any(trading_keyword in content.lower() for trading_keyword in
                       ['wheel', 'option', 'strike', 'premium', 'delta', 'gamma']):
                    analysis["trading_domain"] = True
                    
            except Exception as e:
                analysis["analysis_error"] = str(e)
        
        return analysis
    
    def _learn_from_change(self, file_path: Path, change_type: str) -> Dict[str, Any]:
        """Learn patterns from Claude's changes (E-core background)"""
        
        # Update frequency tracking
        file_key = str(file_path)
        self.claude_patterns['frequent_files'][file_key] = (
            self.claude_patterns['frequent_files'].get(file_key, 0) + 1
        )
        
        # Track editing sequences
        self.claude_patterns['editing_sequences'].append({
            "file": file_key,
            "change_type": change_type,
            "timestamp": time.time()
        })
        
        # Keep only recent sequences (configurable limit)
        limit = self.config.evolution.watcher_observation_retention_count
        if len(self.claude_patterns['editing_sequences']) > limit:
            self.claude_patterns['editing_sequences'] = (
                self.claude_patterns['editing_sequences'][-limit:]
            )
        
        return {
            "pattern_updated": True,
            "total_sequences": len(self.claude_patterns['editing_sequences']),
            "file_frequency": self.claude_patterns['frequent_files'][file_key]
        }
    
    def _infer_claude_context(self, file_path: Path, analysis: Dict[str, Any]) -> str:
        """Infer what Claude was trying to accomplish"""
        
        context_clues = []
        
        # Based on file type and location
        if 'src/unity_wheel/strategy' in str(file_path):
            context_clues.append("trading_strategy_work")
        elif 'src/unity_wheel/api' in str(file_path):
            context_clues.append("api_development")
        elif 'src/unity_wheel/risk' in str(file_path):
            context_clues.append("risk_management")
        
        # Based on analysis
        if analysis.get("performance_related"):
            context_clues.append("performance_optimization")
        if analysis.get("trading_domain"):
            context_clues.append("trading_logic")
        if analysis.get("has_async"):
            context_clues.append("async_programming")
        
        return "_".join(context_clues) if context_clues else "general_development"
    
    def _record_reality_event(self, event: RealityEvent):
        """Record reality event in learning database - synchronous DB operation"""
        
        self.reality_db.execute("""
            INSERT INTO reality_events 
            (timestamp, event_type, target_path, details_json, claude_context)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event.timestamp,
            event.event_type,
            event.target_path,
            json.dumps(event.details),
            event.claude_context
        ))
        
        self.reality_db.commit()
    
    def _feed_to_meta_system(self, event: RealityEvent):
        """Feed reality event to meta system for learning - synchronous processing"""
        
        # Convert to meta system observation
        self.meta_prime.observe("reality_event", {
            "event_type": event.event_type,
            "target_path": event.target_path,
            "claude_context": event.claude_context,
            "details": event.details
        })
        
        # Trigger meta coordination if significant event
        if self._is_significant_event(event):
            self.meta_coordinator.record_coordination_event(
                "reality_significant_change",
                "MetaRealityBridge",
                f"Significant real code change: {event.claude_context}",
                "high"
            )
    
    def _is_significant_event(self, event: RealityEvent) -> bool:
        """Determine if event is significant enough for meta attention"""
        
        # Performance-related changes are significant
        if event.details.get("performance_related"):
            return True
            
        # New files are significant
        if event.event_type == "real_created":
            return True
            
        # Large files are significant
        if event.details.get("line_count", 0) > self.config.evolution.major_file_change_threshold:
            return True
            
        # Trading strategy files are always significant
        if "strategy" in event.target_path.lower():
            return True
            
        return False
    
    def get_claude_learning_report(self) -> Dict[str, Any]:
        """Generate report on Claude's development patterns"""
        
        # Most frequent files
        frequent_files = sorted(
            self.claude_patterns['frequent_files'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Recent activity
        recent_sequences = self.claude_patterns['editing_sequences'][-10:]
        
        # File type distribution
        file_types = {}
        for file_path, freq in self.claude_patterns['frequent_files'].items():
            ext = Path(file_path).suffix
            file_types[ext] = file_types.get(ext, 0) + freq
        
        return {
            "most_edited_files": frequent_files,
            "recent_editing_sequence": recent_sequences,
            "file_type_distribution": file_types,
            "total_events": len(self.claude_patterns['editing_sequences']),
            "learning_database_size": self._get_db_size()
        }
    
    def _get_db_size(self) -> int:
        """Get size of reality learning database"""
        cursor = self.reality_db.execute("SELECT COUNT(*) FROM reality_events")
        return cursor.fetchone()[0]
    
    def shutdown(self):
        """Shutdown reality bridge gracefully - synchronous cleanup"""
        
        if self.reality_observer.is_alive():
            self.reality_observer.stop()
            self.reality_observer.join()
        
        self.p_core_executor.shutdown(wait=True)
        self.e_core_executor.shutdown(wait=True)
        
        self.reality_db.close()
        
        print("🌉 Meta Reality Bridge shutdown complete")


if __name__ == "__main__":
    async def test_reality_bridge():
        """Test the reality bridge"""
        
        bridge = MetaRealityBridge()
        
        try:
            bridge.start_reality_observation()
            
            print("🧪 Reality bridge active - edit some files to see it learn!")
            print("📊 Learning from real development workflow...")
            
            # Let it observe for a bit
            await asyncio.sleep(30)
            
            # Show learning report
            report = bridge.get_claude_learning_report()
            print(f"\n📈 Claude Learning Report:")
            print(f"  Total events observed: {report['total_events']}")
            print(f"  Database size: {report['learning_database_size']}")
            print(f"  Most edited files: {report['most_edited_files'][:3]}")
            
        finally:
            bridge.shutdown()
    
    asyncio.run(test_reality_bridge())