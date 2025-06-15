#!/usr/bin/env python3
"""
Claude Code Integration - Actually monitor Claude's edits and decisions
This hooks into Claude Code's operations to provide real-time feedback
"""

import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from meta_prime import MetaPrime
from meta_coordinator import MetaCoordinator


class ClaudeCodeMonitor(FileSystemEventHandler):
    """Monitors Claude Code's actual file modifications in real-time"""
    
    def __init__(self):
        self.meta_prime = MetaPrime()
        self.meta_coordinator = MetaCoordinator()
        self.claude_session_start = time.time()
        self.files_modified = []
        self.last_modification_time = {}
        
        print("🔍 Claude Code Monitor Active - Watching your edits...")
        
    def on_modified(self, event):
        """Triggered when Claude Code modifies a file"""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Only monitor files Claude Code typically edits
        if self._is_claude_relevant_file(file_path):
            self._record_claude_modification(file_path)
            
    def _is_claude_relevant_file(self, file_path: Path) -> bool:
        """Check if this is a file Claude Code would edit"""
        
        # Python files in the project
        if file_path.suffix == '.py' and 'src' in str(file_path):
            return True
            
        # Configuration files
        if file_path.name in ['config.yaml', 'pyproject.toml', 'requirements.txt']:
            return True
            
        # Meta system files
        if file_path.name.startswith('meta_'):
            return True
            
        return False
        
    def _record_claude_modification(self, file_path: Path):
        """Record Claude Code's modification with context analysis"""
        
        now = time.time()
        
        # Debounce rapid modifications
        if file_path in self.last_modification_time:
            if now - self.last_modification_time[file_path] < 2.0:
                return
                
        self.last_modification_time[file_path] = now
        
        # Analyze the modification
        analysis = self._analyze_claude_change(file_path)
        
        # Record observation
        self.meta_prime.observe("claude_code_edit", {
            "file_path": str(file_path),
            "timestamp": now,
            "session_elapsed": now - self.claude_session_start,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "analysis": analysis,
            "edit_sequence": len(self.files_modified)
        })
        
        self.files_modified.append(str(file_path))
        
        print(f"📝 Claude edited: {file_path.name} ({analysis['change_type']})")
        
        # Check if this should trigger evolution
        if self._should_trigger_evolution():
            self._trigger_meta_evolution()
            
    def _analyze_claude_change(self, file_path: Path) -> Dict[str, Any]:
        """Analyze what type of change Claude made"""
        
        try:
            if not file_path.exists():
                return {"change_type": "file_deleted", "confidence": "high"}
                
            content = file_path.read_text()
            
            # Detect change patterns
            analysis = {
                "change_type": "modification",
                "has_new_functions": "def " in content,
                "has_imports": "import " in content,
                "has_comments": "#" in content,
                "line_count": len(content.split('\n')),
                "confidence": "medium"
            }
            
            # More specific analysis
            if "class " in content and file_path.name not in self.files_modified:
                analysis["change_type"] = "new_class_added"
                analysis["confidence"] = "high"
            elif "def " in content and "test_" in content:
                analysis["change_type"] = "test_added"
                analysis["confidence"] = "high"
            elif "import " in content:
                analysis["change_type"] = "dependency_change"
                analysis["confidence"] = "medium"
            elif file_path.suffix == '.py':
                analysis["change_type"] = "code_modification"
                analysis["confidence"] = "medium"
                
            return analysis
            
        except Exception as e:
            return {"change_type": "analysis_failed", "error": str(e), "confidence": "low"}
            
    def _should_trigger_evolution(self) -> bool:
        """Determine if Claude's changes should trigger meta evolution"""
        
        # Trigger evolution after significant Claude activity
        return (
            len(self.files_modified) >= 3 or  # Multiple files modified
            time.time() - self.claude_session_start > 300  # 5+ minutes of activity
        )
        
    def _trigger_meta_evolution(self):
        """Trigger meta system evolution based on Claude's changes"""
        
        print("🧬 Triggering meta evolution based on Claude Code activity...")
        
        # Create evolution context
        claude_context = {
            "files_modified_count": len(self.files_modified),
            "session_duration": time.time() - self.claude_session_start,
            "recent_files": self.files_modified[-5:],  # Last 5 files
            "trigger_reason": "claude_code_activity"
        }
        
        # Record the evolution trigger
        self.meta_prime.observe("evolution_triggered_by_claude", claude_context)
        
        # Check if system is ready to evolve
        if self.meta_prime.should_evolve():
            success = self.meta_prime.evolve()
            print(f"🚀 Meta evolution {'successful' if success else 'failed'}")
        else:
            print("⏳ Meta system not ready to evolve yet")


class ClaudeCodeFeedbackCollector:
    """Collects feedback about Claude Code's effectiveness"""
    
    def __init__(self):
        self.meta_prime = MetaPrime()
        self.feedback_history = []
        
    def prompt_for_feedback(self, context: str) -> Dict[str, Any]:
        """Prompt user for feedback about Claude's recent changes"""
        
        print(f"\n🤔 Quick feedback about Claude Code's {context}:")
        print("  1. Helpful (h)")
        print("  2. Partially helpful (p)")  
        print("  3. Not helpful (n)")
        print("  4. Made things worse (w)")
        print("  5. Skip feedback (s)")
        
        try:
            response = input("Your feedback: ").lower().strip()
            
            feedback_map = {
                'h': {'rating': 5, 'label': 'helpful'},
                'p': {'rating': 3, 'label': 'partially_helpful'},
                'n': {'rating': 2, 'label': 'not_helpful'},
                'w': {'rating': 1, 'label': 'made_worse'},
                's': {'rating': None, 'label': 'skipped'}
            }
            
            if response in feedback_map:
                feedback = feedback_map[response]
                
                if feedback['rating'] is not None:
                    self.meta_prime.observe("claude_code_feedback", {
                        "context": context,
                        "rating": feedback['rating'],
                        "label": feedback['label'],
                        "timestamp": time.time()
                    })
                    
                    self.feedback_history.append(feedback)
                    print(f"✅ Feedback recorded: {feedback['label']}")
                    
                return feedback
            else:
                print("❓ Invalid input, skipping feedback")
                return {'rating': None, 'label': 'invalid_input'}
                
        except KeyboardInterrupt:
            print("\n⏭️  Feedback skipped")
            return {'rating': None, 'label': 'interrupted'}
            
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback collected"""
        
        if not self.feedback_history:
            return {"average_rating": 0, "total_feedback": 0}
            
        ratings = [f['rating'] for f in self.feedback_history if f['rating'] is not None]
        
        return {
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "total_feedback": len(ratings),
            "recent_trend": ratings[-5:] if len(ratings) >= 5 else ratings
        }


def start_claude_code_monitoring():
    """Start monitoring Claude Code's operations"""
    
    print("🔄 Starting Claude Code Integration Monitor...")
    
    monitor = ClaudeCodeMonitor()
    feedback_collector = ClaudeCodeFeedbackCollector()
    
    # Set up file system watching
    observer = Observer()
    observer.schedule(monitor, path=".", recursive=True)
    observer.start()
    
    try:
        print("✅ Monitoring active. Claude Code edits will be tracked automatically.")
        print("   Press Ctrl+C to stop monitoring")
        
        while True:
            time.sleep(60)  # Check every minute
            
            # Periodically prompt for feedback if there's been activity
            if len(monitor.files_modified) > 0 and len(monitor.files_modified) % 5 == 0:
                feedback_collector.prompt_for_feedback("recent changes")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping Claude Code monitoring...")
        observer.stop()
        
    observer.join()
    
    # Final summary
    summary = feedback_collector.get_feedback_summary()
    print(f"\n📊 Session Summary:")
    print(f"   Files modified: {len(monitor.files_modified)}")
    print(f"   Average feedback: {summary['average_rating']:.1f}/5")
    print(f"   Total feedback: {summary['total_feedback']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--feedback-only":
        # Just collect feedback without monitoring
        collector = ClaudeCodeFeedbackCollector()
        collector.prompt_for_feedback("manual session")
    else:
        # Start full monitoring
        start_claude_code_monitoring()