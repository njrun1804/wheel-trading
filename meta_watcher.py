"""
MetaWatcher - File Watching Component

This component watches the meta system files for changes and records observations.
It's designed to be the sensory system of the meta organism.

Design Decision: Separate watcher component
Rationale: Allows meta_prime.py to focus on core logic while this handles I/O observation
Alternative: Built into MetaPrime
Prediction: Will enable specialized observation strategies without cluttering core
"""

import asyncio
import time
import sqlite3
import hashlib
from pathlib import Path
from typing import Set, Dict, Any
from watchfiles import awatch


class MetaWatcher:
    """Watches meta system files and records changes"""
    
    def __init__(self, meta_db_path: str = None):
        if meta_db_path is None:
            from meta_config import get_meta_config
            config = get_meta_config()
            meta_db_path = config.database.evolution_db
        self.db = sqlite3.connect(meta_db_path)
        self.watched_files: Set[Path] = set()
        self.file_hashes: Dict[Path, str] = {}
        self.birth_time = time.time()
        
        print(f"👁️  MetaWatcher initialized at {time.ctime(self.birth_time)}")
        
    def add_file_to_watch(self, file_path: Path):
        """Add a file to the watch list"""
        self.watched_files.add(file_path)
        self.file_hashes[file_path] = self._get_file_hash(file_path)
        
        self._record_observation("file_watch_started", {
            "file_path": str(file_path),
            "initial_hash": self.file_hashes[file_path],
            "file_size": file_path.stat().st_size if file_path.exists() else 0
        })
        
        print(f"📁 Watching: {file_path}")
        
    def _get_file_hash(self, file_path: Path) -> str:
        """Get SHA-256 hash of file content"""
        try:
            if not file_path.exists():
                return "nonexistent"
            content = file_path.read_text()
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception as e:
            return f"error_{str(e)[:8]}"
            
    def _record_observation(self, event_type: str, details: Dict[str, Any]):
        """Record an observation in the database"""
        import json
        
        self.db.execute("""
            INSERT INTO observations (timestamp, event_type, details, file_hash, context)
            VALUES (?, ?, ?, ?, ?)
        """, (time.time(), event_type, json.dumps(details, default=str), 
              "watcher", "MetaWatcher"))
        
        self.db.commit()
        
    def analyze_file_change(self, file_path: Path) -> Dict[str, Any]:
        """Analyze what changed in a file"""
        
        old_hash = self.file_hashes.get(file_path, "unknown")
        new_hash = self._get_file_hash(file_path)
        
        try:
            old_size = self.file_hashes.get(f"{file_path}_size", 0)
            new_size = file_path.stat().st_size if file_path.exists() else 0
            size_delta = new_size - old_size
        except Exception as e:
            print(f"Warning: Could not get file size for {file_path}: {e}")
            size_delta = 0
            new_size = 0
            
        # Simple change classification
        from meta_config import get_meta_config
        config = get_meta_config()
        
        if new_hash == "nonexistent":
            change_type = "file_deleted"
        elif old_hash == "unknown":
            change_type = "file_created"
        elif size_delta > config.evolution.major_file_change_threshold:
            change_type = "major_addition"
        elif size_delta < -config.evolution.major_file_change_threshold:
            change_type = "major_deletion"
        elif size_delta > 0:
            change_type = "minor_addition"
        elif size_delta < 0:
            change_type = "minor_deletion"
        else:
            change_type = "modification"
            
        return {
            "change_type": change_type,
            "old_hash": old_hash,
            "new_hash": new_hash,
            "size_delta": size_delta,
            "new_size": new_size,
            "analysis_timestamp": time.time()
        }
        
    def handle_file_change(self, file_path: Path):
        """Handle a detected file change"""
        
        change_analysis = self.analyze_file_change(file_path)
        
        # Record the change
        self._record_observation("file_modified", {
            "file_path": str(file_path),
            "timestamp": time.time(),
            **change_analysis
        })
        
        # Update our tracking
        self.file_hashes[file_path] = change_analysis["new_hash"]
        self.file_hashes[f"{file_path}_size"] = change_analysis["new_size"]
        
        # Print immediate feedback
        change_type = change_analysis["change_type"]
        size_delta = change_analysis["size_delta"]
        
        if size_delta > 0:
            emoji = "📈"
        elif size_delta < 0:
            emoji = "📉"
        else:
            emoji = "✏️"
            
        print(f"{emoji} {file_path.name}: {change_type} ({size_delta:+d} bytes)")
        
        # Detect patterns
        self.detect_change_patterns(file_path, change_analysis)
        
    def detect_change_patterns(self, file_path: Path, change_analysis: Dict[str, Any]):
        """Look for patterns in file changes"""
        
        # Get recent changes to this file
        from meta_config import get_meta_config
        config = get_meta_config()
        cursor = self.db.execute("""
            SELECT details, timestamp
            FROM observations 
            WHERE event_type = 'file_modified' 
            AND details LIKE ?
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (f'%{file_path.name}%', config.evolution.watcher_recent_changes_limit))
        
        recent_changes = cursor.fetchall()
        
        if len(recent_changes) >= 3:
            # Check for rapid iteration pattern
            import json
            timestamps = [json.loads(change[0])["timestamp"] for change in recent_changes[:3]]
            intervals = [timestamps[i] - timestamps[i+1] for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            
            from meta_config import get_meta_config
            config = get_meta_config()
            if avg_interval < config.timing.watcher_rapid_iteration_threshold_seconds:
                self._record_observation("pattern_detected", {
                    "pattern_type": "rapid_iteration",
                    "file_path": str(file_path),
                    "change_count": len(recent_changes),
                    "average_interval_seconds": avg_interval,
                    "interpretation": "active_development_session"
                })
                print(f"🔄 Pattern detected: Rapid iteration on {file_path.name}")
                
    async def watch_files(self):
        """Main watching loop"""
        
        if not self.watched_files:
            print("⚠️  No files to watch. Add files with add_file_to_watch()")
            return
            
        # Convert Path objects to strings for watchfiles
        watch_paths = [str(path.parent) for path in self.watched_files]
        
        print(f"🚀 Starting file watcher for {len(self.watched_files)} files...")
        
        async for changes in awatch(*watch_paths):
            for change_type, file_path in changes:
                file_path = Path(file_path)
                
                # Only process files we're explicitly watching
                if file_path in self.watched_files:
                    await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
                    self.handle_file_change(file_path)
                    
    def get_watch_statistics(self) -> Dict[str, Any]:
        """Get statistics about watching activity"""
        
        cursor = self.db.execute("""
            SELECT COUNT(*) as total_changes,
                   COUNT(CASE WHEN timestamp > ? THEN 1 END) as recent_changes
            FROM observations 
            WHERE event_type = 'file_modified'
        """, (time.time() - 300,))  # Last 5 minutes
        
        total_changes, recent_changes = cursor.fetchone()
        
        # Get pattern counts
        cursor = self.db.execute("""
            SELECT COUNT(*) FROM observations WHERE event_type = 'pattern_detected'
        """)
        pattern_count = cursor.fetchone()[0]
        
        return {
            "total_file_changes": total_changes,
            "recent_changes_5min": recent_changes,
            "patterns_detected": pattern_count,
            "files_watched": len(self.watched_files),
            "uptime_minutes": (time.time() - self.birth_time) / 60
        }


async def start_meta_watching():
    """Start watching the meta system files"""
    
    watcher = MetaWatcher()
    
    # Watch the core meta files
    watcher.add_file_to_watch(Path("meta_prime.py"))
    watcher.add_file_to_watch(Path("meta_watcher.py"))
    
    # Record that we're starting coordinated observation
    watcher._record_observation("coordinated_watching_started", {
        "components": ["meta_prime", "meta_watcher"],
        "purpose": "self_observation_during_construction"
    })
    
    try:
        await watcher.watch_files()
    except KeyboardInterrupt:
        print("\n🛑 Watcher stopped by user")
        stats = watcher.get_watch_statistics()
        print(f"📊 Final stats: {stats}")
        

if __name__ == "__main__":
    # Start the meta watching system
    asyncio.run(start_meta_watching())