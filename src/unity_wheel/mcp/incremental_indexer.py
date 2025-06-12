#!/usr/bin/env python3
"""
Incremental filesystem indexer with file-save triggers.
Integrates with VS Code file watchers for real-time updates.
"""

import os
import time
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, List, Optional, Callable
import duckdb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
import json
import logging

logger = logging.getLogger(__name__)

class IncrementalIndexer(FileSystemEventHandler):
    """Incremental indexer with file system watching."""
    
    def __init__(self, project_root: str, index_path: Optional[str] = None):
        self.project_root = Path(project_root)
        self.index_path = index_path or str(self.project_root / ".mcp_index" / "incremental.duckdb")
        self.conn = None
        self.observer = Observer()
        self.update_queue: asyncio.Queue = asyncio.Queue()
        self.processing = False
        self._ensure_index_dir()
        self._ignored_patterns = self._load_ignore_patterns()
        
    def _ensure_index_dir(self):
        """Create index directory if needed."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
    def _load_ignore_patterns(self) -> Set[str]:
        """Load patterns from .claudeignore and .gitignore."""
        patterns = {
            '*.pyc', '__pycache__', '.git', '.venv', 'venv',
            'node_modules', '*.log', '.DS_Store', '*.swp'
        }
        
        # Load from .claudeignore
        claudeignore = self.project_root / '.claudeignore'
        if claudeignore.exists():
            with open(claudeignore) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.add(line)
                        
        return patterns
        
    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        path_str = str(path)
        
        # Check patterns
        for pattern in self._ignored_patterns:
            if pattern in path_str:
                return True
                
        # Check if it's a Python file
        return not path.suffix == '.py'
        
    def connect(self):
        """Establish database connection."""
        self.conn = duckdb.connect(self.index_path)
        self._setup_schema()
        
    def _setup_schema(self):
        """Create schema for incremental indexing."""
        # Main index table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS file_index (
                file_path VARCHAR PRIMARY KEY,
                content TEXT,
                file_hash VARCHAR,
                size_bytes INTEGER,
                last_modified TIMESTAMP,
                last_indexed TIMESTAMP,
                version INTEGER DEFAULT 1,
                deleted BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Change history table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS change_history (
                id INTEGER PRIMARY KEY,
                file_path VARCHAR,
                change_type VARCHAR,  -- 'create', 'modify', 'delete'
                timestamp TIMESTAMP,
                old_hash VARCHAR,
                new_hash VARCHAR,
                version INTEGER
            )
        """)
        
        # Create indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_modified 
            ON file_index(last_modified DESC)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_fts 
            ON file_index USING FTS(content)
        """)
        
    def start_watching(self):
        """Start file system watcher."""
        self.observer.schedule(self, str(self.project_root), recursive=True)
        self.observer.start()
        logger.info(f"Started watching: {self.project_root}")
        
    def stop_watching(self):
        """Stop file system watcher."""
        self.observer.stop()
        self.observer.join()
        logger.info("Stopped file watching")
        
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        path = Path(event.src_path)
        if not self.should_ignore(path):
            asyncio.create_task(self.update_queue.put(('modify', path)))
            
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
            
        path = Path(event.src_path)
        if not self.should_ignore(path):
            asyncio.create_task(self.update_queue.put(('create', path)))
            
    def on_deleted(self, event):
        """Handle file deletion events."""
        if event.is_directory:
            return
            
        path = Path(event.src_path)
        if not self.should_ignore(path):
            asyncio.create_task(self.update_queue.put(('delete', path)))
            
    async def process_updates(self):
        """Process queued file updates."""
        self.processing = True
        batch = []
        
        while self.processing:
            try:
                # Collect updates for batch processing
                timeout = 0.5 if batch else None
                change_type, path = await asyncio.wait_for(
                    self.update_queue.get(), 
                    timeout=timeout
                )
                batch.append((change_type, path))
                
                # Process batch if it gets large
                if len(batch) >= 10:
                    await self._process_batch(batch)
                    batch = []
                    
            except asyncio.TimeoutError:
                # Process remaining batch
                if batch:
                    await self._process_batch(batch)
                    batch = []
                    
    async def _process_batch(self, batch: List[tuple]):
        """Process a batch of file changes."""
        for change_type, path in batch:
            try:
                if change_type == 'delete':
                    await self._handle_delete(path)
                else:
                    await self._handle_update(path, change_type)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                
        self.conn.commit()
        
    async def _handle_update(self, path: Path, change_type: str):
        """Handle file creation or modification."""
        try:
            # Read file content
            content = path.read_text(encoding='utf-8', errors='ignore')
            file_hash = hashlib.md5(content.encode()).hexdigest()
            stats = path.stat()
            
            # Check if file exists in index
            existing = self.conn.execute("""
                SELECT file_hash, version 
                FROM file_index 
                WHERE file_path = ?
            """, [str(path.relative_to(self.project_root))]).fetchone()
            
            if existing and existing[0] == file_hash:
                # No content change
                return
                
            # Calculate new version
            version = (existing[1] + 1) if existing else 1
            
            # Update index
            relative_path = str(path.relative_to(self.project_root))
            self.conn.execute("""
                INSERT OR REPLACE INTO file_index 
                (file_path, content, file_hash, size_bytes, 
                 last_modified, last_indexed, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, FALSE)
            """, [
                relative_path,
                content,
                file_hash,
                stats.st_size,
                datetime.fromtimestamp(stats.st_mtime),
                datetime.now(),
                version
            ])
            
            # Record change
            self.conn.execute("""
                INSERT INTO change_history 
                (file_path, change_type, timestamp, old_hash, new_hash, version)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                relative_path,
                change_type,
                datetime.now(),
                existing[0] if existing else None,
                file_hash,
                version
            ])
            
            logger.info(f"Indexed {change_type}: {relative_path} (v{version})")
            
        except Exception as e:
            logger.error(f"Error indexing {path}: {e}")
            
    async def _handle_delete(self, path: Path):
        """Handle file deletion."""
        relative_path = str(path.relative_to(self.project_root))
        
        # Mark as deleted
        self.conn.execute("""
            UPDATE file_index 
            SET deleted = TRUE, last_indexed = ?
            WHERE file_path = ?
        """, [datetime.now(), relative_path])
        
        # Record deletion
        self.conn.execute("""
            INSERT INTO change_history 
            (file_path, change_type, timestamp, version)
            VALUES (?, 'delete', ?, 0)
        """, [relative_path, datetime.now()])
        
        logger.info(f"Marked deleted: {relative_path}")
        
    def search_incremental(self, query: str, include_deleted: bool = False) -> List[Dict]:
        """Search with incremental index."""
        sql = """
            SELECT 
                file_path,
                content,
                last_modified,
                version,
                SNIPPET(content, -1, '<match>', '</match>', '...', 64) as snippet
            FROM file_index
            WHERE content MATCH ?
        """
        
        if not include_deleted:
            sql += " AND deleted = FALSE"
            
        sql += " ORDER BY last_modified DESC LIMIT 100"
        
        results = self.conn.execute(sql, [query]).fetchall()
        
        return [
            {
                'file_path': r[0],
                'content': r[1],
                'last_modified': r[2],
                'version': r[3],
                'snippet': r[4]
            }
            for r in results
        ]
        
    def get_change_history(self, file_path: Optional[str] = None, 
                          limit: int = 50) -> List[Dict]:
        """Get file change history."""
        if file_path:
            sql = """
                SELECT * FROM change_history 
                WHERE file_path = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = [file_path, limit]
        else:
            sql = """
                SELECT * FROM change_history 
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = [limit]
            
        results = self.conn.execute(sql, params).fetchall()
        
        return [
            {
                'file_path': r[1],
                'change_type': r[2],
                'timestamp': r[3],
                'old_hash': r[4],
                'new_hash': r[5],
                'version': r[6]
            }
            for r in results
        ]
        
    def get_stats(self) -> Dict:
        """Get indexing statistics."""
        stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_files,
                COUNT(CASE WHEN deleted = FALSE THEN 1 END) as active_files,
                COUNT(CASE WHEN deleted = TRUE THEN 1 END) as deleted_files,
                MAX(last_indexed) as last_update,
                SUM(CASE WHEN deleted = FALSE THEN size_bytes ELSE 0 END) as total_size
            FROM file_index
        """).fetchone()
        
        changes = self.conn.execute("""
            SELECT 
                change_type,
                COUNT(*) as count
            FROM change_history
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY change_type
        """).fetchall()
        
        return {
            'total_files': stats[0],
            'active_files': stats[1],
            'deleted_files': stats[2],
            'last_update': stats[3],
            'total_size_mb': stats[4] / 1_000_000 if stats[4] else 0,
            'recent_changes': {r[0]: r[1] for r in changes},
            'watching': self.observer.is_alive()
        }


class VSCodeIntegration:
    """Integration with VS Code for file save events."""
    
    def __init__(self, indexer: IncrementalIndexer):
        self.indexer = indexer
        self.socket_path = Path.home() / ".claude" / "vscode.sock"
        
    def create_vscode_task(self) -> Dict:
        """Create VS Code task configuration for auto-indexing."""
        return {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Update MCP Index",
                    "type": "shell",
                    "command": "python",
                    "args": [
                        "-c",
                        f"from unity_wheel.mcp.incremental_indexer import notify_file_save; notify_file_save('${{file}}')"
                    ],
                    "runOptions": {
                        "runOn": "folderOpen"
                    },
                    "problemMatcher": []
                }
            ]
        }
        
    def create_save_watcher(self) -> Dict:
        """Create VS Code settings for file save watching."""
        return {
            "runOnSave.commands": [
                {
                    "match": "**/*.py",
                    "command": "workbench.action.tasks.runTask",
                    "args": "Update MCP Index",
                    "runIn": "backend",
                    "runningStatusMessage": "Updating MCP index..."
                }
            ]
        }


# Helper function for VS Code integration
def notify_file_save(file_path: str):
    """Notify indexer of file save (called from VS Code)."""
    try:
        # Send notification to running indexer
        socket_path = Path.home() / ".claude" / "vscode.sock"
        if socket_path.exists():
            import socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(socket_path))
            sock.send(json.dumps({
                'type': 'file_save',
                'path': file_path
            }).encode())
            sock.close()
    except Exception:
        pass


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        # Run as watcher
        project_root = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
        
        indexer = IncrementalIndexer(project_root)
        indexer.connect()
        indexer.start_watching()
        
        print(f"Incremental indexer started for: {project_root}")
        print("Watching for file changes...")
        
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(indexer.process_updates())
        except KeyboardInterrupt:
            print("\nStopping indexer...")
            indexer.stop_watching()
            indexer.conn.close()
    else:
        # Show usage
        print("Usage: python incremental_indexer.py watch [project_root]")
        print("\nVS Code Integration:")
        print("1. Add to .vscode/tasks.json:")
        print(json.dumps(VSCodeIntegration(None).create_vscode_task(), indent=2))
        print("\n2. Install 'Run on Save' extension")
        print("3. Add to .vscode/settings.json:")
        print(json.dumps(VSCodeIntegration(None).create_save_watcher(), indent=2))