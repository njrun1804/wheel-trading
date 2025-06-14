"""Experience buffer using DuckDB for efficient storage and querying.

Stores code generation experiences for learning and analysis.
"""
import duckdb
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
import time
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GenerationExperience:
    """Single code generation experience."""
    id: str
    timestamp: float
    query: str
    generated_code: str
    confidence: float
    context: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    metrics: Dict[str, float]
    feedback: Optional[Dict[str, Any]] = None
    tags: List[str] = None


class ExperienceBuffer:
    """Persistent experience storage using DuckDB."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize database schema."""
        if self.initialized:
            return
            
        # Create connection
        self.conn = duckdb.connect(str(self.db_path))
        
        # Create schema
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id VARCHAR PRIMARY KEY,
                timestamp DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                query TEXT,
                generated_code TEXT,
                confidence DOUBLE,
                context JSON,
                alternatives JSON,
                metrics JSON,
                feedback JSON,
                tags JSON
            )
        """)
        
        # Create indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp DESC)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence ON experiences(confidence DESC)
        """)
        
        # Skip has_feedback index since it's not a real column anymore
        
        # Create analysis views
        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS experience_stats AS
            SELECT 
                COUNT(*) as total_experiences,
                AVG(confidence) as avg_confidence,
                AVG(LENGTH(generated_code)) as avg_code_length,
                AVG(JSON_ARRAY_LENGTH(alternatives::VARCHAR)) as avg_alternatives,
                SUM(CASE WHEN feedback IS NOT NULL THEN 1 ELSE 0 END) as feedback_count,
                MIN(timestamp) as earliest_timestamp,
                MAX(timestamp) as latest_timestamp
            FROM experiences
        """)
        
        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS daily_stats AS
            SELECT 
                DATE_TRUNC('day', created_at) as day,
                COUNT(*) as experiences,
                AVG(confidence) as avg_confidence,
                AVG(CAST(JSON_EXTRACT(metrics, '$.generation_time_ms') AS DOUBLE)) as avg_time_ms
            FROM experiences
            GROUP BY DATE_TRUNC('day', created_at)
            ORDER BY day DESC
        """)
        
        self.initialized = True
        logger.info(f"Experience buffer initialized at {self.db_path}")
        
    async def add_experience(self, experience: GenerationExperience):
        """Add new experience to buffer."""
        if not self.initialized:
            await self.initialize()
            
        try:
            self.conn.execute("""
                INSERT INTO experiences 
                (id, timestamp, query, generated_code, confidence, 
                 context, alternatives, metrics, feedback, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experience.id,
                experience.timestamp,
                experience.query,
                experience.generated_code,
                experience.confidence,
                json.dumps(experience.context),
                json.dumps(experience.alternatives),
                json.dumps(experience.metrics),
                json.dumps(experience.feedback) if experience.feedback else None,
                json.dumps(experience.tags) if experience.tags else None
            ))
            
            logger.debug(f"Added experience {experience.id}")
            
        except Exception as e:
            logger.error(f"Failed to add experience: {e}")
            
    async def get_experience(self, experience_id: str) -> Optional[GenerationExperience]:
        """Get experience by ID."""
        if not self.initialized:
            await self.initialize()
            
        result = self.conn.execute("""
            SELECT * FROM experiences WHERE id = ?
        """, (experience_id,)).fetchone()
        
        if result:
            return self._row_to_experience(result)
        return None
        
    async def search_experiences(self, 
                               query: Optional[str] = None,
                               min_confidence: Optional[float] = None,
                               has_feedback: Optional[bool] = None,
                               limit: int = 100) -> List[GenerationExperience]:
        """Search experiences with filters."""
        if not self.initialized:
            await self.initialize()
            
        # Build query
        conditions = []
        params = []
        
        if query:
            conditions.append("query LIKE ?")
            params.append(f"%{query}%")
            
        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
            
        if has_feedback is not None:
            if has_feedback:
                conditions.append("feedback IS NOT NULL")
            else:
                conditions.append("feedback IS NULL")
            
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        sql = f"""
            SELECT * FROM experiences
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)
        
        results = self.conn.execute(sql, params).fetchall()
        return [self._row_to_experience(row) for row in results]
        
    async def get_similar_experiences(self, query: str, 
                                    limit: int = 10) -> List[GenerationExperience]:
        """Find similar past experiences (simple text similarity)."""
        if not self.initialized:
            await self.initialize()
            
        # Use DuckDB's built-in text similarity
        # In production, would use vector similarity
        results = self.conn.execute("""
            SELECT *, 
                   JACCARD(LOWER(query), LOWER(?)) as similarity
            FROM experiences
            WHERE similarity > 0.2
            ORDER BY similarity DESC
            LIMIT ?
        """, (query, limit)).fetchall()
        
        return [self._row_to_experience(row) for row in results]
        
    async def add_feedback(self, experience_id: str, feedback: Dict[str, Any]):
        """Add feedback to an experience."""
        if not self.initialized:
            await self.initialize()
            
        self.conn.execute("""
            UPDATE experiences 
            SET feedback = ?
            WHERE id = ?
        """, (json.dumps(feedback), experience_id))
        
        logger.debug(f"Added feedback to experience {experience_id}")
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self.initialized:
            await self.initialize()
            
        # Overall stats
        stats = self.conn.execute("""
            SELECT * FROM experience_stats
        """).fetchone()
        
        # Recent performance
        recent = self.conn.execute("""
            SELECT 
                AVG(CAST(JSON_EXTRACT(metrics, '$.generation_time_ms') AS DOUBLE)) as avg_time_ms,
                AVG(confidence) as avg_confidence
            FROM experiences
            WHERE timestamp > ?
        """, (time.time() - 86400,)).fetchone()  # Last 24 hours
        
        # Daily trends
        daily = self.conn.execute("""
            SELECT * FROM daily_stats
            LIMIT 7
        """).fetchall()
        
        return {
            'total_experiences': stats[0] if stats else 0,
            'avg_confidence': stats[1] if stats else 0,
            'avg_code_length': stats[2] if stats else 0,
            'avg_alternatives': stats[3] if stats else 0,
            'feedback_count': stats[4] if stats else 0,
            'recent_avg_time_ms': recent[0] if recent else 0,
            'recent_avg_confidence': recent[1] if recent else 0,
            'daily_trends': [
                {
                    'day': str(row[0]),
                    'experiences': row[1],
                    'avg_confidence': row[2],
                    'avg_time_ms': row[3]
                }
                for row in daily
            ]
        }
        
    async def export_for_training(self, 
                                 min_confidence: float = 0.7,
                                 require_feedback: bool = False) -> List[Dict[str, Any]]:
        """Export high-quality experiences for training."""
        if not self.initialized:
            await self.initialize()
            
        conditions = ["confidence >= ?"]
        params = [min_confidence]
        
        if require_feedback:
            conditions.append("feedback IS NOT NULL")
            conditions.append("CAST(JSON_EXTRACT(feedback, '$.rating') AS DOUBLE) >= 4")
            
        results = self.conn.execute(f"""
            SELECT 
                query,
                generated_code,
                confidence,
                JSON_EXTRACT(metrics, '$.generation_time_ms') as time_ms,
                feedback
            FROM experiences
            WHERE {' AND '.join(conditions)}
            ORDER BY confidence DESC
        """, params).fetchall()
        
        return [
            {
                'query': row[0],
                'code': row[1],
                'confidence': row[2],
                'time_ms': row[3],
                'feedback': json.loads(row[4]) if row[4] else None
            }
            for row in results
        ]
        
    async def cleanup_old_experiences(self, days: int = 30):
        """Remove old experiences to save space."""
        if not self.initialized:
            await self.initialize()
            
        cutoff = time.time() - (days * 86400)
        
        # Keep experiences with feedback
        deleted = self.conn.execute("""
            DELETE FROM experiences
            WHERE timestamp < ? 
            AND feedback IS NULL
        """, (cutoff,)).rowcount
        
        logger.info(f"Cleaned up {deleted} old experiences")
        
    def _row_to_experience(self, row: tuple) -> GenerationExperience:
        """Convert database row to experience object."""
        return GenerationExperience(
            id=row[0],
            timestamp=row[1],
            query=row[3],
            generated_code=row[4],
            confidence=row[5],
            context=json.loads(row[6]),
            alternatives=json.loads(row[7]),
            metrics=json.loads(row[8]),
            feedback=json.loads(row[9]) if row[9] else None,
            tags=json.loads(row[10]) if row[10] else None
        )
        
    async def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.initialized = False


# Example usage
async def demo():
    """Demo of experience buffer."""
    buffer = ExperienceBuffer(".jarvis/experience.db")
    await buffer.initialize()
    
    # Add experience
    exp = GenerationExperience(
        id="exp_001",
        timestamp=time.time(),
        query="Create a hello world function",
        generated_code='def hello_world():\n    print("Hello, World!")',
        confidence=0.95,
        context={'platform': 'M4 Pro'},
        alternatives=[],
        metrics={'generation_time_ms': 150}
    )
    
    await buffer.add_experience(exp)
    
    # Search
    results = await buffer.search_experiences(query="hello", min_confidence=0.8)
    print(f"Found {len(results)} experiences")
    
    # Get stats
    stats = await buffer.get_statistics()
    print(f"Statistics: {stats}")
    
    await buffer.close()


if __name__ == "__main__":
    asyncio.run(demo())