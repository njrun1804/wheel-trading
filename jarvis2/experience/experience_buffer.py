"""Experience Replay System for continuous learning.

Stores and manages experiences for training neural networks
and improving code generation over time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.solution import Experience, SolutionMetrics

logger = logging.getLogger(__name__)


class ExperienceReplaySystem:
    """Manages experience storage and replay for learning."""

    def __init__(self, storage_path: Path, buffer_size: int=10000,
        prioritized: bool=True):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.prioritized = prioritized
        self.memory_buffer = deque(maxlen=1000)
        self.priorities = np.ones(buffer_size)
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_epsilon = 1e-06
        self.total_experiences = 0
        self.successful_experiences = 0
        self.db_connection = None

    async def initialize(self):
        """Initialize storage systems."""
        db_path = self.storage_path / 'experiences.db'
        self.db_connection = sqlite3.connect(str(db_path))
        self.db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                query TEXT,
                context_json TEXT,
                solution_json TEXT,
                metrics_json TEXT,
                reward REAL,
                was_selected BOOLEAN,
                user_feedback REAL,
                execution_success BOOLEAN,
                priority REAL DEFAULT 1.0
            )
        """
            )
        self.db_connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp);
        """
            )
        self.db_connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reward ON experiences(reward);
        """
            )
        self.db_connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_priority ON experiences(priority);
        """
            )
        self.db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experience_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                hour_bucket INTEGER,
                total_count INTEGER,
                success_count INTEGER,
                avg_reward REAL,
                avg_confidence REAL,
                avg_generation_time REAL
            )
        """
            )
        self.db_connection.commit()
        await self._load_recent_experiences()
        logger.info(
            f'Experience replay initialized with {len(self.memory_buffer)} recent experiences'
            )

    async def record(self, query: str, solution: Dict[str, Any], metrics:
        SolutionMetrics, evaluations: List[Dict[str, Any]], context:
        Optional[Dict[str, Any]]=None):
        """Record a new experience."""
        experience = Experience(query=query, context=context or {},
            solution=solution, metrics={'generation_time_ms': metrics.
            generation_time_ms, 'simulations_performed': metrics.
            simulations_performed, 'confidence_score': metrics.
            confidence_score, 'complexity_score': metrics.complexity_score,
            'gpu_utilization': metrics.gpu_utilization}, timestamp=time.time())
        priority = self._calculate_priority(experience, evaluations)
        self.memory_buffer.append((experience, priority))
        await self._persist_experience(experience, priority)
        self.total_experiences += 1
        if experience.metrics.get('confidence_score', 0) > 0.7:
            self.successful_experiences += 1
        if self.total_experiences % 100 == 0:
            await self._update_statistics()

    def _calculate_priority(self, experience: Experience, evaluations: List
        [Dict[str, Any]]) ->float:
        """Calculate priority for experience replay."""
        confidence = experience.metrics.get('confidence_score', 0.5)
        complexity = experience.metrics.get('complexity_score', 0.5)
        confidence_factor = 4 * confidence * (1 - confidence)
        complexity_factor = complexity
        if evaluations:
            eval_scores = [e.get('overall', 0) for e in evaluations]
            diversity_factor = np.std(eval_scores) if len(eval_scores
                ) > 1 else 0.5
        else:
            diversity_factor = 0.5
        priority = (0.4 * confidence_factor + 0.3 * complexity_factor + 0.3 *
            diversity_factor)
        return max(self.priority_epsilon, priority)

    async def _persist_experience(self, experience: Experience, priority: float
        ):
        """Save experience to database."""
        await asyncio.sleep(0)
        self.db_connection.execute(
            """
            INSERT INTO experiences 
            (timestamp, query, context_json, solution_json, metrics_json,
             reward, was_selected, user_feedback, execution_success, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
            , (experience.timestamp, experience.query, json.dumps(
            experience.context), json.dumps(experience.solution), json.
            dumps(experience.metrics), experience._calculate_reward(),
            experience.was_selected, experience.user_feedback, experience.
            execution_success, priority))
        self.db_connection.commit()

    async def sample(self, batch_size: int=32, beta: Optional[float]=None
        ) ->List[Dict[str, Any]]:
        """Sample experiences for training."""
        if len(self.memory_buffer) < batch_size:
            return []
        if self.prioritized:
            return await self._prioritized_sample(batch_size, beta)
        else:
            return await self._uniform_sample(batch_size)

    async def _prioritized_sample(self, batch_size: int, beta: Optional[
        float]=None) ->List[Dict[str, Any]]:
        """Sample using prioritized experience replay."""
        await asyncio.sleep(0)
        beta = beta or self.priority_beta
        priorities = np.array([p for _, p in self.memory_buffer])
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory_buffer), batch_size, p=probs
            )
        weights = (len(self.memory_buffer) * probs[indices]) ** -beta
        weights /= weights.max()
        samples = []
        for i, idx in enumerate(indices):
            exp, _ = self.memory_buffer[idx]
            sample = {'experience': exp, 'index': idx, 'weight': weights[i],
                'query': exp.query, 'context': exp.context, 'solution': exp
                .solution, 'metrics': exp.metrics}
            samples.append(sample)
        return samples

    async def _uniform_sample(self, batch_size: int) ->List[Dict[str, Any]]:
        """Uniform random sampling."""
        await asyncio.sleep(0)
        indices = np.random.choice(len(self.memory_buffer), batch_size,
            replace=False)
        samples = []
        for idx in indices:
            exp, _ = self.memory_buffer[idx]
            sample = {'experience': exp, 'index': idx, 'weight': 1.0,
                'query': exp.query, 'context': exp.context, 'solution': exp
                .solution, 'metrics': exp.metrics}
            samples.append(sample)
        return samples

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on TD errors from training."""
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self.memory_buffer):
                priority = (abs(td_error) + self.priority_epsilon
                    ) ** self.priority_alpha
                exp, _ = self.memory_buffer[idx]
                self.memory_buffer[idx] = exp, priority

    async def _load_recent_experiences(self, limit: int=1000):
        """Load recent experiences from database."""
        await asyncio.sleep(0)
        cursor = self.db_connection.execute(
            """
            SELECT timestamp, query, context_json, solution_json, 
                   metrics_json, reward, priority
            FROM experiences
            ORDER BY timestamp DESC
            LIMIT ?
        """
            , (limit,))
        for row in cursor.fetchall():
            exp = Experience(query=row[1], context=json.loads(row[2]),
                solution=json.loads(row[3]), metrics=json.loads(row[4]),
                timestamp=row[0])
            priority = row[6]
            self.memory_buffer.append((exp, priority))
        self.memory_buffer = deque(reversed(self.memory_buffer), maxlen=1000)

    async def get_statistics(self) ->Dict[str, Any]:
        """Get experience buffer statistics."""
        await asyncio.sleep(0)
        if self.memory_buffer:
            rewards = [exp._calculate_reward() for exp, _ in self.memory_buffer
                ]
            confidences = [exp.metrics.get('confidence_score', 0) for exp,
                _ in self.memory_buffer]
            complexities = [exp.metrics.get('complexity_score', 0) for exp,
                _ in self.memory_buffer]
        else:
            rewards = confidences = complexities = []
        db_stats = self.db_connection.execute(
            """
            SELECT 
                COUNT(*) as total,
                AVG(reward) as avg_reward,
                MAX(reward) as max_reward,
                AVG(priority) as avg_priority
            FROM experiences
        """
            ).fetchone()
        return {'buffer_size': len(self.memory_buffer), 'total_experiences':
            self.total_experiences, 'successful_experiences': self.
            successful_experiences, 'success_rate': self.
            successful_experiences / max(1, self.total_experiences),
            'memory_stats': {'avg_reward': np.mean(rewards) if rewards else
            0, 'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_complexity': np.mean(complexities) if complexities else 0},
            'database_stats': {'total_stored': db_stats[0], 'avg_reward': 
            db_stats[1] or 0, 'max_reward': db_stats[2] or 0,
            'avg_priority': db_stats[3] or 0}}

    async def _update_statistics(self):
        """Update aggregated statistics."""
        await asyncio.sleep(0)
        current_hour = int(time.time() // 3600)
        stats = self.db_connection.execute(
            """
            SELECT 
                COUNT(*) as count,
                SUM(CASE WHEN reward > 0.7 THEN 1 ELSE 0 END) as success_count,
                AVG(reward) as avg_reward,
                AVG(json_extract(metrics_json, '$.confidence_score')) as avg_confidence,
                AVG(json_extract(metrics_json, '$.generation_time_ms')) as avg_time
            FROM experiences
            WHERE timestamp >= ? AND timestamp < ?
        """
            , (current_hour * 3600, (current_hour + 1) * 3600)).fetchone()
        if stats[0] > 0:
            self.db_connection.execute(
                """
                INSERT OR REPLACE INTO experience_stats
                (timestamp, hour_bucket, total_count, success_count, 
                 avg_reward, avg_confidence, avg_generation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
                , (time.time(), current_hour, stats[0], stats[1] or 0, 
                stats[2] or 0, stats[3] or 0, stats[4] or 0))
            self.db_connection.commit()

    async def get_learning_curves(self, window_size: int=100) ->Dict[str,
        List[float]]:
        """Get learning curves over time."""
        await asyncio.sleep(0)
        cursor = self.db_connection.execute(
            """
            SELECT 
                reward,
                json_extract(metrics_json, '$.confidence_score') as confidence,
                json_extract(metrics_json, '$.generation_time_ms') as time_ms
            FROM experiences
            ORDER BY timestamp
        """
            )
        rewards = []
        confidences = []
        times = []
        for row in cursor:
            rewards.append(row[0] or 0)
            confidences.append(row[1] or 0)
            times.append(row[2] or 0)

        def moving_average(data, window):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window) / window, mode='valid'
                ).tolist()
        return {'rewards': moving_average(rewards, window_size),
            'confidences': moving_average(confidences, window_size),
            'generation_times': moving_average(times, window_size)}

    def flush(self):
        """Flush any pending data."""
        if self.db_connection:
            self.db_connection.commit()

    def size(self) ->int:
        """Get current buffer size."""
        return len(self.memory_buffer)

    def close(self):
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()


class ExperienceAnalyzer:
    """Analyze experiences to find patterns and insights."""

    def __init__(self, experience_buffer: ExperienceReplaySystem):
        self.buffer = experience_buffer

    async def analyze_patterns(self) ->Dict[str, Any]:
        """Analyze patterns in experiences."""
        await asyncio.sleep(0)
        query_patterns = self._analyze_query_patterns()
        success_patterns = self._analyze_success_patterns()
        failure_patterns = self._analyze_failure_patterns()
        complexity_patterns = self._analyze_complexity_patterns()
        return {'query_patterns': query_patterns, 'success_patterns':
            success_patterns, 'failure_patterns': failure_patterns,
            'complexity_patterns': complexity_patterns}

    def _analyze_query_patterns(self) ->Dict[str, Any]:
        """Analyze patterns in queries."""
        cursor = self.buffer.db_connection.execute(
            """
            SELECT query, COUNT(*) as count, AVG(reward) as avg_reward
            FROM experiences
            GROUP BY query
            ORDER BY count DESC
            LIMIT 20
        """
            )
        patterns = []
        for row in cursor:
            patterns.append({'query': row[0], 'count': row[1], 'avg_reward':
                row[2] or 0})
        return {'most_common_queries': patterns, 'total_unique_queries':
            self.buffer.db_connection.execute(
            'SELECT COUNT(DISTINCT query) FROM experiences').fetchone()[0]}

    def _analyze_success_patterns(self) ->Dict[str, Any]:
        """Find patterns in successful experiences."""
        cursor = self.buffer.db_connection.execute(
            """
            SELECT 
                json_extract(metrics_json, '$.complexity_score') as complexity,
                json_extract(metrics_json, '$.simulations_performed') as simulations,
                AVG(reward) as avg_reward,
                COUNT(*) as count
            FROM experiences
            WHERE reward > 0.8
            GROUP BY complexity, simulations
            ORDER BY count DESC
            LIMIT 10
        """
            )
        patterns = []
        for row in cursor:
            patterns.append({'complexity': row[0] or 0, 'simulations': row[
                1] or 0, 'avg_reward': row[2] or 0, 'count': row[3]})
        return patterns

    def _analyze_failure_patterns(self) ->Dict[str, Any]:
        """Find patterns in failed experiences."""
        cursor = self.buffer.db_connection.execute(
            """
            SELECT 
                query,
                json_extract(solution_json, '$.approach') as approach,
                COUNT(*) as count
            FROM experiences
            WHERE reward < 0.3
            GROUP BY query, approach
            ORDER BY count DESC
            LIMIT 10
        """
            )
        patterns = []
        for row in cursor:
            patterns.append({'query': row[0], 'approach': row[1] or
                'unknown', 'failure_count': row[2]})
        return patterns

    def _analyze_complexity_patterns(self) ->Dict[str, Any]:
        """Analyze relationship between complexity and performance."""
        cursor = self.buffer.db_connection.execute(
            """
            SELECT 
                ROUND(json_extract(metrics_json, '$.complexity_score'), 1) as complexity_bucket,
                AVG(reward) as avg_reward,
                AVG(json_extract(metrics_json, '$.generation_time_ms')) as avg_time,
                COUNT(*) as count
            FROM experiences
            GROUP BY complexity_bucket
            ORDER BY complexity_bucket
        """
            )
        buckets = []
        for row in cursor:
            buckets.append({'complexity': row[0] or 0, 'avg_reward': row[1] or
                0, 'avg_generation_time': row[2] or 0, 'count': row[3]})
        return buckets
