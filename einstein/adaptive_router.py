#!/usr/bin/env python3
"""
Einstein Adaptive Query Router

Self-tuning query router that learns from user patterns and success rates.
Implements contextual bandit for optimizing search strategy selection.
"""

import logging
import os
import pickle
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .query_router import QueryPlan, QueryRouter
from .einstein_config import get_einstein_config

logger = logging.getLogger(__name__)


@dataclass
class QueryOutcome:
    """Records the outcome of a query execution."""
    query: str
    plan: QueryPlan
    latency_ms: float
    result_count: int
    user_satisfaction: float  # 0.0-1.0 based on user interaction
    timestamp: float
    success: bool


@dataclass
class QueryFeatures:
    """Features extracted from query for learning."""
    query_length: int
    has_quotes: bool
    has_math_terms: bool
    has_code_terms: bool
    has_complexity_terms: bool
    word_count: int
    hour_of_day: int
    contains_symbol: bool


class ContextualBandit:
    """Simple contextual bandit for route selection."""
    
    def __init__(self, n_arms: int = 4, learning_rate: float = None):
        self.n_arms = n_arms  # Number of search modality combinations
        config = get_einstein_config()
        self.learning_rate = learning_rate or config.ml.adaptive_learning_rate
        self.arm_weights = np.ones(n_arms) * 0.5  # Initial weights
        self.arm_counts = np.zeros(n_arms)
        self.exploration_rate = 0.1
        
        # Arm mappings (search modality combinations)
        self.arms = [
            ['text'],                           # 0: Fast text only
            ['text', 'structural'],             # 1: Text + structure
            ['text', 'semantic'],               # 2: Text + semantic
            ['text', 'structural', 'semantic', 'analytical']  # 3: Full search
        ]
    
    def select_arm(self, features: QueryFeatures) -> int:
        """Select search strategy arm based on features."""
        
        # Feature-based bias
        feature_bias = np.zeros(self.n_arms)
        
        # Bias towards text-only for quoted strings
        if features.has_quotes:
            feature_bias[0] += 0.3
        
        # Bias towards semantic for complex queries
        if features.query_length > 20 or features.has_math_terms:
            feature_bias[2] += 0.2
            feature_bias[3] += 0.1
        
        # Bias towards structural for symbol queries
        if features.contains_symbol or features.has_code_terms:
            feature_bias[1] += 0.2
            feature_bias[3] += 0.1
        
        # Bias towards analytical for complexity queries
        if features.has_complexity_terms:
            feature_bias[3] += 0.3
        
        # Combine weights with feature bias
        combined_scores = self.arm_weights + feature_bias
        
        # Epsilon-greedy exploration
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(combined_scores)
    
    def update(self, arm: int, reward: float):
        """Update arm weights based on reward."""
        
        self.arm_counts[arm] += 1
        
        # Update weight with learning rate
        error = reward - self.arm_weights[arm]
        self.arm_weights[arm] += self.learning_rate * error
        
        # Decay exploration rate over time
        total_pulls = np.sum(self.arm_counts)
        self.exploration_rate = max(0.05, 0.1 * np.exp(-total_pulls / 1000))
    
    def get_stats(self) -> dict[str, Any]:
        """Get bandit statistics."""
        
        return {
            'arm_weights': self.arm_weights.tolist(),
            'arm_counts': self.arm_counts.tolist(),
            'exploration_rate': self.exploration_rate,
            'total_pulls': int(np.sum(self.arm_counts))
        }


class AdaptiveQueryRouter(QueryRouter):
    """Enhanced query router with learning capabilities."""
    
    def __init__(self, model_path: Path = None):
        super().__init__()
        
        config = get_einstein_config()
        self.model_path = model_path or (config.paths.cache_dir / 'adaptive_router.pkl')
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Learning components
        self.bandit = ContextualBandit()
        self.query_history = deque(maxlen=config.cache.max_cache_entries)  # Keep last queries
        self.outcome_buffer = deque(maxlen=1000)  # Buffer for batch learning
        
        # Statistics
        self.stats = {
            'queries_processed': 0,
            'learning_updates': 0,
            'average_latency': 0.0,
            'success_rate': 0.0
        }
        
        # Load existing model if available
        self._load_model()
        
        # Start background learning task
        self._learning_task = None
    
    def _extract_features(self, query: str) -> QueryFeatures:
        """Extract features from query for learning."""
        
        query_lower = query.lower()
        
        math_terms = ['delta', 'gamma', 'theta', 'vega', 'calculation', 'formula', 'equation']
        code_terms = ['class', 'function', 'def ', 'import', 'return', 'if ', 'for ']
        complexity_terms = ['complex', 'slow', 'performance', 'optimize', 'bottleneck']
        
        return QueryFeatures(
            query_length=len(query),
            has_quotes='"' in query or "'" in query,
            has_math_terms=any(term in query_lower for term in math_terms),
            has_code_terms=any(term in query_lower for term in code_terms),
            has_complexity_terms=any(term in query_lower for term in complexity_terms),
            word_count=len(query.split()),
            hour_of_day=int(time.strftime('%H')),
            contains_symbol='.' in query and any(c.isupper() for c in query)
        )
    
    def analyze_query_adaptive(self, query: str, context: dict[str, Any] = None) -> QueryPlan:
        """Analyze query with adaptive learning."""
        
        # Extract features
        features = self._extract_features(query)
        
        # Get bandit recommendation
        arm = self.bandit.select_arm(features)
        recommended_modalities = self.bandit.arms[arm]
        
        # Create base plan using parent logic
        base_plan = super().analyze_query(query)
        
        # Override modalities with bandit recommendation
        adaptive_plan = QueryPlan(
            query=query,
            query_type=base_plan.query_type,
            search_modalities=recommended_modalities,
            confidence=base_plan.confidence * 0.9,  # Slightly lower confidence for exploration
            estimated_time_ms=sum(self.modality_performance[m] for m in recommended_modalities),
            reasoning=f"Adaptive: {base_plan.reasoning} [Arm {arm}]"
        )
        
        # Store for learning
        self.query_history.append({
            'query': query,
            'features': asdict(features),
            'arm': arm,
            'plan': asdict(adaptive_plan),
            'timestamp': time.time()
        })
        
        self.stats['queries_processed'] += 1
        
        return adaptive_plan
    
    def record_outcome(self, query: str, plan: QueryPlan, latency_ms: float, 
                      result_count: int, user_satisfaction: float = 0.5):
        """Record query outcome for learning."""
        
        # Calculate success based on multiple factors
        latency_score = max(0, 1.0 - (latency_ms / 1000.0))  # 1.0 for <1s, 0.0 for >1s
        result_score = min(1.0, result_count / 10.0)          # 1.0 for 10+ results
        
        # Combined success score
        success_score = (
            0.4 * user_satisfaction +
            0.3 * latency_score +
            0.3 * result_score
        )
        
        outcome = QueryOutcome(
            query=query,
            plan=plan,
            latency_ms=latency_ms,
            result_count=result_count,
            user_satisfaction=user_satisfaction,
            timestamp=time.time(),
            success=success_score > 0.6
        )
        
        self.outcome_buffer.append(outcome)
        
        # Update statistics
        self._update_stats(outcome)
        
        # Trigger learning if buffer is full
        if len(self.outcome_buffer) >= 50:
            self._learn_from_outcomes()
    
    def _learn_from_outcomes(self):
        """Learn from recent query outcomes."""
        
        if not self.outcome_buffer:
            return
        
        # Process recent outcomes
        outcomes = list(self.outcome_buffer)
        self.outcome_buffer.clear()
        
        for outcome in outcomes:
            # Find corresponding query in history
            for query_record in reversed(self.query_history):
                if query_record['query'] == outcome.query:
                    arm = query_record['arm']
                    
                    # Calculate reward
                    reward = self._calculate_reward(outcome)
                    
                    # Update bandit
                    self.bandit.update(arm, reward)
                    break
        
        self.stats['learning_updates'] += 1
        
        # Save model periodically
        if self.stats['learning_updates'] % 10 == 0:
            self._save_model()
        
        logger.debug(f"Learning update: {len(outcomes)} outcomes processed")
    
    def _calculate_reward(self, outcome: QueryOutcome) -> float:
        """Calculate reward signal for bandit learning."""
        
        if not outcome.success:
            return -0.1
        
        # Reward based on latency (faster is better)
        latency_reward = max(0, 1.0 - (outcome.latency_ms / 1000.0))
        
        # Reward based on user satisfaction
        satisfaction_reward = outcome.user_satisfaction
        
        # Reward based on result count (more results often better)
        result_reward = min(1.0, outcome.result_count / 20.0)
        
        # Combined reward
        total_reward = (
            0.4 * satisfaction_reward +
            0.4 * latency_reward +
            0.2 * result_reward
        )
        
        return total_reward
    
    def _update_stats(self, outcome: QueryOutcome):
        """Update router statistics."""
        
        # Update average latency
        alpha = 0.1  # Learning rate for moving average
        self.stats['average_latency'] = (
            alpha * outcome.latency_ms + 
            (1 - alpha) * self.stats['average_latency']
        )
        
        # Update success rate
        self.stats['success_rate'] = (
            alpha * (1.0 if outcome.success else 0.0) +
            (1 - alpha) * self.stats['success_rate']
        )
    
    def _save_model(self):
        """Save bandit model and statistics."""
        
        try:
            model_data = {
                'bandit_weights': self.bandit.arm_weights,
                'bandit_counts': self.bandit.arm_counts,
                'exploration_rate': self.bandit.exploration_rate,
                'stats': self.stats,
                'timestamp': time.time()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.debug(f"Saved adaptive router model to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save adaptive model: {e}", exc_info=True,
                        extra={
                            'operation': 'save_adaptive_model',
                            'error_type': type(e).__name__,
                            'model_path': str(self.model_path),
                            'model_exists': self.model_path.exists(),
                            'parent_exists': self.model_path.parent.exists(),
                            'parent_writable': os.access(self.model_path.parent, os.W_OK) if self.model_path.parent.exists() else False,
                            'stats': self.stats,
                            'bandit_weights': self.bandit.arm_weights.tolist() if hasattr(self.bandit, 'arm_weights') else None
                        })
    
    def _load_model(self):
        """Load bandit model and statistics."""
        
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.bandit.arm_weights = model_data.get('bandit_weights', self.bandit.arm_weights)
                self.bandit.arm_counts = model_data.get('bandit_counts', self.bandit.arm_counts)
                self.bandit.exploration_rate = model_data.get('exploration_rate', self.bandit.exploration_rate)
                self.stats.update(model_data.get('stats', {}))
                
                logger.info(f"Loaded adaptive router model from {self.model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load adaptive model: {e}", exc_info=True,
                        extra={
                            'operation': 'load_adaptive_model',
                            'error_type': type(e).__name__,
                            'model_path': str(self.model_path),
                            'model_exists': self.model_path.exists(),
                            'file_size': self.model_path.stat().st_size if self.model_path.exists() else 0,
                            'file_readable': os.access(self.model_path, os.R_OK) if self.model_path.exists() else False,
                            'bandit_stats': self.bandit.get_stats() if hasattr(self.bandit, 'get_stats') else None
                        })
    
    def get_learning_stats(self) -> dict[str, Any]:
        """Get comprehensive learning statistics."""
        
        return {
            'router_stats': self.stats,
            'bandit_stats': self.bandit.get_stats(),
            'query_history_size': len(self.query_history),
            'outcome_buffer_size': len(self.outcome_buffer),
            'model_path': str(self.model_path)
        }


# Factory function for easy integration
def create_adaptive_router(model_path: Path = None) -> AdaptiveQueryRouter:
    """Create and initialize adaptive query router."""
    
    router = AdaptiveQueryRouter(model_path)
    
    logger.info("🧠 Adaptive query router initialized")
    logger.info(f"   Model path: {router.model_path}")
    logger.info(f"   Bandit arms: {len(router.bandit.arms)}")
    
    return router


if __name__ == "__main__":
    # Test the adaptive router
    router = create_adaptive_router()
    
    # Test queries
    test_queries = [
        "WheelStrategy class",
        "complex functions with high complexity",
        '"delta calculation"',
        "optimize options pricing performance"
    ]
    
    for query in test_queries:
        plan = router.analyze_query_adaptive(query)
        print(f"Query: {query}")
        print(f"  Modalities: {plan.search_modalities}")
        print(f"  Reasoning: {plan.reasoning}")
        
        # Simulate outcome
        router.record_outcome(query, plan, 25.0, 15, 0.8)
        print()
    
    # Show learning stats
    stats = router.get_learning_stats()
    print(f"Learning stats: {stats}")
