"""Adaptive assignment probability model that learns from outcomes."""
from __future__ import annotations


from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from unity_wheel.analytics.unity_assignment import AssignmentProbability
from src.config.loader import get_config
from unity_wheel.utils.logging import get_logger
from .feedback_loop import FeedbackLoop, ParameterUpdate

logger = get_logger(__name__)


@dataclass
class AssignmentOutcome:
    """Records actual assignment outcome for learning."""
    decision_id: str
    symbol: str
    strike: float
    underlying_price: float
    days_to_expiry: int
    implied_volatility: float
    delta: float
    theta: float
    volume: int
    open_interest: int
    bid_ask_spread: float
    predicted_probability: float
    was_assigned: bool
    timestamp: datetime
    pnl: Optional[float] = None


class AdaptiveAssignmentModel(FeedbackLoop):
    """
    Machine learning model for assignment probability that learns from actual outcomes.
    
    Replaces static assignment calculations with adaptive predictions based on
    historical assignment patterns.
    """
    
    def __init__(self):
        """Initialize adaptive assignment model."""
        super().__init__("AdaptiveAssignmentModel", learning_rate=0.001)
        
        # ML components
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            warm_start=True  # Allow incremental learning
        )
        self.scaler = StandardScaler()
        
        # Track outcomes
        self.assignment_history: List[AssignmentOutcome] = []
        self.is_trained = False
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Calibration parameters (evolving)
        self.calibration_params = {
            'threshold_adjustment': 0.0,  # Learned bias correction
            'confidence_scaling': 1.0,     # Learned confidence adjustment
            'regime_adjustments': {        # Per-regime calibration
                'low_vol': 0.0,
                'normal': 0.0,
                'high_vol': 0.0
            }
        }
    
    def predict_assignment(
        self,
        symbol: str,
        strike: float,
        underlying_price: float,
        days_to_expiry: int,
        implied_volatility: float,
        delta: float,
        theta: float,
        volume: int,
        open_interest: int,
        bid_ask_spread: float,
        market_regime: Optional[str] = None
    ) -> AssignmentProbability:
        """
        Predict assignment probability using ML model.
        
        Returns AssignmentProbability with learned prediction.
        """
        # Extract features
        features = self._extract_features_for_prediction({
            'symbol': symbol,
            'strike': strike,
            'underlying_price': underlying_price,
            'days_to_expiry': days_to_expiry,
            'implied_volatility': implied_volatility,
            'delta': delta,
            'theta': theta,
            'volume': volume,
            'open_interest': open_interest,
            'bid_ask_spread': bid_ask_spread
        })
        
        if self.is_trained and len(self.assignment_history) >= 100:
            # Use ML model prediction
            features_scaled = self.scaler.transform([features])
            base_probability = self.model.predict_proba(features_scaled)[0, 1]
            
            # Apply learned calibrations
            adjusted_probability = self._apply_calibration(
                base_probability,
                market_regime
            )
            
            # Calculate confidence based on training data
            confidence = self._calculate_prediction_confidence(features_scaled)
            
        else:
            # Fallback to physics-based model until enough data
            moneyness = strike / underlying_price
            time_factor = np.sqrt(days_to_expiry / 365.0)
            vol_factor = implied_volatility
            
            if moneyness < 1.0:  # ITM put
                base_probability = 0.5 + 0.5 * (1.0 - moneyness) / (vol_factor * time_factor)
            else:  # OTM put
                base_probability = 0.5 * np.exp(-2 * (moneyness - 1.0) / (vol_factor * time_factor))
            
            adjusted_probability = np.clip(base_probability, 0.01, 0.99)
            confidence = 0.5  # Low confidence for fallback model
        
        return AssignmentProbability(
            probability=adjusted_probability,
            confidence=confidence,
            factors={
                'moneyness': strike / underlying_price,
                'time_value': days_to_expiry / 365.0,
                'volatility': implied_volatility,
                'delta': delta,
                'model_type': 'ml' if self.is_trained else 'fallback',
                'training_samples': len(self.assignment_history)
            }
        )
    
    def record_assignment_outcome(self, outcome: AssignmentOutcome) -> None:
        """Record actual assignment outcome for learning."""
        self.assignment_history.append(outcome)
        
        # Convert to feedback loop format
        context = {
            'symbol': outcome.symbol,
            'strike': outcome.strike,
            'underlying_price': outcome.underlying_price,
            'days_to_expiry': outcome.days_to_expiry,
            'implied_volatility': outcome.implied_volatility,
            'delta': outcome.delta,
            'theta': outcome.theta,
            'volume': outcome.volume,
            'open_interest': outcome.open_interest,
            'bid_ask_spread': outcome.bid_ask_spread
        }
        
        outcome_dict = {
            'was_assigned': outcome.was_assigned,
            'predicted_probability': outcome.predicted_probability,
            'pnl': outcome.pnl
        }
        
        self.record_outcome(context, outcome_dict)
        
        # Retrain if enough new data
        if len(self.outcome_buffer) >= 50:
            self._retrain_model()
    
    def _extract_features_for_prediction(self, context: Dict) -> np.ndarray:
        """Extract features for prediction."""
        return np.array([
            context['strike'] / context['underlying_price'],  # Moneyness
            context['days_to_expiry'] / 30.0,  # Normalized DTE
            context['implied_volatility'],
            abs(context['delta']),
            context['theta'] / context['underlying_price'],
            np.log1p(context['volume']),
            np.log1p(context['open_interest']),
            context['bid_ask_spread'] / context['strike'],
            context['days_to_expiry'] <= 7,  # Near expiry flag
            context['strike'] / context['underlying_price'] < 0.95,  # Deep ITM flag
        ])
    
    def extract_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract features from context (for parent class)."""
        return self._extract_features_for_prediction(context)
    
    def calculate_reward(self, outcome: Dict[str, Any]) -> float:
        """Calculate reward based on prediction accuracy."""
        predicted = outcome['predicted_probability']
        actual = 1.0 if outcome['was_assigned'] else 0.0
        
        # Prediction error
        error = abs(predicted - actual)
        
        # Reward is negative error (we want to minimize error)
        base_reward = -error
        
        # Bonus for correct high-confidence predictions
        if error < 0.1 and abs(predicted - 0.5) > 0.3:
            base_reward += 0.5
        
        # Include P&L if available
        if outcome.get('pnl') is not None:
            pnl_factor = np.tanh(outcome['pnl'] / 1000)  # Normalize P&L
            base_reward += 0.3 * pnl_factor
        
        return base_reward
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current calibration parameters."""
        params = {
            'threshold_adjustment': self.calibration_params['threshold_adjustment'],
            'confidence_scaling': self.calibration_params['confidence_scaling']
        }
        
        # Add regime adjustments
        for regime, adjustment in self.calibration_params['regime_adjustments'].items():
            params[f'regime_adj_{regime}'] = adjustment
        
        return params
    
    def apply_parameter_update(self, updates: List[ParameterUpdate]) -> None:
        """Apply calibration parameter updates."""
        for update in updates:
            if update.parameter_name == 'threshold_adjustment':
                self.calibration_params['threshold_adjustment'] = update.recommended_value
            elif update.parameter_name == 'confidence_scaling':
                self.calibration_params['confidence_scaling'] = update.recommended_value
            elif update.parameter_name.startswith('regime_adj_'):
                regime = update.parameter_name.replace('regime_adj_', '')
                self.calibration_params['regime_adjustments'][regime] = update.recommended_value
    
    def _retrain_model(self) -> None:
        """Retrain the ML model with accumulated data."""
        if len(self.assignment_history) < 100:
            return
        
        logger.info(f"Retraining assignment model with {len(self.assignment_history)} samples")
        
        # Prepare training data
        X = []
        y = []
        
        for outcome in self.assignment_history:
            features = self._extract_features_for_prediction({
                'symbol': outcome.symbol,
                'strike': outcome.strike,
                'underlying_price': outcome.underlying_price,
                'days_to_expiry': outcome.days_to_expiry,
                'implied_volatility': outcome.implied_volatility,
                'delta': outcome.delta,
                'theta': outcome.theta,
                'volume': outcome.volume,
                'open_interest': outcome.open_interest,
                'bid_ask_spread': outcome.bid_ask_spread
            })
            X.append(features)
            y.append(1 if outcome.was_assigned else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        if not self.is_trained:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Train model
        if not self.is_trained:
            self.model.fit(X_scaled, y)
            self.is_trained = True
        else:
            # Incremental learning
            self.model.n_estimators += 10
            self.model.fit(X_scaled, y)
        
        # Update feature importance
        self.feature_importance = {
            'moneyness': self.model.feature_importances_[0],
            'dte_normalized': self.model.feature_importances_[1],
            'implied_vol': self.model.feature_importances_[2],
            'delta': self.model.feature_importances_[3],
            'theta_normalized': self.model.feature_importances_[4],
            'volume_log': self.model.feature_importances_[5],
            'oi_log': self.model.feature_importances_[6],
            'spread_normalized': self.model.feature_importances_[7],
            'near_expiry': self.model.feature_importances_[8],
            'deep_itm': self.model.feature_importances_[9]
        }
        
        logger.info(f"Model retrained. Top features: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    def _apply_calibration(self, base_probability: float, market_regime: Optional[str]) -> float:
        """Apply learned calibrations to base probability."""
        # Apply threshold adjustment
        adjusted = base_probability + self.calibration_params['threshold_adjustment']
        
        # Apply regime-specific adjustment
        if market_regime and market_regime in self.calibration_params['regime_adjustments']:
            adjusted += self.calibration_params['regime_adjustments'][market_regime]
        
        # Ensure valid probability
        return np.clip(adjusted, 0.01, 0.99)
    
    def _calculate_prediction_confidence(self, features_scaled: np.ndarray) -> float:
        """Calculate confidence in prediction based on training data density."""
        if len(self.assignment_history) < 100:
            return 0.5
        
        # Get prediction probabilities from all trees
        tree_predictions = []
        for estimator in self.model.estimators_:
            tree_pred = estimator[0].predict_proba(features_scaled)[0, 1]
            tree_predictions.append(tree_pred)
        
        # Confidence based on agreement among trees
        pred_std = np.std(tree_predictions)
        base_confidence = 1.0 - min(pred_std * 2, 0.5)
        
        # Scale by calibration parameter
        scaled_confidence = base_confidence * self.calibration_params['confidence_scaling']
        
        return np.clip(scaled_confidence, 0.1, 0.99)
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get model performance metrics."""
        if len(self.assignment_history) < 100:
            return {'status': 'insufficient_data', 'samples': len(self.assignment_history)}
        
        # Calculate accuracy on recent predictions
        recent_outcomes = self.assignment_history[-100:]
        
        errors = []
        correct_predictions = 0
        
        for outcome in recent_outcomes:
            error = abs(outcome.predicted_probability - (1.0 if outcome.was_assigned else 0.0))
            errors.append(error)
            
            # Count correct predictions (within 0.3 threshold)
            if outcome.predicted_probability > 0.5 and outcome.was_assigned:
                correct_predictions += 1
            elif outcome.predicted_probability <= 0.5 and not outcome.was_assigned:
                correct_predictions += 1
        
        return {
            'mean_absolute_error': np.mean(errors),
            'accuracy': correct_predictions / len(recent_outcomes),
            'samples_trained': len(self.assignment_history),
            'model_type': 'gradient_boosting',
            'is_trained': self.is_trained,
            'top_features': list(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]) if self.feature_importance else []
        }