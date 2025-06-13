# Second-Generation MCTS Improvements for Wheel Trading

Based on comprehensive codebase analysis, here are targeted MCTS improvements that address the actual complexity of the wheel trading system.

## 1. Sequential Decision MCTS for Complete Wheel Cycles

The wheel strategy is inherently sequential. Current code evaluates each decision in isolation. MCTS can optimize entire cycles:

```python
class WheelCycleMCTS:
    """MCTS for optimizing complete put→stock→call cycles."""
    
    def __init__(self, market_model, risk_constraints):
        self.market_model = market_model
        self.risk_constraints = risk_constraints
        # Key insight: exploration constant should vary by market volatility
        self.c = self._adaptive_exploration_constant()
    
    def _adaptive_exploration_constant(self):
        """Tune exploration based on market regime."""
        vix = self.market_model.get_vix()
        if vix < 15:
            return 0.5  # Low volatility = exploit known strategies
        elif vix > 30:
            return 2.5  # High volatility = explore more
        else:
            return 1.414
    
    def simulate_wheel_cycle(self, starting_state):
        """Simulate complete wheel cycle with market dynamics."""
        states = []
        total_return = 0
        
        # Stage 1: Put selling
        put_strike = self._select_put_strike(starting_state)
        put_premium = self._calculate_premium(put_strike, starting_state)
        total_return += put_premium
        
        # Stage 2: Assignment probability
        if self._simulate_assignment(put_strike, starting_state):
            # Stage 3: Stock ownership period
            holding_period = self._simulate_holding_period()
            stock_return = self._simulate_stock_movement(holding_period)
            
            # Stage 4: Call selling
            call_strike = self._select_call_strike(starting_state, stock_return)
            call_premium = self._calculate_premium(call_strike, starting_state)
            total_return += call_premium + stock_return
            
        return total_return, states
```

## 2. Portfolio-Level MCTS with Correlation Awareness

Current portfolio optimization treats positions independently. MCTS can explore correlated position combinations:

```python
class PortfolioConstructionMCTS:
    """Build portfolios considering correlations and risk limits."""
    
    def __init__(self, correlation_matrix, risk_engine):
        self.correlations = correlation_matrix
        self.risk_engine = risk_engine
        self.position_cache = {}  # Cache evaluated positions
        
    def expand_node(self, node):
        """Generate child nodes with portfolio constraints."""
        current_positions = node.state['positions']
        current_risk = self.risk_engine.calculate_portfolio_risk(current_positions)
        
        # Smart action generation based on current portfolio
        possible_actions = []
        
        # Only consider positions that improve risk-adjusted returns
        for symbol in self.universe:
            for strike in self._get_viable_strikes(symbol):
                new_position = Position(symbol, strike)
                
                # Quick correlation check
                avg_correlation = self._get_avg_correlation(new_position, current_positions)
                
                if avg_correlation < 0.7:  # Diversification benefit
                    marginal_var = self._calculate_marginal_var(new_position, current_positions)
                    
                    if marginal_var < self.risk_constraints.max_marginal_var:
                        possible_actions.append(new_position)
        
        # Create child nodes with different position sizes
        for position in possible_actions[:10]:  # Limit branching factor
            for size_factor in [0.5, 1.0, 1.5]:  # Kelly fraction multipliers
                child = node.add_child(
                    action=('add_position', position, size_factor),
                    state=self._apply_action(node.state, position, size_factor)
                )
                
        return node.children
```

## 3. Market Regime-Aware MCTS

The dynamic optimizer adjusts parameters based on market conditions. MCTS can learn optimal strategies per regime:

```python
class MarketRegimeMCTS:
    """MCTS that adapts to different market regimes."""
    
    def __init__(self):
        # Separate trees for different regimes
        self.regime_trees = {
            'bull_low_vol': MCTSTree(c=0.5),
            'bull_high_vol': MCTSTree(c=1.5),
            'bear_low_vol': MCTSTree(c=1.0),
            'bear_high_vol': MCTSTree(c=2.0),
            'sideways': MCTSTree(c=1.414)
        }
        
        # Transfer learning between regimes
        self.regime_transitions = np.array([
            # Transition probabilities between regimes
            [0.7, 0.1, 0.1, 0.05, 0.05],  # bull_low_vol
            [0.2, 0.4, 0.1, 0.2, 0.1],     # bull_high_vol
            # ...
        ])
        
    def get_action(self, market_state):
        """Select action considering possible regime changes."""
        current_regime = self._detect_regime(market_state)
        
        # Weight trees by transition probability
        weighted_values = {}
        for next_regime, prob in self._get_transition_probs(current_regime):
            tree = self.regime_trees[next_regime]
            value = tree.get_best_action_value(market_state)
            weighted_values[next_regime] = value * prob
            
        # Meta-MCTS: explore which regime assumption is best
        return self._meta_mcts_selection(weighted_values, market_state)
```

## 4. Real-Time MCTS with GPU Acceleration

Leverage the GPU components for parallel MCTS simulations:

```python
class GPUAcceleratedWheelMCTS:
    """MCTS using MLX for massive parallel simulations."""
    
    def __init__(self, gpu_accelerator):
        self.gpu = gpu_accelerator
        self.batch_size = 1024  # M4 Pro can handle large batches
        
    async def parallel_rollouts(self, nodes: List[MCTSNode], market_data):
        """Run parallel rollouts on GPU."""
        # Prepare batch data
        states = mx.array([node.state.to_vector() for node in nodes])
        
        # Parallel market simulations
        future_prices = self._simulate_gbm_batch(
            S0=market_data.price,
            mu=market_data.drift,
            sigma=market_data.vol,
            T=45/365,  # 45 DTE
            paths=self.batch_size
        )
        
        # Parallel option pricing
        put_values = self._batch_black_scholes(
            S=future_prices,
            K=states[:, 0],  # strikes
            T=states[:, 1],  # time to expiry
            r=market_data.rate,
            sigma=states[:, 2]  # implied vol
        )
        
        # Parallel profit calculation
        profits = self._calculate_wheel_profits_batch(put_values, future_prices)
        
        return profits.tolist()
    
    def _simulate_gbm_batch(self, S0, mu, sigma, T, paths):
        """Geometric Brownian Motion on GPU."""
        dt = T / 252  # Daily steps
        Z = mx.random.normal((paths, int(T * 252)))
        
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * mx.sqrt(dt) * Z
        log_prices = mx.cumsum(log_returns, axis=1)
        
        return S0 * mx.exp(log_prices)
```

## 5. Learning-Enhanced MCTS with Pattern Recognition

Integrate with the incremental learner to recognize profitable patterns:

```python
class LearningEnhancedMCTS:
    """MCTS that learns from historical patterns."""
    
    def __init__(self, incremental_learner, pattern_db):
        self.learner = incremental_learner
        self.pattern_db = pattern_db
        self.value_network = self._init_value_network()
        
    def evaluate_node(self, node):
        """Use learned patterns to guide evaluation."""
        # Extract features
        features = self._extract_node_features(node)
        
        # Check pattern database
        similar_patterns = self.pattern_db.find_similar(features, k=10)
        
        if similar_patterns:
            # Use historical outcomes to bias evaluation
            historical_value = np.mean([p.outcome for p in similar_patterns])
            confidence = len(similar_patterns) / 10.0
            
            # Blend with neural network prediction
            nn_value = self.value_network.predict(features)
            
            return confidence * historical_value + (1 - confidence) * nn_value
        else:
            # Pure neural network evaluation for novel situations
            return self.value_network.predict(features)
    
    def update_from_outcome(self, path, actual_return):
        """Learn from actual trading outcomes."""
        # Store pattern
        self.pattern_db.add(path, actual_return)
        
        # Update value network
        training_data = [(node.features, actual_return) for node in path]
        self.value_network.train_on_batch(training_data)
        
        # Adjust exploration based on prediction error
        prediction_error = abs(path[-1].value - actual_return)
        if prediction_error > 0.1:  # 10% error
            self.c *= 1.1  # Increase exploration
        else:
            self.c *= 0.98  # Decrease exploration
```

## 6. Hyperparameter Auto-Tuning Based on Market Conditions

```python
class AdaptiveMCTSHyperparameters:
    """Auto-tune MCTS parameters based on market conditions."""
    
    def __init__(self):
        self.param_performance = defaultdict(list)
        self.market_states = []
        
    def get_optimal_parameters(self, market_state):
        """Get best parameters for current market."""
        # Key parameters that matter for wheel trading
        iv_rank = market_state.iv_rank
        trend_strength = market_state.trend_strength
        days_to_earnings = market_state.days_to_earnings
        
        # Exploration constant
        if iv_rank > 70:  # High IV
            c = 2.0  # Explore more strikes
        elif iv_rank < 30:  # Low IV
            c = 0.8  # Stick to tested strategies
        else:
            c = 1.414
            
        # Rollout depth
        if days_to_earnings < 30:
            depth = 20  # Deeper search near events
        else:
            depth = 10
            
        # Simulation count based on available time
        if market_state.minutes_to_close < 30:
            simulations = 1000  # Quick decisions
        else:
            simulations = 10000  # Thorough analysis
            
        return {
            'exploration_constant': c,
            'rollout_depth': depth,
            'simulation_count': simulations,
            'use_pattern_db': iv_rank < 50,  # Use patterns in normal markets
            'gpu_batch_size': 256 if market_state.is_volatile else 1024
        }
```

## 7. Multi-Objective MCTS for Risk-Return Optimization

```python
class MultiObjectiveMCTS:
    """Optimize for multiple objectives: return, risk, drawdown."""
    
    def __init__(self, risk_tolerance):
        self.risk_tolerance = risk_tolerance
        self.pareto_frontier = []
        
    def evaluate_node_multi_objective(self, node):
        """Evaluate node on multiple criteria."""
        simulation_results = self.run_simulations(node, n=1000)
        
        # Calculate metrics
        expected_return = np.mean(simulation_results)
        downside_risk = np.percentile(simulation_results, 5)  # CVaR
        max_drawdown = self.calculate_max_drawdown(simulation_results)
        sharpe_ratio = expected_return / np.std(simulation_results)
        
        # Composite score based on risk tolerance
        if self.risk_tolerance == 'conservative':
            score = 0.2 * expected_return + 0.5 * (-downside_risk) + 0.3 * sharpe_ratio
        elif self.risk_tolerance == 'aggressive':
            score = 0.6 * expected_return + 0.2 * (-downside_risk) + 0.2 * sharpe_ratio
        else:  # balanced
            score = 0.4 * expected_return + 0.3 * (-downside_risk) + 0.3 * sharpe_ratio
            
        # Update Pareto frontier
        self._update_pareto_frontier(expected_return, downside_risk, node)
        
        return score
```

## Implementation Priority

1. **Sequential Decision MCTS** - Highest value, directly addresses wheel cycle optimization
2. **Portfolio-Level MCTS** - Critical for multi-position traders
3. **GPU Acceleration** - Enables real-time decision making
4. **Learning Enhancement** - Improves over time
5. **Market Regime Awareness** - Handles changing conditions
6. **Multi-Objective Optimization** - Balances risk and return
7. **Auto-Tuning** - Maintains peak performance

## Key Hyperparameters for Wheel Trading MCTS

Based on the analysis, these parameters matter most:

1. **Exploration Constant (c)**: 
   - Low (0.5-0.8) in stable markets
   - High (2.0-3.0) in volatile/uncertain markets
   - Adaptive based on IV rank

2. **Rollout Depth**:
   - 5-10 for normal decisions
   - 15-20 near earnings or major events
   - Deeper for multi-leg strategies

3. **Simulation Budget**:
   - 1,000 for real-time decisions
   - 10,000 for daily portfolio optimization
   - 100,000 for backtesting strategy discovery

4. **Branching Factor**:
   - Limit to top 10-15 strikes (already done via bucketing)
   - 3-5 position size options
   - 2-3 expiration dates

The key insight is that MCTS should optimize **sequences of decisions** in the wheel strategy, not just individual trades. This matches the true nature of the strategy and can capture path-dependent profits that current optimization misses.