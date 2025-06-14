"""Code Policy Network for action selection.

Guides MCTS exploration by predicting promising code transformations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any


class CodePolicyNetwork(nn.Module):
    """Neural network to predict code transformation actions."""
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 512,
                 num_actions: int = 50,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder for context understanding
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Action prediction heads
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Specialized heads for different action types
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 100)  # 100 possible locations
        )
        
        self.parameter_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # Parameter embedding
        )
        
        # Value prediction for action
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # [-1, 1] range
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass returning action probabilities."""
        # Project input
        x = self.input_projection(x)
        
        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, hidden)
        
        # Encode context
        encoded = self.context_encoder(x, src_key_padding_mask=mask)
        
        # Global pooling
        if encoded.dim() == 3:
            encoded = encoded.mean(dim=1)  # (batch, hidden)
        
        # Get action logits
        action_logits = self.action_type_head(encoded)
        
        return action_logits
    
    def get_action_distribution(self, x: torch.Tensor, 
                               temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """Get full action distribution with all components."""
        # Project and encode
        x_proj = self.input_projection(x)
        if x_proj.dim() == 2:
            x_proj = x_proj.unsqueeze(1)
        
        encoded = self.context_encoder(x_proj)
        if encoded.dim() == 3:
            encoded = encoded.mean(dim=1)
        
        # Get all predictions
        action_logits = self.action_type_head(encoded) / temperature
        location_logits = self.location_head(encoded) / temperature
        parameter_features = self.parameter_head(encoded)
        action_value = self.value_head(encoded)
        
        return {
            'action_probs': F.softmax(action_logits, dim=-1),
            'location_probs': F.softmax(location_logits, dim=-1),
            'parameter_features': parameter_features,
            'action_value': action_value,
            'action_logits': action_logits  # For training
        }
    
    def sample_action(self, x: torch.Tensor, 
                     temperature: float = 1.0,
                     top_k: Optional[int] = None) -> Dict[str, Any]:
        """Sample an action from the policy."""
        with torch.no_grad():
            dist = self.get_action_distribution(x, temperature)
            
            # Sample action type
            action_probs = dist['action_probs']
            
            if top_k is not None:
                # Top-k sampling
                top_probs, top_indices = torch.topk(action_probs, min(top_k, action_probs.size(-1)))
                action_idx = top_indices[0, torch.multinomial(top_probs[0], 1)].item()
            else:
                # Full distribution sampling
                action_idx = torch.multinomial(action_probs[0], 1).item()
            
            # Sample location
            location_idx = torch.multinomial(dist['location_probs'][0], 1).item()
            
            return {
                'action_type': action_idx,
                'location': location_idx,
                'parameters': dist['parameter_features'][0].cpu().numpy(),
                'value': dist['action_value'][0].item(),
                'confidence': action_probs[0, action_idx].item()
            }
    
    async def train_batch(self, data: List[Tuple[Any, Any]]) -> float:
        """Train on a batch of state-action pairs."""
        # Convert data
        states = torch.stack([torch.tensor(x[0], dtype=torch.float32) for x in data])
        actions = torch.tensor([x[1].get('action_type', 0) for x in data], dtype=torch.long)
        
        # Forward pass
        logits = self(states)
        
        # Compute loss
        loss = F.cross_entropy(logits, actions)
        
        # In practice, would do actual backprop
        return loss.item()


class ActionEncoder:
    """Encodes code actions into a fixed vocabulary."""
    
    def __init__(self):
        # Define action vocabulary
        self.action_types = [
            # Structural actions
            'add_function',
            'add_class',
            'add_method',
            'add_import',
            'add_variable',
            
            # Modification actions
            'modify_function',
            'rename_variable',
            'extract_method',
            'inline_function',
            'move_code',
            
            # Optimization actions
            'add_cache',
            'vectorize_loop',
            'parallelize',
            'optimize_memory',
            'add_index',
            
            # Quality actions
            'add_type_hints',
            'add_docstring',
            'add_error_handling',
            'add_validation',
            'add_logging',
            
            # Refactoring actions
            'split_function',
            'merge_functions',
            'extract_constant',
            'introduce_parameter',
            'remove_duplication',
            
            # Pattern actions
            'apply_singleton',
            'apply_factory',
            'apply_strategy',
            'apply_observer',
            'apply_decorator',
            
            # Testing actions
            'add_unit_test',
            'add_integration_test',
            'add_benchmark',
            'add_assertion',
            'add_mock',
            
            # Architecture actions
            'create_module',
            'create_package',
            'add_interface',
            'add_abstract_class',
            'restructure_hierarchy',
            
            # Performance actions
            'add_async',
            'add_threading',
            'add_multiprocessing',
            'optimize_algorithm',
            'reduce_complexity',
            
            # Special actions
            'terminal',  # End of modifications
            'no_op',  # No operation
            'revert',  # Undo last action
        ]
        
        self.action_to_idx = {a: i for i, a in enumerate(self.action_types)}
        self.idx_to_action = {i: a for a, i in self.action_to_idx.items()}
    
    def encode(self, action: str) -> int:
        """Encode action to index."""
        return self.action_to_idx.get(action, self.action_to_idx['no_op'])
    
    def decode(self, idx: int) -> str:
        """Decode index to action."""
        return self.idx_to_action.get(idx, 'no_op')
    
    def get_valid_actions(self, context: Dict[str, Any]) -> List[int]:
        """Get valid actions for current context."""
        valid = []
        
        # Always valid
        always_valid = ['add_function', 'add_class', 'add_import', 
                       'add_docstring', 'terminal', 'no_op']
        
        for action in always_valid:
            valid.append(self.encode(action))
        
        # Context-dependent actions
        if context.get('has_functions', False):
            valid.extend([
                self.encode('modify_function'),
                self.encode('add_type_hints'),
                self.encode('split_function'),
                self.encode('add_unit_test')
            ])
        
        if context.get('has_loops', False):
            valid.extend([
                self.encode('vectorize_loop'),
                self.encode('parallelize'),
                self.encode('optimize_algorithm')
            ])
        
        if context.get('complexity', 0) > 10:
            valid.extend([
                self.encode('extract_method'),
                self.encode('reduce_complexity'),
                self.encode('restructure_hierarchy')
            ])
        
        return list(set(valid))  # Remove duplicates


class PolicyGradientTrainer:
    """Trains policy network using policy gradient methods."""
    
    def __init__(self, policy_net: CodePolicyNetwork, 
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99):
        self.policy_net = policy_net
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
    
    def record_trajectory(self, state: torch.Tensor, 
                         action: int, 
                         reward: float):
        """Record a single step."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train_episode(self) -> float:
        """Train on collected episode."""
        if not self.states:
            return 0.0
        
        # Calculate returns
        returns = []
        running_return = 0
        for reward in reversed(self.rewards):
            running_return = reward + self.gamma * running_return
            returns.insert(0, running_return)
        
        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        
        # Forward pass
        logits = self.policy_net(states)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Policy gradient loss
        loss = -(selected_log_probs * returns).mean()
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        
        return loss.item()