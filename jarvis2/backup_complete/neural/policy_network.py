"""Simplified Code Policy Network for debugging.

A simpler version without TransformerEncoder to avoid initialization issues.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any


class CodePolicyNetwork(nn.Module):
    """Simplified neural network to predict code transformation actions."""
    
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
        
        # Simplified architecture - just feedforward layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass returning action probabilities."""
        # Simple forward pass
        logits = self.layers(x)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_action_distribution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action distribution and entropy."""
        logits = self.layers(x)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Entropy for exploration
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return probs, entropy
    
    async def train_batch(self, data: List[Tuple[Any, Any]], 
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         device: Optional[torch.device] = None) -> Dict[str, float]:
        """Train policy network using policy gradient methods."""
        if device is None:
            device = next(self.parameters()).device
            
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        
        # Unpack data: (states, actions, advantages)
        states = torch.stack([torch.tensor(x[0], dtype=torch.float32) for x in data]).to(device)
        actions = torch.tensor([x[1][0] for x in data], dtype=torch.long).to(device)
        advantages = torch.tensor([x[1][1] for x in data], dtype=torch.float32).to(device)
        
        # Ensure model is in training mode
        self.train()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        probs, entropy = self(states)
        
        # Get log probabilities of selected actions
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        
        # Policy gradient loss (negative because we want to maximize)
        policy_loss = -(log_probs * advantages).mean()
        
        # Add entropy bonus for exploration
        entropy_bonus = -0.01 * entropy.mean()  # Small entropy coefficient
        
        # Total loss
        loss = policy_loss + entropy_bonus
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        
        # Update weights
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Action diversity (higher is better)
            action_diversity = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            # Confidence in best action
            max_prob = probs.max(dim=1)[0].mean()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.mean().item(),
            'action_diversity': action_diversity.item(),
            'max_prob': max_prob.item(),
            'batch_size': len(data)
        }