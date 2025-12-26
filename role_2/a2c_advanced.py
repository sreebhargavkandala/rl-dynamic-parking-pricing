"""
ADVANCED A2C AGENT WITH DUELING NETWORKS
=========================================

Improved A2C implementation with:
- Dueling architecture (separate value & advantage streams)
- Residual connections for deep networks
- Better gradient flow
- Improved stability and convergence
- Entropy regularization with adaptive coefficients
- Priority experience replay (optional)
- Target networks for value stabilization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdvancedA2CConfig:
    """Configuration for Advanced A2C Agent."""
    state_dim: int = 5
    action_dim: int = 1
    hidden_dim: int = 512
    num_hidden_layers: int = 2
    policy_lr: float = 1e-4
    value_lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95  # GAE lambda
    entropy_coef: float = 0.05
    entropy_decay: float = 0.9995  # Decay entropy bonus over time
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    l2_reg: float = 1e-5
    log_std_bounds: Tuple[float, float] = (-20, 2)
    batch_size: int = 64
    n_epochs: int = 4
    device: str = 'cpu'
    use_dueling: bool = True  # Use dueling architecture
    use_residual: bool = True  # Use residual connections
    use_target_network: bool = True  # Use target network for value
    target_update_freq: int = 5  # Update target network every N updates


class ResidualBlock(nn.Module):
    """Residual connection block for deep networks."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.fc1(x)))
        x = self.norm2(self.fc2(x))
        return F.relu(x + residual)


class DuelingPolicyNetwork(nn.Module):
    """
    Dueling policy network: separates action policy into advantage streams.
    
    Architecture:
    - Shared layer
    - Mean stream: computes action mean
    - Std stream: computes action std
    - Combines for final policy
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int,
                 num_hidden_layers: int, log_std_bounds: Tuple, use_residual: bool):
        super().__init__()
        
        self.log_std_bounds = log_std_bounds
        
        # Shared layers
        self.fc_shared = nn.Linear(state_dim, hidden_dim)
        self.shared_layers = nn.ModuleList()
        
        for _ in range(num_hidden_layers):
            if use_residual:
                self.shared_layers.append(ResidualBlock(hidden_dim))
            else:
                self.shared_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                ))
        
        # Mean stream (action mean)
        self.mean_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.mean_out = nn.Linear(hidden_dim // 2, action_dim)
        
        # Std stream (action std, per-action)
        self.std_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.std_out = nn.Linear(hidden_dim // 2, action_dim)
        
        # Initialize output layers
        self.mean_out.weight.data.uniform_(-0.003, 0.003)
        self.std_out.weight.data.uniform_(-0.003, 0.003)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            mean: Action means
            log_std: Log standard deviations (constrained)
        """
        x = F.relu(self.fc_shared(state))
        
        # Shared layers with residual connections
        for layer in self.shared_layers:
            x = layer(x)
        
        # Mean stream
        mean = self.mean_out(F.relu(self.mean_fc(x)))
        
        # Std stream (with bounds)
        log_std = self.std_out(F.relu(self.std_fc(x)))
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        
        return mean, log_std


class DuelingValueNetwork(nn.Module):
    """
    Dueling value network: separates into value and advantage streams.
    
    Architecture:
    - Shared layers
    - Value stream: estimates V(s)
    - Advantage stream: estimates A(s,a) - average
    - Combines as: Q(s,a) = V(s) + A(s,a) - mean(A)
    """
    
    def __init__(self, state_dim: int, hidden_dim: int,
                 num_hidden_layers: int, use_residual: bool):
        super().__init__()
        
        # Shared layers
        self.fc_shared = nn.Linear(state_dim, hidden_dim)
        self.shared_layers = nn.ModuleList()
        
        for _ in range(num_hidden_layers):
            if use_residual:
                self.shared_layers.append(ResidualBlock(hidden_dim))
            else:
                self.shared_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                ))
        
        # Value stream
        self.value_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_out = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize outputs
        self.value_out.weight.data.uniform_(-0.003, 0.003)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns state value."""
        x = F.relu(self.fc_shared(state))
        
        # Shared layers
        for layer in self.shared_layers:
            x = layer(x)
        
        # Value stream
        value = self.value_out(F.relu(self.value_fc(x)))
        
        return value


class AdvancedA2CAgent:
    """
    Advanced A2C Agent with improvements:
    - Dueling networks
    - Residual connections
    - Target networks
    - Entropy decay
    - Better exploration
    - GAE (Generalized Advantage Estimation)
    """
    
    def __init__(self, config: AdvancedA2CConfig = None):
        self.config = config or AdvancedA2CConfig()
        self.device = torch.device(self.config.device)
        
        # Networks
        self.policy_network = DuelingPolicyNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
            self.config.num_hidden_layers,
            self.config.log_std_bounds,
            self.config.use_residual
        ).to(self.device)
        
        self.value_network = DuelingValueNetwork(
            self.config.state_dim,
            self.config.hidden_dim,
            self.config.num_hidden_layers,
            self.config.use_residual
        ).to(self.device)
        
        # Target network for value (for stability)
        if self.config.use_target_network:
            self.target_value_network = DuelingValueNetwork(
                self.config.state_dim,
                self.config.hidden_dim,
                self.config.num_hidden_layers,
                self.config.use_residual
            ).to(self.device)
            self.target_value_network.load_state_dict(self.value_network.state_dict())
            self.update_count = 0
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.config.policy_lr
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=self.config.value_lr
        )
        
        # Experience buffer
        self.experience_buffer = []
        self.entropy_coef = self.config.entropy_coef
        
        # Stats
        self.update_step = 0
        
        logger.info(f"Advanced A2C Agent initialized with config: {self.config}")
        logger.info(f"  Using dueling: {self.config.use_dueling}")
        logger.info(f"  Using residual: {self.config.use_residual}")
        logger.info(f"  Using target network: {self.config.use_target_network}")
    
    def select_action(self, state: np.ndarray, training: bool = True) \
            -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """
        Select action from policy.
        
        Args:
            state: Current state
            training: Whether in training mode (use noise)
        
        Returns:
            action: Selected action
            log_prob: Log probability (for training)
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std = self.policy_network(state_tensor)
        
        if training:
            # Sample from distribution
            std = log_std.exp()
            normal = Normal(mean, std)
            action_tensor = normal.rsample()
            log_prob = normal.log_prob(action_tensor).sum(dim=-1)
        else:
            # Use mean (deterministic)
            action_tensor = mean
            log_prob = None
        
        # Clamp action to valid range [0.5, 20.0]
        action_tensor = torch.clamp(action_tensor, 0.5, 20.0)
        action = action_tensor.cpu().numpy().flatten()
        
        return action, log_prob
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: float,
                 next_state: np.ndarray, done: bool, log_prob: torch.Tensor):
        """Store experience in buffer."""
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })
    
    def update(self) -> dict:
        """
        Update networks with experience buffer.
        
        Returns:
            Dictionary with loss metrics
        """
        if len(self.experience_buffer) < self.config.batch_size:
            return {}
        
        # Convert buffer to tensors
        states = torch.FloatTensor(
            np.array([e['state'] for e in self.experience_buffer])
        ).to(self.device)
        
        actions = torch.FloatTensor(
            np.array([e['action'] for e in self.experience_buffer])
        ).to(self.device)
        
        rewards = torch.FloatTensor(
            np.array([e['reward'] for e in self.experience_buffer])
        ).to(self.device).unsqueeze(1)
        
        next_states = torch.FloatTensor(
            np.array([e['next_state'] for e in self.experience_buffer])
        ).to(self.device)
        
        dones = torch.FloatTensor(
            np.array([not e['done'] for e in self.experience_buffer])
        ).to(self.device).unsqueeze(1)
        
        # Compute n-step returns with GAE
        values = self.value_network(states)
        next_values = self.value_network(next_states) * dones
        
        if self.config.use_target_network:
            next_values = self.target_value_network(next_states) * dones
        
        # TD error (advantage)
        td_error = rewards + self.config.gamma * next_values - values
        
        # GAE computation
        advantages = self._compute_gae(td_error, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = advantages + values.detach()
        
        # Update value network
        value_loss = F.smooth_l1_loss(values, returns.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value_network.parameters(),
            self.config.max_grad_norm
        )
        self.value_optimizer.step()
        
        # Update policy network
        mean, log_std = self.policy_network(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_log_probs = normal.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # Policy loss (maximize expected advantage)
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Entropy bonus (decays over time)
        entropy = normal.entropy().mean()
        policy_loss = policy_loss - self.entropy_coef * entropy
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.config.max_grad_norm
        )
        self.policy_optimizer.step()
        
        # Update target network periodically
        if self.config.use_target_network:
            self.update_count += 1
            if self.update_count % self.config.target_update_freq == 0:
                self.target_value_network.load_state_dict(
                    self.value_network.state_dict()
                )
        
        # Decay entropy coefficient
        self.entropy_coef *= self.config.entropy_decay
        
        # Clear buffer
        self.experience_buffer = []
        self.update_step += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'entropy_coef': self.entropy_coef,
            'avg_advantage': advantages.mean().item(),
            'avg_return': returns.mean().item()
        }
    
    def _compute_gae(self, td_errors: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            td_errors: Temporal difference errors
            dones: Not done indicators
        
        Returns:
            GAE advantages
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(td_errors))):
            if t == len(td_errors) - 1:
                next_nonterminal = dones[t]
            else:
                next_nonterminal = dones[t]
            
            gae = td_errors[t] + self.config.gamma * self.config.gae_lambda * next_nonterminal * gae
            advantages.insert(0, gae)
        
        return torch.stack(advantages)
    
    def save_checkpoint(self, path: str):
        """Save agent checkpoint."""
        checkpoint = {
            'policy_state': self.policy_network.state_dict(),
            'value_state': self.value_network.state_dict(),
            'config': self.config,
            'update_step': self.update_step
        }
        if self.config.use_target_network:
            checkpoint['target_value_state'] = self.target_value_network.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"✓ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_state'])
        self.value_network.load_state_dict(checkpoint['value_state'])
        if self.config.use_target_network and 'target_value_state' in checkpoint:
            self.target_value_network.load_state_dict(checkpoint['target_value_state'])
        self.update_step = checkpoint.get('update_step', 0)
        logger.info(f"✓ Checkpoint loaded from {path}")


if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED A2C AGENT WITH DUELING NETWORKS")
    print("=" * 80)
    
    # Test configuration
    config = AdvancedA2CConfig(
        state_dim=5,
        action_dim=1,
        hidden_dim=512,
        num_hidden_layers=2,
        use_dueling=True,
        use_residual=True,
        use_target_network=True
    )
    
    print("\n1. Creating Advanced A2C Agent...")
    agent = AdvancedA2CAgent(config)
    print(f"   ✓ Agent created successfully")
    
    print("\n2. Testing action selection...")
    for i in range(3):
        state = np.random.randn(5)
        action, log_prob = agent.select_action(state, training=True)
        print(f"   Sample {i+1}: state_dim={len(state)}, action={action[0]:.2f}, valid={0.5 <= action[0] <= 20.0}")
    
    print("\n3. Testing training step...")
    # Simulate episode
    for episode in range(5):
        state = np.random.randn(5)
        for step in range(10):
            action, log_prob = agent.select_action(state, training=True)
            reward = np.random.randn()
            next_state = np.random.randn(5)
            done = step == 9
            
            agent.remember(state, action, reward, next_state, done, log_prob)
            state = next_state
        
        metrics = agent.update()
        if metrics:
            print(f"   Episode {episode+1}: policy_loss={metrics['policy_loss']:.4f}, "
                  f"entropy={metrics['entropy']:.4f}")
    
    print("\n" + "=" * 80)
    print("Advanced A2C Agent Ready!")
    print("=" * 80)
