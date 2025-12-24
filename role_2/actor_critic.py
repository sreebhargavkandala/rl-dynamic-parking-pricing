from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .networks import PolicyNetwork, ValueNetwork


class ActorCriticAgent:
    """Actor-Critic Agent for continuous control.
    
    Implements Advantage Actor-Critic (A2C) algorithm with:
    - Gaussian policy network (actor)
    - Value function network (critic)
    - Advantage-based learning
    - Numerical stability improvements
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        device: str = "cpu"
    ) -> None:
        """Initialize Actor-Critic agent.
        
        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            lr: Learning rate for optimizers (default: 3e-4)
            gamma: Discount factor (default: 0.99)
            entropy_coef: Coefficient for entropy regularization (default: 0.01)
            device: Device to run on ('cpu' or 'cuda')
            
        Raises:
            ValueError: If learning rates are not positive
        """
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0 <= gamma <= 1):
            raise ValueError(f"Gamma must be in [0, 1], got {gamma}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)

        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)

        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Loss tracking
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Select action from policy network.
        
        Args:
            state: Current state observation (shape: state_dim,)
            
        Returns:
            action: Sampled action (numpy array, shape: action_dim,)
            log_prob: Log probability of sampled action
            
        Raises:
            ValueError: If state shape doesn't match expected state_dim
        """
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"State dimension mismatch. Expected {self.state_dim}, got {state.shape[0]}"
            )
        
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            mean, std = self.policy_net(state_tensor)
        
        dist = Normal(mean, std)
        action = dist.rsample()  # Use rsample for gradient flow
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Clamp action to valid range if needed
        action_np = action.squeeze(0).cpu().numpy()

        return action_np, log_prob

    def update(
        self,
        state: np.ndarray,
        action_log_prob: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        max_grad_norm: float = 10.0
    ) -> Dict[str, float]:
        """Update agent using single transition.
        
        Args:
            state: Current state
            action_log_prob: Log probability of action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            max_grad_norm: Maximum gradient norm for clipping (default: 10.0)
            
        Returns:
            Dictionary with losses: {'policy_loss': float, 'value_loss': float, 'entropy': float}
        """
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)

        # Compute value estimates
        value = self.value_net(state_tensor)
        with torch.no_grad():
            next_value = self.value_net(next_state_tensor)

        # Compute TD target with clipped reward for stability
        target = reward_tensor + self.gamma * next_value * (1 - done_tensor)
        advantage = (target.detach() - value).clamp(-10, 10)  # Clamp advantage

        # ----- Policy Loss -----
        # Entropy regularization for exploration
        mean, std = self.policy_net(state_tensor)
        dist = Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1)
        
        policy_loss = -(action_log_prob * advantage.detach()).mean()
        policy_loss = policy_loss - self.entropy_coef * entropy.mean()  # Entropy bonus

        # ----- Policy Update with Gradient Clipping -----
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)
        self.policy_optimizer.step()

        # ----- Value Loss -----
        value_loss = nn.functional.mse_loss(value, target.detach())
        
        # Add L2 regularization for stability
        l2_penalty = 0.01 * sum(p.pow(2).sum() for p in self.value_net.parameters())
        value_loss = value_loss + l2_penalty

        # ----- Value Update with Gradient Clipping -----
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), max_grad_norm)
        self.value_optimizer.step()

        # Track metrics
        policy_loss_val = policy_loss.item()
        value_loss_val = value_loss.item()
        entropy_val = entropy.item()
        
        self.policy_loss_history.append(policy_loss_val)
        self.value_loss_history.append(value_loss_val)
        self.entropy_history.append(entropy_val)

        return {
            'policy_loss': policy_loss_val,
            'value_loss': value_loss_val,
            'entropy': entropy_val
        }

    def save(self, filepath: str) -> None:
        """Save agent networks to file.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load agent networks from file.
        
        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        print(f"Agent loaded from {filepath}")

    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics.
        
        Returns:
            Dictionary with average losses over last episode
        """
        if not self.policy_loss_history:
            return {}
        
        return {
            'avg_policy_loss': np.mean(self.policy_loss_history[-100:]),
            'avg_value_loss': np.mean(self.value_loss_history[-100:]),
            'avg_entropy': np.mean(self.entropy_history[-100:])
        }

    def reset_metrics(self) -> None:
        """Reset training history."""
        self.policy_loss_history.clear()
        self.value_loss_history.clear()
        self.entropy_history.clear()
