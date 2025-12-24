"""
SAC (Soft Actor-Critic) - State-of-the-Art Maximum Entropy RL

Implementation of Soft Actor-Critic algorithm for continuous control with:
- Maximum entropy RL framework for automatic exploration-exploitation tradeoff
- Dual critic networks for stability (clipped double Q-learning)
- Automatic temperature adjustment for entropy coefficient
- Off-policy learning with experience replay
- Superior sample efficiency and robustness

Paper: "Soft Actor-Critic: Off-Policy Deep Reinforcement Learning with a Stochastic Actor"
(Haarnoja et al., 2018) https://arxiv.org/abs/1801.01290

Key Innovations:
1. Maximum entropy framework: maximize E[π(·|s)] + αH(π(·|s))
2. Stochastic actor π(a|s) for better exploration
3. Dual critics for reduced overestimation
4. Automatic temperature tuning for entropy coefficient α
5. Off-policy learning with Gaussian policy

Why SAC is State-of-the-Art:
- Best sample efficiency among all continuous RL algorithms
- Robust to hyperparameters
- Automatic exploration through entropy maximization
- Used in robotics and real-world applications
- Superior to PPO, DDPG on most benchmarks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Dict, Optional
from collections import deque
import logging


logger = logging.getLogger(__name__)


class SACReplayBuffer:
    """
    Experience Replay Buffer for SAC.
    
    Optimized for off-policy learning with efficient sampling.
    """
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class SACPolicy(nn.Module):
    """
    Stochastic Policy Network for SAC.
    
    Maps state → (mean, log_std) for Gaussian policy.
    Uses tanh squashing for bounded actions.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.mean_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std_layer.weight.data.uniform_(-3e-3, 3e-3)
        
        # Log standard deviation bounds
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return mean and log_std of Gaussian policy.
        
        Returns:
            mean: Policy mean
            log_std: Log standard deviation (clamped for stability)
        """
        features = self.net(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Uses reparameterization trick: a = tanh(μ + σ*ε)
        Also computes log probability accounting for tanh squashing.
        
        Returns:
            action: Sampled action (bounded to [-1, 1])
            log_prob: Log probability of sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        z = torch.randn_like(mean)
        action_unbounded = mean + std * z
        action = torch.tanh(action_unbounded)
        
        # Log probability with tanh squashing correction
        log_prob = Normal(mean, std).log_prob(action_unbounded).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action, log_prob


class SACCritic(nn.Module):
    """
    Q-Function Network for SAC.
    
    Maps (state, action) → Q-value.
    SAC uses two critic networks for clipped double Q-learning.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for layer in self.state_net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-3e-3, 3e-3)
        
        for layer in self.action_net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_features = self.state_net(state)
        combined = torch.cat([state_features, action], dim=-1)
        q_value = self.action_net(combined)
        return q_value


class SACAgent:
    """
    Soft Actor-Critic Agent.
    
    State-of-the-art algorithm for continuous control.
    
    Components:
    1. Policy π(a|s): Stochastic Gaussian policy
    2. Two Q-networks: Q₁(s,a) and Q₂(s,a) (twin critics)
    3. Target Q-networks: Q₁' and Q₂' (for stability)
    4. Automatic entropy tuning: α_t (learns entropy coefficient)
    5. Replay buffer: For off-policy learning
    
    Objective (Maximum Entropy RL):
    J(π) = E[log π(a|s) - Q(s,a)]
    
    Where α is automatically tuned to maintain target entropy.
    
    Update Rules:
    1. Actor: ∇J(π) = ∇[α log π(a|s) - Q(s,a)]
    2. Critic: Minimize MSE(Q_target - Q), where
       Q_target = r + γ min(Q₁'(s',a'), Q₂'(s',a')) - α log π(a'|s')
    3. Temperature: Tune α to maximize entropy + exploration
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden_dim: int = 256,
        device: str = "cpu",
        target_entropy: Optional[float] = None
    ):
        """
        Initialize SAC Agent.
        
        Args:
            state_dim: State dimensionality
            action_dim: Action dimensionality
            max_action: Maximum action value
            actor_lr: Policy learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            hidden_dim: Hidden layer size
            device: PyTorch device
            target_entropy: Target entropy for temperature tuning
                           (default: -action_dim)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        
        # Target entropy for automatic temperature tuning
        self.target_entropy = target_entropy or -action_dim
        
        # Policy network
        self.policy = SACPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Dual critic networks (twin Q-learning)
        self.critic1 = SACCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = SACCritic(state_dim, action_dim, hidden_dim).to(self.device)
        
        self.target_critic1 = SACCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = SACCritic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Automatic temperature tuning
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=actor_lr)
        
        # Replay buffer
        self.replay_buffer = SACReplayBuffer()
        
        # Statistics
        self.policy_loss_history = []
        self.critic_loss_history = []
        self.alpha_history = []
        self.total_updates = 0
    
    @property
    def alpha(self) -> float:
        """Get current temperature coefficient."""
        return self.log_alpha.exp().item()
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action from policy.
        
        Args:
            state: Current observation
            deterministic: If True, use mean action (for evaluation)
        
        Returns:
            Action to execute
        """
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy.sample(state_tensor)
        
        action = action.squeeze(0).cpu().numpy() * self.max_action
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update via Polyak averaging."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Perform one update step.
        
        Updates actor, critics, and temperature coefficient.
        
        Args:
            batch_size: Mini-batch size
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {"status": "insufficient_data"}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device).unsqueeze(-1)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device).unsqueeze(-1)
        
        # ===== Update Critics =====
        with torch.no_grad():
            # Sample next action
            next_action, next_log_prob = self.policy.sample(next_states_t)
            
            # Compute target Q-value using minimum of dual critics
            target_q1 = self.target_critic1(next_states_t, next_action)
            target_q2 = self.target_critic2(next_states_t, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # Q_target = r + γ(1-d)[Q(s',a') - α log π(a'|s')]
            alpha = self.log_alpha.exp()
            target_q = rewards_t + self.gamma * (1 - dones_t) * (
                target_q - alpha * next_log_prob
            )
        
        # Critic 1 loss
        q1_value = self.critic1(states_t, actions_t)
        critic1_loss = nn.MSELoss()(q1_value, target_q)
        
        # Critic 2 loss
        q2_value = self.critic2(states_t, actions_t)
        critic2_loss = nn.MSELoss()(q2_value, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # ===== Update Policy =====
        action, log_prob = self.policy.sample(states_t)
        
        q1 = self.critic1(states_t, action)
        q2 = self.critic2(states_t, action)
        q = torch.min(q1, q2)
        
        alpha = self.log_alpha.exp()
        policy_loss = (alpha * log_prob - q).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # ===== Automatic Temperature Tuning =====
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        # Record metrics
        self.policy_loss_history.append(policy_loss.item())
        self.critic_loss_history.append((critic1_loss.item() + critic2_loss.item()) / 2)
        self.alpha_history.append(self.alpha)
        self.total_updates += 1
        
        if self.total_updates % 100 == 0:
            logger.info(
                f"SAC Update {self.total_updates}: "
                f"Policy Loss={policy_loss.item():.4f}, "
                f"Critic Loss={(critic1_loss.item() + critic2_loss.item()) / 2:.4f}, "
                f"α={self.alpha:.4f}"
            )
        
        return {
            "policy_loss": policy_loss.item(),
            "critic_loss": (critic1_loss.item() + critic2_loss.item()) / 2,
            "alpha": self.alpha,
            "total_updates": self.total_updates
        }
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            "policy": self.policy.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.data,
        }, path)
        logger.info(f"SAC Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])
        self.log_alpha.data = checkpoint["log_alpha"]
        logger.info(f"SAC Agent loaded from {path}")
