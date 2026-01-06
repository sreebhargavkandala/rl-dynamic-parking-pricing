 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Dict, Optional
from collections import deque
import logging


logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience Replay Buffer for off-policy learning.
    
    Stores transitions and provides random sampling for stable training.
    This breaks correlations between consecutive samples and enables
    off-policy learning algorithms like DDPG.
    """
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of samples to return
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
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


class DDPGActor(nn.Module):
    """
    Deterministic Policy Network for DDPG.
    
    Maps state → action (deterministic).
    Uses tanh activation to bound actions to [-1, 1].
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DDPGCritic(nn.Module):
    """
    Q-Function Network for DDPG.
    
    Maps (state, action) → Q-value estimate.
    Uses Batch Normalization for stability.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # First layer processes state
        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Second layer combines with action
        self.action_layers = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for layer in self.state_layers:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -3e-3, 3e-3)
        
        for layer in self.action_layers:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_features = self.state_layers(state)
        combined = torch.cat([state_features, action], dim=-1)
        q_value = self.action_layers(combined)
        return q_value


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck Process for exploration.
    
    Generates correlated noise for smooth exploration in continuous control.
    Better than Gaussian noise because it maintains temporal correlation.
    
    dx_t = θ(μ - x_t)dt + σ dW_t
    """
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x = np.ones(size) * mu
    
    def sample(self) -> np.ndarray:
        """Generate next sample."""
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.size)
        self.x = self.x + dx
        return self.x
    
    def reset(self):
        """Reset process."""
        self.x = np.ones(self.size) * self.mu


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient Agent.
    
    Off-policy actor-critic algorithm for continuous control.
    
    Components:
    1. Actor (μ): Deterministic policy π(a|s) = μ(s)
    2. Critic (Q): Action-value function Q(s, a)
    3. Target networks: μ' and Q' for training stability
    4. Replay buffer: For off-policy learning
    5. Exploration noise: Ornstein-Uhlenbeck process
    
    Algorithm:
    1. Collect experience with actor + noise
    2. Store in replay buffer
    3. Sample random batch from buffer
    4. Update critic: minimize MSE(Q_target - Q)
    5. Update actor: maximize E[Q(s, μ(s))]
    6. Soft update target networks: θ' ← τθ + (1-τ)θ'
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.001,
        hidden_dim: int = 256,
        device: str = "cpu"
    ):
        """
        Initialize DDPG Agent.
        
        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            max_action: Maximum action value
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            tau: Soft update coefficient
            hidden_dim: Hidden layer dimension
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        
        # Actor networks
        self.actor = DDPGActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_actor = DDPGActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        # Critic networks
        self.critic = DDPGCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = DDPGCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Exploration
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Statistics
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.total_updates = 0
    
    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        Select action using actor network + exploration noise.
        
        Args:
            state: Current observation
            noise_scale: Standard deviation of exploration noise
        
        Returns:
            Action to execute
        """
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()
        
        # Add exploration noise
        noise = self.noise.sample() * noise_scale
        action = action + noise
        
        # Clip to valid range
        action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                         next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """
        Soft update target network using Polyak averaging.
        
        θ' ← τθ + (1-τ)θ'
        
        This provides training stability by slowly moving targets.
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Perform one update step.
        
        Updates both actor and critic using sampled batch from replay buffer.
        
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
        
        # Critic update
        with torch.no_grad():
            # Target Q-value: r + γ(1-d)Q'(s', μ'(s'))
            next_actions = self.target_actor(next_states_t)
            target_q = rewards_t + self.gamma * (1 - dones_t) * self.target_critic(
                next_states_t, next_actions
            )
        
        # Current Q-value
        q_value = self.critic(states_t, actions_t)
        
        # Critic loss
        critic_loss = nn.MSELoss()(q_value, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor update (policy gradient)
        # J = E[Q(s, μ(s))] - maximize
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        # Record metrics
        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())
        self.total_updates += 1
        
        if self.total_updates % 100 == 0:
            logger.info(
                f"DDPG Update {self.total_updates}: "
                f"Actor Loss={actor_loss.item():.4f}, "
                f"Critic Loss={critic_loss.item():.4f}"
            )
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "total_updates": self.total_updates
        }
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }, path)
        logger.info(f"DDPG Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        logger.info(f"DDPG Agent loaded from {path}")
