

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


class PPOMemory:
    """Memory buffer for storing transitions during episode collection.
    
    Efficiently stores states, actions, rewards, values, log-probs and masks
    for mini-batch sampling during PPO training.
    """
    
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
    def store(self, state, action, reward, value, log_prob, done):
        """Store a single transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def compute_gae(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE provides a bias-variance tradeoff for advantage estimation:
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        
        Where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD residual.
        
        This reduces variance compared to raw returns while keeping low bias.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE parameter (0-1). Higher = lower bias, higher variance
        """
        self.advantages = []
        self.returns = []
        
        gae = 0
        for t in reversed(range(len(self.rewards))):
            # Get next value (0 if done)
            next_value = self.values[t + 1] if t + 1 < len(self.values) else 0
            
            # Compute TD residual
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            
            # Update GAE accumulator
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + self.values[t])
        
        # Normalize advantages for stability
        advantages_array = np.array(self.advantages)
        self.advantages = (advantages_array - advantages_array.mean()) / (advantages_array.std() + 1e-8)
        self.advantages = self.advantages.tolist()
        
    def get_batches(self, batch_size: int = 64):
        """
        Generate mini-batches for SGD updates.
        
        Yields batches of transitions for PPO training.
        """
        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            yield (
                np.array([self.states[i] for i in batch_indices]),
                np.array([self.actions[i] for i in batch_indices]),
                np.array([self.returns[i] for i in batch_indices]),
                np.array([self.advantages[i] for i in batch_indices]),
                np.array([self.log_probs[i] for i in batch_indices])
            )
    
    def clear(self):
        """Clear the memory buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []


class PPOAgent:
   
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        policy_net: nn.Module,
        value_net: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize PPO Agent.
        
        Args:
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            policy_net: Policy network (must output mean and std)
            value_net: Value network
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE parameter
            clip_ratio: PPO clipping ratio ε
            entropy_coef: Entropy regularization coefficient
            value_coef: Value function loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: PyTorch device (cpu or cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        
        self.policy_net = policy_net.to(self.device)
        self.value_net = value_net.to(self.device)
        
        # Store old policy for ratio computation
        self.old_policy_net = type(policy_net)(state_dim, action_dim).to(self.device)
        self._copy_weights(self.policy_net, self.old_policy_net)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Memory
        self.memory = PPOMemory()
        
        # Statistics
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        self.total_updates = 0
        
    def _copy_weights(self, source: nn.Module, target: nn.Module):
        """Copy weights from source to target network."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(source_param.data)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Select action using the policy network.
        
        Args:
            state: Current observation
            training: If True, sample action. If False, use mean action.
        
        Returns:
            action: Sampled or mean action
            log_prob: Log probability of the action
            value: Value estimate V(s)
        """
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            mean, std = self.policy_net(state_tensor)
            value = self.value_net(state_tensor)
        
        if training:
            # Sample action during training
            dist = Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            # Use mean action during evaluation
            action = mean
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.squeeze(0).item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in memory."""
        self.memory.store(state, action, reward, value, log_prob, done)
    
    def update(self, epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Collects trajectories, computes advantages, and updates policy/value networks.
        
        Args:
            epochs: Number of training epochs over the collected data
            batch_size: Mini-batch size for SGD
        
        Returns:
            Dictionary with training metrics
        """
        # Compute advantages
        self.memory.compute_gae(self.gamma, self.gae_lambda)
        
        # Store old policy
        self._copy_weights(self.policy_net, self.old_policy_net)
        
        # Training loop
        policy_losses = []
        value_losses = []
        entropies = []
        
        for epoch in range(epochs):
            for batch in self.memory.get_batches(batch_size):
                states, actions, returns, advantages, old_log_probs = batch
                
                # Convert to tensors
                states_t = torch.FloatTensor(states).to(self.device)
                actions_t = torch.FloatTensor(actions).to(self.device)
                returns_t = torch.FloatTensor(returns).to(self.device).unsqueeze(-1)
                advantages_t = torch.FloatTensor(advantages).to(self.device).unsqueeze(-1)
                old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device).unsqueeze(-1)
                
                # Policy update
                mean, std = self.policy_net(states_t)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions_t).sum(dim=-1, keepdim=True)
                
                # Probability ratio
                ratio = torch.exp(new_log_probs - old_log_probs_t)
                
                # Clipped PPO objective
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus for exploration
                entropy = dist.entropy().mean()
                
                # Total policy loss
                total_policy_loss = policy_loss - self.entropy_coef * entropy
                
                # Value function loss
                values = self.value_net(states_t)
                value_loss = nn.MSELoss()(values, returns_t)
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                total_policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                # Record metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
        
        # Clear memory
        self.memory.clear()
        
        # Update statistics
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy = np.mean(entropies)
        
        self.policy_loss_history.append(avg_policy_loss)
        self.value_loss_history.append(avg_value_loss)
        self.entropy_history.append(avg_entropy)
        self.total_updates += 1
        
        logger.info(
            f"PPO Update {self.total_updates}: "
            f"Policy Loss={avg_policy_loss:.4f}, "
            f"Value Loss={avg_value_loss:.4f}, "
            f"Entropy={avg_entropy:.4f}"
        )
        
        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "total_updates": self.total_updates
        }
    
    def save(self, path: str):
        """Save agent to disk."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "optimizer_state": self.policy_optimizer.state_dict()
        }, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent from disk."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.policy_optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info(f"Agent loaded from {path}")
