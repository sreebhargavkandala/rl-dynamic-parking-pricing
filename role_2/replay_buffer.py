"""
Experience Replay Buffer for Actor-Critic Training

Stores and samples experiences for efficient mini-batch training.
Supports both on-policy and off-policy algorithms.
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: Optional[float] = None
    value: Optional[float] = None


class ReplayBuffer:
    """Experience replay buffer for RL training.
    
    Stores and samples experiences with priority sampling support.
    Useful for off-policy algorithms (DQN, SAC) and experience reuse.
    """
    
    def __init__(self, capacity: int = 10000) -> None:
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store (default: 10000)
            
        Raises:
            ValueError: If capacity is not positive
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: Optional[float] = None,
        value: Optional[float] = None
    ) -> None:
        """Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action (for on-policy methods)
            value: State value estimate (for on-policy methods)
        """
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        )
        self.buffer.append(experience)
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample random batch from buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            
        Raises:
            ValueError: If batch_size > buffer size
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Batch size {batch_size} exceeds buffer size {len(self.buffer)}"
            )
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[i] for i in indices]
        
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def sample_episode(self) -> List[Experience]:
        """Sample all experiences in episode order (for on-policy methods).
        
        Returns:
            List of all stored experiences
        """
        return list(self.buffer)
    
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) == self.capacity


class EpisodeBuffer:
    """Buffer for collecting single episode trajectory.
    
    Used in on-policy methods like A2C, PPO to collect trajectories
    before computing advantages and updating.
    """
    
    def __init__(self) -> None:
        """Initialize episode buffer."""
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: Optional[float] = None,
        value: Optional[float] = None
    ) -> None:
        """Add experience to episode buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action
            value: State value estimate
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)
    
    def get_trajectory(self) -> Dict[str, np.ndarray]:
        """Get complete trajectory as arrays.
        
        Returns:
            Dictionary with trajectory data
        """
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'next_states': np.array(self.next_states),
            'dones': np.array(self.dones, dtype=np.float32),
            'log_probs': np.array(self.log_probs) if self.log_probs else None,
            'values': np.array(self.values) if self.values else None,
        }
    
    def get_length(self) -> int:
        """Get number of steps in episode."""
        return len(self.states)
    
    def get_return(self, gamma: float = 0.99) -> float:
        """Get cumulative discounted return.
        
        Args:
            gamma: Discount factor
            
        Returns:
            Cumulative discounted return
        """
        returns = 0.0
        for i, reward in enumerate(self.rewards):
            returns += (gamma ** i) * reward
        return returns
    
    def clear(self) -> None:
        """Clear all trajectory data."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay Buffer.
    
    Implements prioritized experience sampling based on TD-error.
    Improves learning efficiency by focusing on important experiences.
    
    Reference: Schaul et al. (2015) "Prioritized Experience Replay"
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4
    ) -> None:
        """Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0=uniform, 1=full priority)
            beta: Importance sampling exponent
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float = 1.0,
        log_prob: Optional[float] = None,
        value: Optional[float] = None
    ) -> None:
        """Add experience with priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            td_error: TD-error for priority (default: max priority)
            log_prob: Log probability of action
            value: State value estimate
        """
        super().add(state, action, reward, next_state, done, log_prob, value)
        
        # Store priority
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample prioritized batch.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights)
            where weights are importance sampling weights
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Batch size {batch_size} exceeds buffer size {len(self.buffer)}"
            )
        
        # Compute sampling probabilities
        priorities = np.array(list(self.priorities))
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            p=probabilities,
            replace=False
        )
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences], dtype=np.float32)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize
        
        return states, actions, rewards, next_states, dones, weights
    
    def update_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        """Update priorities based on new TD-errors.
        
        Args:
            indices: Indices of experiences to update
            td_errors: New TD-errors for prioritization
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
