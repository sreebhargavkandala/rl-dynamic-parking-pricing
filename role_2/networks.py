from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Policy network for Actor-Critic algorithm (Gaussian policy).
    
    Outputs mean and standard deviation of action distribution.
    Learns continuous action policy for parking pricing.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        """Initialize policy network.
        
        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            hidden_dim: Size of hidden layers (default: 128)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Main network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output heads
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for layer in [self.fc1, self.fc2, self.mean]:
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(layer.bias, 0.0)
        # Initialize log_std to small negative value for exploration
        nn.init.constant_(self.log_std, -1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            mean: Action mean of shape (batch_size, action_dim)
            std: Action standard deviation of shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        # Clamp log_std for numerical stability
        std = torch.exp(torch.clamp(self.log_std, min=-20, max=2))
        
        return mean, std


class ValueNetwork(nn.Module):
    """Value network for Actor-Critic algorithm.
    
    Estimates state value function V(s) for advantage computation.
    Helps stabilize policy gradient learning.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        """Initialize value network.
        
        Args:
            state_dim: Dimensionality of state space
            hidden_dim: Size of hidden layers (default: 128)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Main network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for layer in [self.fc1, self.fc2, self.value]:
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            value: State value estimate of shape (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value
