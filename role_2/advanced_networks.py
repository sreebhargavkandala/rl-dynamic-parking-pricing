 

import torch
import torch.nn as nn
import numpy as np
import math


class Mish(nn.Module):
    """
    Mish activation function.
    
    Better alternative to ReLU with smoother gradients.
    f(x) = x * tanh(softplus(x))
    """
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class SwiGLU(nn.Module):
    """
    SwiGLU activation: gating mechanism for better feature selection.
    
    Used in modern transformers and deep networks.
    More expressive than standard MLP layers.
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out * 2)
    
    def forward(self, x):
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


def orthogonal_init(module, gain=np.sqrt(2)):
    """
    Orthogonal initialization for better convergence.
    
    Ensures weight matrices have good conditioning and
    helps avoid gradient flow problems in deep networks.
    
    Reference: "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    return module


class AdvancedPolicyNetwork(nn.Module):
    """
    State-of-the-art Policy Network for continuous control.
    
    Features:
    - Batch normalization for stability
    - Mish activations for smooth gradients
    - Proper initialization
    - Output layer for action distribution
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "relu",
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "mish":
            self.activation = Mish
        elif activation == "elu":
            self.activation = nn.ELU
        else:
            self.activation = nn.ReLU
        
        # Build network
        layers = []
        in_dim = state_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation())
            in_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
        # Output layers for Gaussian policy
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize output layers with small weights
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)
        
        # Apply orthogonal initialization to hidden layers
        self.net.apply(orthogonal_init)
        
        # Log std bounds for numerical stability
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state: torch.Tensor) -> tuple:
        """
        Forward pass returning mean and log_std.
        
        Args:
            state: State observation
        
        Returns:
            mean: Policy mean (action distribution center)
            log_std: Log standard deviation (clamped for stability)
        """
        features = self.net(state)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), self.log_std_min, self.log_std_max)
        return mean, log_std


class AdvancedValueNetwork(nn.Module):
    """
    State-of-the-art Value Network for baseline subtraction.
    
    Maps state → scalar value estimate.
    Uses modern architectures and initialization.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "relu",
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        # Build network
        layers = []
        in_dim = state_dim
        
        if activation == "relu":
            act = nn.ReLU
        elif activation == "mish":
            act = Mish
        else:
            act = nn.ReLU
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act())
            in_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
        # Output layer for value
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize output with small weights
        nn.init.uniform_(self.value_head.weight, -3e-3, 3e-3)
        
        # Apply orthogonal initialization
        self.net.apply(orthogonal_init)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning value estimate.
        
        Args:
            state: State observation
        
        Returns:
            value: Estimated state value V(s)
        """
        features = self.net(state)
        value = self.value_head(features)
        return value


class DuelingNetwork(nn.Module):
    """
    Dueling Network Architecture.
    
    Separates value and advantage streams:
    Q(s,a) = V(s) + (A(s,a) - mean(A))
    
    This decomposition helps the network learn value and advantage
    more effectively, especially in environments with many actions
    where most actions have similar value.
    
    Paper: "Dueling Network Architectures for Deep Reinforcement Learning"
    (Wang et al., 2015) https://arxiv.org/abs/1511.06581
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Shared feature extraction
        layers = []
        in_dim = state_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Value stream: outputs single scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream: outputs A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize with orthogonal weights
        self.feature_net.apply(orthogonal_init)
        self.value_stream.apply(orthogonal_init)
        self.advantage_stream.apply(orthogonal_init)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values using dueling architecture.
        
        Q(s,a) = V(s) + [A(s,a) - mean(A(s,*))]
        
        The advantage subtraction (mean) ensures stable learning
        by removing scale differences between value and advantage.
        """
        features = self.feature_net(state)
        
        # Value component
        value = self.value_stream(features)
        
        # Advantage component
        advantages = self.advantage_stream(features)
        
        # Combine with mean-normalized advantages
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values


class NoiseInjectionNetwork(nn.Module):
    """
    Network with built-in noise injection for exploration.
    
    Uses noise layers at certain points to encourage exploration
    while maintaining deterministic inference.
    
    Useful for deep exploration in hard exploration problems.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        noise_std: float = 0.5
    ):
        super().__init__()
        
        self.noise_std = noise_std
        self.training_mode = True
        
        # Build network
        layers = []
        in_dim = state_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            # Add noise injection layers at certain depths
            if i % 2 == 0:
                layers.append(nn.Dropout(p=0.1))  # Dropout acts like noise
            in_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, action_dim)
        
        # Initialize
        self.net.apply(orthogonal_init)
        nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
        """
        Forward pass with optional noise injection.
        
        Args:
            state: State observation
            noise_scale: Scaling factor for injected noise
        
        Returns:
            Output with optionally injected noise
        """
        features = self.net(state)
        output = self.output(features)
        
        if self.training_mode and noise_scale > 0:
            # Inject Gaussian noise proportional to output magnitude
            noise = torch.randn_like(output) * self.noise_std * noise_scale
            output = output + noise
        
        return output


class ResidualPolicyNetwork(nn.Module):
    """
    Deep Policy Network with Residual Connections.
    
    Residual connections allow training very deep networks
    by enabling gradient flow through skip connections.
    
    f(x) = x + g(x)
    
    Helps with:
    - Training stability in deep networks
    - Better feature reuse
    - Faster convergence
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_residual_blocks: int = 4
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            self._build_residual_block(hidden_dim)
            for _ in range(num_residual_blocks)
        ])
        
        # Output layers
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize
        self.input_proj.apply(orthogonal_init)
        for block in self.residual_blocks:
            block.apply(orthogonal_init)
        
        self.log_std_min = -20
        self.log_std_max = 2
    
    def _build_residual_block(self, dim: int) -> nn.Module:
        """Build a single residual block."""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, state: torch.Tensor) -> tuple:
        """
        Forward pass with residual connections.
        
        x_{i+1} = x_i + f(x_i)
        """
        x = self.input_proj(state)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = x + block(x)  # Residual connection
        
        # Output
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        
        return mean, log_std


# Factory function for network creation
def create_policy_network(
    state_dim: int,
    action_dim: int,
    architecture: str = "advanced",
    **kwargs
) -> nn.Module:
    """
    Factory function to create policy networks.
    
    Args:
        state_dim: State dimensionality
        action_dim: Action dimensionality
        architecture: "advanced", "residual", or "dueling"
        **kwargs: Additional arguments for the network
    
    Returns:
        Policy network module
    """
    if architecture == "advanced":
        return AdvancedPolicyNetwork(state_dim, action_dim, **kwargs)
    elif architecture == "residual":
        return ResidualPolicyNetwork(state_dim, action_dim, **kwargs)
    elif architecture == "dueling":
        return DuelingNetwork(state_dim, action_dim, **kwargs)
    else:
        return AdvancedPolicyNetwork(state_dim, action_dim, **kwargs)


def create_value_network(
    state_dim: int,
    architecture: str = "advanced",
    **kwargs
) -> nn.Module:
    """Create value network."""
    if architecture == "advanced":
        return AdvancedValueNetwork(state_dim, **kwargs)
    else:
        return AdvancedValueNetwork(state_dim, **kwargs)
