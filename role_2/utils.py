"""
Utility functions for Actor-Critic training and evaluation.
"""

import numpy as np
from typing import List, Dict, Tuple
import torch


def compute_running_mean_std(
    data: np.ndarray,
    window_size: int = 100
) -> Tuple[float, float]:
    """Compute running mean and standard deviation.
    
    Args:
        data: Array of values
        window_size: Size of rolling window (default: 100)
        
    Returns:
        Tuple of (mean, std_dev)
    """
    if len(data) < window_size:
        return np.mean(data), np.std(data)
    
    windowed = data[-window_size:]
    return np.mean(windowed), np.std(windowed)


def normalize_advantages(
    advantages: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """Normalize advantages for stability.
    
    Args:
        advantages: Array of advantage values
        epsilon: Small constant for numerical stability
        
    Returns:
        Normalized advantages
    """
    return (advantages - np.mean(advantages)) / (np.std(advantages) + epsilon)


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lambda_gae: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of state values
        gamma: Discount factor
        lambda_gae: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = []
    gae = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_gae * gae
        advantages.insert(0, gae)
    
    advantages = np.array(advantages)
    returns = advantages + np.array(values)
    
    return advantages, returns


def clip_by_global_norm(
    tensors: List[torch.Tensor],
    max_norm: float = 10.0
) -> float:
    """Clip gradients by global norm.
    
    Args:
        tensors: List of tensors
        max_norm: Maximum gradient norm
        
    Returns:
        Global gradient norm
    """
    total_norm = 0.0
    for tensor in tensors:
        if tensor.grad is not None:
            param_norm = tensor.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    
    if total_norm > max_norm > 0:
        clip_coef = max_norm / (total_norm + 1e-6)
        for tensor in tensors:
            if tensor.grad is not None:
                tensor.grad.data.mul_(clip_coef)
    
    return total_norm


def soft_update(
    target_net: torch.nn.Module,
    source_net: torch.nn.Module,
    tau: float = 0.001
) -> None:
    """Soft update of target network (polyak averaging).
    
    Args:
        target_net: Target network to update
        source_net: Source network
        tau: Update coefficient (default: 0.001)
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            (1.0 - tau) * target_param.data + tau * source_param.data
        )


def format_metrics(metrics: Dict, decimal_places: int = 4) -> str:
    """Format metrics dictionary for printing.
    
    Args:
        metrics: Dictionary of metrics
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.{decimal_places}f}")
        else:
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)
