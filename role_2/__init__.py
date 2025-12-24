"""
Role 2: RL Algorithm (Actor-Critic) Implementation

Production-ready implementation of Advantage Actor-Critic (A2C) algorithm.
"""

from .actor_critic import ActorCriticAgent
from .networks import PolicyNetwork, ValueNetwork
from .train import Trainer

__version__ = "1.0.0"
__all__ = [
    "ActorCriticAgent",
    "PolicyNetwork",
    "ValueNetwork",
    "Trainer",
]
