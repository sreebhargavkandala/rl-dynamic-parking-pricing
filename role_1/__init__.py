"""
Role 1: Problem Formulation & Environment

Complete RL environment for dynamic parking pricing with:
- State/action/reward formulation
- Parking lot simulator
- Metrics computation
- Reward functions
"""

from .env import ParkingPricingEnv
from .metrics import compute_all_metrics, ParkingMetrics
from .reward_function import RewardFunction

__version__ = "1.0.0"
__all__ = [
    "ParkingPricingEnv",
    "compute_all_metrics",
    "ParkingMetrics",
    "RewardFunction",
]
