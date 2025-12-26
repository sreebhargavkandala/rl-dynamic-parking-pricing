"""
Evaluation Metrics Module
=========================

Provides metric computation functions for RL agent evaluation.
"""

import numpy as np
from typing import List, Dict


def compute_revenue_metrics(revenues: List[float]) -> Dict[str, float]:
    """Compute revenue statistics."""
    return {
        'mean': float(np.mean(revenues)),
        'std': float(np.std(revenues)),
        'min': float(np.min(revenues)),
        'max': float(np.max(revenues))
    }


def compute_occupancy_metrics(occupancies: List[float]) -> Dict[str, float]:
    """Compute occupancy statistics."""
    return {
        'mean': float(np.mean(occupancies)),
        'std': float(np.std(occupancies)),
        'min': float(np.min(occupancies)),
        'max': float(np.max(occupancies))
    }


def compute_price_volatility(prices: List[float]) -> float:
    """Compute price volatility as standard deviation of price changes."""
    if len(prices) < 2:
        return 0.0
    price_changes = np.diff(prices)
    return float(np.std(price_changes))


def compute_all_metrics(
    revenues: List[float],
    occupancies: List[float],
    prices: List[float]
) -> Dict[str, Dict]:
    """Compute all evaluation metrics."""
    return {
        'revenue': compute_revenue_metrics(revenues),
        'occupancy': compute_occupancy_metrics(occupancies),
        'price_volatility': compute_price_volatility(prices)
    }
