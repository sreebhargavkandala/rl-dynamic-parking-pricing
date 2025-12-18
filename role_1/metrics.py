"""
Metrics Functions
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ParkingMetrics:
    """Container for parking environment metrics."""
    total_revenue: float
    avg_occupancy: float
    occupancy_std: float
    min_occupancy: float
    max_occupancy: float
    price_volatility: float
    avg_price: float
    min_price: float
    max_price: float
    episode_length: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_revenue": self.total_revenue,
            "avg_occupancy": self.avg_occupancy,
            "occupancy_std": self.occupancy_std,
            "min_occupancy": self.min_occupancy,
            "max_occupancy": self.max_occupancy,
            "price_volatility": self.price_volatility,
            "avg_price": self.avg_price,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "episode_length": self.episode_length,
        }
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        lines = [
            "=" * 60,
            "PARKING PRICING METRICS",
            "=" * 60,
            f"Total Revenue:        ${self.total_revenue:>10.2f}",
            f"Average Occupancy:    {self.avg_occupancy:>10.1%}",
            f"Occupancy Std Dev:    {self.occupancy_std:>10.1%}",
            f"Min Occupancy:        {self.min_occupancy:>10.1%}",
            f"Max Occupancy:        {self.max_occupancy:>10.1%}",
            f"Price Volatility:     ${self.price_volatility:>10.2f}",
            f"Average Price:        ${self.avg_price:>10.2f}",
            f"Min Price:            ${self.min_price:>10.2f}",
            f"Max Price:            ${self.max_price:>10.2f}",
            f"Episode Length:       {self.episode_length:>10d} steps",
            "=" * 60,
        ]
        return "\n".join(lines)




def compute_revenue_metric(occupancies: List[float], prices: List[float], 
                           capacity: int = 100) -> float:
    """
    Compute total revenue over an episode.
    
    Revenue = Σ(occupancy_t × capacity × price_t)
    
    Args:
        occupancies: List of occupancy rates per step
        prices: List of prices per step
        capacity: Number of parking spots
    
    Returns:
        Total revenue in dollars
    """
    occupancies = np.array(occupancies)
    prices = np.array(prices)
    revenue = np.sum(occupancies * capacity * prices)
    return float(revenue)


def compute_occupancy_metrics(occupancies: List[float]) -> Dict[str, float]:
    """
    Compute occupancy-related metrics.
    
    Args:
        occupancies: List of occupancy rates per step
    
    Returns:
        Dict with: avg, std, min, max occupancy
    """
    occ_array = np.array(occupancies)
    return {
        "avg_occupancy": float(np.mean(occ_array)),
        "occupancy_std": float(np.std(occ_array)),
        "min_occupancy": float(np.min(occ_array)),
        "max_occupancy": float(np.max(occ_array)),
    }


def compute_price_metrics(prices: List[float]) -> Dict[str, float]:
    """
    Compute price-related metrics.
    
    Args:
        prices: List of prices per step
    
    Returns:
        Dict with: volatility, avg, min, max price
    """
    price_array = np.array(prices)
    return {
        "price_volatility": float(np.std(price_array)),
        "avg_price": float(np.mean(price_array)),
        "min_price": float(np.min(price_array)),
        "max_price": float(np.max(price_array)),
    }


def compute_all_metrics(occupancies: List[float], prices: List[float],
                        capacity: int = 100) -> ParkingMetrics:
    """
    Compute all parking metrics.
    
    Args:
        occupancies: List of occupancy rates per step
        prices: List of prices per step
        capacity: Number of parking spots (default 100)
    
    Returns:
        ParkingMetrics dataclass with all metrics
    """
    revenue = compute_revenue_metric(occupancies, prices, capacity)
    occ_metrics = compute_occupancy_metrics(occupancies)
    price_metrics = compute_price_metrics(prices)
    
    return ParkingMetrics(
        total_revenue=revenue,
        avg_occupancy=occ_metrics["avg_occupancy"],
        occupancy_std=occ_metrics["occupancy_std"],
        min_occupancy=occ_metrics["min_occupancy"],
        max_occupancy=occ_metrics["max_occupancy"],
        price_volatility=price_metrics["price_volatility"],
        avg_price=price_metrics["avg_price"],
        min_price=price_metrics["min_price"],
        max_price=price_metrics["max_price"],
        episode_length=len(occupancies),
    )




def analyze_occupancy_stability(occupancies: List[float], 
                                target_occupancy: float = 0.8) -> Dict[str, float]:
    """
    Analyze how well occupancy was controlled.
    
    Args:
        occupancies: List of occupancy rates per step
        target_occupancy: Target occupancy rate
    
    Returns:
        Dict with stability metrics
    """
    occ_array = np.array(occupancies)
    errors = np.abs(occ_array - target_occupancy)
    
    return {
        "target_occupancy": target_occupancy,
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "percent_within_5pct": float(
            np.sum(errors <= 0.05) / len(occ_array) * 100
        ),
        "percent_within_10pct": float(
            np.sum(errors <= 0.10) / len(occ_array) * 100
        ),
    }


def analyze_price_stability(prices: List[float]) -> Dict[str, float]:
    """
    Analyze price change patterns.
    
    Args:
        prices: List of prices per step
    
    Returns:
        Dict with price change statistics
    """
    price_array = np.array(prices)
    price_changes = np.abs(np.diff(price_array))
    
    return {
        "avg_price_change": float(np.mean(price_changes)),
        "max_price_change": float(np.max(price_changes)),
        "std_price_change": float(np.std(price_changes)),
        "percent_unchanged": float(
            np.sum(price_changes < 0.01) / len(price_changes) * 100
        ),
    }


def compute_revenue_per_spot(total_revenue: float, capacity: int = 100) -> float:
    """
    Compute average revenue per spot per hour.
    
    Args:
        total_revenue: Total revenue over episode
        capacity: Number of parking spots
    
    Returns:
        Revenue per spot in dollars
    """
    return total_revenue / capacity


def analyze_demand_response(occupancies: List[float], prices: List[float],
                           window_size: int = 12) -> Dict[str, float]:
    """
    Analyze relationship between price and occupancy changes.
    
    Uses sliding window correlation.
    
    Args:
        occupancies: List of occupancy rates
        prices: List of prices
        window_size: Window size for correlation (steps)
    
    Returns:
        Dict with correlation statistics
    """
    occ_array = np.array(occupancies)
    price_array = np.array(prices)
    
    # Compute changes
    occ_changes = np.diff(occ_array)
    price_changes = np.diff(price_array)
    
    # Compute correlation
    if len(occ_changes) > 0 and len(price_changes) > 0:
        correlation = np.corrcoef(occ_changes, price_changes)[0, 1]
    else:
        correlation = 0.0
    
    return {
        "price_occupancy_correlation": float(correlation),
        "correlation_interpretation": (
            "negative" if correlation < -0.2 else 
            "weak" if abs(correlation) <= 0.2 else 
            "positive"
        ),
    }


 

def compare_two_episodes(metrics1: ParkingMetrics, metrics2: ParkingMetrics,
                        label1: str = "Policy 1", label2: str = "Policy 2") -> str:
    """
    Compare two episodes side by side.
    
    Args:
        metrics1: First episode metrics
        metrics2: Second episode metrics
        label1: Label for first policy
        label2: Label for second policy
    
    Returns:
        Formatted comparison string
    """
    lines = [
        "=" * 80,
        f"COMPARISON: {label1} vs {label2}",
        "=" * 80,
        f"{'Metric':<30} {label1:>20} {label2:>20} {'Difference':>8}",
        "-" * 80,
        f"{'Total Revenue':<30} ${metrics1.total_revenue:>18.2f} ${metrics2.total_revenue:>18.2f}",
        f"{'  Difference':<30} {'':<20} {'':<20} "
        f"{(metrics2.total_revenue - metrics1.total_revenue):>+7.2f}%",
        f"{'Avg Occupancy':<30} {metrics1.avg_occupancy:>19.1%} {metrics2.avg_occupancy:>19.1%}",
        f"{'Occupancy Std':<30} {metrics1.occupancy_std:>19.1%} {metrics2.occupancy_std:>19.1%}",
        f"{'Price Volatility':<30} ${metrics1.price_volatility:>18.2f} ${metrics2.price_volatility:>18.2f}",
        f"{'Avg Price':<30} ${metrics1.avg_price:>18.2f} ${metrics2.avg_price:>18.2f}",
        "=" * 80,
    ]
    return "\n".join(lines)



 

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    occupancies = 0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, 288))
    occupancies = np.clip(occupancies, 0, 1).tolist()
    prices = (5 + 3 * np.sin(np.linspace(0, 2*np.pi, 288))).tolist()
    
    # Compute metrics
    metrics = compute_all_metrics(occupancies, prices)
    print(metrics)
    
    # Analyze
    occ_stability = analyze_occupancy_stability(occupancies)
    print("\n" + "=" * 60)
    print("OCCUPANCY STABILITY ANALYSIS")
    print("=" * 60)
    for key, value in occ_stability.items():
        if isinstance(value, float) and key != "target_occupancy":
            if "percent" in key:
                print(f"{key:<30} {value:>10.1f}%")
            else:
                print(f"{key:<30} {value:>10.4f}")
        else:
            print(f"{key:<30} {value}")
    
    price_stability = analyze_price_stability(prices)
    print("\n" + "=" * 60)
    print("PRICE STABILITY ANALYSIS")
    print("=" * 60)
    for key, value in price_stability.items():
        if "percent" in key:
            print(f"{key:<30} {value:>10.1f}%")
        else:
            print(f"{key:<30} {value:>10.4f}")
