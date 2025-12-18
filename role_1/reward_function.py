"""
ROLE 1: Reward Function Implementation

Detailed implementation and explanation of the reward function.
"""

import numpy as np
from typing import Dict, Tuple


# ==============================================================================
# REWARD FUNCTION MATHEMATICAL DEFINITION
# ==============================================================================

REWARD_FUNCTION_MATH = """
REWARD FUNCTION: MATHEMATICAL FORMULATION
==========================================

The reward function r(s, a) is a composite of three components:

r(s, a) = λ_rev × R_revenue(s, a) + λ_occ × R_occupancy(s) + λ_vol × R_volatility(a, a_prev)

COMPONENT 1: REVENUE REWARD
============================
R_revenue(s, a) = occupancy × capacity × price / capacity
                = occupancy × price
                
Mathematical Form:
- Direct product of occupancy and price
- Incentivizes: high occupancy AND high prices simultaneously
- Range: [0.5 × 0, 20.0 × 1] = [0, 20]

Example:
- At 80% occupancy, $10/hour price: R_rev = 0.8 × 10 = 8.0
- At 50% occupancy, $5/hour price: R_rev = 0.5 × 5 = 2.5
- At 100% occupancy, $20/hour price: R_rev = 1.0 × 20 = 20.0
- At 0% occupancy, any price: R_rev = 0 × price = 0

Why this component?
- Primary business objective: maximize revenue
- Revenue = occupancy × capacity × price
- Normalized by capacity for scale independence

COMPONENT 2: OCCUPANCY CONTROL REWARD
======================================
R_occupancy(s) = -α × (target_occ - occupancy)²

Where:
- target_occ = 0.8 (80% occupancy target)
- α = 0.5 (occupancy penalty weight)
- (target_occ - occupancy)² = quadratic error

Mathematical Form:
- Penalty for deviating from target occupancy
- Quadratic penalty (larger deviations penalized more)
- Always non-positive (R_occupancy ≤ 0)

Example:
- At target (80%): R_occ = -0.5 × (0.8 - 0.8)² = 0
- At 70% occ: R_occ = -0.5 × (0.8 - 0.7)² = -0.5 × 0.01 = -0.005
- At 60% occ: R_occ = -0.5 × (0.8 - 0.6)² = -0.5 × 0.04 = -0.02
- At 90% occ: R_occ = -0.5 × (0.8 - 0.9)² = -0.5 × 0.01 = -0.005
- At 40% occ: R_occ = -0.5 × (0.8 - 0.4)² = -0.5 × 0.16 = -0.08

Why this component?
- Ensures lot doesn't get overcrowded (>95%) or too empty (<20%)
- Stabilizes occupancy near target
- Quadratic penalty encourages large deviations more than small ones
- Prevents: extreme occupancy swings

COMPONENT 3: VOLATILITY PENALTY
================================
R_volatility(a, a_prev) = -β × |price_t - price_t-1|

Where:
- β = 0.1 (volatility penalty weight)
- |price_t - price_t-1| = absolute price change

Mathematical Form:
- Penalty proportional to price change magnitude
- Always non-positive (R_vol ≤ 0)
- Linear penalty (vs quadratic for occupancy)

Example:
- No price change ($10 → $10): R_vol = -0.1 × 0 = 0
- Small change ($10 → $10.50): R_vol = -0.1 × 0.5 = -0.05
- Medium change ($10 → $12): R_vol = -0.1 × 2 = -0.2
- Large change ($10 → $15): R_vol = -0.1 × 5 = -0.5
- Extreme change ($5 → $20): R_vol = -0.1 × 15 = -1.5

Why this component?
- Users expect stable, predictable pricing
- Frequent price changes erode trust
- Prevents: constant price oscillations
- Encourages: smooth, gradual pricing adjustments

COMBINED REWARD FUNCTION
========================
r(s, a) = 1.0 × R_revenue + (-0.5) × R_occupancy + (-0.1) × R_volatility

With weights: λ_rev = 1.0, λ_occ = 0.5, λ_vol = 0.1

Example Scenario:
State: occupancy = 75%, time = noon, demand = high
Action: price = $11/hour (up from $10)

R_revenue = 0.75 × 11 = 8.25
R_occupancy = -0.5 × (0.8 - 0.75)² = -0.5 × 0.0025 = -0.00125
R_volatility = -0.1 × |11 - 10| = -0.1 × 1 = -0.1
r(s,a) = 8.25 - 0.00125 - 0.1 = 8.149

Interpretation:
- Strong positive signal (8.149) because high revenue outweighs small occupancy error
- Small price change (-0.1) is acceptable for revenue gain
- Agent learns to adjust prices to maximize revenue while maintaining stability

WEIGHT INTERPRETATION
====================
Why λ_rev = 1.0, λ_occ = 0.5, λ_vol = 0.1?

Priority Ranking:
1. Revenue maximization (highest weight: 1.0)
   - Primary business objective
   - Drives agent to find optimal pricing
   
2. Occupancy control (medium weight: 0.5)
   - Secondary objective
   - Prevents extreme occupancy
   - Ensures lot remains useful
   
3. Price stability (lowest weight: 0.1)
   - Tertiary objective (fairness)
   - Prevents chaotic pricing
   - Lower weight allows flexibility for revenue/occupancy

Real-world interpretation:
- Business wants max revenue (weight 1.0)
- But also needs stable occupancy (weight 0.5)
- And should maintain customer trust with stable pricing (weight 0.1)
"""


 

class RewardFunction:
    """Complete reward function implementation."""
    
    def __init__(
        self,
        capacity: int = 100,
        target_occupancy: float = 0.8,
        revenue_weight: float = 1.0,
        occupancy_weight: float = 0.5,
        volatility_weight: float = 0.1,
    ):
        """
        Initialize reward function with parameters.
        
        Args:
            capacity: Number of parking spots
            target_occupancy: Target occupancy rate [0, 1]
            revenue_weight: Weight for revenue component
            occupancy_weight: Weight for occupancy penalty
            volatility_weight: Weight for volatility penalty
        """
        self.capacity = capacity
        self.target_occupancy = target_occupancy
        self.revenue_weight = revenue_weight
        self.occupancy_weight = occupancy_weight
        self.volatility_weight = volatility_weight
    
    def compute_revenue_reward(self, occupancy: float, price: float) -> float:
        """
        Compute revenue component of reward.
        
        R_rev = occupancy × price
        
        Args:
            occupancy: Current occupancy [0, 1]
            price: Parking price [0.5, 20.0]
        
        Returns:
            Revenue reward component
        """
        return occupancy * price
    
    def compute_occupancy_reward(self, occupancy: float) -> float:
        """
        Compute occupancy control component.
        
        R_occ = -α × (target_occ - occupancy)²
        
        Args:
            occupancy: Current occupancy [0, 1]
        
        Returns:
            Occupancy reward component (non-positive)
        """
        error = self.target_occupancy - occupancy
        return -self.occupancy_weight * (error ** 2)
    
    def compute_volatility_reward(self, price: float, price_prev: float) -> float:
        """
        Compute price volatility penalty.
        
        R_vol = -β × |price_t - price_t-1|
        
        Args:
            price: Current price
            price_prev: Previous price
        
        Returns:
            Volatility reward component (non-positive)
        """
        price_change = abs(price - price_prev)
        return -self.volatility_weight * price_change
    
    def compute_total_reward(
        self,
        occupancy: float,
        price: float,
        price_prev: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward and component breakdown.
        
        Args:
            occupancy: Current occupancy [0, 1]
            price: Current price [0.5, 20.0]
            price_prev: Previous price
        
        Returns:
            (total_reward, components_dict)
        """
        r_rev = self.compute_revenue_reward(occupancy, price)
        r_occ = self.compute_occupancy_reward(occupancy)
        r_vol = self.compute_volatility_reward(price, price_prev)
        
        total = (
            self.revenue_weight * r_rev +
            r_occ +
            r_vol
        )
        
        components = {
            "revenue": float(r_rev),
            "occupancy": float(r_occ),
            "volatility": float(r_vol),
            "total": float(total),
        }
        
        return float(total), components
    
    def batch_compute_reward(
        self,
        occupancies: np.ndarray,
        prices: np.ndarray,
        prices_prev: np.ndarray,
    ) -> np.ndarray:
        """
        Compute rewards for batch of transitions.
        
        Args:
            occupancies: Array of occupancies [batch_size]
            prices: Array of prices [batch_size]
            prices_prev: Array of previous prices [batch_size]
        
        Returns:
            Array of rewards [batch_size]
        """
        r_rev = occupancies * prices
        r_occ = -self.occupancy_weight * (self.target_occupancy - occupancies) ** 2
        r_vol = -self.volatility_weight * np.abs(prices - prices_prev)
        
        total = (
            self.revenue_weight * r_rev +
            r_occ +
            r_vol
        )
        
        return total


 
def analyze_reward_surface(
    occupancy_range: tuple = (0.0, 1.0),
    price_range: tuple = (0.5, 20.0),
    price_prev: float = 10.0,
    n_points: int = 20,
) -> Dict:
    """
    Analyze reward surface across occupancy and price.
    
    Args:
        occupancy_range: (min, max) occupancy
        price_range: (min, max) price
        price_prev: Fixed previous price
        n_points: Resolution (n_points x n_points grid)
    
    Returns:
        Dict with grid data and analysis
    """
    reward_fn = RewardFunction()
    
    occupancies = np.linspace(occupancy_range[0], occupancy_range[1], n_points)
    prices = np.linspace(price_range[0], price_range[1], n_points)
    
    reward_grid = np.zeros((n_points, n_points))
    
    for i, occ in enumerate(occupancies):
        for j, price in enumerate(prices):
            r, _ = reward_fn.compute_total_reward(occ, price, price_prev)
            reward_grid[i, j] = r
    
    return {
        "occupancies": occupancies,
        "prices": prices,
        "rewards": reward_grid,
        "max_reward": float(np.max(reward_grid)),
        "max_at": (
            float(occupancies[np.unravel_index(np.argmax(reward_grid), reward_grid.shape)[0]]),
            float(prices[np.unravel_index(np.argmax(reward_grid), reward_grid.shape)[1]]),
        ),
    }


def show_reward_examples():
    """Show example reward calculations."""
    print("\n" + "=" * 80)
    print("REWARD FUNCTION EXAMPLES")
    print("=" * 80)
    
    reward_fn = RewardFunction()
    
    scenarios = [
        ("Low occupancy, cheap price", 0.3, 3.0, 3.0),
        ("Low occupancy, expensive price", 0.3, 15.0, 15.0),
        ("Target occupancy, medium price", 0.8, 10.0, 10.0),
        ("Target occupancy, price increase", 0.8, 11.0, 10.0),
        ("High occupancy, high price", 0.95, 18.0, 18.0),
        ("High occupancy, price drop", 0.95, 8.0, 18.0),
    ]
    
    print(f"\n{'Scenario':<35} {'Occupancy':<12} {'Price':<10} {'Reward':<10}")
    print("-" * 80)
    
    for scenario_name, occ, price, price_prev in scenarios:
        reward, components = reward_fn.compute_total_reward(occ, price, price_prev)
        print(f"{scenario_name:<35} {occ:>6.0%}       ${price:>6.2f}      {reward:>8.3f}")
    
    print("\nComponent breakdown for target state (80% occupancy, $10 price):")
    print("-" * 80)
    reward, components = reward_fn.compute_total_reward(0.8, 10.0, 10.0)
    for comp_name, comp_value in components.items():
        print(f"  {comp_name:<20} {comp_value:>10.4f}")


if __name__ == "__main__":
    print(REWARD_FUNCTION_MATH)
    show_reward_examples()
    
    # Analyze reward surface
    analysis = analyze_reward_surface()
    print("\n" + "=" * 80)
    print("REWARD SURFACE ANALYSIS")
    print("=" * 80)
    print(f"Maximum reward: {analysis['max_reward']:.4f}")
    print(f"Achieved at: occupancy={analysis['max_at'][0]:.2%}, price=${analysis['max_at'][1]:.2f}")
