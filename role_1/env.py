"""
RL Problem Formulation + Environment


Dynamic Parking Pricing Environment 

This module implements the ParkingPricingEnv, a custom reinforcement learning environment
that models a parking facility's pricing problem as a Markov Decision Process (MDP).

Features:
- Continuous action space: Real-valued parking prices [$0.50 - $20.00]
- Continuous state space: Occupancy, time, demand, price history
- Production-ready: Type hints, logging, error handling, validation
- Realistic dynamics: Price elasticity, time-dependent demand, stochasticity
- Comprehensive metrics: Revenue, occupancy, volatility, stability

MDP Formulation:
- State space: Continuous features (occupancy rate, time of day, demand, price history)
- Action space: Continuous pricing decisions [price_min, price_max]
- Reward: Revenue maximization with occupancy constraints
- Transition: Stochastic demand response to pricing (continuous)

Justification for Non-Tabular RL:
The state space is inherently continuous and high-dimensional, making tabular methods
(Q-learning, SARSA) infeasible. The curse of dimensionality prevents us from enumerating
all state-action pairs. Function approximation (neural networks) is necessary.
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable
import gymnasium as gym
from gymnasium import spaces


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ParkingPricingEnv(gym.Env):
    """
    
    
    STATE SPACE (S):
    - occupancy_rate: float in [0, 1] - current parking lot occupancy
    - time_of_day: float in [0, 1] - normalized hour (0 = midnight, 1 = next midnight)
    - demand_level: float in [0, 1] - estimated demand intensity
    - price_history: array of last N prices - previous pricing decisions
    
    Continuous observation space: Box([0, 0, 0, -inf], [1, 1, 1, inf], shape=(5,))
    
    ACTION SPACE (A):
    - Continuous pricing action: float in [min_price, max_price]
    - Action represents the hourly parking price in dollars
    
    Range: [0.5, 20.0] dollars per hour
    
    TRANSITION FUNCTION (P):
    - Stochastic transition: demand = f(price, time, external_factors) + noise
    - occupancy_new = occupancy_old + (demand - departures) / capacity
    - time_new = (time_old + dt/24) % 1
    
    REWARD FUNCTION (R):
    - r(s, a) = revenue - penalty_for_low_occupancy - penalty_for_volatility
    - revenue = occupancy * capacity * price
    - penalty = alpha * (target_occupancy - occupancy)^2
    
    DISCOUNT FACTOR: γ = 0.99
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(
        self,
        capacity: int = 100,
        max_steps: int = 288,  
        target_occupancy: float = 0.8,
        demand_model: Optional[Callable] = None,
        min_price: float = 0.5,
        max_price: float = 20.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the parking pricing environment (production-ready).
        
        Args:
            capacity: Number of parking spots (default: 100)
            max_steps: Episode length in 5-min intervals (default: 288 = 24 hours)
            target_occupancy: Desired occupancy rate (default: 0.8 = 80%)
            demand_model: Callable demand model, if None uses default stochastic model
            min_price: Minimum allowed price in dollars (default: $0.50)
            max_price: Maximum allowed price in dollars (default: $20.00)
            seed: Random seed for reproducibility
        
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if max_steps <= 0:
            raise ValueError(f"Max steps must be positive, got {max_steps}")
        if not (0 < target_occupancy < 1):
            raise ValueError(f"Target occupancy must be in (0, 1), got {target_occupancy}")
        if min_price >= max_price:
            raise ValueError(f"Min price ({min_price}) must be < max price ({max_price})")
        
   
        self.capacity = capacity
        self.max_steps = max_steps
        self.target_occupancy = target_occupancy
        self.demand_model = demand_model
        self.seed_value = seed
        
        
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.price_range = self.max_price - self.min_price
        
        
        self.volatility_penalty_weight = 0.05  # Soft penalty on price changes
        
       
        self.action_space = spaces.Box(
            low=np.array([self.min_price], dtype=np.float32),
            high=np.array([self.max_price], dtype=np.float32),
            dtype=np.float32,
            shape=(1,)
        )
        
       
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
            shape=(5,)
        )
        
      
        self.occupancy = 0.5
        self.time_step = 0
        self.demand_level = 0.5
        self.price_history = [self.min_price, self.min_price] 
        self.episode_metrics = {
            'occupancies': [],
            'prices': [],
            'rewards': [],
            'revenues': []
        }
        
       
        self.rng = np.random.RandomState(seed)
        
        logger.info(f"Initialized ParkingPricingEnv: capacity={capacity}, max_steps={max_steps}, "
                   f"price_range=[${min_price:.2f}, ${max_price:.2f}]")
        
        
        self.action_space = spaces.Box(
            low=np.array([self.min_price], dtype=np.float32),
            high=np.array([self.max_price], dtype=np.float32),
            dtype=np.float32,
            shape=(1,)
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            observation: Initial state (normalized to [0, 1])
            info: Additional info dict
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        self.time_step = 0
        self.occupancy = self.rng.uniform(0.4, 0.6)  
        self.demand_level = 0.5
        self.price_history = [self.min_price + self.price_range * 0.5] * 2
        
       
        self.episode_metrics = {
            'occupancies': [self.occupancy],
            'prices': [],
            'rewards': [],
            'revenues': []
        }
        
        observation = self._get_observation()
        info = {
            'occupancy': float(self.occupancy),
            'time_step': self.time_step,
            'demand_level': float(self.demand_level)
        }
        
        logger.debug(f"Environment reset: occupancy={self.occupancy:.2%}, time=0")
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
      
        
        price = float(np.clip(action[0], self.min_price, self.max_price))
        
       
        revenue = self.occupancy * self.capacity * price
        
        
        occupancy_change = self._compute_occupancy_change(price)
        occupancy_change += self.rng.normal(0, 0.02)  
        
        
        self.occupancy = np.clip(
            self.occupancy + occupancy_change / self.capacity,
            0.0, 1.0
        )
        
       
        self.time_step += 1
        time_normalized = self.time_step / self.max_steps
        
       
        self.demand_level = self._compute_demand_level(time_normalized)
        
        
        reward = self._compute_reward(revenue, price)
        
        
        self.price_history = [price, self.price_history[0]]
        
       
        self.episode_metrics['occupancies'].append(self.occupancy)
        self.episode_metrics['prices'].append(price)
        self.episode_metrics['rewards'].append(reward)
        self.episode_metrics['revenues'].append(revenue)
        
        
        terminated = (self.time_step >= self.max_steps)
        
    
        observation = self._get_observation()
        
        info = {
            'revenue': float(revenue),
            'occupancy': float(self.occupancy),
            'price': float(price),
            'time_step': self.time_step,
        }
        
        logger.debug(f"Step {self.time_step}: price=${price:.2f}, occ={self.occupancy:.2%}, reward={reward:.4f}")
        
        return observation, float(reward), terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state) normalized to [0, 1].
        
        Returns:
            Array: [occupancy, time_of_day, demand_level, price_t-1_norm, price_t-2_norm]
        """
      
        price_norm_1 = (self.price_history[0] - self.min_price) / self.price_range
        price_norm_2 = (self.price_history[1] - self.min_price) / self.price_range
        
        return np.array([
            float(self.occupancy),
            float(self.time_step / self.max_steps),  
            float(self.demand_level),
            float(price_norm_1),
            float(price_norm_2),
        ], dtype=np.float32)
    
    def _compute_demand_level(self, time_normalized: float) -> float:
        """
        Compute demand level based on time of day (continuous).
        
        Peak demand during business hours (10am-6pm).
        
        Args:
            time_normalized: Normalized time [0, 1], where 0.5 ≈ noon
        """
        #
        demand = 0.5 + 0.35 * np.sin(2 * np.pi * (time_normalized - 0.33))
        return float(np.clip(demand, 0.0, 1.0))
    
    def _compute_occupancy_change(self, price: float) -> float:
        """
        Compute change in occupied spots based on price and demand dynamics (continuous).
        
        PRODUCTION: Price-elastic demand with continuous price values
        
        Args:
            price: Continuous price in [min_price, max_price]
        
        Returns:
            Change in occupancy (positive = arrivals, negative = departures)
        """
        if self.demand_model is not None:
            
            return self.demand_model(price, self.time_step / self.max_steps, self.occupancy)
     
        reference_price = (self.min_price + self.max_price) / 2
        
       
        price_elasticity = -0.5
        price_factor = (price - reference_price) / reference_price
        demand_response = price_elasticity * price_factor
        
      
        base_demand = self.demand_level
        
       
        arrivals = base_demand * (1 + demand_response)
        
        
        departures = 0.1 * self.occupancy  
        
        occupancy_change = arrivals - departures
        return float(occupancy_change)
    
    def _compute_reward(self, revenue: float, price: float) -> float:
        """
        REWARD FUNCTION (REVENUE-OPTIMIZED WITH OCCUPANCY THRESHOLD):
        
        r(s, a) = R_revenue + R_price_bonus - λ_occ * R_occupancy - λ_vol * R_volatility
        
        Components:
        -----------
        1. BASE REVENUE REWARD:
           R_revenue = occupancy × price (empirical revenue per timestep)
           Encourages both high prices AND high occupancy
        
        2. PRICE BONUS (NEW):
           If occupancy > 0.75 (high demand):
               Bonus = price × (occupancy - 0.75) × 10
           Else: Bonus = 0
           
           This incentivizes RAISING PRICES when lot is busy!
           
        3. OCCUPANCY THRESHOLD:
           If occupancy < 0.6: penalty = 5.0 (force filling when low occupancy)
           Else: penalty = very light (don't over-constrain when full)
           
        4. VOLATILITY PENALTY (SOFT):
           R_volatility = |price_t - price_t-1| × 0.01
           Minimal penalty for smooth but responsive pricing
        
        Key Improvements for Revenue:
        ---
        ✓ Bonus for HIGH prices when occupancy HIGH
        ✓ Strong penalty for EMPTY lot (occupancy < 60%)
        ✓ Minimal penalty for FULL lot (allows high prices)
        ✓ Direct revenue × price maximization
        
        This enables agent to:
        ✓ Learn to RAISE prices when demand is high
        ✓ FILL lot when empty (low prices)
        ✓ MAXIMIZE revenue (high price × high occupancy)
        
        Args:
            revenue: Total revenue from this step
            price: Current continuous price
        
        Returns:
            Scalar reward value
        """
       
        # 1. Base revenue reward
        revenue_reward = self.occupancy * price
        
        # 2. PRICE BONUS: reward high prices when occupancy is high (>75%)
        price_bonus = 0.0
        if self.occupancy > 0.75:
            # Extra reward for raising prices in high-demand situations
            price_bonus = price * (self.occupancy - 0.75) * 10.0
        
        # 3. OCCUPANCY CONSTRAINT: Strong penalty if too empty, soft if full
        if self.occupancy < 0.6:
            # Low occupancy = underutilized lot = big penalty
            occupancy_penalty = 5.0 * ((0.6 - self.occupancy) ** 2)
        else:
            # High occupancy = good, very light penalty
            occupancy_penalty = 0.05 * ((self.target_occupancy - self.occupancy) ** 2)
        
        # 4. Minimal volatility penalty for smooth pricing
        price_change = abs(price - self.price_history[1])
        volatility_penalty = 0.01 * price_change
        
        # Total reward: encourage revenue, penalize empty lot, allow exploration
        total_reward = revenue_reward + price_bonus - occupancy_penalty - volatility_penalty
        
        return float(total_reward)
    
    
    # EVALUATION & METRICS
    
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive metrics for completed episode (PRODUCTION-READY).
        
        Returns:
            Dictionary with:
            - total_revenue: Cumulative revenue earned
            - avg_occupancy: Mean occupancy rate
            - occupancy_std: Occupancy standard deviation (stability)
            - min_occupancy: Minimum occupancy
            - max_occupancy: Maximum occupancy
            - avg_price: Mean price charged
            - min_price: Minimum price charged
            - max_price: Maximum price charged
            - price_volatility: Price std dev (smoothness)
            - episode_length: Number of steps in episode
        """
        occupancies = np.array(self.episode_metrics['occupancies'])
        prices = np.array(self.episode_metrics['prices'])
        revenues = np.array(self.episode_metrics['revenues'])
        
        metrics = {
            'total_revenue': float(np.sum(revenues)),
            'avg_occupancy': float(np.mean(occupancies)),
            'occupancy_std': float(np.std(occupancies)),
            'min_occupancy': float(np.min(occupancies)) if len(occupancies) > 0 else 0.0,
            'max_occupancy': float(np.max(occupancies)) if len(occupancies) > 0 else 0.0,
            'avg_price': float(np.mean(prices)) if len(prices) > 0 else 0.0,
            'min_price': float(np.min(prices)) if len(prices) > 0 else 0.0,
            'max_price': float(np.max(prices)) if len(prices) > 0 else 0.0,
            'price_volatility': float(np.std(np.diff(prices))) if len(prices) > 1 else 0.0,
            'episode_length': len(prices),
        }
        
        logger.info(f"Episode metrics: revenue=${metrics['total_revenue']:.2f}, "
                   f"occupancy={metrics['avg_occupancy']:.2%}, "
                   f"price_volatility=${metrics['price_volatility']:.2f}")
        
        return metrics
    
    def render(self) -> None:
        """Render the environment (placeholder for gymnasium compatibility)."""
        pass
    
    def close(self) -> None:
        """Clean up resources."""
        logger.debug("Environment closed")



# EXAMPLE USAGE


if __name__ == "__main__":
    # Initialize environment
    env = ParkingPricingEnv(
        capacity=100,
        max_steps=288,
        target_occupancy=0.8,
        min_price=0.5,
        max_price=20.0
    )
    
    print("=" * 70)
    print("ROLE 1: PARKING PRICING ENVIRONMENT - PRODUCTION VERSION")
    print("=" * 70)
    
  
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation: {obs}")
    print(f"Initial info: {info}")
    
    
    sample_action = env.action_space.sample()
    print(f"\nSample continuous price action: ${sample_action[0]:.2f}")
    
   
    obs, reward, terminated, truncated, info = env.step(sample_action)
    print(f"Reward: {reward:.4f}")
    print(f"Info: {info}")
    
   
    print("\nRunning full episode with random policy...")
    obs, _ = env.reset()
    total_reward = 0.0
    
    while True:
        action = env.action_space.sample()  
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            break
    
   
    metrics = env.get_episode_metrics()
    print("\n" + "=" * 70)
    print("EPISODE METRICS (CONTINUOUS PRICING)")
    print("=" * 70)
    for key, value in metrics.items():
        if 'revenue' in key or 'price' in key:
            print(f"{key:.<40} ${value:>12.2f}")
        elif 'occupancy' in key:
            print(f"{key:.<40} {value*100:>11.1f}%")
        else:
            print(f"{key:.<40} {value:>12.4f}")




