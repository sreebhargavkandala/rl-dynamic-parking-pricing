"""

Improved environment with:
- Curriculum learning
- Domain randomization
- Weather simulator
- Event simulator
- Competitor simulator
- Demand history in state
- Multi-objective rewards
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging
from role_1.advanced_features import (
    CurriculumLearner, CurriculumConfig,
    DomainRandomizer, DomainRandomizationConfig,
    WeatherSimulator, EventSimulator, CompetitorSimulator
)

logger = logging.getLogger(__name__)


class EnhancedParkingEnvironment:
    """
    Enhanced parking lot simulator with advanced features.
    
    Features:
    - Curriculum learning for progressive difficulty
    - Domain randomization for robustness
    - Weather effects on demand
    - Special events affecting occupancy
    - Competitor price awareness
    - Demand history tracking
    """
    
    def __init__(self, capacity: int = 100, max_steps: int = 288,
                 use_curriculum: bool = True, use_randomization: bool = True,
                 use_advanced_features: bool = True):
        
        self.capacity = capacity
        self.max_steps = max_steps
        self.current_step = 0
        
        # Advanced features
        self.use_curriculum = use_curriculum
        self.use_randomization = use_randomization
        self.use_advanced_features = use_advanced_features
        
        if self.use_curriculum:
            self.curriculum = CurriculumLearner()
        
        if self.use_randomization:
            self.randomizer = DomainRandomizer()
            self.randomized_params = {}
        
        if self.use_advanced_features:
            self.weather = WeatherSimulator()
            self.events = EventSimulator()
            self.competitors = CompetitorSimulator(num_competitors=2)
        
        # State tracking
        self.occupancy = 0.5
        self.price = 10.0
        self.demand_factor = 0.6
        self.revenue = 0.0
        self.price_history = np.ones(5) * 10.0  # Last 5 prices
        
        # Environment parameters (can be randomized)
        self.price_elasticity = -0.5
        self.base_demand = 0.6
        self.demand_std = 0.08
        self.noise_std = 0.05
        self.capacity_ratio = 1.0
        
        logger.info(f"Enhanced Parking Environment initialized")
        logger.info(f"  Capacity: {self.capacity}")
        logger.info(f"  Max steps: {self.max_steps}")
        logger.info(f"  Curriculum learning: {self.use_curriculum}")
        logger.info(f"  Domain randomization: {self.use_randomization}")
        logger.info(f"  Advanced features: {self.use_advanced_features}")
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.current_step = 0
        self.occupancy = np.random.uniform(0.4, 0.8)
        self.price = 10.0
        self.demand_factor = self.base_demand + np.random.randn() * 0.1
        self.revenue = 0.0
        self.price_history = np.ones(5) * 10.0
        
        # Advance curriculum if enabled
        if self.use_curriculum:
            self.curriculum.advance_episode()
            stage_params = self.curriculum.get_stage_params()
            self.demand_std = stage_params['demand_std']
            # Note: would also set use_competitors from stage_params
        
        # Randomize domain if enabled
        if self.use_randomization:
            self.randomized_params = self.randomizer.randomize_episode()
            
            if 'capacity_ratio' in self.randomized_params:
                self.capacity_ratio = self.randomized_params['capacity_ratio']
            if 'demand_mean' in self.randomized_params:
                self.demand_factor = self.randomized_params['demand_mean']
            if 'price_elasticity' in self.randomized_params:
                self.price_elasticity = self.randomized_params['price_elasticity']
            if 'noise_std' in self.randomized_params:
                self.noise_std = self.randomized_params['noise_std']
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Price action [0.5, 20.0]
        
        Returns:
            state, reward, done, info
        """
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Parse action
        if isinstance(action, np.ndarray):
            self.price = float(np.clip(action[0], 0.5, 20.0))
        else:
            self.price = float(np.clip(action, 0.5, 20.0))
        
        # Update demand with external factors
        hour = (self.current_step / self.max_steps) * 24
        
        # Base demand
        occupancy_demand = self.demand_factor
        
        # Time-of-day effect (peak at noon and evening)
        time_effect = 0.15 * np.sin(2 * np.pi * (hour - 6) / 24)
        occupancy_demand += time_effect
        
        # Price elasticity effect
        price_effect = self.price_elasticity * ((self.price - 10.0) / 10.0)
        occupancy_demand = occupancy_demand * (1.0 + price_effect)
        
        # Weather effect
        if self.use_advanced_features:
            weather_type, weather_mult = self.weather.update()
            occupancy_demand *= weather_mult
            
            # Event effects
            event_mult = self.events.update()
            occupancy_demand *= event_mult
            
            # Competitor pressure
            competitor_pressure = self.competitors.update(self.price)
            # Negative pressure suggests competitor undercuts, reduce demand
            occupancy_demand *= (1.0 - 0.1 * max(0, -competitor_pressure))
        
        # Add noise
        noise = np.random.randn() * self.noise_std
        self.occupancy = np.clip(occupancy_demand + noise, 0.0, 1.0)
        
        # Compute revenue
        self.revenue = self.occupancy * self.price * self.capacity * self.capacity_ratio
        
        # Update price history
        self.price_history = np.roll(self.price_history, 1)
        self.price_history[0] = self.price
        
        # Multi-objective reward
        reward = self._compute_reward()
        
        # Info
        info = {
            'occupancy': self.occupancy,
            'price': self.price,
            'revenue': self.revenue,
            'step': self.current_step,
            'hour': hour % 24
        }
        
        if self.use_advanced_features:
            info['weather'] = weather_type
            info['active_events'] = len(self.events.active_events)
        
        if self.use_curriculum:
            info['curriculum_stage'] = self.curriculum.current_stage.name
        
        state = self._get_state()
        
        return state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state with enhanced features.
        
        State includes:
        - Occupancy (normalized)
        - Time of day (hour/24)
        - Demand factor
        - Current price
        - Revenue
        - Price trend (optional)
        - Demand velocity (optional)
        """
        hour = (self.current_step / self.max_steps) * 24
        time_of_day = (hour % 24) / 24.0
        
        # Basic state
        state = np.array([
            self.occupancy,
            time_of_day,
            self.demand_factor,
            self.price / 20.0,  # Normalize price
            self.revenue / (self.capacity * 20.0)  # Normalize revenue
        ], dtype=np.float32)
        
        # Extended state (optional)
        extended_state = []
        
        # Price trend
        if len(self.price_history) > 1:
            price_trend = (self.price_history[0] - self.price_history[2]) / 10.0
            extended_state.append(price_trend)
        
        # Demand trend
        extended_state.append((self.demand_factor - 0.6) / 0.2)  # Normalized
        
        if extended_state:
            state = np.concatenate([state, np.array(extended_state, dtype=np.float32)])
        
        return state
    
    def _compute_reward(self) -> float:
        """
        Compute multi-objective reward.
        
        Objectives:
        1. Revenue generation
        2. Occupancy control
        3. Price stability
        4. Demand responsiveness
        """
        # Revenue reward
        revenue_reward = self.revenue / (self.capacity * 15.0)  # Normalized
        
        # Occupancy reward (penalize if too high or too low)
        occupancy_target = 0.75
        occupancy_penalty = 0.5 * ((self.occupancy - occupancy_target) ** 2)
        
        # Price stability reward (encourage smooth pricing)
        if len(self.price_history) > 1:
            price_change = abs(self.price_history[0] - self.price_history[1]) / 20.0
            stability_reward = -0.01 * price_change
        else:
            stability_reward = 0.0
        
        # Price bonus for high prices when occupancy is high
        if self.occupancy > 0.75:
            price_bonus = self.price * (self.occupancy - 0.75) * 5.0 / 20.0
        else:
            price_bonus = 0.0
        
        # Strong penalty for very low occupancy
        if self.occupancy < 0.60:
            low_occ_penalty = 5.0 * ((0.6 - self.occupancy) ** 2)
        else:
            low_occ_penalty = 0.0
        
        # Total reward
        total_reward = (revenue_reward + 
                       price_bonus - 
                       occupancy_penalty - 
                       low_occ_penalty + 
                       stability_reward)
        
        return float(total_reward)


if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED PARKING ENVIRONMENT")
    print("=" * 80)
    
    # Create environment with all features
    env = EnhancedParkingEnvironment(
        capacity=100,
        max_steps=288,
        use_curriculum=True,
        use_randomization=True,
        use_advanced_features=True
    )
    
    print("\n1. Testing environment reset...")
    state = env.reset()
    print(f"   ✓ Initial state shape: {state.shape}")
    print(f"   ✓ State: occ={state[0]:.2f}, time={state[1]:.2f}, demand={state[2]:.2f}, "
          f"price={state[3]:.2f}, revenue={state[4]:.2f}")
    
    print("\n2. Running episode with advanced features...")
    total_reward = 0
    occupancies = []
    prices = []
    
    for step in range(288):
        # Random action
        action = np.array([np.random.uniform(0.5, 20.0)])
        state, reward, done, info = env.step(action)
        
        total_reward += reward
        occupancies.append(info['occupancy'])
        prices.append(info['price'])
    
    print(f"   ✓ Episode completed")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Avg occupancy: {np.mean(occupancies):.2%}")
    print(f"   Avg price: ${np.mean(prices):.2f}")
    print(f"   Price std: ${np.std(prices):.2f}")
    print(f"   Final revenue: ${env.revenue:.2f}")
    
    print("\n" + "=" * 80)
    print("Enhanced Environment Ready!")
    print("=" * 80)
