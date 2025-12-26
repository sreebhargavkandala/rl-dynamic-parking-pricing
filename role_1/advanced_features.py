"""
CURRICULUM LEARNING & DOMAIN RANDOMIZATION MODULE
==================================================

Implements:
1. Curriculum Learning - Start simple, gradually increase difficulty
2. Domain Randomization - Randomize environment parameters
3. Advanced Features - Weather, events, competitor prices
4. Multi-scale Training - Different time horizons
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CurriculumStage(Enum):
    """Curriculum learning stages."""
    STAGE1_BASIC = 1         # Constant demand, simple patterns
    STAGE2_DEMAND_VAR = 2    # Variable demand, low variance
    STAGE3_SEASONALITY = 3   # Time-based seasonality
    STAGE4_COMPETITION = 4   # Competitor effects
    STAGE5_ADVANCED = 5      # All features combined


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    enabled: bool = True
    stage_episodes: int = 30  # Episodes per stage before advancing
    # Stage 1: Basic (constant demand)
    stage1_demand_std: float = 0.02  # Very low variance
    stage1_competitors: bool = False
    
    # Stage 2: Demand variance
    stage2_demand_std: float = 0.05
    stage2_competitors: bool = False
    
    # Stage 3: Seasonality
    stage3_demand_std: float = 0.08
    stage3_competitors: bool = False
    
    # Stage 4: Competition
    stage4_demand_std: float = 0.10
    stage4_competitors: bool = True
    
    # Stage 5: Full challenge
    stage5_demand_std: float = 0.15
    stage5_competitors: bool = True


@dataclass
class DomainRandomizationConfig:
    """Domain randomization configuration."""
    enabled: bool = True
    randomize_capacity: bool = True
    capacity_range: Tuple[float, float] = (0.8, 1.2)  # Ratio of base capacity
    
    randomize_demand: bool = True
    demand_mean_range: Tuple[float, float] = (0.55, 0.75)
    demand_std_range: Tuple[float, float] = (0.05, 0.15)
    
    randomize_elasticity: bool = True
    elasticity_range: Tuple[float, float] = (-0.7, -0.3)  # Price elasticity
    
    randomize_noise: bool = True
    noise_std_range: Tuple[float, float] = (0.01, 0.10)


class CurriculumLearner:
    """Manages curriculum learning progression."""
    
    def __init__(self, config: CurriculumConfig = None):
        self.config = config or CurriculumConfig()
        self.current_stage = CurriculumStage.STAGE1_BASIC
        self.episode_count = 0
        self.stage_episode_count = 0
        
        logger.info(f"Curriculum Learning initialized")
        logger.info(f"  Stage 1 (0-{self.config.stage_episodes}): Basic - Constant demand, no competitors")
        logger.info(f"  Stage 2 ({self.config.stage_episodes}-{self.config.stage_episodes*2}): Demand variance")
        logger.info(f"  Stage 3 ({self.config.stage_episodes*2}-{self.config.stage_episodes*3}): Seasonality")
        logger.info(f"  Stage 4 ({self.config.stage_episodes*3}-{self.config.stage_episodes*4}): Competitor effects")
        logger.info(f"  Stage 5 ({self.config.stage_episodes*4}+): Full challenge")
    
    def get_stage_params(self) -> Dict:
        """Get parameters for current curriculum stage."""
        if self.current_stage == CurriculumStage.STAGE1_BASIC:
            return {
                'demand_std': self.config.stage1_demand_std,
                'use_competitors': self.config.stage1_competitors,
                'stage_name': 'BASIC'
            }
        elif self.current_stage == CurriculumStage.STAGE2_DEMAND_VAR:
            return {
                'demand_std': self.config.stage2_demand_std,
                'use_competitors': self.config.stage2_competitors,
                'stage_name': 'DEMAND_VAR'
            }
        elif self.current_stage == CurriculumStage.STAGE3_SEASONALITY:
            return {
                'demand_std': self.config.stage3_demand_std,
                'use_competitors': self.config.stage3_competitors,
                'stage_name': 'SEASONALITY'
            }
        elif self.current_stage == CurriculumStage.STAGE4_COMPETITION:
            return {
                'demand_std': self.config.stage4_demand_std,
                'use_competitors': self.config.stage4_competitors,
                'stage_name': 'COMPETITION'
            }
        else:
            return {
                'demand_std': self.config.stage5_demand_std,
                'use_competitors': self.config.stage5_competitors,
                'stage_name': 'FULL'
            }
    
    def advance_episode(self) -> bool:
        """
        Advance curriculum learning.
        
        Returns:
            True if stage advanced
        """
        self.episode_count += 1
        self.stage_episode_count += 1
        
        if self.stage_episode_count >= self.config.stage_episodes:
            if self.current_stage == CurriculumStage.STAGE1_BASIC:
                self.current_stage = CurriculumStage.STAGE2_DEMAND_VAR
            elif self.current_stage == CurriculumStage.STAGE2_DEMAND_VAR:
                self.current_stage = CurriculumStage.STAGE3_SEASONALITY
            elif self.current_stage == CurriculumStage.STAGE3_SEASONALITY:
                self.current_stage = CurriculumStage.STAGE4_COMPETITION
            elif self.current_stage == CurriculumStage.STAGE4_COMPETITION:
                self.current_stage = CurriculumStage.STAGE5_ADVANCED
            
            self.stage_episode_count = 0
            stage_params = self.get_stage_params()
            logger.info(f"✓ Advanced to curriculum stage: {stage_params['stage_name']} "
                       f"(episode {self.episode_count})")
            return True
        
        return False


class DomainRandomizer:
    """Applies domain randomization to environment."""
    
    def __init__(self, config: DomainRandomizationConfig = None):
        self.config = config or DomainRandomizationConfig()
        self.randomized_params = {}
        
        logger.info(f"Domain Randomization initialized")
        if self.config.randomize_capacity:
            logger.info(f"  Capacity randomization: {self.config.capacity_range}")
        if self.config.randomize_demand:
            logger.info(f"  Demand mean range: {self.config.demand_mean_range}")
            logger.info(f"  Demand std range: {self.config.demand_std_range}")
        if self.config.randomize_elasticity:
            logger.info(f"  Elasticity range: {self.config.elasticity_range}")
    
    def randomize_episode(self) -> Dict:
        """
        Randomize environment parameters for new episode.
        
        Returns:
            Dictionary of randomized parameters
        """
        params = {}
        
        # Capacity randomization
        if self.config.randomize_capacity:
            ratio = np.random.uniform(*self.config.capacity_range)
            params['capacity_ratio'] = ratio
        
        # Demand randomization
        if self.config.randomize_demand:
            demand_mean = np.random.uniform(*self.config.demand_mean_range)
            demand_std = np.random.uniform(*self.config.demand_std_range)
            params['demand_mean'] = demand_mean
            params['demand_std'] = demand_std
        
        # Elasticity randomization
        if self.config.randomize_elasticity:
            elasticity = np.random.uniform(*self.config.elasticity_range)
            params['price_elasticity'] = elasticity
        
        # Noise randomization
        if self.config.randomize_noise:
            noise_std = np.random.uniform(*self.config.noise_std_range)
            params['noise_std'] = noise_std
        
        self.randomized_params = params
        return params


class WeatherSimulator:
    """Simulates weather effects on parking demand."""
    
    # Weather patterns: (type, demand_multiplier, likelihood)
    WEATHER_PATTERNS = {
        'sunny': (1.1, 0.6),        # +10% demand, 60% chance
        'cloudy': (1.0, 0.25),      # No change, 25% chance
        'rainy': (0.85, 0.10),      # -15% demand, 10% chance
        'snow': (0.7, 0.05)         # -30% demand, 5% chance
    }
    
    def __init__(self):
        self.current_weather = 'sunny'
        self.weather_change_prob = 0.1  # 10% chance to change weather each timestep
    
    def update(self) -> Tuple[str, float]:
        """
        Update weather and return demand multiplier.
        
        Returns:
            (weather_type, demand_multiplier)
        """
        if np.random.random() < self.weather_change_prob:
            # Change weather
            weather_types = list(self.WEATHER_PATTERNS.keys())
            self.current_weather = np.random.choice(weather_types)
        
        multiplier = self.WEATHER_PATTERNS[self.current_weather][0]
        return self.current_weather, multiplier


class EventSimulator:
    """Simulates special events affecting parking demand."""
    
    # Events: (type, demand_multiplier, duration, likelihood)
    EVENTS = {
        'sports_event': (1.5, 4, 0.02),         # +50% demand, 4 hours, 2% chance
        'concert': (1.4, 6, 0.01),              # +40% demand, 6 hours, 1% chance
        'holiday': (1.3, 24, 0.005),            # +30% demand, all day, 0.5% chance
        'school_closed': (1.2, 24, 0.03),       # +20% demand, 0.3% chance
        'strike': (0.6, 8, 0.01)                # -40% demand, 0.1% chance
    }
    
    def __init__(self):
        self.active_events = []  # List of (event_type, remaining_duration)
    
    def update(self) -> float:
        """
        Update events and return demand multiplier.
        
        Returns:
            Combined demand multiplier from all active events
        """
        # Check for new events
        if np.random.random() < 0.02:  # 2% chance for any event
            event_types = list(self.EVENTS.keys())
            event_probs = [self.EVENTS[e][2] for e in event_types]
            event_probs = np.array(event_probs) / sum(event_probs)
            
            event = np.random.choice(event_types, p=event_probs)
            duration = self.EVENTS[event][1]
            self.active_events.append([event, duration])
        
        # Update active events
        multiplier = 1.0
        remaining_events = []
        
        for event, duration in self.active_events:
            if duration > 0:
                multiplier *= self.EVENTS[event][0]
                remaining_events.append([event, duration - 1])
        
        self.active_events = remaining_events
        
        return multiplier


class CompetitorSimulator:
    """Simulates competitor pricing behavior."""
    
    def __init__(self, num_competitors: int = 2):
        self.num_competitors = num_competitors
        self.competitor_prices = [10.0] * num_competitors
        self.competitor_strategies = ['matching', 'aggressive', 'conservative']
    
    def update(self, our_price: float) -> float:
        """
        Update competitor prices and return price pressure.
        
        Args:
            our_price: Our current price
        
        Returns:
            Price pressure indicator (-1 to 1, where -1 = pressure to lower)
        """
        pressure = 0.0
        
        for i in range(self.num_competitors):
            # Random strategy for competitor
            strategy = np.random.choice(self.competitor_strategies)
            
            if strategy == 'matching':
                # Match our price with small noise
                target_price = our_price + np.random.randn() * 0.5
            elif strategy == 'aggressive':
                # Undercut our price
                target_price = our_price - 1.0 + np.random.randn() * 0.5
            else:  # conservative
                # Overcharge
                target_price = our_price + 2.0 + np.random.randn() * 0.5
            
            # Move toward target price
            self.competitor_prices[i] = 0.7 * self.competitor_prices[i] + 0.3 * target_price
            self.competitor_prices[i] = np.clip(self.competitor_prices[i], 0.5, 20.0)
            
            # Compute pressure from this competitor
            if self.competitor_prices[i] < our_price - 2.0:
                pressure -= 0.5
            elif self.competitor_prices[i] > our_price + 2.0:
                pressure += 0.5
        
        pressure /= self.num_competitors
        return np.clip(pressure, -1.0, 1.0)


if __name__ == "__main__":
    print("=" * 80)
    print("CURRICULUM LEARNING & DOMAIN RANDOMIZATION")
    print("=" * 80)
    
    # Test curriculum learning
    print("\n1. Testing Curriculum Learning...")
    curriculum = CurriculumLearner()
    
    for ep in range(160):
        curriculum.advance_episode()
        if ep % 30 == 0:
            params = curriculum.get_stage_params()
            print(f"   Episode {ep}: Stage {params['stage_name']}, "
                  f"demand_std={params['demand_std']:.3f}, "
                  f"competitors={params['use_competitors']}")
    
    # Test domain randomization
    print("\n2. Testing Domain Randomization...")
    randomizer = DomainRandomizer()
    
    for ep in range(3):
        params = randomizer.randomize_episode()
        print(f"   Episode {ep}:")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.3f}")
            else:
                print(f"     {key}: {value}")
    
    # Test weather simulator
    print("\n3. Testing Weather Simulator...")
    weather = WeatherSimulator()
    
    weather_count = {'sunny': 0, 'cloudy': 0, 'rainy': 0, 'snow': 0}
    for step in range(1000):
        weather_type, multiplier = weather.update()
        weather_count[weather_type] += 1
    
    print("   Weather distribution (1000 steps):")
    for wtype, count in sorted(weather_count.items(), key=lambda x: -x[1]):
        print(f"     {wtype}: {count/10:.1f}%")
    
    # Test event simulator
    print("\n4. Testing Event Simulator...")
    events = EventSimulator()
    
    event_count = {}
    for step in range(1000):
        multiplier = events.update()
        if events.active_events:
            for event, _ in events.active_events:
                event_count[event] = event_count.get(event, 0) + 1
    
    print("   Active event hours (1000 steps):")
    for event, hours in sorted(event_count.items(), key=lambda x: -x[1]):
        print(f"     {event}: {hours} hours")
    
    # Test competitor simulator
    print("\n5. Testing Competitor Simulator...")
    competitors = CompetitorSimulator(num_competitors=2)
    
    pressures = []
    for step in range(100):
        our_price = 10.0 + np.sin(step / 10) * 5
        pressure = competitors.update(our_price)
        pressures.append(pressure)
    
    print(f"   Average price pressure: {np.mean(pressures):.3f}")
    print(f"   Competitor 1 final price: ${competitors.competitor_prices[0]:.2f}")
    print(f"   Competitor 2 final price: ${competitors.competitor_prices[1]:.2f}")
    
    print("\n" + "=" * 80)
    print("Advanced Features Ready!")
    print("=" * 80)
