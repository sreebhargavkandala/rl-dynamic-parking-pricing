"""


Saran will replace this with real data-trained models.
"""

import logging
import numpy as np
from typing import Optional

# Configure logging for production
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimulatorDemandModel:
    """
    PRODUCTION-READY: Synthetic demand model with continuous pricing.
    
    Implements realistic parking demand dynamics with full continuous price support:
    - Price elasticity: continuous demand response to price changes
    - Time-dependent: peak demand during business hours
    - Stochastic: realistic variability with noise
    
    Mathematical Model:
    -------------------
    Base demand:
        D_base(t) = 0.5 + 0.35 * sin(2π(t - 0.33))
        where t ∈ [0, 1] is normalized time of day
        Peak around 10am-2pm
    
    Price response (continuous):
        ε = -0.5 (price elasticity)
        Δq/q = ε * Δp/p
        
    Occupancy change:
        ΔOcc = D_base * (1 + ε * (p - p_ref) / p_ref) - λ_depart * Occ + noise
    """
    
    def __init__(
        self,
        elasticity: float = -0.5,
        reference_price: float = 10.0,
        noise_std: float = 0.02,
        departure_rate: float = 0.1,
    ):
        """
        Initialize production-ready demand model with continuous pricing.
        
        Args:
            elasticity: Price elasticity of demand
                       Default: -0.5 (1% price ↑ → 0.5% demand ↓)
                       Range: typically [-1.0, 0.0]
            reference_price: Reference price for elasticity calculation (typically median)
                            Default: $10.00
            noise_std: Standard deviation of stochastic noise
                      Default: 0.02
            departure_rate: Rate at which customers leave (fraction of occupancy per step)
                           Default: 0.1 (10% per step)
        
        Raises:
            ValueError: If parameters are invalid
        """
        if elasticity >= 0:
            raise ValueError(f"Elasticity must be negative, got {elasticity}")
        if reference_price <= 0:
            raise ValueError(f"Reference price must be positive, got {reference_price}")
        if noise_std < 0:
            raise ValueError(f"Noise std must be non-negative, got {noise_std}")
        
        self.elasticity = float(elasticity)
        self.reference_price = float(reference_price)
        self.noise_std = float(noise_std)
        self.departure_rate = float(departure_rate)
        
        logger.info(f"Initialized SimulatorDemandModel: elasticity={elasticity}, "
                   f"ref_price=${reference_price:.2f}, noise_std={noise_std}")
    
    def __call__(
        self,
        price: float,
        time_of_day: float,
        occupancy: float,
        seed: Optional[int] = None
    ) -> float:
        """
        Predict occupancy change given continuous price and time (PRODUCTION).
        
        CONTINUOUS PRICING: Supports full floating-point price values [$0.50 - $20.00]
        
        Args:
            price: Current parking price (continuous, ∈ [0.5, 20.0])
            time_of_day: Normalized time of day ∈ [0.0, 1.0]
                        0.0 = midnight, 0.5 = noon, 1.0 = next midnight
            occupancy: Current occupancy rate ∈ [0.0, 1.0]
            seed: Optional random seed for reproducibility
        
        Returns:
            float: Occupancy change (ΔOcc/capacity), negative = departures
        
        Raises:
            ValueError: If inputs are out of valid ranges
        """
      
        if not (0 <= time_of_day <= 1):
            raise ValueError(f"time_of_day must be in [0, 1], got {time_of_day}")
        if not (0 <= occupancy <= 1):
            raise ValueError(f"occupancy must be in [0, 1], got {occupancy}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        
        
        rng = np.random.RandomState(seed)
        
        
        base_demand = self._compute_base_demand(time_of_day)
        
        
        price_response = self._compute_price_response(price)
        
       
        arrivals = base_demand * (1 + price_response)
        
       
        departures = self.departure_rate * occupancy
        
      
        noise = rng.normal(0, self.noise_std)
        
        
        occupancy_change = arrivals - departures + noise
        
        logger.debug(f"Demand model: price=${price:.2f}, time={time_of_day:.2f}, "
                    f"base_demand={base_demand:.3f}, price_response={price_response:.3f}, "
                    f"delta_occ={occupancy_change:.3f}")
        
        return float(occupancy_change)
    
    def _compute_base_demand(self, time_of_day: float) -> float:
        """
        Compute base demand level from time of day (CONTINUOUS).
        
        PRODUCTION: Realistic sinusoidal pattern with peak during business hours
        
        Args:
            time_of_day: Normalized time ∈ [0, 1]
        
        Returns:
            Base demand level ∈ [0.15, 0.85] (approximately)
        """
        
        phase_shift = 0.33   
        
        base = 0.5 + 0.35 * np.sin(2 * np.pi * (time_of_day - phase_shift))
        
   
        return float(np.clip(base, 0.15, 0.85))
    
    def _compute_price_response(self, price: float) -> float:
        """
        Compute demand response to price (CONTINUOUS).
        
        PRODUCTION: Full floating-point price precision
        
        Mathematical relationship:
            ΔQ/Q = ε * ΔP/P
            where ε is price elasticity
        
        Args:
            price: Continuous price ∈ [0.5, 20.0]
        
        Returns:
            Price response factor (typically ∈ [-0.5, 0.5])
        """
        
        price_ratio = price / self.reference_price
        
        # Elasticity relationship (continuous)
        price_response = self.elasticity * np.log(price_ratio)
        
        # Clip to realistic range
        return float(np.clip(price_response, -1.0, 1.0))
    
    def get_demand_at_price_points(
        self,
        prices: np.ndarray,
        time_of_day: float,
        occupancy: float = 0.5
    ) -> np.ndarray:
        """
        Compute occupancy change for multiple continuous prices (PRODUCTION).
        
        Useful for analyzing demand curve at different price points.
        
        Args:
            prices: Array of continuous prices ∈ [0.5, 20.0]
            time_of_day: Time of day ∈ [0, 1]
            occupancy: Current occupancy ∈ [0, 1]
        
        Returns:
            Array of occupancy changes, shape same as input prices
        """
        return np.array([
            self(float(p), time_of_day, occupancy)
            for p in prices
        ])
    
    def get_elasticity_curve(self, time_of_day: float = 0.5) -> tuple:
        """
        Generate elasticity curve for visualization (PRODUCTION).
        
        Args:
            time_of_day: Time for which to compute curve
        
        Returns:
            Tuple of (prices, occupancy_changes) for plotting
        """
        # Generate continuous price points
        prices = np.linspace(0.5, 20.0, 100)
        changes = self.get_demand_at_price_points(prices, time_of_day)
        
        return prices, changes


 

if __name__ == "__main__":
    print("=" * 70)
    print("ROLE 1: DEMAND MODEL - PRODUCTION VERSION")
    print("=" * 70)
    
    # Initialize production model
    model = SimulatorDemandModel(
        elasticity=-0.5,
        reference_price=10.0,
        noise_std=0.02
    )
    
    print("\n1. BASIC OCCUPANCY CHANGE AT VARIOUS CONTINUOUS PRICES")
    print("-" * 70)
    test_prices = [0.5, 2.5, 5.0, 10.0, 15.0, 20.0]
    
    for price in test_prices:
        change = model(price=price, time_of_day=0.5, occupancy=0.5, seed=42)
        print(f"  Price: ${price:>5.1f} → Occupancy change: {change:>8.4f}")
    
    print("\n2. TIME-DEPENDENT DEMAND")
    print("-" * 70)
    times = np.linspace(0, 1, 5)
    
    for t in times:
        change = model(price=10.0, time_of_day=float(t), occupancy=0.5, seed=42)
        hour = int(t * 24)
        print(f"  Time: {hour:>2d}:00 → Occupancy change: {change:>8.4f}")
    
    print("\n3. PRICE ELASTICITY CURVE (Continuous)")
    print("-" * 70)
    prices, changes = model.get_elasticity_curve(time_of_day=0.5)
    
    # Show sample points from elasticity curve
    indices = [0, 25, 50, 75, 99]
    for idx in indices:
        print(f"  Price: ${prices[idx]:>5.1f} → Change: {changes[idx]:>8.4f}")
    
    print("\n4. BATCH COMPUTATION (PRODUCTION)")
    print("-" * 70)
    batch_prices = np.array([5.0, 10.0, 15.0])
    batch_changes = model.get_demand_at_price_points(
        batch_prices,
        time_of_day=0.5,
        occupancy=0.6
    )
    
    for price, change in zip(batch_prices, batch_changes):
        print(f"  Price: ${price:>5.1f} → Change: {change:>8.4f}")
    
    print("\n" + "=" * 70)
    print(f"Model parameters:")
    print(f"  - Price elasticity: {model.elasticity}")
    print(f"  - Reference price: ${model.reference_price:.2f}")
    print(f"  - Noise std dev: {model.noise_std}")
    print(f"  - Departure rate: {model.departure_rate:.2%}")
    print("=" * 70)
