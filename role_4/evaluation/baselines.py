"""
Role 4: Evaluation Framework for Parking Pricing
=================================================

Implements baseline strategies and evaluation tools for comparing
RL agents against traditional pricing approaches.

Baselines:
- Fixed Price: Constant hourly rate
- Time-Based: Peak/off-peak pricing
- Random: Random prices (worst-case baseline)
- Demand-Based: Simple rule matching current occupancy

Author: Role 4 - Evaluation, Baselines & Presentation
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from role_1.env import ParkingPricingEnv
from role_1.metrics import compute_all_metrics, ParkingMetrics


# =============================================================================
# BASELINE STRATEGY INTERFACE
# =============================================================================

class PricingStrategy(ABC):
    """Abstract base class for pricing strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""
        pass
    
    @abstractmethod
    def get_price(self, observation: np.ndarray, env: ParkingPricingEnv) -> float:
        """
        Determine price based on current state.
        
        Args:
            observation: Environment observation [occupancy, time, demand, price_t-1, price_t-2]
            env: Environment instance (for accessing config)
            
        Returns:
            Price in dollars
        """
        pass
    
    def reset(self):
        """Reset any internal state (called at episode start)."""
        pass


# =============================================================================
# BASELINE IMPLEMENTATIONS
# =============================================================================

class FixedPriceStrategy(PricingStrategy):
    """
    Fixed Price Baseline
    --------------------
    Charges a constant price regardless of conditions.
    Represents traditional static pricing with no optimization.
    
    Common in real parking: flat hourly rate signs.
    """
    
    def __init__(self, price: float = 5.0):
        """
        Args:
            price: Fixed hourly price in dollars
        """
        self.price = price
    
    @property
    def name(self) -> str:
        return f"Fixed Price (${self.price:.2f})"
    
    def get_price(self, observation: np.ndarray, env: ParkingPricingEnv) -> float:
        return self.price


class TimeBasedStrategy(PricingStrategy):
    """
    Time-Based Pricing Baseline
    ---------------------------
    Higher prices during peak hours (8am-6pm), lower off-peak.
    Represents simple time-of-day pricing without demand sensing.
    
    Common approach: event-based or rush hour pricing.
    """
    
    def __init__(self, peak_price: float = 8.0, offpeak_price: float = 3.0,
                 peak_start: float = 0.33, peak_end: float = 0.75):
        """
        Args:
            peak_price: Price during peak hours
            offpeak_price: Price during off-peak hours
            peak_start: Normalized time when peak starts (0.33 ≈ 8am)
            peak_end: Normalized time when peak ends (0.75 ≈ 6pm)
        """
        self.peak_price = peak_price
        self.offpeak_price = offpeak_price
        self.peak_start = peak_start
        self.peak_end = peak_end
    
    @property
    def name(self) -> str:
        return f"Time-Based (${self.offpeak_price:.0f}/${self.peak_price:.0f})"
    
    def get_price(self, observation: np.ndarray, env: ParkingPricingEnv) -> float:
        time_of_day = observation[1]  # Normalized time [0, 1]
        
        if self.peak_start <= time_of_day <= self.peak_end:
            return self.peak_price
        else:
            return self.offpeak_price


class RandomPriceStrategy(PricingStrategy):
    """
    Random Pricing Baseline
    -----------------------
    Uniformly random prices - worst case baseline.
    Useful for showing RL improvement over random policy.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    @property
    def name(self) -> str:
        return "Random Policy"
    
    def get_price(self, observation: np.ndarray, env: ParkingPricingEnv) -> float:
        return self.rng.uniform(env.min_price, env.max_price)
    
    def reset(self):
        pass  # Keep same random sequence for reproducibility


class DemandBasedStrategy(PricingStrategy):
    """
    Simple Demand-Based Pricing
    ---------------------------
    Rule-based pricing that increases price when occupancy is high.
    Represents simple reactive pricing without learning.
    
    Rule: price = base + (occupancy - target) * sensitivity
    """
    
    def __init__(self, base_price: float = 5.0, sensitivity: float = 10.0):
        """
        Args:
            base_price: Base price at target occupancy
            sensitivity: How much price changes per 10% occupancy deviation
        """
        self.base_price = base_price
        self.sensitivity = sensitivity
    
    @property
    def name(self) -> str:
        return "Demand-Based (Rule)"
    
    def get_price(self, observation: np.ndarray, env: ParkingPricingEnv) -> float:
        occupancy = observation[0]
        target = env.target_occupancy
        
        # Price adjustment based on occupancy vs target
        adjustment = (occupancy - target) * self.sensitivity
        price = self.base_price + adjustment
        
        return np.clip(price, env.min_price, env.max_price)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

@dataclass
class EvaluationResult:
    """Results from evaluating a strategy over multiple episodes."""
    strategy_name: str
    num_episodes: int
    
    # Revenue metrics
    total_revenues: List[float] = field(default_factory=list)
    avg_revenue: float = 0.0
    std_revenue: float = 0.0
    
    # Occupancy metrics
    avg_occupancies: List[float] = field(default_factory=list)
    mean_occupancy: float = 0.0
    std_occupancy: float = 0.0
    
    # Price metrics
    avg_prices: List[float] = field(default_factory=list)
    mean_price: float = 0.0
    price_volatility: float = 0.0
    
    # Episode rewards (for RL comparison)
    episode_rewards: List[float] = field(default_factory=list)
    mean_reward: float = 0.0
    
    def compute_statistics(self):
        """Compute aggregate statistics from per-episode data."""
        if self.total_revenues:
            self.avg_revenue = np.mean(self.total_revenues)
            self.std_revenue = np.std(self.total_revenues)
        
        if self.avg_occupancies:
            self.mean_occupancy = np.mean(self.avg_occupancies)
            self.std_occupancy = np.std(self.avg_occupancies)
        
        if self.avg_prices:
            self.mean_price = np.mean(self.avg_prices)
        
        if self.episode_rewards:
            self.mean_reward = np.mean(self.episode_rewards)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy_name": self.strategy_name,
            "num_episodes": int(self.num_episodes),
            "avg_revenue": float(round(self.avg_revenue, 2)),
            "std_revenue": float(round(self.std_revenue, 2)),
            "mean_occupancy": float(round(self.mean_occupancy, 4)),
            "std_occupancy": float(round(self.std_occupancy, 4)),
            "mean_price": float(round(self.mean_price, 2)),
            "price_volatility": float(round(self.price_volatility, 4)),
            "mean_reward": float(round(self.mean_reward, 2)),
        }


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

def evaluate_strategy(
    env: ParkingPricingEnv,
    strategy: PricingStrategy,
    num_episodes: int = 20,
    seed: int = 42,
    verbose: bool = False
) -> EvaluationResult:
    """
    Evaluate a pricing strategy over multiple episodes.
    
    Args:
        env: ParkingPricingEnv instance
        strategy: PricingStrategy implementation
        num_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        EvaluationResult with aggregate metrics
    """
    result = EvaluationResult(
        strategy_name=strategy.name,
        num_episodes=num_episodes
    )
    
    all_price_changes = []
    
    for ep in range(num_episodes):
        strategy.reset()
        obs, _ = env.reset(seed=seed + ep)
        
        done = False
        episode_reward = 0.0
        episode_prices = []
        episode_occupancies = []
        
        while not done:
            price = strategy.get_price(obs, env)
            episode_prices.append(price)
            
            action = np.array([price], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_occupancies.append(info['occupancy'])
            done = terminated or truncated
        
        # Collect episode metrics
        metrics = env.get_episode_metrics()
        result.total_revenues.append(metrics['total_revenue'])
        result.avg_occupancies.append(metrics['avg_occupancy'])
        result.avg_prices.append(np.mean(episode_prices))
        result.episode_rewards.append(episode_reward)
        
        # Track price changes for volatility
        if len(episode_prices) > 1:
            changes = np.abs(np.diff(episode_prices))
            all_price_changes.extend(changes)
        
        if verbose:
            print(f"  Episode {ep+1}: Revenue=${metrics['total_revenue']:.2f}, "
                  f"Occupancy={metrics['avg_occupancy']:.1%}")
    
    # Compute price volatility
    if all_price_changes:
        result.price_volatility = np.std(all_price_changes)
    
    result.compute_statistics()
    
    return result


def evaluate_rl_agent(
    env: ParkingPricingEnv,
    agent,
    num_episodes: int = 20,
    seed: int = 42,
    verbose: bool = False
) -> EvaluationResult:
    """
    Evaluate trained RL agent.
    
    Args:
        env: ParkingPricingEnv instance
        agent: Trained A2C agent (or any agent with select_action method)
        num_episodes: Number of evaluation episodes
        seed: Random seed
        verbose: Print progress
        
    Returns:
        EvaluationResult with aggregate metrics
    """
    import torch
    
    result = EvaluationResult(
        strategy_name="RL Agent (A2C)",
        num_episodes=num_episodes
    )
    
    all_price_changes = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        
        done = False
        episode_reward = 0.0
        episode_prices = []
        episode_occupancies = []
        
        while not done:
            # Agent action selection
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            action, _, _ = agent.select_action(state_tensor, training=False)
            
            # Handle tensor/numpy conversion
            if isinstance(action, torch.Tensor):
                price = action.cpu().detach().numpy().flatten()[0]
            elif isinstance(action, np.ndarray):
                price = action.flatten()[0]
            else:
                price = float(action)
            
            episode_prices.append(price)
            
            action_arr = np.array([price], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action_arr)
            
            episode_reward += reward
            episode_occupancies.append(info['occupancy'])
            done = terminated or truncated
        
        # Collect episode metrics
        metrics = env.get_episode_metrics()
        result.total_revenues.append(metrics['total_revenue'])
        result.avg_occupancies.append(metrics['avg_occupancy'])
        result.avg_prices.append(np.mean(episode_prices))
        result.episode_rewards.append(episode_reward)
        
        # Track price changes
        if len(episode_prices) > 1:
            changes = np.abs(np.diff(episode_prices))
            all_price_changes.extend(changes)
        
        if verbose:
            print(f"  Episode {ep+1}: Revenue=${metrics['total_revenue']:.2f}, "
                  f"Occupancy={metrics['avg_occupancy']:.1%}")
    
    if all_price_changes:
        result.price_volatility = np.std(all_price_changes)
    
    result.compute_statistics()
    
    return result


# =============================================================================
# COMPARISON & REPORTING
# =============================================================================

def compare_strategies(
    env: ParkingPricingEnv,
    strategies: List[PricingStrategy],
    rl_agent=None,
    num_episodes: int = 20,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, EvaluationResult]:
    """
    Compare multiple strategies including RL agent.
    
    Args:
        env: Environment instance
        strategies: List of baseline strategies
        rl_agent: Optional trained RL agent
        num_episodes: Episodes per strategy
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary mapping strategy name to EvaluationResult
    """
    results = {}
    
    print("\n" + "="*70)
    print("  STRATEGY COMPARISON - EVALUATION")
    print("="*70)
    
    # Evaluate each baseline
    for strategy in strategies:
        if verbose:
            print(f"\nEvaluating: {strategy.name}")
        
        result = evaluate_strategy(env, strategy, num_episodes, seed, verbose)
        results[strategy.name] = result
        
        if verbose:
            print(f"  → Avg Revenue: ${result.avg_revenue:,.2f} ± ${result.std_revenue:.2f}")
            print(f"  → Avg Occupancy: {result.mean_occupancy:.1%}")
    
    # Evaluate RL agent if provided
    if rl_agent is not None:
        if verbose:
            print(f"\nEvaluating: RL Agent (A2C)")
        
        result = evaluate_rl_agent(env, rl_agent, num_episodes, seed, verbose)
        results["RL Agent (A2C)"] = result
        
        if verbose:
            print(f"  → Avg Revenue: ${result.avg_revenue:,.2f} ± ${result.std_revenue:.2f}")
            print(f"  → Avg Occupancy: {result.mean_occupancy:.1%}")
    
    return results


def generate_comparison_table(results: Dict[str, EvaluationResult]) -> str:
    """Generate formatted comparison table."""
    lines = [
        "",
        "=" * 90,
        "  STRATEGY COMPARISON RESULTS",
        "=" * 90,
        f"{'Strategy':<30} {'Avg Revenue':>15} {'Avg Occupancy':>15} {'Avg Price':>12} {'Volatility':>12}",
        "-" * 90
    ]
    
    for name, result in results.items():
        lines.append(
            f"{name:<30} ${result.avg_revenue:>13,.2f} {result.mean_occupancy:>14.1%} "
            f"${result.mean_price:>10.2f} ${result.price_volatility:>10.4f}"
        )
    
    lines.append("=" * 90)
    
    # Find best performer
    best_revenue = max(results.values(), key=lambda r: r.avg_revenue)
    lines.append(f"\n  BEST REVENUE: {best_revenue.strategy_name} (${best_revenue.avg_revenue:,.2f})")
    
    return "\n".join(lines)


def save_results(
    results: Dict[str, EvaluationResult],
    output_dir: Path
) -> None:
    """Save evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_data = {name: result.to_dict() for name, result in results.items()}
    with open(output_dir / "comparison_results.json", 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save CSV
    import csv
    with open(output_dir / "comparison_table.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Strategy", "Avg Revenue", "Std Revenue", "Mean Occupancy", 
                        "Mean Price", "Price Volatility", "Mean Reward"])
        for name, result in results.items():
            writer.writerow([
                name, result.avg_revenue, result.std_revenue, result.mean_occupancy,
                result.mean_price, result.price_volatility, result.mean_reward
            ])
    
    print(f"\n✓ Results saved to {output_dir}")


# =============================================================================
# DEFAULT BASELINES
# =============================================================================

def get_default_baselines() -> List[PricingStrategy]:
    """Get standard set of baseline strategies."""
    return [
        FixedPriceStrategy(price=5.0),   # $5 flat rate
        FixedPriceStrategy(price=10.0),  # $10 flat rate (higher)
        TimeBasedStrategy(peak_price=10.0, offpeak_price=3.0),  # Peak pricing
        RandomPriceStrategy(seed=42),     # Random baseline
        DemandBasedStrategy(base_price=5.0, sensitivity=10.0),  # Rule-based
    ]


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ROLE 4: EVALUATION FRAMEWORK - STANDALONE TEST")
    print("="*70)
    
    # Create environment
    env = ParkingPricingEnv(
        capacity=100,
        max_steps=288,
        target_occupancy=0.8,
        min_price=0.5,
        max_price=20.0,
        seed=42
    )
    
    # Get baselines
    baselines = get_default_baselines()
    
    # Run comparison (no RL agent for standalone test)
    results = compare_strategies(env, baselines, num_episodes=5, verbose=True)
    
    # Print table
    print(generate_comparison_table(results))
    
    print("\n✓ Evaluation framework test complete!")
