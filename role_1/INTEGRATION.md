# Integration Guide - Role 1 to Role 2

## Quick Setup

```bash
pip install gymnasium numpy
```

---

## Basic Usage (Role 2 - RL Training)

```python
import numpy as np
from env import ParkingPricingEnv

# Create environment
env = ParkingPricingEnv(seed=42)

# Reset episode
obs, info = env.reset()

# Your agent takes action (continuous price)
action = np.array([10.50])  # Price between 0.5 and 20.0

# Step
obs, reward, terminated, truncated, info = env.step(action)

# After episode
if terminated:
    metrics = env.get_episode_metrics()
    print(f"Revenue: ${metrics['total_revenue']:.2f}")
```

---

## Training Loop

```python
from env import ParkingPricingEnv
import numpy as np

env = ParkingPricingEnv(seed=42)

for episode in range(1000):
    obs, _ = env.reset()
    episode_reward = 0
    
    for step in range(288):  # 24 hours
        # Your agent selects price action
        action = agent.select_action(obs)
        
        # Environment step
        obs, reward, done, _, info = env.step(action)
        episode_reward += reward
        
        # Your training here
        agent.learn(obs, reward, done)
        
        if done:
            break
    
    # Get metrics
    metrics = env.get_episode_metrics()
    print(f"Ep {episode}: Reward={episode_reward:.1f}, Revenue=${metrics['total_revenue']:.0f}")
```

---

## Environment Specification

### Action Space
- **Type:** Box(0.5, 20.0)
- **Meaning:** Parking price in dollars
- **Example:** `action = np.array([7.50])`

### State Space (5D)
```python
[occupancy_rate,    # 0-1: current lot occupancy
 time_progress,     # 0-1: hour of day (0=midnight, 1=next midnight)
 demand_level,      # 0-1: current demand intensity
 price_t-1,         # 0-1: previous price normalized
 price_t-2]         # 0-1: price 2 steps ago normalized
```

### Episode
- **Duration:** 288 steps (24 hours at 5-min intervals)
- **Termination:** After 288 steps

### Reward Function
```
reward = revenue - occupancy_penalty - volatility_penalty

Where:
  revenue = occupancy × price
  occupancy_penalty = 0.5 × (0.8 - occupancy)²
  volatility_penalty = 0.1 × |price_t - price_t-1|
```

---

## Available Metrics

```python
metrics = env.get_episode_metrics()

# Returns dictionary with:
- total_revenue       # Sum of (occupancy × price × capacity)
- avg_occupancy       # Mean occupancy
- occupancy_std       # Occupancy variability
- min_occupancy       # Lowest occupancy
- max_occupancy       # Highest occupancy
- avg_price           # Mean price
- min_price           # Lowest price
- max_price           # Highest price
- price_volatility    # Price stability (std of price changes)
- episode_length      # Steps completed (always 288)
- total_demand        # Total customer demand
- total_served        # Total customers served
```

---

## API Reference

| Function | Returns | Purpose |
|----------|---------|---------|
| `env.reset()` | (obs, info) | Start new episode |
| `env.step(action)` | (obs, reward, done, truncated, info) | Execute one step |
| `env.get_episode_metrics()` | dict | Get all metrics |

---

## Example: Complete Episode

```python
from env import ParkingPricingEnv
import numpy as np

env = ParkingPricingEnv(seed=42)
obs, _ = env.reset()

print(f"Initial state shape: {obs.shape}")

# Run one complete episode
total_reward = 0
for step in range(288):
    # Random policy (replace with your agent)
    action = env.action_space.sample()
    
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    
    if done:
        break

# Results
metrics = env.get_episode_metrics()
print(f"\nEpisode Results:")
print(f"  Total Reward: {total_reward:.2f}")
print(f"  Total Revenue: ${metrics['total_revenue']:.2f}")
print(f"  Avg Occupancy: {metrics['avg_occupancy']:.1%}")
print(f"  Price Volatility: ${metrics['price_volatility']:.2f}")
```

---

## Testing

Run the test suite:

```bash
python test_role1.py
```

Expected output: **8/8 tests PASSING** ✓

---

## File Reference

| File | Purpose |
|------|---------|
| `env.py` | Main environment (what you import) |
| `reward_function.py` | Reward computation |
| `metrics.py` | Metrics calculation |
| `data_processing.py` | Demand model |
| `test_role1.py` | Test suite |
| `state_action_documentation.py` | Mathematical specifications |
| `README_ROLE1.md` | Full documentation |

---

## Notes

- All observations are normalized to [0, 1]
- All prices are clipped to [0.5, 20.0]
- Use seed for reproducibility: `ParkingPricingEnv(seed=42)`
- For custom demand model: `ParkingPricingEnv(demand_model=YourModel())`
