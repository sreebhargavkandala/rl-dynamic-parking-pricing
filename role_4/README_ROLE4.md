# ROLE 4: Evaluation, Baselines & Presentation



**Responsibilities:**
- Implement baselines (fixed price, time-based)
- Compare RL vs baselines
- Generate plots (revenue, occupancy, volatility)
- Record before/after agent videos
- Write Results, Discussion, Future Work

**Deliverables:**
- evaluation.py - Baselines and evaluation framework
- Plots & tables
- Videos
- Results section of report

## Integration with ROLE 1 & ROLE 2

```python
from role_1.env import ParkingPricingEnv
from role_1.metrics import compute_all_metrics
from role_2.agent import ActorCriticAgent

# Evaluate agent on environment
env = ParkingPricingEnv(...)
agent = ActorCriticAgent(...)

# Get metrics
occupancies = [...]
prices = [...]
metrics = compute_all_metrics(occupancies, prices)

# Compare with baselines
```

Start implementing when ready!
