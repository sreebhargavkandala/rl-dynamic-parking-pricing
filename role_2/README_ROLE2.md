# ROLE 2: RL Algorithm (From Scratch) Implementation



**Responsibilities:**
- Implement Actor-Critic (A2C) from scratch
- Policy network
- Value network
- Advantage computation
- Training loop (NO RL libraries like stable-baselines3)
- Write Method section (math + algorithm)

**Deliverables:**
- agent.py - Complete A2C implementation
- Training loop
- Loss functions
- Pseudocode for report

## Integration with ROLE 1

```python
from role_1.env import ParkingPricingEnv
from role_1.data_processing import SimulatorDemandModel

env = ParkingPricingEnv(demand_model=SimulatorDemandModel())

# Train your agent here
agent = ActorCriticAgent(
    state_dim=env.observation_space.shape[0],  # 5
    action_dim=env.action_space.shape[0],      # 1
    action_bounds=(env.min_price, env.max_price),
)
```

Start implementing when ready!
