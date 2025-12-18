# ROLE 3: Data & Demand Modelling (Real-World Grounding)



**Responsibilities:**
- Source real parking dataset
- Clean & preprocess data
- Train demand model (price → occupancy)
- Provide callable simulator interface
- Write Data / Environment Modelling subsection

**Deliverables:**
- data_processing.py - Data loading and preprocessing
- demand_model.py - Trained regression model
- Trained regression model (pickled)
- Dataset description

## Integration with ROLE 1

Replace the SimulatorDemandModel with your trained model:

```python
from role_1.env import ParkingPricingEnv
from role_3.demand_model import TrainedDemandModel
import pickle

# Load your trained model
with open('demand_model.pkl', 'rb') as f:
    demand_model = pickle.load(f)

# Use it in the environment
env = ParkingPricingEnv(demand_model=demand_model)
```

Start implementing when ready!
