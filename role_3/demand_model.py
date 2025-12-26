"""
ROLE 3: Demand Modeling Module
================================

Implements demand forecasting for dynamic parking pricing.
Provides models to predict occupancy based on:
- Current price
- Time of day
- Day of week
- Historical demand patterns

Can be trained on real data or used with synthetic simulator.
"""

import numpy as np
import pickle
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DemandModelConfig:
    """Configuration for demand model."""
    model_type: str = "polynomial"  # "polynomial", "neural", "rule_based"
    degree: int = 2  # For polynomial
    price_elasticity: float = -0.5  # Price-demand elasticity
    base_demand: float = 0.6  # Base occupancy without price effects
    time_seasonality: bool = True  # Include time-of-day effects
    day_seasonality: bool = True  # Include day-of-week effects


class BaseDemandModel:
    """Base class for demand models."""
    
    def __init__(self, config: DemandModelConfig):
        self.config = config
        self.is_trained = False
    
    def predict(self, state: np.ndarray) -> float:
        """
        Predict occupancy given state.
        
        State format: [occupancy, time_of_day, demand_factor, price, revenue]
        Returns: occupancy in [0, 1]
        """
        raise NotImplementedError
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model to data."""
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str) -> 'BaseDemandModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class PolynomialDemandModel(BaseDemandModel):
    """
    Polynomial demand model: occupancy = f(price, time, day).
    
    Uses price elasticity + time-of-day + day-of-week effects.
    Suitable for simple, interpretable demand patterns.
    """
    
    def __init__(self, config: DemandModelConfig):
        super().__init__(config)
        self.coefficients = {}
        self.scaler_params = {}
    
    def predict(self, state: np.ndarray) -> float:
        """
        Predict occupancy from state vector.
        
        Args:
            state: [occupancy, time_of_day, demand_factor, price, revenue]
        
        Returns:
            Predicted occupancy in [0, 1]
        """
        if not self.is_trained:
            # Return rule-based estimate if not trained
            return self._rule_based_demand(state)
        
        occupancy, time_of_day, demand_factor, price, revenue = state
        
        # Base demand from demand_factor
        occupancy_pred = demand_factor
        
        # Price elasticity effect
        price_effect = self.config.price_elasticity * (price - 10.0) / 10.0
        occupancy_pred += price_effect
        
        # Time-of-day effect (peak hours)
        if self.config.time_seasonality:
            hour = (time_of_day * 24) % 24
            time_effect = 0.15 * np.sin(2 * np.pi * hour / 24)  # Peak at noon
            occupancy_pred += time_effect
        
        # Clamp to valid range
        return np.clip(occupancy_pred, 0.0, 1.0)
    
    def _rule_based_demand(self, state: np.ndarray) -> float:
        """Fallback rule-based demand model."""
        occupancy, time_of_day, demand_factor, price, revenue = state
        
        # Base demand
        base_demand = demand_factor
        
        # Lower demand with higher prices (price elasticity)
        price_penalty = -0.5 * (price - 10.0) / 10.0
        
        # Time-of-day effects
        hour = (time_of_day * 24) % 24
        time_effect = 0.15 * np.sin(2 * np.pi * hour / 24)
        
        occupancy_pred = base_demand + price_penalty + time_effect
        return np.clip(occupancy_pred, 0.0, 1.0)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit polynomial model to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target occupancy values (n_samples,)
        """
        # Simple polynomial fit on price dimension
        if len(X) > 0:
            prices = X[:, 3]  # Price is 4th feature
            coeffs = np.polyfit(prices, y, self.config.degree)
            self.coefficients['price_poly'] = coeffs
            
            # Store mean/std for normalization
            self.scaler_params['price_mean'] = np.mean(prices)
            self.scaler_params['price_std'] = np.std(prices) + 1e-6
        
        self.is_trained = True


class RuleBasedDemandModel(BaseDemandModel):
    """
    Rule-based demand model using domain knowledge.
    
    Good for when no training data is available.
    Uses interpretable rules about price elasticity and time patterns.
    """
    
    def __init__(self, config: DemandModelConfig):
        super().__init__(config)
        self.is_trained = True  # Pre-trained with rules
    
    def predict(self, state: np.ndarray) -> float:
        """
        Predict occupancy using rules.
        
        Rules:
        1. Base demand from demand_factor (external demand)
        2. Price elasticity: -0.5 elasticity (1% price increase → 0.5% demand decrease)
        3. Time-of-day: Peak at noon (hour 12), low at night
        4. Day-of-week: (Optional) Weekends busier
        """
        occupancy, time_of_day, demand_factor, price, revenue = state
        
        # Start with external demand factor
        occupancy_pred = demand_factor
        
        # Rule 1: Price elasticity effect
        # Reference price = $10
        # Elasticity = -0.5: 1% price change → 0.5% occupancy change
        price_effect = -0.5 * ((price - 10.0) / 10.0)
        occupancy_pred = occupancy_pred * (1.0 + price_effect)
        
        # Rule 2: Time-of-day seasonality
        if self.config.time_seasonality:
            hour = (time_of_day * 24) % 24
            # Peak at noon (hour 12), low at night (0-6am)
            time_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)
            occupancy_pred = occupancy_pred * time_factor
        
        # Rule 3: Day-of-week seasonality (optional)
        if self.config.day_seasonality:
            # Assume demand_factor already encodes day effects
            pass
        
        # Ensure valid occupancy
        return np.clip(occupancy_pred, 0.0, 1.0)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No-op: rules are pre-defined."""
        self.is_trained = True


class NeuralDemandModel(BaseDemandModel):
    """
    Neural network-based demand model.
    
    Uses a simple 2-layer network for demand prediction.
    Requires training data.
    
    Note: Full implementation would require TensorFlow/PyTorch.
    This is a placeholder showing the interface.
    """
    
    def __init__(self, config: DemandModelConfig):
        super().__init__(config)
        self.weights = None
        self.biases = None
    
    def predict(self, state: np.ndarray) -> float:
        """Neural network forward pass."""
        if not self.is_trained:
            # Fallback to rule-based
            return RuleBasedDemandModel(self.config).predict(state)
        
        # This would require actual NN implementation
        raise NotImplementedError("Full NN model requires PyTorch/TensorFlow")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train neural network on data."""
        raise NotImplementedError("Full NN model requires PyTorch/TensorFlow")


class DemandModelFactory:
    """Factory for creating demand models."""
    
    @staticmethod
    def create(config: DemandModelConfig) -> BaseDemandModel:
        """Create appropriate demand model based on config."""
        if config.model_type == "polynomial":
            return PolynomialDemandModel(config)
        elif config.model_type == "rule_based":
            return RuleBasedDemandModel(config)
        elif config.model_type == "neural":
            return NeuralDemandModel(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")


# ============================================================================
# SYNTHETIC DATA GENERATION (for training/testing)
# ============================================================================

def generate_synthetic_demand_data(
    n_samples: int = 1000,
    price_range: Tuple[float, float] = (0.5, 20.0),
    noise_level: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic parking demand data.
    
    Features:
    - Price elasticity: -0.5 (1% price → 0.5% demand change)
    - Time-of-day seasonality: Peak at noon
    - Day-of-week seasonality: Weekends busier
    
    Args:
        n_samples: Number of samples to generate
        price_range: Min/max prices
        noise_level: Gaussian noise std dev
    
    Returns:
        X: Feature matrix (n_samples, 5)
        y: Target occupancy (n_samples,)
    """
    
    # Generate random features
    occupancy_current = np.random.uniform(0.3, 0.9, n_samples)
    time_of_day = np.random.uniform(0, 1, n_samples)
    demand_factor = np.random.uniform(0.5, 0.95, n_samples)
    price = np.random.uniform(price_range[0], price_range[1], n_samples)
    revenue = np.zeros(n_samples)  # Will be computed
    
    # Compute true occupancy following demand model
    occupancy_true = demand_factor.copy()
    
    # Price elasticity: -0.5
    price_effect = -0.5 * ((price - 10.0) / 10.0)
    occupancy_true = occupancy_true * (1.0 + price_effect)
    
    # Time-of-day seasonality
    hour = (time_of_day * 24) % 24
    time_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)
    occupancy_true = occupancy_true * time_factor
    
    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    occupancy_true = np.clip(occupancy_true + noise, 0.0, 1.0)
    
    # Compute revenue
    revenue = occupancy_true * price
    
    # Construct feature matrix
    X = np.column_stack([
        occupancy_current,
        time_of_day,
        demand_factor,
        price,
        revenue
    ])
    
    return X, occupancy_true


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_demand_model(
    model: BaseDemandModel,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate demand model performance.
    
    Args:
        model: Trained demand model
        X_test: Test feature matrix
        y_test: Test target occupancy
    
    Returns:
        Dictionary with metrics (MAE, RMSE, R²)
    """
    
    predictions = np.array([model.predict(x) for x in X_test])
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - y_test))
    
    # Root Mean Squared Error
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    
    # R² Score
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_occupancy': np.mean(y_test),
        'pred_mean_occupancy': np.mean(predictions)
    }


# ============================================================================
# MAIN: Demo usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ROLE 3: DEMAND MODELING MODULE")
    print("=" * 80)
    
    # Create config
    config = DemandModelConfig(
        model_type="rule_based",
        price_elasticity=-0.5,
        base_demand=0.6,
        time_seasonality=True,
        day_seasonality=False
    )
    
    print("\n1. Creating demand model...")
    model = DemandModelFactory.create(config)
    print(f"   Model type: {config.model_type}")
    print(f"   Model trained: {model.is_trained}")
    
    # Generate test data
    print("\n2. Generating synthetic demand data...")
    X_test, y_test = generate_synthetic_demand_data(n_samples=100)
    print(f"   Generated {len(X_test)} test samples")
    print(f"   Features: occupancy, time_of_day, demand_factor, price, revenue")
    
    # Evaluate
    print("\n3. Evaluating demand model...")
    metrics = evaluate_demand_model(model, X_test, y_test)
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   R²: {metrics['r2']:.4f}")
    print(f"   Mean occupancy (true): {metrics['mean_occupancy']:.2%}")
    print(f"   Mean occupancy (pred): {metrics['pred_mean_occupancy']:.2%}")
    
    # Demo predictions
    print("\n4. Sample predictions:")
    for i in range(3):
        state = X_test[i]
        pred = model.predict(state)
        true = y_test[i]
        error = abs(pred - true)
        print(f"   Price: ${state[3]:.2f} → Pred occupancy: {pred:.2%}, True: {true:.2%}, Error: {error:.2%}")
    
    print("\n" + "=" * 80)
    print("Role 3 demand modeling ready for integration!")
    print("=" * 80)
