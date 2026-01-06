 

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StateStatistics:
    """Statistics for state normalization."""
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using z-score."""
        return (state - self.mean) / (self.std + 1e-8)
    
    def denormalize(self, state: np.ndarray) -> np.ndarray:
        """Denormalize state."""
        return state * (self.std + 1e-8) + self.mean
    
    def clip(self, state: np.ndarray) -> np.ndarray:
        """Clip state to observed range."""
        return np.clip(state, self.min, self.max)


class StatePreprocessor:
    """Advanced state preprocessing with multiple normalization strategies."""
    
    def __init__(self, state_dim: int, strategy: str = "zscore"):
        """
        Initialize preprocessor.
        
        Args:
            state_dim: Dimension of state
            strategy: "zscore", "minmax", or "none"
        """
        self.state_dim = state_dim
        self.strategy = strategy
        self.stats: Optional[StateStatistics] = None
        self.buffer = []
        self.initialized = False
    
    def collect_statistics(self, states: np.ndarray):
        """
        Collect statistics from batch of states.
        
        Args:
            states: Array of shape (N, state_dim)
        """
        states = np.array(states)
        
        self.stats = StateStatistics(
            mean=np.mean(states, axis=0),
            std=np.std(states, axis=0),
            min=np.min(states, axis=0),
            max=np.max(states, axis=0)
        )
        
        self.initialized = True
        logger.info(f"✓ State statistics collected from {len(states)} samples")
        logger.info(f"  Mean: {self.stats.mean}")
        logger.info(f"  Std: {self.stats.std}")
    
    def preprocess(self, state: np.ndarray) -> np.ndarray:
        """
        Preprocess state.
        
        Args:
            state: State vector
            
        Returns:
            Preprocessed state
        """
        if self.strategy == "zscore":
            if self.stats:
                return self.stats.normalize(state)
            return state
        
        elif self.strategy == "minmax":
            if self.stats:
                return (state - self.stats.min) / (self.stats.max - self.stats.min + 1e-8)
            return state
        
        else:  # "none"
            return state
    
    def add_batch(self, states: np.ndarray):
        """Add states to buffer for statistics collection."""
        self.buffer.extend(states)
    
    def finalize(self):
        """Compute statistics from buffered states."""
        if self.buffer:
            self.collect_statistics(self.buffer)


class ActionScaler:
    """Smart action scaling and clipping."""
    
    def __init__(self, action_dim: int, min_action: float, max_action: float,
                 strategy: str = "linear"):
        """
        Initialize action scaler.
        
        Args:
            action_dim: Dimension of action
            min_action: Minimum action value
            max_action: Maximum action value
            strategy: "linear", "tanh", or "softplus"
        """
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action
        self.strategy = strategy
        self.action_mean = (min_action + max_action) / 2
        self.action_std = (max_action - min_action) / 2
    
    def scale(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Scale action from network output to action space.
        
        Args:
            raw_action: Raw network output (typically [-∞, ∞])
            
        Returns:
            Scaled action in [min_action, max_action]
        """
        if self.strategy == "linear":
            # Simple clipping
            return np.clip(raw_action, self.min_action, self.max_action)
        
        elif self.strategy == "tanh":
            # Tanh squashes to [-1, 1], then rescale
            scaled = np.tanh(raw_action)
            return self.action_mean + scaled * self.action_std
        
        elif self.strategy == "softplus":
            # Softplus (smooth ReLU), then clip
            scaled = np.log(1 + np.exp(raw_action))
            normalized = (scaled - self.action_mean) / (self.action_std + 1e-8)
            return np.clip(self.action_mean + normalized * self.action_std,
                         self.min_action, self.max_action)
        
        else:
            return np.clip(raw_action, self.min_action, self.max_action)
    
    def unscale(self, action: np.ndarray) -> np.ndarray:
        """
        Unscale action back to network output range (inverse of scale).
        
        Args:
            action: Scaled action
            
        Returns:
            Raw network output
        """
        if self.strategy == "linear":
            return action
        
        elif self.strategy == "tanh":
            normalized = (action - self.action_mean) / (self.action_std + 1e-8)
            return np.arctanh(np.clip(normalized, -0.999, 0.999))
        
        elif self.strategy == "softplus":
            return np.log(np.exp(action) - 1)
        
        else:
            return action
    
    def add_noise(self, action: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """
        Add exploration noise to action.
        
        Args:
            action: Action to add noise to
            noise_std: Standard deviation of noise
            
        Returns:
            Action with added noise
        """
        noise = np.random.normal(0, noise_std, size=action.shape)
        noisy_action = action + noise
        return np.clip(noisy_action, self.min_action, self.max_action)


class MultiModalPreprocessor:
    """Handle multi-modal state with different preprocessing per component."""
    
    def __init__(self, component_dims: Dict[str, int]):
        """
        Initialize with state components.
        
        Args:
            component_dims: Dict mapping component names to dimensions
                Example: {"occupancy": 1, "time": 1, "demand": 1, "price": 1}
        """
        self.component_dims = component_dims
        self.preprocessors = {}
        self.start_idx = 0
        self.component_ranges = {}
        
        # Create preprocessor for each component
        for name, dim in component_dims.items():
            self.preprocessors[name] = StatePreprocessor(dim)
            self.component_ranges[name] = (self.start_idx, self.start_idx + dim)
            self.start_idx += dim
        
        self.total_dim = self.start_idx
    
    def split_state(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Split state into components."""
        components = {}
        for name, (start, end) in self.component_ranges.items():
            components[name] = state[start:end]
        return components
    
    def merge_state(self, components: Dict[str, np.ndarray]) -> np.ndarray:
        """Merge components into single state."""
        state_parts = []
        for name in sorted(self.component_ranges.keys()):
            state_parts.append(components[name])
        return np.concatenate(state_parts)
    
    def preprocess(self, state: np.ndarray) -> np.ndarray:
        """Preprocess multi-modal state."""
        components = self.split_state(state)
        processed = {}
        
        for name, component in components.items():
            processed[name] = self.preprocessors[name].preprocess(component)
        
        return self.merge_state(processed)
    
    def collect_statistics(self, states: np.ndarray):
        """Collect statistics per component."""
        components = [self.split_state(s) for s in states]
        
        for name in self.component_dims.keys():
            component_data = np.array([c[name] for c in components])
            self.preprocessors[name].collect_statistics(component_data)


class ObservationFilter:
    """Filter observations to reduce noise and smooth trajectories."""
    
    def __init__(self, window_size: int = 5):
        """
        Initialize filter.
        
        Args:
            window_size: Moving average window size
        """
        self.window_size = window_size
        self.buffer = []
    
    def filter(self, observation: np.ndarray) -> np.ndarray:
        """
        Apply moving average filter.
        
        Args:
            observation: Current observation
            
        Returns:
            Filtered observation
        """
        self.buffer.append(observation.copy())
        
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        filtered = np.mean(self.buffer, axis=0)
        return filtered
    
    def reset(self):
        """Reset filter buffer."""
        self.buffer = []


def create_preprocessing_pipeline(state_dim: int, action_dim: int, min_action: float,
                                 max_action: float) -> Tuple[StatePreprocessor, ActionScaler]:
   
    state_preprocessor = StatePreprocessor(state_dim, strategy="zscore")
    action_scaler = ActionScaler(action_dim, min_action, max_action, strategy="tanh")
    
    return state_preprocessor, action_scaler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("State Preprocessor initialized with z-score normalization")
    print("Action Scaler initialized with tanh strategy")
    print("Ready for use in agents and environments")
