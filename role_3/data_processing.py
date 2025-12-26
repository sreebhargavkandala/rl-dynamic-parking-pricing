"""
ROLE 3: Data Processing Module
================================

Loads, cleans, and prepares parking demand data for training.
Supports both real datasets and synthetic data generation.

Functions:
- load_parking_dataset(): Load real parking data
- preprocess_data(): Clean and normalize
- create_train_test_split(): Prepare for model training
- generate_synthetic_dataset(): Create synthetic data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class DataConfig:
    """Configuration for data processing."""
    normalize: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 3.0  # std dev threshold
    test_split: float = 0.2
    random_seed: int = 42
    min_samples: int = 100


class DataProcessor:
    """Handles parking data loading, cleaning, and preprocessing."""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.data = None
        self.metadata = {}
        np.random.seed(self.config.random_seed)
    
    def load_csv(self, filepath: str) -> np.ndarray:
        """
        Load parking data from CSV file.
        
        Expected columns:
        - occupancy: Current occupancy ratio [0, 1]
        - time_of_day: Hour of day / 24 [0, 1]
        - demand_factor: External demand level [0, 1]
        - price: Current price [$]
        - revenue: occupancy × price [$]
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            Data as numpy array
        """
        try:
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['occupancy', 'time_of_day', 'demand_factor', 'price', 'revenue']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Extract features in order
            data = df[required_cols].values
            
            self.metadata = {
                'source': 'csv',
                'filepath': filepath,
                'n_samples': len(data),
                'columns': required_cols
            }
            
            self.data = data
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading CSV: {e}")
    
    def load_json(self, filepath: str) -> np.ndarray:
        """
        Load parking data from JSON file.
        
        Expected format:
        {
            "data": [[occupancy, time_of_day, demand_factor, price, revenue], ...],
            "metadata": {...}
        }
        """
        try:
            with open(filepath, 'r') as f:
                file_data = json.load(f)
            
            data = np.array(file_data['data'])
            self.metadata = file_data.get('metadata', {})
            self.metadata['source'] = 'json'
            self.data = data
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading JSON: {e}")
    
    def remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Remove statistical outliers from data.
        
        Uses z-score method: removes points > 3 std dev from mean.
        
        Args:
            data: Feature matrix (n_samples, n_features)
        
        Returns:
            Cleaned data
        """
        if not self.config.remove_outliers:
            return data
        
        # Compute z-scores
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-6
        z_scores = np.abs((data - mean) / std)
        
        # Keep rows where all features have |z| < threshold
        mask = np.all(z_scores < self.config.outlier_threshold, axis=1)
        cleaned_data = data[mask]
        
        n_removed = len(data) - len(cleaned_data)
        if n_removed > 0:
            self.metadata['outliers_removed'] = n_removed
        
        return cleaned_data
    
    def normalize(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalize data to [0, 1] or [-1, 1].
        
        Args:
            data: Feature matrix
        
        Returns:
            Normalized data and normalization parameters
        """
        if not self.config.normalize:
            return data, {}
        
        # Compute normalization parameters
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        ranges = max_vals - min_vals + 1e-6
        
        # Normalize to [0, 1]
        normalized_data = (data - min_vals) / ranges
        
        norm_params = {
            'method': 'minmax',
            'min': min_vals.tolist(),
            'max': max_vals.tolist(),
            'range': ranges.tolist()
        }
        
        return normalized_data, norm_params
    
    def preprocess(self, data: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete preprocessing pipeline.
        
        Steps:
        1. Remove outliers (if enabled)
        2. Normalize to [0, 1] (if enabled)
        3. Validate minimum samples
        
        Args:
            data: Feature matrix (uses self.data if None)
        
        Returns:
            Preprocessed data and statistics
        """
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data loaded. Call load_csv() or load_json() first.")
        
        if len(data) < self.config.min_samples:
            raise ValueError(f"Insufficient data: {len(data)} < {self.config.min_samples}")
        
        # Remove outliers
        data = self.remove_outliers(data)
        
        # Normalize
        data, norm_params = self.normalize(data)
        
        # Update metadata
        self.metadata.update({
            'n_samples_final': len(data),
            'normalization': norm_params
        })
        
        return data, self.metadata
    
    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        test_size: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Split data into train/test sets.
        
        Args:
            X: Feature matrix
            y: Target values (if None, last column of X is target)
            test_size: Test fraction (uses config if None)
        
        Returns:
            Dictionary with X_train, X_test, y_train, y_test
        """
        if test_size is None:
            test_size = self.config.test_split
        
        if y is None:
            # Assume last column is target
            X_data = X[:, :-1]
            y_data = X[:, -1]
        else:
            X_data = X
            y_data = y
        
        # Random split
        n = len(X_data)
        indices = np.random.permutation(n)
        split_idx = int((1 - test_size) * n)
        
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        return {
            'X_train': X_data[train_idx],
            'X_test': X_data[test_idx],
            'y_train': y_data[train_idx],
            'y_test': y_data[test_idx],
            'indices_train': train_idx,
            'indices_test': test_idx
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get processing metadata."""
        return self.metadata.copy()


def generate_synthetic_parking_dataset(
    n_samples: int = 5000,
    n_days: int = 30,
    price_range: Tuple[float, float] = (0.5, 20.0),
    noise_level: float = 0.1,
    random_seed: int = 42,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic parking demand dataset.
    
    Realistic simulation of parking lot occupancy based on:
    - Time of day (peak hours noon and evening)
    - Day of week (weekends busier)
    - Price (negative elasticity)
    - Random demand fluctuations
    
    Args:
        n_samples: Total number of samples
        n_days: Number of days to simulate
        price_range: Min/max price range
        noise_level: Gaussian noise std dev
        random_seed: Random seed for reproducibility
        output_path: Path to save dataset (CSV or JSON)
    
    Returns:
        Generated data and metadata dictionary
    """
    np.random.seed(random_seed)
    
    # Generate time features
    timestamps = np.arange(n_samples)
    day_number = (timestamps // (24 * 3))  % n_days  # ~3 samples per hour
    hour_of_day = (timestamps // 3) % 24
    time_of_day = hour_of_day / 24.0
    
    # Day of week (0=Monday, 6=Sunday)
    day_of_week = (day_number + np.random.randint(0, 7)) % 7
    is_weekend = (day_of_week >= 5).astype(float)
    
    # Random prices
    price = np.random.uniform(price_range[0], price_range[1], n_samples)
    
    # Base demand factor (varies by day, day of week, hour)
    base_demand = np.ones(n_samples) * 0.65
    
    # Day-of-week effect (weekends +15%)
    base_demand += 0.15 * is_weekend
    
    # Hour-of-day seasonality (peak 11am-3pm, 6pm-8pm)
    hour_effect = 0.3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    base_demand += hour_effect
    
    # Random daily fluctuations
    daily_variation = np.repeat(
        np.random.uniform(-0.1, 0.1, n_days),
        (n_samples // n_days) + 1
    )[:n_samples]
    base_demand += daily_variation
    
    # Clamp demand to [0, 1]
    base_demand = np.clip(base_demand, 0.0, 1.0)
    
    # Price elasticity: -0.5 (1% price → 0.5% demand decrease)
    price_effect = -0.5 * ((price - 10.0) / 10.0)
    demand_factor = base_demand * np.exp(price_effect)
    demand_factor = np.clip(demand_factor, 0.0, 1.0)
    
    # Current occupancy (with lag and smoothing)
    occupancy = demand_factor.copy()
    for i in range(1, len(occupancy)):
        occupancy[i] = 0.7 * occupancy[i] + 0.3 * occupancy[i-1]
    
    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    occupancy = np.clip(occupancy + noise, 0.0, 1.0)
    
    # Revenue
    revenue = occupancy * price
    
    # Construct feature matrix
    data = np.column_stack([
        occupancy,
        time_of_day,
        demand_factor,
        price,
        revenue
    ])
    
    # Metadata
    metadata = {
        'generation_method': 'synthetic',
        'n_samples': n_samples,
        'n_days': n_days,
        'price_range': price_range,
        'noise_level': noise_level,
        'random_seed': random_seed,
        'features': ['occupancy', 'time_of_day', 'demand_factor', 'price', 'revenue'],
        'elasticity': -0.5,
        'statistics': {
            'occupancy_mean': float(np.mean(occupancy)),
            'occupancy_std': float(np.std(occupancy)),
            'price_mean': float(np.mean(price)),
            'price_std': float(np.std(price)),
            'revenue_mean': float(np.mean(revenue)),
            'revenue_std': float(np.std(revenue))
        }
    }
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            df = pd.DataFrame(data, columns=['occupancy', 'time_of_day', 'demand_factor', 'price', 'revenue'])
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump({
                    'data': data.tolist(),
                    'metadata': metadata
                }, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}")
        
        print(f"Saved synthetic dataset to {output_path}")
    
    return data, metadata


def load_or_create_dataset(
    data_path: Optional[str] = None,
    generate_synthetic: bool = True,
    n_samples: int = 5000
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load parking dataset or generate synthetic data.
    
    Args:
        data_path: Path to existing dataset
        generate_synthetic: Generate synthetic if file not found
        n_samples: Samples for synthetic data
    
    Returns:
        Data array and metadata
    """
    if data_path and Path(data_path).exists():
        processor = DataProcessor()
        if data_path.endswith('.csv'):
            data = processor.load_csv(data_path)
        elif data_path.endswith('.json'):
            data = processor.load_json(data_path)
        else:
            raise ValueError(f"Unsupported format: {data_path}")
        
        data, metadata = processor.preprocess()
        return data, metadata
    
    elif generate_synthetic:
        return generate_synthetic_parking_dataset(n_samples=n_samples)
    
    else:
        raise FileNotFoundError(f"Dataset not found: {data_path}")


# ============================================================================
# MAIN: Demo usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ROLE 3: DATA PROCESSING MODULE")
    print("=" * 80)
    
    # Generate synthetic dataset
    print("\n1. Generating synthetic parking dataset...")
    data, metadata = generate_synthetic_parking_dataset(
        n_samples=2000,
        n_days=30,
        output_path="./parking_dataset_synthetic.json"
    )
    print(f"   Generated {len(data)} samples over {metadata['n_days']} days")
    print(f"   Features: {metadata['features']}")
    
    # Data statistics
    print("\n2. Dataset statistics:")
    for key, value in metadata['statistics'].items():
        print(f"   {key}: {value:.4f}")
    
    # Preprocessing
    print("\n3. Preprocessing data...")
    processor = DataProcessor()
    processor.data = data
    processed_data, proc_metadata = processor.preprocess()
    print(f"   Samples after processing: {len(processed_data)}")
    print(f"   Normalization method: {proc_metadata['normalization']['method']}")
    
    # Train/test split
    print("\n4. Train/test split...")
    split = processor.train_test_split(processed_data)
    print(f"   Training samples: {len(split['X_train'])}")
    print(f"   Test samples: {len(split['X_test'])}")
    print(f"   Test fraction: {len(split['X_test']) / len(processed_data):.1%}")
    
    print("\n" + "=" * 80)
    print("Data processing complete! Ready for demand model training.")
    print("=" * 80)
