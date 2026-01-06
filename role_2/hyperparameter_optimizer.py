 

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass, asdict
from itertools import product
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""
    policy_lr: float
    value_lr: float
    entropy_coef: float
    gamma: float
    gae_lambda: float
    hidden_dim: int
    batch_size: int


class HyperparameterSpace:
    """Define hyperparameter search space."""
    
    def __init__(self):
        self.space = {}
    
    def add_float(self, name: str, low: float, high: float, log_scale: bool = False):
        """Add float parameter."""
        self.space[name] = {
            'type': 'float',
            'low': low,
            'high': high,
            'log_scale': log_scale
        }
    
    def add_int(self, name: str, low: int, high: int):
        """Add integer parameter."""
        self.space[name] = {
            'type': 'int',
            'low': low,
            'high': high
        }
    
    def add_choice(self, name: str, choices: List[Any]):
        """Add choice parameter."""
        self.space[name] = {
            'type': 'choice',
            'choices': choices
        }
    
    def sample_random(self) -> Dict:
        """Sample random configuration."""
        config = {}
        
        for name, spec in self.space.items():
            if spec['type'] == 'float':
                if spec['log_scale']:
                    value = np.exp(np.random.uniform(
                        np.log(spec['low']), np.log(spec['high'])
                    ))
                else:
                    value = np.random.uniform(spec['low'], spec['high'])
                config[name] = value
            
            elif spec['type'] == 'int':
                config[name] = np.random.randint(spec['low'], spec['high'] + 1)
            
            elif spec['type'] == 'choice':
                config[name] = np.random.choice(spec['choices'])
        
        return config
    
    def grid_search(self) -> List[Dict]:
        """Generate all configurations for grid search."""
        grid_configs = []
        
        # Build lists for each parameter
        param_lists = {}
        for name, spec in self.space.items():
            if spec['type'] == 'float':
                # For grid search, use 3 values: low, mid, high
                param_lists[name] = [
                    spec['low'],
                    (spec['low'] + spec['high']) / 2,
                    spec['high']
                ]
            elif spec['type'] == 'int':
                # Use low, mid, high
                mid = (spec['low'] + spec['high']) // 2
                param_lists[name] = [spec['low'], mid, spec['high']]
            elif spec['type'] == 'choice':
                param_lists[name] = spec['choices']
        
        # Generate all combinations
        names = list(param_lists.keys())
        for values in product(*[param_lists[n] for n in names]):
            grid_configs.append(dict(zip(names, values)))
        
        return grid_configs


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    config: Dict
    score: float
    episode: int
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class HyperparameterOptimizer:
    """Main optimizer class."""
    
    def __init__(self, output_dir: str = "hyperparameter_tuning"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results: List[OptimizationResult] = []
    
    def grid_search(self, space: HyperparameterSpace, 
                   objective_fn: Callable[[Dict], float],
                   num_episodes: int = 10) -> List[OptimizationResult]:
        """
        Grid search over hyperparameter space.
        
        Args:
            space: HyperparameterSpace object
            objective_fn: Function that takes config and returns score
            num_episodes: Episodes per evaluation
            
        Returns:
            Sorted list of results
        """
        logger.info("\n" + "="*80)
        logger.info("GRID SEARCH HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80)
        
        configs = space.grid_search()
        logger.info(f"Evaluating {len(configs)} configurations\n")
        
        for i, config in enumerate(configs):
            logger.info(f"Config {i+1}/{len(configs)}: {config}")
            score = objective_fn(config)
            
            result = OptimizationResult(
                config=config,
                score=score,
                episode=num_episodes,
                timestamp=datetime.now().isoformat()
            )
            self.results.append(result)
            logger.info(f"  Score: {score:.4f}\n")
        
        # Sort by score (descending)
        self.results.sort(key=lambda x: x.score, reverse=True)
        return self.results
    
    def random_search(self, space: HyperparameterSpace,
                     objective_fn: Callable[[Dict], float],
                     num_samples: int = 20,
                     num_episodes: int = 10) -> List[OptimizationResult]:
        """
        Random search over hyperparameter space.
        
        Args:
            space: HyperparameterSpace object
            objective_fn: Function that takes config and returns score
            num_samples: Number of random samples to evaluate
            num_episodes: Episodes per evaluation
            
        Returns:
            Sorted list of results
        """
        logger.info("\n" + "="*80)
        logger.info("RANDOM SEARCH HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80)
        logger.info(f"Evaluating {num_samples} random configurations\n")
        
        for i in range(num_samples):
            config = space.sample_random()
            logger.info(f"Sample {i+1}/{num_samples}: {config}")
            score = objective_fn(config)
            
            result = OptimizationResult(
                config=config,
                score=score,
                episode=num_episodes,
                timestamp=datetime.now().isoformat()
            )
            self.results.append(result)
            logger.info(f"  Score: {score:.4f}\n")
        
        # Sort by score (descending)
        self.results.sort(key=lambda x: x.score, reverse=True)
        return self.results
    
    def save_results(self, filename: str = "optimization_results.json"):
        """Save optimization results."""
        results_data = [r.to_dict() for r in self.results]
        
        file_path = self.output_dir / filename
        with open(file_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"✓ Results saved to {file_path}")
    
    def get_best_config(self) -> Dict:
        """Get best configuration found."""
        if not self.results:
            return {}
        return self.results[0].config
    
    def get_top_configs(self, k: int = 5) -> List[Dict]:
        """Get top K configurations."""
        return [r.config for r in self.results[:k]]
    
    def print_summary(self):
        """Print optimization summary."""
        if not self.results:
            logger.info("No results to summarize")
            return
        
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("="*80)
        
        best = self.results[0]
        logger.info(f"Best Score: {best.score:.4f}")
        logger.info(f"Best Config: {best.config}")
        
        logger.info("\nTop 5 Configurations:")
        for i, result in enumerate(self.results[:5]):
            logger.info(f"{i+1}. Score: {result.score:.4f}")
            logger.info(f"   Config: {result.config}")


class BayesianOptimizer:
    """Simple Bayesian optimization using Gaussian Process surrogate."""
    
    def __init__(self, output_dir: str = "bayesian_tuning"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.history = []
    
    def suggest(self, space: HyperparameterSpace, n_suggestions: int = 1) -> List[Dict]:
        """
        Suggest next configurations based on history.
        
        Args:
            space: HyperparameterSpace object
            n_suggestions: Number of suggestions
            
        Returns:
            List of suggested configurations
        """
        if not self.history:
            # First iteration: return random samples
            return [space.sample_random() for _ in range(n_suggestions)]
        
        # For simplicity, use best historical config with small perturbation
        best_config = max(self.history, key=lambda x: x['score'])['config']
        
        suggestions = []
        for _ in range(n_suggestions):
            # Small perturbation
            perturbed = best_config.copy()
            for key in perturbed:
                if isinstance(perturbed[key], (int, float)):
                    perturbed[key] *= np.random.uniform(0.95, 1.05)
            suggestions.append(perturbed)
        
        return suggestions
    
    def tell(self, config: Dict, score: float):
        """Tell optimizer about a result."""
        self.history.append({
            'config': config,
            'score': score,
            'timestamp': datetime.now().isoformat()
        })


def create_agent_hyperparameter_space() -> HyperparameterSpace:
    """Create standard hyperparameter space for RL agents."""
    space = HyperparameterSpace()
    
    # Learning rates (log scale)
    space.add_float("policy_lr", 1e-5, 1e-3, log_scale=True)
    space.add_float("value_lr", 1e-5, 1e-3, log_scale=True)
    
    # Coefficients
    space.add_float("entropy_coef", 0.001, 0.1, log_scale=True)
    space.add_float("gamma", 0.95, 0.99)
    space.add_float("gae_lambda", 0.90, 0.99)
    
    # Network architecture
    space.add_choice("hidden_dim", [128, 256, 512])
    space.add_choice("batch_size", [32, 64, 128])
    
    return space


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("Hyperparameter Optimizer initialized")
    print("Methods: grid_search(), random_search(), BayesianOptimizer()")
