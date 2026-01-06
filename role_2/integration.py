 

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import torch
import sys
from pathlib import Path


class IntegrationConfig:
    """Configuration for integrated training."""
    
    # Environment settings
    CAPACITY: int = 100
    MAX_STEPS: int = 288  # 24 hours in 5-min intervals
    TARGET_OCCUPANCY: float = 0.8
    MIN_PRICE: float = 0.5
    MAX_PRICE: float = 20.0
    
    # Agent settings
    STATE_DIM: int = 5
    ACTION_DIM: int = 1
    LEARNING_RATE: float = 3e-4
    GAMMA: float = 0.99
    ENTROPY_COEF: float = 0.01
    
    # Training settings
    NUM_EPISODES: int = 1000
    BATCH_SIZE: int = 32
    EVAL_EPISODES: int = 5
    LOG_INTERVAL: int = 10
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED: int = 42


class EnvironmentWrapper:
    """Wraps Role 1 environment with validation and utilities.
    
    Ensures type safety and handles edge cases between environment and agent.
    """
    
    def __init__(self, env_class, **env_kwargs):
        """Initialize environment wrapper.
        
        Args:
            env_class: Environment class (typically ParkingPricingEnv)
            **env_kwargs: Arguments to pass to environment
        """
        self.env = env_class(**env_kwargs)
        self.config = IntegrationConfig()
        
        # Validate action/state spaces
        self._validate_spaces()
    
    def _validate_spaces(self) -> None:
        """Validate environment action and state spaces match configuration."""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        if state_dim != self.config.STATE_DIM:
            raise ValueError(
                f"State dimension mismatch. Expected {self.config.STATE_DIM}, "
                f"got {state_dim}. Check environment configuration."
            )
        
        if action_dim != self.config.ACTION_DIM:
            raise ValueError(
                f"Action dimension mismatch. Expected {self.config.ACTION_DIM}, "
                f"got {action_dim}. Check environment configuration."
            )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation and info dict
        """
        obs, info = self.env.reset(seed=seed)
        return self._validate_obs(obs), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment.
        
        Args:
            action: Action from agent (shape: action_dim,)
            
        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        # Validate and clip action
        action = self._validate_action(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Validate outputs
        obs = self._validate_obs(obs)
        reward = self._validate_reward(reward)
        
        return obs, reward, terminated, truncated, info
    
    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        """Validate and clip action to valid range.
        
        Args:
            action: Raw action from agent
            
        Returns:
            Validated and clipped action
        """
        if not isinstance(action, (np.ndarray, float)):
            raise TypeError(f"Action must be ndarray or float, got {type(action)}")
        
        action = np.array(action).flatten()
        if action.shape[0] != self.config.ACTION_DIM:
            raise ValueError(
                f"Action dimension mismatch. Expected {self.config.ACTION_DIM}, "
                f"got {action.shape[0]}"
            )
        
        # Clip to valid price range
        action = np.clip(
            action,
            self.config.MIN_PRICE,
            self.config.MAX_PRICE
        )
        
        return action
    
    def _validate_obs(self, obs: np.ndarray) -> np.ndarray:
        """Validate observation shape and values.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Validated observation
        """
        if not isinstance(obs, np.ndarray):
            raise TypeError(f"Observation must be ndarray, got {type(obs)}")
        
        obs = obs.flatten()
        if obs.shape[0] != self.config.STATE_DIM:
            raise ValueError(
                f"Observation dimension mismatch. Expected {self.config.STATE_DIM}, "
                f"got {obs.shape[0]}"
            )
        
        # Check for NaN/Inf
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            raise ValueError(f"Observation contains NaN/Inf: {obs}")
        
        return obs
    
    def _validate_reward(self, reward: float) -> float:
        """Validate reward value.
        
        Args:
            reward: Reward from environment
            
        Returns:
            Validated reward
        """
        if not isinstance(reward, (float, int, np.number)):
            raise TypeError(f"Reward must be numeric, got {type(reward)}")
        
        reward = float(reward)
        if np.isnan(reward) or np.isinf(reward):
            raise ValueError(f"Reward is NaN/Inf: {reward}")
        
        return reward
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get episode metrics from environment.
        
        Returns:
            Dictionary of metrics
        """
        if hasattr(self.env, 'get_episode_metrics'):
            return self.env.get_episode_metrics()
        return {}


class AgentWrapper:
    """Wraps Role 2 agent with validation and utilities."""
    
    def __init__(self, agent_class, **agent_kwargs):
        """Initialize agent wrapper.
        
        Args:
            agent_class: Agent class (typically ActorCriticAgent)
            **agent_kwargs: Arguments to pass to agent
        """
        self.agent = agent_class(**agent_kwargs)
        self.config = IntegrationConfig()
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Select action from policy.
        
        Args:
            state: Current state from environment
            
        Returns:
            Tuple of (action, log_prob)
        """
        state = self._validate_state(state)
        action, log_prob = self.agent.select_action(state)
        return action, log_prob
    
    def update(
        self,
        state: np.ndarray,
        log_prob: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """Update agent parameters.
        
        Args:
            state: Current state
            log_prob: Log probability from action selection
            reward: Reward from environment
            next_state: Next state
            done: Episode termination flag
            
        Returns:
            Dictionary of loss values
        """
        state = self._validate_state(state)
        next_state = self._validate_state(next_state)
        reward = float(reward)
        
        losses = self.agent.update(state, log_prob, reward, next_state, done)
        return losses
    
    def _validate_state(self, state: np.ndarray) -> np.ndarray:
        """Validate state shape and values.
        
        Args:
            state: State from environment
            
        Returns:
            Validated state
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        state = state.flatten()
        if state.shape[0] != self.config.STATE_DIM:
            raise ValueError(
                f"State dimension mismatch. Expected {self.config.STATE_DIM}, "
                f"got {state.shape[0]}"
            )
        
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise ValueError(f"State contains NaN/Inf: {state}")
        
        return state
    
    def save(self, filepath: str) -> None:
        """Save agent checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        self.agent.save(filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent checkpoint.
        
        Args:
            filepath: Path to load checkpoint
        """
        self.agent.load(filepath)


class IntegratedTrainer:
    """Full integration of Role 1 environment and Role 2 agent.
    
    Provides complete training, evaluation, and analysis pipeline.
    """
    
    def __init__(
        self,
        env_wrapper: EnvironmentWrapper,
        agent_wrapper: AgentWrapper,
        checkpoint_dir: str = "./checkpoints"
    ):
        """Initialize integrated trainer.
        
        Args:
            env_wrapper: Wrapped environment
            agent_wrapper: Wrapped agent
            checkpoint_dir: Directory for checkpoints
        """
        self.env = env_wrapper
        self.agent = agent_wrapper
        self.config = IntegrationConfig()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history
        self.episode_rewards = []
        self.episode_revenues = []
        self.episode_occupancies = []
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode.
        
        Returns:
            Episode statistics
        """
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_losses = []
        
        done = False
        step = 0
        while not done and step < self.config.MAX_STEPS:
            # Agent selects action
            action, log_prob = self.agent.select_action(obs)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Agent update
            losses = self.agent.update(obs, log_prob, reward, next_obs, done)
            episode_losses.append(losses)
            
            episode_reward += reward
            obs = next_obs
            step += 1
        
        # Get environment metrics
        env_metrics = self.env.get_metrics()
        
        # Compile episode statistics
        stats = {
            'episode_reward': episode_reward,
            'episode_steps': step,
            'avg_loss': np.mean([l.get('value_loss', 0) for l in episode_losses])
            if episode_losses else 0.0,
        }
        stats.update(env_metrics)
        
        return stats
    
    def train(self, num_episodes: int, save_best: bool = True) -> None:
        """Train agent for specified episodes.
        
        Args:
            num_episodes: Number of episodes to train
            save_best: Whether to save best performing model
        """
        best_reward = -np.inf
        
        print(f"{'='*70}")
        print(f"Starting Integrated Training: Role 1 (Env) + Role 2 (Agent)")
        print(f"{'='*70}")
        print(f"Episodes: {num_episodes} | Device: {self.config.DEVICE}")
        print(f"State Dim: {self.config.STATE_DIM} | Action Dim: {self.config.ACTION_DIM}")
        print(f"{'='*70}\n")
        
        for episode in range(1, num_episodes + 1):
            stats = self.train_episode()
            
            self.episode_rewards.append(stats['episode_reward'])
            if 'total_revenue' in stats:
                self.episode_revenues.append(stats['total_revenue'])
            if 'avg_occupancy' in stats:
                self.episode_occupancies.append(stats['avg_occupancy'])
            
            # Logging
            if episode % self.config.LOG_INTERVAL == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config.LOG_INTERVAL:])
                
                print(f"Episode {episode:4d}/{num_episodes} | "
                      f"Reward: {avg_reward:8.2f} | "
                      f"Steps: {stats['episode_steps']:3d}")
                
                if 'total_revenue' in stats:
                    avg_revenue = np.mean(
                        self.episode_revenues[-self.config.LOG_INTERVAL:]
                    )
                    print(f"                  | Revenue: ${avg_revenue:8.2f} | "
                          f"Occupancy: {stats.get('avg_occupancy', 0):5.1%}")
                
                # Save checkpoint
                if save_best and avg_reward > best_reward:
                    best_reward = avg_reward
                    self.agent.save(str(self.checkpoint_dir / "best_model.pt"))
                    print(f"✓ Best model saved (reward: {best_reward:.2f})\n")
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate agent without training.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        eval_rewards = []
        eval_revenues = []
        eval_occupancies = []
        
        print(f"\n{'='*70}")
        print(f"Evaluation: {num_episodes} Episodes")
        print(f"{'='*70}\n")
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False
            step = 0
            
            while not done and step < self.config.MAX_STEPS:
                with torch.no_grad():
                    action, _ = self.agent.select_action(obs)
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                obs = next_obs
                step += 1
            
            eval_rewards.append(episode_reward)
            env_metrics = self.env.get_metrics()
            if 'total_revenue' in env_metrics:
                eval_revenues.append(env_metrics['total_revenue'])
            if 'avg_occupancy' in env_metrics:
                eval_occupancies.append(env_metrics['avg_occupancy'])
            
            print(f"Eval Episode {ep+1:2d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Revenue: ${env_metrics.get('total_revenue', 0):8.2f} | "
                  f"Occupancy: {env_metrics.get('avg_occupancy', 0):5.1%}")
        
        results = {
            'mean_reward': float(np.mean(eval_rewards)),
            'std_reward': float(np.std(eval_rewards)),
            'max_reward': float(np.max(eval_rewards)),
            'min_reward': float(np.min(eval_rewards)),
        }
        
        if eval_revenues:
            results['mean_revenue'] = float(np.mean(eval_revenues))
            results['max_revenue'] = float(np.max(eval_revenues))
        
        if eval_occupancies:
            results['mean_occupancy'] = float(np.mean(eval_occupancies))
        
        print(f"\n{'='*70}")
        print(f"Evaluation Results:")
        for key, value in results.items():
            if 'revenue' in key:
                print(f"  {key:20s}: ${value:8.2f}")
            elif 'occupancy' in key:
                print(f"  {key:20s}: {value:5.1%}")
            else:
                print(f"  {key:20s}: {value:8.2f}")
        print(f"{'='*70}\n")
        
        return results
