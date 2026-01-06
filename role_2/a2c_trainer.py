 

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
import pickle


logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics collected during an episode."""
    episode_num: int
    total_reward: float
    episode_length: int
    avg_value: float
    avg_policy_loss: float
    avg_value_loss: float
    timestamp: str


class TrajectoryBuffer:
    """
    Buffer for storing trajectories during episode collection.
    
    Efficiently manages states, actions, rewards, and values
    for batch processing during training updates.
    """
    
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_values = []
        
    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        next_value: float
    ):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.next_values.append(next_value)
    
    def get_batch(self) -> Tuple:
        """Retrieve entire batch."""
        return (
            self.states,
            self.actions,
            self.rewards,
            self.values,
            self.log_probs,
            self.dones,
            self.next_values
        )
    
    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_values = []
    
    def __len__(self):
        return len(self.states)


class LearningRateScheduler:
    """
    Learning Rate Scheduling for improved convergence.
    
    Implements various scheduling strategies:
    - Linear decay
    - Exponential decay
    - Step decay
    - Cosine annealing
    """
    
    def __init__(
        self,
        initial_lr: float,
        schedule_type: str = "constant",
        total_steps: int = 10000,
        warmup_steps: int = 0
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            initial_lr: Starting learning rate
            schedule_type: "constant", "linear", "exponential", "step", "cosine"
            total_steps: Total training steps
            warmup_steps: Number of warmup steps
        """
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (self.current_step / self.warmup_steps)
        
        if self.schedule_type == "constant":
            return self.initial_lr
        
        elif self.schedule_type == "linear":
            # Linear decay to 0
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return self.initial_lr * (1 - progress)
        
        elif self.schedule_type == "exponential":
            # Exponential decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return self.initial_lr * np.exp(-progress * 5)
        
        elif self.schedule_type == "step":
            # Step decay (halve every 25% of training)
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            steps = int(progress * 4)
            return self.initial_lr / (2 ** steps)
        
        elif self.schedule_type == "cosine":
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return self.initial_lr * (1 + np.cos(np.pi * progress)) / 2
        
        return self.initial_lr
    
    def step(self):
        """Advance to next step."""
        self.current_step += 1


class A2CTrainer:
    """
    A2C Training System - FROM SCRATCH.
    
    Manages:
    - Episode collection
    - Model training
    - Checkpointing
    - Metric tracking
    - Early stopping
    - Progress monitoring
    """
    
    def __init__(
        self,
        agent,
        env,
        config: Dict,
        save_dir: str = "./checkpoints"
    ):
        """
        Initialize trainer.
        
        Args:
            agent: A2CAgent instance
            env: RL environment (must support gym interface)
            config: Training configuration dictionary
            save_dir: Directory for saving checkpoints
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.episode_num = 0
        self.total_steps = 0
        self.best_reward = -np.inf
        self.patience_counter = 0
        
        # Buffers and scheduling
        self.trajectory_buffer = TrajectoryBuffer()
        self.lr_scheduler = LearningRateScheduler(
            initial_lr=agent.config.policy_lr,
            schedule_type=config.get("lr_schedule", "constant"),
            total_steps=config.get("total_episodes", 1000),
            warmup_steps=config.get("warmup_episodes", 0)
        )
        
        # Metrics
        self.episode_metrics: List[EpisodeMetrics] = []
        self.training_history = {
            "episode_num": [],
            "total_reward": [],
            "episode_length": [],
            "avg_policy_loss": [],
            "avg_value_loss": [],
            "entropy": []
        }
        
        logger.info(f"Trainer initialized. Save directory: {self.save_dir}")
    
    def collect_episode(self) -> EpisodeMetrics:
        """
        Collect one episode of experience.
        
        Interaction loop:
        1. Reset environment
        2. For each timestep:
           a. Select action using policy
           b. Execute action, get reward and next state
           c. Store transition in buffer
           d. Compute advantage and value
        3. Return episode metrics
        
        Returns:
            EpisodeMetrics object with episode statistics
        """
        state, _ = self.env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_values = []
        
        self.trajectory_buffer.clear()
        
        while not done:
            # Select action from policy
            action, log_prob, value = self.agent.select_action(state, training=True)
            episode_values.append(value)
            
            # Step environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Get next value for bootstrapping
            _, _, next_value = self.agent.select_action(next_state, training=True)
            
            # Store transition
            self.trajectory_buffer.add_transition(
                state=state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                next_value=next_value
            )
            
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            state = next_state
        
        # Train on collected trajectory
        update_metrics = self.agent.update(*self.trajectory_buffer.get_batch())
        
        # Create metrics object
        metrics = EpisodeMetrics(
            episode_num=self.episode_num,
            total_reward=episode_reward,
            episode_length=episode_length,
            avg_value=np.mean(episode_values),
            avg_policy_loss=update_metrics.get("policy_loss", 0.0),
            avg_value_loss=update_metrics.get("value_loss", 0.0),
            timestamp=datetime.now().isoformat()
        )
        
        return metrics
    
    def train(
        self,
        num_episodes: int,
        eval_interval: int = 100,
        max_patience: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Main training loop.
        
        Args:
            num_episodes: Number of episodes to train
            eval_interval: Evaluate every N episodes
            max_patience: Max episodes without improvement before stopping
            verbose: Print training progress
        
        Returns:
            Training summary dictionary
        """
        logger.info(f"Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            self.episode_num = episode
            
            # Collect episode
            metrics = self.collect_episode()
            self.episode_metrics.append(metrics)
            
            # Update learning rate
            self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_lr()
            
            # Update training history
            self.training_history["episode_num"].append(episode)
            self.training_history["total_reward"].append(metrics.total_reward)
            self.training_history["episode_length"].append(metrics.episode_length)
            self.training_history["avg_policy_loss"].append(metrics.avg_policy_loss)
            self.training_history["avg_value_loss"].append(metrics.avg_value_loss)
            
            # Check for improvement
            if metrics.total_reward > self.best_reward:
                self.best_reward = metrics.total_reward
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(tag="best")
                
                if verbose:
                    print(f"Episode {episode}: New best reward: {metrics.total_reward:.2f}")
            else:
                self.patience_counter += 1
            
            # Log progress
            if verbose and (episode + 1) % eval_interval == 0:
                avg_reward = np.mean([
                    m.total_reward for m in self.episode_metrics[-eval_interval:]
                ])
                avg_length = np.mean([
                    m.episode_length for m in self.episode_metrics[-eval_interval:]
                ])
                
                logger.info(
                    f"Episode {episode}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Length: {avg_length:.0f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Best: {self.best_reward:.2f}"
                )
            
            # Early stopping
            if self.patience_counter >= max_patience:
                logger.info(
                    f"Early stopping at episode {episode}. "
                    f"No improvement for {max_patience} episodes."
                )
                break
        
        # Save final model
        self.save_checkpoint(tag="final")
        
        # Create summary
        summary = {
            "total_episodes": self.episode_num + 1,
            "total_steps": self.total_steps,
            "best_reward": self.best_reward,
            "final_reward": self.episode_metrics[-1].total_reward,
            "training_time": datetime.now().isoformat()
        }
        
        logger.info(f"Training completed. Summary: {summary}")
        
        return summary
    
    def save_checkpoint(self, tag: str = "latest"):
        """
        Save training checkpoint.
        
        Args:
            tag: Checkpoint tag (e.g., "best", "final", "latest")
        """
        checkpoint = {
            "agent_state": {
                "policy_params": [
                    p.detach().cpu() for p in self.agent.policy_network.parameters()
                ],
                "value_params": [
                    p.detach().cpu() for p in self.agent.value_network.parameters()
                ]
            },
            "episode_num": self.episode_num,
            "total_steps": self.total_steps,
            "best_reward": self.best_reward,
            "metrics": self.training_history
        }
        
        path = self.save_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, tag: str = "best"):
        """
        Load training checkpoint.
        
        Args:
            tag: Checkpoint tag to load
        """
        path = self.save_dir / f"checkpoint_{tag}.pt"
        
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path)
        
        # Restore agent state
        for param, saved_param in zip(
            self.agent.policy_network.parameters(),
            checkpoint["agent_state"]["policy_params"]
        ):
            param.data = saved_param.to(self.agent.device)
        
        for param, saved_param in zip(
            self.agent.value_network.parameters(),
            checkpoint["agent_state"]["value_params"]
        ):
            param.data = saved_param.to(self.agent.device)
        
        # Restore training state
        self.episode_num = checkpoint["episode_num"]
        self.total_steps = checkpoint["total_steps"]
        self.best_reward = checkpoint["best_reward"]
        self.training_history = checkpoint["metrics"]
        
        logger.info(f"Checkpoint loaded: {path}")
    
    def get_metrics_summary(self) -> Dict:
        """Get summary statistics of training metrics."""
        if not self.episode_metrics:
            return {}
        
        rewards = [m.total_reward for m in self.episode_metrics]
        lengths = [m.episode_length for m in self.episode_metrics]
        
        return {
            "total_episodes": len(self.episode_metrics),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "mean_episode_length": np.mean(lengths),
            "total_steps": self.total_steps
        }
    
    def save_metrics(self, path: Optional[str] = None):
        """
        Save training metrics to JSON.
        
        Args:
            path: Save path (default: save_dir/metrics.json)
        """
        if path is None:
            path = self.save_dir / "metrics.json"
        
        metrics_dict = {
            "training_history": self.training_history,
            "summary": self.get_metrics_summary()
        }
        
        with open(path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Metrics saved to {path}")
