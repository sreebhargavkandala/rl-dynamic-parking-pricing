"""
Training script for Actor-Critic agent on parking pricing environment.

Implements complete training loop with:
- Episode management
- Metrics logging
- Model checkpointing
- Integration with Role 1 environment
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import os
from pathlib import Path

from .actor_critic import ActorCriticAgent


class Trainer:
    """Trainer class for Actor-Critic agent."""
    
    def __init__(
        self,
        env,
        agent: ActorCriticAgent,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10
    ) -> None:
        """Initialize trainer.
        
        Args:
            env: Gym-compatible environment
            agent: ActorCriticAgent instance
            checkpoint_dir: Directory to save checkpoints
            log_interval: Episodes between logging (default: 10)
        """
        self.env = env
        self.agent = agent
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_interval = log_interval
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.all_metrics = []

    def train_episode(self) -> Dict[str, float]:
        """Train for one episode.
        
        Returns:
            Dictionary with episode metrics
        """
        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_policy_losses = []
        episode_value_losses = []
        episode_entropies = []

        done = False
        while not done:
            # Select action
            action, log_prob = self.agent.select_action(state)
            
            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Update agent
            metrics = self.agent.update(
                state=state,
                action_log_prob=log_prob,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            # Accumulate metrics
            episode_policy_losses.append(metrics['policy_loss'])
            episode_value_losses.append(metrics['value_loss'])
            episode_entropies.append(metrics['entropy'])
            
            episode_reward += reward
            episode_length += 1
            state = next_state

        # Compute episode statistics
        episode_stats = {
            'reward': episode_reward,
            'length': episode_length,
            'avg_policy_loss': np.mean(episode_policy_losses),
            'avg_value_loss': np.mean(episode_value_losses),
            'avg_entropy': np.mean(episode_entropies),
        }
        
        # Add environment-specific metrics if available
        if hasattr(self.env, 'get_episode_metrics'):
            episode_stats.update(self.env.get_episode_metrics())
        
        return episode_stats

    def train(
        self,
        num_episodes: int,
        eval_episodes: int = 5,
        save_best: bool = True
    ) -> None:
        """Train agent for specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train
            eval_episodes: Episodes between evaluation (default: 5)
            save_best: Save best performing model (default: True)
        """
        best_reward = -np.inf
        
        for episode in range(1, num_episodes + 1):
            # Train episode
            episode_stats = self.train_episode()
            
            self.episode_rewards.append(episode_stats['reward'])
            self.episode_lengths.append(episode_stats['length'])
            self.all_metrics.append(episode_stats)
            
            # Logging
            if episode % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.log_interval:])
                
                print(f"\n{'='*60}")
                print(f"Episode {episode}/{num_episodes}")
                print(f"{'='*60}")
                print(f"Avg Reward (last {self.log_interval}): {avg_reward:.2f}")
                print(f"Avg Episode Length: {avg_length:.1f}")
                print(f"Avg Policy Loss: {episode_stats['avg_policy_loss']:.4f}")
                print(f"Avg Value Loss: {episode_stats['avg_value_loss']:.4f}")
                print(f"Avg Entropy: {episode_stats['avg_entropy']:.4f}")
                
                # Save checkpoint
                if save_best and avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save_checkpoint(f"best_model.pt")
                    print(f"✓ New best model saved (reward: {best_reward:.2f})")
                
                # Regular checkpoint
                if episode % (self.log_interval * 5) == 0:
                    self.save_checkpoint(f"checkpoint_ep{episode}.pt")

    def evaluate(
        self,
        num_episodes: int = 5,
        render: bool = False
    ) -> Dict[str, float]:
        """Evaluate agent without training.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
            
        Returns:
            Evaluation metrics
        """
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                # No gradient computation during evaluation
                with torch.no_grad():
                    action, _ = self.agent.select_action(state)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'max_reward': np.max(eval_rewards),
            'min_reward': np.min(eval_rewards),
        }

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        filepath = self.checkpoint_dir / filename
        self.agent.save(str(filepath))

    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        filepath = self.checkpoint_dir / filename
        self.agent.load(str(filepath))


def main():
    """Main training function.
    
    Example integration with Role 1 environment.
    """
    # Import environment from Role 1
    try:
        from env import ParkingPricingEnv
    except ImportError:
        print("Error: Could not import ParkingPricingEnv from Role 1")
        print("Make sure role_1 is in the Python path")
        return
    
    # Configuration
    CONFIG = {
        'num_episodes': 1000,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
    }
    
    print(f"Training on device: {CONFIG['device']}")
    
    # Set random seeds
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    # Create environment
    env = ParkingPricingEnv(seed=CONFIG['seed'])
    
    # Create agent
    agent = ActorCriticAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr=CONFIG['learning_rate'],
        gamma=CONFIG['gamma'],
        entropy_coef=CONFIG['entropy_coef'],
        device=CONFIG['device']
    )
    
    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        checkpoint_dir="./checkpoints",
        log_interval=10
    )
    
    # Train
    print("Starting training...")
    trainer.train(
        num_episodes=CONFIG['num_episodes'],
        eval_episodes=5,
        save_best=True
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluation Phase")
    print("="*60)
    eval_metrics = trainer.evaluate(num_episodes=10)
    print(f"Evaluation Results:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nTraining completed!")
    print(f"Best model saved to: {trainer.checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
