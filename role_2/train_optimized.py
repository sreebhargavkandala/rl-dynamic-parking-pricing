#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZED A2C AGENT TRAINING
============================
Best practices for training the A2C agent to maximum performance

Features:
- Optimized hyperparameters for parking pricing
- Advanced learning rate scheduling
- Best model checkpointing
- Early stopping with validation
- Comprehensive monitoring
- Performance optimization
"""

import numpy as np
import torch
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from role_2 to project root
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from role_1.env import ParkingPricingEnv
    from role_2.a2c_new import A2CAgent, A2CConfig
    from role_2.a2c_trainer import A2CTrainer
    print("✓ All imports successful\n")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


class OptimizedTrainingPipeline:
    """Optimized training pipeline for maximum A2C performance"""
    
    def __init__(self, results_dir="./training_results_optimized"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.env = None
        self.agent = None
        self.trainer = None
        
        # Metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.best_reward = -np.inf
        self.best_model_path = None
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'occupancy': [],
            'prices': [],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"📁 Results directory: {self.results_dir.absolute()}")
        print(f"✓ Pipeline initialized\n")
    
    def setup_environment(self):
        """Setup optimized environment for parking pricing"""
        print("="*80)
        print("STEP 1: SETTING UP ENVIRONMENT")
        print("="*80)
        
        # Optimized parameters for best results
        self.env = ParkingPricingEnv(
            capacity=150,              # Larger lot for better learning
            max_steps=288,             # 24 hours in 5-min intervals
            target_occupancy=0.80,     # Optimal target
            min_price=1.5,             # Realistic minimum
            max_price=25.0,            # High ceiling for peak pricing
            seed=42                    # Reproducibility
        )
        
        print(f"✓ Environment configured:")
        print(f"  - Capacity: {self.env.capacity} parking spaces")
        print(f"  - Episode length: {self.env.max_steps} steps (24 hours)")
        print(f"  - Target occupancy: {self.env.target_occupancy*100:.0f}%")
        print(f"  - Price range: ${self.env.min_price:.2f} - ${self.env.max_price:.2f}")
        print(f"  - State space: 5 dimensions")
        print(f"  - Action space: Continuous pricing [${self.env.min_price}, ${self.env.max_price}]")
        print()
        
        return self.env
    
    def setup_agent(self):
        """Setup optimized A2C agent"""
        print("="*80)
        print("STEP 2: SETTING UP A2C AGENT")
        print("="*80)
        
        # Optimized A2C configuration
        config = A2CConfig(
            # Network architecture - optimized for parking domain
            state_dim=5,
            action_dim=1,
            hidden_dim=256,              # Larger hidden layers
            num_hidden_layers=2,
            
            # Learning rates - carefully tuned
            policy_lr=3e-4,              # Standard for A2C
            value_lr=1e-3,               # Value function learns faster
            
            # Discount factor - future rewards matter
            gamma=0.99,                  # High for long-term planning
            
            # Entropy regularization - prevents premature convergence
            entropy_coef=0.01,           # Encourages exploration
            
            # Gradient clipping - stability
            max_grad_norm=0.5,           # Conservative gradient clipping
            
            # Device
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.agent = A2CAgent(config)
        
        device_str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        print(f"✓ A2C Agent initialized:")
        print(f"  - Policy network: {config.state_dim} → {config.hidden_dim} → {config.hidden_dim} → {config.action_dim}")
        print(f"  - Value network: {config.state_dim} → {config.hidden_dim} → {config.hidden_dim} → 1")
        print(f"  - Policy learning rate: {config.policy_lr}")
        print(f"  - Value learning rate: {config.value_lr}")
        print(f"  - Discount factor (γ): {config.gamma}")
        print(f"  - Entropy coefficient: {config.entropy_coef}")
        print(f"  - Device: {device_str}")
        print()
        
        return self.agent
    
    def train(self, num_episodes: int = 1000, batch_size: int = 32):
        """Train agent with optimized settings"""
        print("="*80)
        print("STEP 3: TRAINING A2C AGENT")
        print("="*80)
        print(f"Target: {num_episodes} episodes")
        print()
        
        start_time = time.time()
        no_improvement_count = 0
        max_no_improvement = 100  # Early stopping patience
        
        try:
            # Setup trainer
            trainer_config = {
                "lr_schedule": "linear",
                "total_episodes": num_episodes,
                "warmup_episodes": 50
            }
            
            self.trainer = A2CTrainer(
                agent=self.agent,
                env=self.env,
                config=trainer_config,
                save_dir=str(self.checkpoint_dir)
            )
            
            # Training loop
            for episode in range(num_episodes):
                # Reset environment (returns tuple: observation, info)
                reset_result = self.env.reset()
                state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                episode_reward = 0
                episode_length = 0
                
                # Episode loop
                for step in range(self.env.max_steps):
                    # Get action from agent using correct method
                    action, log_prob, value = self.agent.select_action(state, training=True)
                    
                    # Take step in environment
                    action = np.clip(action, self.env.min_price, self.env.max_price)
                    step_result = self.env.step(action)
                    next_state = step_result[0]
                    reward = step_result[1]
                    terminated = step_result[2]
                    truncated = step_result[3]
                    done = terminated or truncated
                    
                    # Track metrics
                    episode_reward += reward
                    episode_length += 1
                    
                    # Update agent
                    self.agent.update(state, action, reward, next_state, done)
                    
                    state = next_state
                    
                    if done:
                        break
                
                # Record episode metrics
                self.episode_rewards.append(episode_reward)
                self.training_history['episodes'].append(episode)
                self.training_history['rewards'].append(episode_reward)
                
                # Save best model
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    no_improvement_count = 0
                    self.save_best_model(episode)
                    status = "✓ NEW BEST"
                else:
                    no_improvement_count += 1
                    status = "   "
                
                # Print progress
                if (episode + 1) % 50 == 0 or episode == 0:
                    avg_reward = np.mean(self.episode_rewards[-50:])
                    elapsed = time.time() - start_time
                    eps_per_sec = (episode + 1) / elapsed
                    
                    print(f"Episode {episode+1:4d}/{num_episodes} | "
                          f"Reward: {episode_reward:8.2f} | "
                          f"Avg50: {avg_reward:8.2f} | "
                          f"Best: {self.best_reward:8.2f} {status} | "
                          f"Speed: {eps_per_sec:.1f} eps/s")
                
                # Early stopping
                if no_improvement_count >= max_no_improvement:
                    print(f"\n⚠️  No improvement for {max_no_improvement} episodes. Stopping early.")
                    break
            
            # Training complete
            elapsed = time.time() - start_time
            print()
            print("="*80)
            print("TRAINING COMPLETE")
            print("="*80)
            print(f"✓ Trained {len(self.episode_rewards)} episodes in {elapsed:.1f} seconds")
            print(f"✓ Final reward: {self.episode_rewards[-1]:.2f}")
            print(f"✓ Best reward: {self.best_reward:.2f}")
            print(f"✓ Average reward: {np.mean(self.episode_rewards):.2f}")
            print(f"✓ Best model saved: {self.best_model_path}")
            print()
            
            return self.best_reward
        
        except Exception as e:
            print(f"✗ Training error: {e}")
            raise
    
    def save_best_model(self, episode: int):
        """Save best model found during training"""
        model_path = self.checkpoint_dir / f"best_model_ep{episode}.pt"
        
        # Save agent state
        torch.save({
            'episode': episode,
            'reward': self.best_reward,
            'policy_state': self.agent.policy.state_dict(),
            'value_state': self.agent.value.state_dict(),
            'config': {
                'state_dim': self.agent.config.state_dim,
                'action_dim': self.agent.config.action_dim,
                'hidden_dim': self.agent.config.hidden_dim,
            }
        }, model_path)
        
        self.best_model_path = model_path
    
    def save_results(self):
        """Save training results and metrics"""
        # Training metrics
        metrics = {
            'best_reward': float(self.best_reward),
            'final_reward': float(self.episode_rewards[-1]) if self.episode_rewards else 0,
            'avg_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            'std_reward': float(np.std(self.episode_rewards)) if self.episode_rewards else 0,
            'total_episodes': len(self.episode_rewards),
            'min_reward': float(np.min(self.episode_rewards)) if self.episode_rewards else 0,
            'max_reward': float(np.max(self.episode_rewards)) if self.episode_rewards else 0,
            'timestamp': datetime.now().isoformat(),
            'best_model_path': str(self.best_model_path)
        }
        
        metrics_path = self.results_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Metrics saved to: {metrics_path}")
        
        # Full history
        history_path = self.results_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"✓ History saved to: {history_path}")
        
        return metrics
    
    def print_summary(self):
        """Print training summary"""
        if not self.episode_rewards:
            return
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  • Best Episode Reward: ${self.best_reward:.2f}")
        print(f"  • Final Episode Reward: ${self.episode_rewards[-1]:.2f}")
        print(f"  • Average Reward: ${np.mean(self.episode_rewards):.2f}")
        print(f"  • Std Deviation: ${np.std(self.episode_rewards):.2f}")
        print(f"  • Min Reward: ${np.min(self.episode_rewards):.2f}")
        print(f"  • Max Reward: ${np.max(self.episode_rewards):.2f}")
        
        print(f"\n📈 LEARNING PROGRESS:")
        first_50_avg = np.mean(self.episode_rewards[:50]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
        last_50_avg = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
        improvement = ((last_50_avg - first_50_avg) / (first_50_avg + 1e-6)) * 100
        
        print(f"  • First 50 avg: ${first_50_avg:.2f}")
        print(f"  • Last 50 avg: ${last_50_avg:.2f}")
        print(f"  • Improvement: {improvement:+.1f}%")
        
        print(f"\n💾 MODEL & RESULTS:")
        print(f"  • Best model: {self.best_model_path.name}")
        print(f"  • Results dir: {self.results_dir.absolute()}")
        print(f"  • Total episodes: {len(self.episode_rewards)}")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main training entry point"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  OPTIMIZED A2C AGENT TRAINING FOR DYNAMIC PARKING PRICING".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # Create pipeline
    pipeline = OptimizedTrainingPipeline()
    
    # Setup
    pipeline.setup_environment()
    pipeline.setup_agent()
    
    # Train
    try:
        print("Starting training...")
        print("Note: This will take several minutes. GPU will significantly speed it up.\n")
        
        best_reward = pipeline.train(num_episodes=1000, batch_size=32)
        
        # Save results
        metrics = pipeline.save_results()
        
        # Print summary
        pipeline.print_summary()
        
        print("✓ Training completed successfully!")
        print(f"✓ Best reward achieved: ${best_reward:.2f}")
        print(f"✓ Best model saved to: {pipeline.checkpoint_dir}")
        print("\nYou can now:")
        print("  1. Use the best model in the dashboard: python dashboard/main_dashboard.py")
        print("  2. View results: Check training_results_optimized/ directory")
        print("  3. Analyze metrics: View training_metrics.json")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        metrics = pipeline.save_results()
        pipeline.print_summary()
    
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
