#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Training Script - Train RL Agent for Parking Pricing

This script trains the A2C agent from scratch with:
- Environment setup
- Agent initialization
- Complete training loop (500 episodes)
- Performance tracking
- Model checkpointing
- Results visualization
"""

import numpy as np
import torch
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from role_1.env import ParkingPricingEnv
    from role_2.a2c_new import A2CAgent, A2CConfig
    from role_2.a2c_trainer import A2CTrainer
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Make sure all role_1, role_2, etc. files are in place")
    sys.exit(1)


class TrainingOrchestrator:
    """Orchestrates complete RL training pipeline."""
    
    def __init__(self, results_dir="./training_results"):
        """Initialize training orchestrator."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.env = None
        self.agent = None
        self.trainer = None
        
        # Tracking
        self.episode_rewards = []
        self.episode_losses = []
        self.best_reward = -np.inf
        
        print(f"\nResults directory: {self.results_dir.absolute()}")
        print(f"✓ Orchestrator initialized\n")
    
    def setup_environment(self) -> ParkingPricingEnv:
        """Setup parking pricing environment."""
        print("="*80)
        print("STEP 1: SETTING UP ENVIRONMENT")
        print("="*80)
        
        try:
            # Simple demand model (will be created by env)
            self.env = ParkingPricingEnv(
                capacity=100,
                max_steps=288,  # 24 hours * 12 (5-min intervals)
                target_occupancy=0.80,
                min_price=0.5,
                max_price=20.0,
                seed=42
            )
            
            print(f"✓ Environment created:")
            print(f"  - Capacity: {self.env.capacity} spaces")
            print(f"  - Max steps: {self.env.max_steps}")
            print(f"  - Target occupancy: {self.env.target_occupancy*100:.1f}%")
            print(f"  - Price range: ${self.env.min_price} - ${self.env.max_price}")
            print(f"  - State dim: 5")
            print(f"  - Action dim: 1 (continuous price)")
            
            return self.env
        
        except Exception as e:
            print(f" Error setting up environment: {e}")
            raise
    
    def setup_agent(self) -> A2CAgent:
        """Setup A2C agent."""
        print("\n" + "="*80)
        print("STEP 2: SETTING UP A2C AGENT")
        print("="*80)
        
        try:
            config = A2CConfig(
                state_dim=5,
                action_dim=1,
                hidden_dim=256,
                policy_lr=3e-4,
                value_lr=1e-3,
                gamma=0.99,
                entropy_coef=0.01,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            self.agent = A2CAgent(config)
            
            device = config.device
            print(f"✓ A2C Agent created:")
            print(f"  - State dimension: {config.state_dim}")
            print(f"  - Action dimension: {config.action_dim}")
            print(f"  - Hidden layer size: {config.hidden_dim}")
            print(f"  - Policy learning rate: {config.policy_lr}")
            print(f"  - Value learning rate: {config.value_lr}")
            print(f"  - Discount factor (γ): {config.gamma}")
            print(f"  - Entropy coefficient: {config.entropy_coef}")
            print(f"  - Device: {device.upper()}")
            
            return self.agent
        
        except Exception as e:
            print(f"Error setting up agent: {e}")
            raise
    
    def train_agent(self, num_episodes: int = 500, eval_interval: int = 50):
        """Train the A2C agent."""
        print("\n" + "="*80)
        print("STEP 3: TRAINING A2C AGENT")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Create config for trainer
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
            
            print(f"\nStarting training for {num_episodes} episodes...\n")
            print(f"{'Episode':<10} {'Reward':<15} {'Avg Reward':<15} {'Loss':<15} {'Time':<10}")
            print("-" * 65)
            
            episode_rewards = []
            episode_losses = []
            
            for episode in range(1, num_episodes + 1):
                # Collect episode using trainer's method
                metrics = self.trainer.collect_episode()
                
                episode_reward = metrics.total_reward
                episode_loss = metrics.avg_policy_loss + metrics.avg_value_loss
                
                episode_rewards.append(episode_reward)
                episode_losses.append(episode_loss)
                
                # Calculate metrics
                avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) > 0 else 0
                elapsed_time = time.time() - start_time
                
                # Track best reward
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self._save_checkpoint(episode)
                
                # Print progress
                if episode % eval_interval == 0 or episode == 1:
                    print(f"{episode:<10} {episode_reward:<15.2f} {avg_reward:<15.2f} "
                          f"{episode_loss:<15.4f} {elapsed_time:<10.1f}s")
            
            # Training complete
            print("\n" + "-" * 65)
            print(f"✓ Training completed!")
            print(f"  - Total episodes: {num_episodes}")
            print(f"  - Best reward: {self.best_reward:.2f}")
            print(f"  - Final avg reward (last 50): {np.mean(episode_rewards[-50:]):.2f}")
            print(f"  - Total time: {(time.time() - start_time)/60:.1f} minutes")
            
            self.episode_rewards = episode_rewards
            self.episode_losses = episode_losses
            
            return {
                'final_reward': episode_rewards[-1],
                'best_reward': self.best_reward,
                'avg_reward': np.mean(episode_rewards[-50:]),
                'total_episodes': num_episodes,
                'training_time': time.time() - start_time
            }
        
        except Exception as e:
            print(f" Training error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def evaluate_agent(self, num_episodes: int = 20):
        """Evaluate trained agent."""
        print("\n" + "="*80)
        print("STEP 4: EVALUATING AGENT")
        print("="*80)
        
        eval_rewards = []
        
        print(f"\n running evaluation for {num_episodes} episodes...\n")
        
        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _, _ = self.agent.select_action(
                    torch.FloatTensor(state).unsqueeze(0).to(self.agent.device),
                    training=False
                )
                
                # Handle both tensor and numpy array returns
                if isinstance(action, torch.Tensor):
                    action_value = action.cpu().detach().numpy()[0]
                else:
                    action_value = action[0] if isinstance(action, np.ndarray) else action
                
                state, reward, done, truncated, _ = self.env.step(action_value)
                episode_reward += reward
                done = done or truncated
            
            eval_rewards.append(episode_reward)
            
            if ep % 5 == 0 or ep == 1:
                avg = np.mean(eval_rewards)
                print(f"  Episode {ep}: Reward = {episode_reward:.2f}, Avg = {avg:.2f}")
        
        final_avg = np.mean(eval_rewards)
        final_std = np.std(eval_rewards)
        
        print(f"\n✓ Evaluation Results:")
        print(f"  - Average reward: {final_avg:.2f}")
        print(f"  - Std deviation: {final_std:.2f}")
        print(f"  - Min reward: {np.min(eval_rewards):.2f}")
        print(f"  - Max reward: {np.max(eval_rewards):.2f}")
        
        return {
            'avg_reward': final_avg,
            'std_reward': final_std,
            'rewards': eval_rewards
        }
    
        def save_results(self):
            """Save training results and metrics."""
            print("\n" + "="*80)
            print("STEP 5: SAVING RESULTS")
            print("="*80)
            
            try:
                # Save metrics
                metrics = {
                    'episode_rewards': self.episode_rewards,
                    'episode_losses': self.episode_losses,
                    'best_reward': float(self.best_reward),
                    'final_avg_reward': float(np.mean(self.episode_rewards[-50:])),
                }
                
                metrics_file = self.results_dir / "training_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                print(f"✓ Metrics saved to: {metrics_file}")
                
                # Create summary report
                report = f"""
    TRAINING SUMMARY
    ================
    Best Reward: {self.best_reward:.2f}
    Final Avg Reward: {np.mean(self.episode_rewards[-50:]):.2f}
                """
                
                report_file = self.results_dir / "TRAINING_REPORT.txt"
                with open(report_file, 'w') as f:
                    f.write(report)
                
                print(f"✓ Report saved to: {report_file}")
                
                print(report)
            
            except Exception as e:
                print(f"Warning: Could not save results - {e}")
    
    def _save_checkpoint(self, episode: int):
        
        try:
            checkpoint_file = self.checkpoint_dir / f"agent_episode_{episode}.pt"
            torch.save(self.agent.state_dict(), checkpoint_file)
        except Exception as e:
            print(f"Warning: Could not save checkpoint - {e}")
    
    def run_complete_pipeline(self, num_episodes: int = 500):
     
        print("\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*78 + "║")
        print("║" + "    COMPLETE RL TRAINING PIPELINE - PARKING PRICING ".center(78) + "║")
        print("║" + " "*78 + "║")
        print("╚" + "="*78 + "╝")
        
        try:
            # Step 1: Setup environment
            self.setup_environment()
            
            # Step 2: Setup agent
            self.setup_agent()
            
            # Step 3: Train agent
            self.train_agent(num_episodes=num_episodes)
            
            # Step 4: Evaluate agent
            self.evaluate_agent(num_episodes=20)
            
            # Step 5: Save results
            self.save_results()
            
            print("\n" + "="*80)
            print("  PIPELINE EXECUTION COMPLETE! ")
            print("="*80)
            print(f"\nAll results saved to: {self.results_dir.absolute()}\n")
            
        except Exception as e:
            print(f"\n  Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    
    print(torch.__version__)
    
    orchestrator = TrainingOrchestrator(results_dir="./training_results")
    orchestrator.run_complete_pipeline(num_episodes=500)


if __name__ == "__main__":
    main()
