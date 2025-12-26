#!/usr/bin/env python3
"""
MASTER INTEGRATION: All Roles United
=====================================

Complete end-to-end pipeline connecting:
- ROLE 1: Environment & Metrics
- ROLE 2: RL Agents (A2C, PPO, SAC, DDPG)
- ROLE 3: Demand Modeling (placeholder - use Role 1 simulator)
- ROLE 4: Evaluation & Visualization

This script orchestrates the full workflow:
1. Setup environment (Role 1)
2. Train RL agent (Role 2)
3. Evaluate against baselines (Role 4)
4. Generate comparison visualizations (Role 4)

Usage:
    python integrate_all_roles.py [options]
    
Options:
    --episodes N        Number of training episodes (default: 100)
    --eval-episodes N   Number of evaluation episodes (default: 20)
    --agent {a2c,ppo}   Which agent to train (default: a2c)
    --generate-video    Create comparison videos
    --output DIR        Output directory for results
"""

import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse
import logging
from datetime import datetime
import json
import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# IMPORTS: All Roles
# ============================================================================

logger.info("Importing Role 1: Environment & Metrics...")
try:
    from role_1.env import ParkingPricingEnv
    from role_1.metrics import compute_all_metrics, ParkingMetrics
    from role_1.reward_function import RewardFunction
    logger.info("✓ Role 1 imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import Role 1: {e}")
    sys.exit(1)

logger.info("Importing Role 2: RL Agents...")
try:
    from role_2.a2c_new import A2CAgent, A2CConfig
    from role_2.ppo import PPOAgent
    logger.info("✓ Role 2 imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import Role 2: {e}")
    sys.exit(1)

logger.info("Importing Role 4: Evaluation & Visualization...")
try:
    from role_4.evaluation import (
        get_default_baselines,
        compare_strategies,
        generate_comparison_table,
        save_results,
        FixedPriceStrategy,
        TimeBasedStrategy,
    )
    from role_4.evaluation.visualise import (
        plot_training_progress,
        plot_revenue_comparison,
        plot_occupancy_comparison,
        plot_price_volatility,
        create_summary_dashboard,
    )
    from role_4.evaluation.video_recording import (
        collect_episode_data,
        create_comparison_static,
        create_comparison_animation,
    )
    logger.info("✓ Role 4 imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import Role 4: {e}")
    sys.exit(1)

# ============================================================================
# INTEGRATION PIPELINE
# ============================================================================

class RolePipeline:
    """Unified pipeline for all roles."""
    
    def __init__(self, output_dir: Path = None, seed: int = 42):
        """Initialize pipeline."""
        self.seed = seed
        self.output_dir = output_dir or Path("./integration_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Will be populated during pipeline
        self.env = None
        self.agent = None
        self.training_metrics = {}
        self.evaluation_results = {}
        
        logger.info(f"Pipeline initialized. Output: {self.output_dir}")
    
    def setup_environment(self, capacity: int = 100, max_steps: int = 288) -> ParkingPricingEnv:
        """STEP 1: Setup Role 1 Environment."""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Setting up Environment (Role 1)")
        logger.info("="*80)
        
        self.env = ParkingPricingEnv(
            capacity=capacity,
            max_steps=max_steps,
            target_occupancy=0.8,
            min_price=0.5,
            max_price=20.0,
            seed=self.seed
        )
        logger.info(f"✓ Environment created: {capacity} spots, max {max_steps} steps")
        logger.info(f"  State space: 5-dim (occupancy, time, demand, price, revenue)")
        logger.info(f"  Action space: 1-dim (price ∈ [{self.env.min_price}, {self.env.max_price}])")
        
        return self.env
    
    def train_agent(self, agent_type: str = "a2c", num_episodes: int = 100) -> None:
        """STEP 2: Train Role 2 Agent."""
        logger.info("\n" + "="*80)
        logger.info(f"STEP 2: Training {agent_type.upper()} Agent (Role 2)")
        logger.info("="*80)
        
        if agent_type.lower() == "a2c":
            config = A2CConfig(
                state_dim=5,
                action_dim=1,
                hidden_dim=512,  # Increased from 256 for better capacity
                policy_lr=1e-4,  # Reduced from 3e-4 for stability
                value_lr=5e-4,   # Reduced from 1e-3 for stability
                gamma=0.99,
                entropy_coef=0.05,  # Increased from 0.01 for more exploration
                value_loss_coef=0.5,
                max_grad_norm=0.5,
                l2_reg=1e-5,
                device='cpu'
            )
            self.agent = A2CAgent(config)
            logger.info(f"✓ A2C Agent (Enhanced) created:")
            logger.info(f"  Hidden dim: {config.hidden_dim} (increased for capacity)")
            logger.info(f"  Learning rate (policy): {config.policy_lr} (reduced for stability)")
            logger.info(f"  Learning rate (value): {config.value_lr}")
            logger.info(f"  Entropy coef: {config.entropy_coef} (increased for exploration)")
            logger.info(f"  This configuration encourages dynamic pricing by:")
        else:
            raise ValueError(f"Agent type {agent_type} not supported")
        
        # Training loop
        episode_rewards = []
        logger.info(f"\nTraining for {num_episodes} episodes...")
        logger.info(f"Goal: Learn dynamic pricing that responds to occupancy and demand\n")
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset(seed=self.seed + episode)
            episode_reward = 0.0
            done = False
            
            # Track pricing dynamics
            prices_in_episode = []
            occupancies_in_episode = []
            
            # Collect trajectory data
            states, actions, rewards, values, log_probs, dones, next_values = [], [], [], [], [], [], []
            
            while not done:
                # Get current occupancy and demand from state
                occupancy = obs[0]
                time_of_day = obs[1]
                demand_level = obs[2]
                
                occupancies_in_episode.append(occupancy)
                
                # Select action
                action, log_prob, value = self.agent.select_action(obs, training=True)
                
                # Ensure action is an array
                if not isinstance(action, np.ndarray):
                    action = np.array([action])
                
                # Scale from [-1, 1] to [min_price, max_price]
                action_scaled = self.env.min_price + (action[0] + 1) / 2 * (self.env.max_price - self.env.min_price)
                prices_in_episode.append(action_scaled)
                
                # Step environment with action as array
                obs_next, reward, terminated, truncated, info = self.env.step(np.array([action_scaled]))
                done = terminated or truncated
                
                # Get next value estimate for advantage calculation
                if not done:
                    _, _, next_value = self.agent.select_action(obs_next, training=False)
                else:
                    next_value = 0.0
                
                # Store trajectory data
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)
                next_values.append(next_value)
                
                episode_reward += reward
                obs = obs_next
            
            # Update agent with trajectory
            if len(states) > 0:
                self.agent.update(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    values=values,
                    log_probs=log_probs,
                    dones=dones,
                    next_values=next_values
                )
            
            episode_rewards.append(episode_reward)
            
            # Check if agent is learning dynamic pricing
            if len(prices_in_episode) > 1:
                price_std = np.std(prices_in_episode)
                avg_price = np.mean(prices_in_episode)
                occ_range = max(occupancies_in_episode) - min(occupancies_in_episode)
            else:
                price_std = 0.0
                avg_price = prices_in_episode[0] if prices_in_episode else 0
                occ_range = 0.0
            
            if (episode + 1) % 5 == 0:
                avg_reward = sum(episode_rewards[-5:]) / 5
                logger.info(f"  Episode {episode+1}/{num_episodes}, Reward: {avg_reward:.2f}, " +
                           f"Price: ${avg_price:.2f} (std=${price_std:.2f}), " +
                           f"Occ range: {occ_range:.1%}")
        
        self.training_metrics['episode_rewards'] = episode_rewards
        final_avg = sum(episode_rewards[-5:]) / 5 if len(episode_rewards) >= 5 else sum(episode_rewards) / len(episode_rewards)
        logger.info(f"✓ Training complete. Final avg reward: {final_avg:.2f}")
    
    def evaluate_agent(self, num_episodes: int = 20) -> Dict:
        """STEP 3: Evaluate Agent vs Baselines (Role 4)."""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Evaluating Agent (Role 4)")
        logger.info("="*80)
        
        if self.agent is None:
            logger.warning("No trained agent found. Skipping evaluation.")
            return {}
        
        baselines = get_default_baselines()
        logger.info(f"✓ Loaded {len(baselines)} baseline strategies:")
        for baseline in baselines:
            logger.info(f"  - {baseline.name}")
        
        logger.info(f"\nRunning comparison for {num_episodes} episodes...")
        results = compare_strategies(
            env=self.env,
            strategies=baselines,
            rl_agent=self.agent,
            num_episodes=num_episodes,
            seed=self.seed,
            verbose=False
        )
        
        self.evaluation_results = results
        logger.info("✓ Evaluation complete")
        
        # Print comparison table
        table = generate_comparison_table(results)
        logger.info("\nComparison Results:\n" + table)
        
        return results
    
    def generate_visualizations(self, generate_videos: bool = False) -> None:
        """STEP 4: Generate Visualizations (Role 4)."""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Generating Visualizations (Role 4)")
        logger.info("="*80)
        
        if not self.evaluation_results:
            logger.warning("No evaluation results. Skipping visualizations.")
            return
        
        # Save results
        logger.info("Saving results...")
        save_results(self.evaluation_results, self.output_dir)
        logger.info(f"✓ Results saved to {self.output_dir}")
        
        # Generate plots
        logger.info("Generating comparison plots...")
        
        try:
            plot_revenue_comparison(
                self.evaluation_results,
                output_path=str(self.output_dir / "revenue_comparison.png"),
                show=False
            )
            logger.info("✓ Revenue comparison plot")
        except Exception as e:
            logger.warning(f"Could not generate revenue plot: {e}")
        
        try:
            plot_occupancy_comparison(
                self.evaluation_results,
                output_path=str(self.output_dir / "occupancy_comparison.png"),
                show=False
            )
            logger.info("✓ Occupancy comparison plot")
        except Exception as e:
            logger.warning(f"Could not generate occupancy plot: {e}")
        
        try:
            plot_price_volatility(
                self.evaluation_results,
                output_path=str(self.output_dir / "volatility_comparison.png"),
                show=False
            )
            logger.info("✓ Price volatility plot")
        except Exception as e:
            logger.warning(f"Could not generate volatility plot: {e}")
        
        # Generate comparison video/image
        if generate_videos and self.agent:
            logger.info("Generating before/after comparison...")
            try:
                baseline = FixedPriceStrategy(price=5.0)
                baseline_data = collect_episode_data(self.env, baseline, is_rl_agent=False)
                rl_data = collect_episode_data(self.env, self.agent, is_rl_agent=True)
                
                create_comparison_static(
                    baseline_data=baseline_data[:2],
                    rl_data=rl_data[:2],
                    baseline_name="Fixed Price ($5)",
                    output_path=str(self.output_dir / "baseline_vs_rl.png"),
                    show=False
                )
                logger.info("✓ Comparison image created")
            except Exception as e:
                logger.warning(f"Could not generate comparison image: {e}")
    
    def run_full_pipeline(self, num_train_episodes: int = 100, num_eval_episodes: int = 20,
                          agent_type: str = "a2c", generate_videos: bool = False) -> None:
        """Run complete pipeline."""
        logger.info("\n" + "█"*80)
        logger.info("█ MASTER INTEGRATION: ALL ROLES UNITED")
        logger.info("█"*80)
        logger.info(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info("█"*80 + "\n")
        
        try:
            # Run all steps
            self.setup_environment()
            self.train_agent(agent_type=agent_type, num_episodes=num_train_episodes)
            self.evaluate_agent(num_episodes=num_eval_episodes)
            self.generate_visualizations(generate_videos=generate_videos)
            
            logger.info("\n" + "█"*80)
            logger.info("█ PIPELINE COMPLETE ✓")
            logger.info("█"*80)
            logger.info(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"  Results saved to: {self.output_dir}")
            logger.info("█"*80 + "\n")
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}")
            raise


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Master integration script for all roles"
    )
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--agent", choices=["a2c", "ppo"], default="a2c", help="Agent type")
    parser.add_argument("--generate-video", action="store_true", help="Generate videos")
    parser.add_argument("--output", type=str, default="./integration_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = RolePipeline(output_dir=Path(args.output))
    pipeline.run_full_pipeline(
        num_train_episodes=args.episodes,
        num_eval_episodes=args.eval_episodes,
        agent_type=args.agent,
        generate_videos=args.generate_video
    )


if __name__ == "__main__":
    main()
