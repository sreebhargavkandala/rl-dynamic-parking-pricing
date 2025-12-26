"""
ADVANCED TRAINING SCRIPT - Simplified
====================================

Trains the advanced agent without baseline comparison.
Results are saved for later analysis.
"""

import argparse
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
try:
    from role_2.a2c_advanced import AdvancedA2CAgent, AdvancedA2CConfig
    from role_1.env_enhanced import EnhancedParkingEnvironment
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)


class SimpleAdvancedPipeline:
    """Simplified pipeline - training only."""
    
    def __init__(self, output_dir: str = "advanced_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.training_history = {
            'rewards': [],
            'prices': [],
            'occupancies': [],
            'revenues': [],
            'entropy': [],
            'value_loss': [],
            'policy_loss': []
        }
        
        logger.info(f"Pipeline initialized. Output: {self.output_dir}")
    
    def train(self, episodes: int = 150):
        """Train agent."""
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING ADVANCED AGENT FOR {episodes} EPISODES")
        logger.info("="*80)
        
        # Setup
        logger.info("\nSetting up environment...")
        env = EnhancedParkingEnvironment(
            capacity=100,
            max_steps=288,
            use_curriculum=True,
            use_randomization=True,
            use_advanced_features=True
        )
        
        logger.info("Setting up agent...")
        config = AdvancedA2CConfig(
            state_dim=7,
            action_dim=1,
            hidden_dim=512,
            num_hidden_layers=2,
            policy_lr=1e-4,
            value_lr=5e-4,
            entropy_coef=0.05,
            entropy_decay=0.9995,
            gae_lambda=0.95,
            use_dueling=True,
            use_residual=True,
            use_target_network=True,
            target_update_freq=5
        )
        agent = AdvancedA2CAgent(config)
        
        # Training loop
        logger.info("\n✓ Setup complete. Starting training...\n")
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_prices = []
            episode_occupancies = []
            
            # Run episode
            step_count = 0
            for step in range(env.max_steps):
                action, log_prob = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done, log_prob)
                
                episode_reward += reward
                episode_prices.append(info['price'])
                episode_occupancies.append(info['occupancy'])
                
                state = next_state
                step_count += 1
                
                if done:
                    break
            
            # Update agent
            update_metrics = agent.update()
            
            # Track metrics
            self.training_history['rewards'].append(episode_reward)
            self.training_history['prices'].append(np.mean(episode_prices))
            self.training_history['occupancies'].append(np.mean(episode_occupancies))
            self.training_history['revenues'].append(env.revenue)
            
            if update_metrics:
                self.training_history['entropy'].append(update_metrics.get('entropy', 0))
                self.training_history['value_loss'].append(update_metrics.get('value_loss', 0))
                self.training_history['policy_loss'].append(update_metrics.get('policy_loss', 0))
            
            # Progress logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_history['rewards'][-10:])
                avg_price = np.mean(self.training_history['prices'][-10:])
                avg_occ = np.mean(self.training_history['occupancies'][-10:])
                entropy = self.training_history['entropy'][-1] if self.training_history['entropy'] else 0
                
                logger.info(f"Episode {episode+1}/{episodes}")
                logger.info(f"  Reward (10-ep avg): {avg_reward:7.2f}")
                logger.info(f"  Price: ${avg_price:6.2f} (std: ${np.std(episode_prices):5.2f})")
                logger.info(f"  Occupancy: {avg_occ:6.1%}")
                logger.info(f"  Revenue: ${env.revenue:8.2f}")
                logger.info(f"  Entropy: {entropy:7.4f}")
                if hasattr(env, 'curriculum') and env.curriculum:
                    logger.info(f"  Curriculum: {env.curriculum.current_stage.name}")
                logger.info("")
        
        logger.info("✓ Training completed!")
        logger.info(f"  Final reward: {self.training_history['rewards'][-1]:.2f}")
        logger.info(f"  Best reward: {max(self.training_history['rewards']):.2f}")
        logger.info(f"  Avg reward (last 10): {np.mean(self.training_history['rewards'][-10:]):.2f}")
    
    def save_results(self):
        """Save results."""
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS")
        logger.info("="*80)
        
        # Metrics
        metrics_file = self.output_dir / "advanced_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"✓ Metrics saved to {metrics_file}")
        
        # Summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'training_episodes': len(self.training_history['rewards']),
            'final_reward': float(self.training_history['rewards'][-1]),
            'best_reward': float(max(self.training_history['rewards'])),
            'avg_reward_last_10': float(np.mean(self.training_history['rewards'][-10:])),
            'avg_price_final': float(np.mean(self.training_history['prices'][-10:])),
            'avg_occupancy_final': float(np.mean(self.training_history['occupancies'][-10:])),
            'avg_revenue_final': float(np.mean(self.training_history['revenues'][-10:])),
        }
        
        summary_file = self.output_dir / "advanced_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Advanced RL Training")
    parser.add_argument('--episodes', type=int, default=150, help='Training episodes')
    parser.add_argument('--output', type=str, default='advanced_results', help='Output dir')
    
    args = parser.parse_args()
    
    logger.info("\n" + "█"*80)
    logger.info("█ ADVANCED RL DYNAMIC PARKING PRICING - TRAINING")
    logger.info("█"*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("█"*80 + "\n")
    
    pipeline = SimpleAdvancedPipeline(output_dir=args.output)
    pipeline.train(episodes=args.episodes)
    pipeline.save_results()
    
    logger.info("\n" + "█"*80)
    logger.info("█ TRAINING COMPLETE ✓")
    logger.info("█"*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results saved to: {pipeline.output_dir}")
    logger.info("█"*80 + "\n")


if __name__ == "__main__":
    main()
