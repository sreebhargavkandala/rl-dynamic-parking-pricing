 

import argparse
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
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
    from role_4.evaluation.baselines import (
        FixedPriceStrategy,
        TimeBasedStrategy,
        RandomPriceStrategy,
        DemandBasedStrategy
    )
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)


class AdvancedRolePipeline:
    """
    Advanced integration pipeline with all features.
    """
    
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
        
        self.evaluation_results = {}
        
        logger.info(f"Advanced Pipeline initialized. Output: {self.output_dir}")
    
    def setup_environment(self) -> EnhancedParkingEnvironment:
        """Create enhanced environment with all features."""
        logger.info("\n" + "="*80)
        logger.info("SETTING UP ENHANCED ENVIRONMENT")
        logger.info("="*80)
        
        env = EnhancedParkingEnvironment(
            capacity=100,
            max_steps=288,
            use_curriculum=True,
            use_randomization=True,
            use_advanced_features=True
        )
        
        logger.info("✓ Environment created with:")
        logger.info("  - Curriculum learning")
        logger.info("  - Domain randomization")
        logger.info("  - Weather simulator")
        logger.info("  - Event simulator")
        logger.info("  - Competitor simulator")
        
        return env
    
    def setup_agent(self) -> AdvancedA2CAgent:
        """Create advanced A2C agent."""
        logger.info("\n" + "="*80)
        logger.info("SETTING UP ADVANCED A2C AGENT")
        logger.info("="*80)
        
        config = AdvancedA2CConfig(
            state_dim=7,  # Extended state with trends
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
        
        logger.info("✓ Agent created with:")
        logger.info("  - Dueling networks")
        logger.info("  - Residual connections")
        logger.info("  - Target network")
        logger.info("  - GAE (Generalized Advantage Estimation)")
        logger.info(f"  - Initial entropy coef: {config.entropy_coef}")
        
        return agent
    
    def train(self, agent: AdvancedA2CAgent, env: EnhancedParkingEnvironment,
              episodes: int = 150):
        """
        Train agent with advanced features.
        
        Args:
            agent: Advanced A2C agent
            env: Enhanced environment
            episodes: Number of training episodes
        """
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING AGENT FOR {episodes} EPISODES")
        logger.info("="*80)
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_prices = []
            episode_occupancies = []
            
            # Run episode
            for step in range(env.max_steps):
                action, log_prob = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done, log_prob)
                
                episode_reward += reward
                episode_prices.append(info['price'])
                episode_occupancies.append(info['occupancy'])
                
                state = next_state
                
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
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_history['rewards'][-10:])
                avg_price = np.mean(self.training_history['prices'][-10:])
                avg_occ = np.mean(self.training_history['occupancies'][-10:])
                entropy = self.training_history['entropy'][-1] if self.training_history['entropy'] else 0
                
                logger.info(f"Episode {episode+1}/{episodes}")
                logger.info(f"  Avg Reward (10 ep): {avg_reward:.2f}")
                logger.info(f"  Avg Price: ${avg_price:.2f} (std: ${np.std(episode_prices):.2f})")
                logger.info(f"  Avg Occupancy: {avg_occ:.2%}")
                logger.info(f"  Entropy: {entropy:.4f}")
                if env.use_curriculum:
                    logger.info(f"  Curriculum Stage: {env.curriculum.current_stage.name}")
        
        logger.info(f"✓ Training completed. Final reward: {self.training_history['rewards'][-1]:.2f}")
    
    def evaluate(self, agent: AdvancedA2CAgent, env: EnhancedParkingEnvironment,
                 episodes: int = 15):
        """
        Evaluate agent against baselines.
        
        Args:
            agent: Trained agent
            env: Environment
            episodes: Number of evaluation episodes
        """
        logger.info("\n" + "="*80)
        logger.info(f"EVALUATING AGENT AGAINST BASELINES")
        logger.info("="*80)
        
        # Create baselines
        baselines = {
            'Fixed $5.00': FixedPriceStrategy(price=5.0),
            'Fixed $10.00': FixedPriceStrategy(price=10.0),
            'Time-Based': TimeBasedStrategy(peak_price=10.0, offpeak_price=3.0),
            'Random Policy': RandomPriceStrategy(),
            'Demand-Based': DemandBasedStrategy(),
            'Advanced RL Agent': None  # Our agent
        }
        
        results = {}
        
        for strategy_name, baseline in baselines.items():
            logger.info(f"\nEvaluating: {strategy_name}")
            
            revenues = []
            occupancies = []
            prices = []
            volatilities = []
            
            for episode in range(episodes):
                state = env.reset()
                episode_revenue = 0.0
                episode_prices = []
                episode_occupancies = []
                
                for step in range(env.max_steps):
                    # Get action from strategy
                    if strategy_name == 'Advanced RL Agent':
                        action, _ = agent.select_action(state, training=False)
                        price = action[0] if isinstance(action, np.ndarray) else action
                    else:
                        # Baselines expect observation and env
                        price = baseline.get_price(state, env)
                    
                    # Step with price (convert to action format expected by env)
                    action = np.array([price])
                    next_state, reward, done, info = env.step(action)
                    
                    episode_prices.append(info['price'])
                    episode_occupancies.append(info['occupancy'])
                    episode_revenue = info['revenue']
                    
                    state = next_state
                    
                    if done:
                        break
                
                # Compute metrics
                revenues.append(episode_revenue)
                occupancies.append(np.mean(episode_occupancies))
                prices.append(np.mean(episode_prices))
                
                if len(episode_prices) > 1:
                    volatility = np.std(np.diff(episode_prices))
                    volatilities.append(volatility)
                else:
                    volatilities.append(0.0)
            
            # Store results
            results[strategy_name] = {
                'avg_revenue': float(np.mean(revenues)),
                'avg_occupancy': float(np.mean(occupancies)),
                'avg_price': float(np.mean(prices)),
                'price_volatility': float(np.mean(volatilities)) if volatilities else 0.0
            }
            
            logger.info(f"  Avg Revenue: ${results[strategy_name]['avg_revenue']:,.2f}")
            logger.info(f"  Avg Occupancy: {results[strategy_name]['avg_occupancy']:.2%}")
            logger.info(f"  Avg Price: ${results[strategy_name]['avg_price']:.2f}")
            logger.info(f"  Price Volatility: ${results[strategy_name]['price_volatility']:.2f}")
        
        self.evaluation_results = results
        logger.info(f"\n✓ Evaluation completed")
    
    def save_results(self):
        """Save all results to files."""
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS")
        logger.info("="*80)
        
        # Training metrics
        metrics_file = self.output_dir / "advanced_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"✓ Saved training metrics to {metrics_file}")
        
        # Evaluation results
        eval_file = self.output_dir / "advanced_evaluation.json"
        with open(eval_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        logger.info(f"✓ Saved evaluation results to {eval_file}")
        
        # Summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'training_episodes': len(self.training_history['rewards']),
            'final_reward': float(self.training_history['rewards'][-1]),
            'avg_reward_last_10': float(np.mean(self.training_history['rewards'][-10:])),
            'best_reward': float(max(self.training_history['rewards'])),
            'evaluation_results': self.evaluation_results
        }
        
        summary_file = self.output_dir / "advanced_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Advanced RL Parking Pricing Pipeline")
    parser.add_argument('--episodes', type=int, default=150, help='Training episodes')
    parser.add_argument('--eval-episodes', type=int, default=15, help='Evaluation episodes')
    parser.add_argument('--output', type=str, default='advanced_results', help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("\n" + "█"*80)
    logger.info("█ ADVANCED RL DYNAMIC PARKING PRICING PIPELINE")
    logger.info("█"*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("█"*80)
    
    # Create pipeline
    pipeline = AdvancedRolePipeline(output_dir=args.output)
    
    # Setup
    env = pipeline.setup_environment()
    agent = pipeline.setup_agent()
    
    # Train
    pipeline.train(agent, env, episodes=args.episodes)
    
    # Evaluate
    pipeline.evaluate(agent, env, episodes=args.eval_episodes)
    
    # Save
    pipeline.save_results()
    
    # Summary
    logger.info("\n" + "█"*80)
    logger.info("█ PIPELINE COMPLETE ✓")
    logger.info("█"*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results saved to: {pipeline.output_dir}")
    logger.info("█"*80 + "\n")


if __name__ == "__main__":
    main()
