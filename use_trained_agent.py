#!/usr/bin/env python3
 

import sys
from pathlib import Path
import torch
import numpy as np

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from role_2.a2c_new import A2CAgent, A2CConfig
from role_1.env import ParkingPricingEnv


def load_best_agent():
    """Load the best trained A2C agent from checkpoint."""
    model_path = PROJECT_ROOT / "training_results" / "a2c_best" / "best_model_ep84.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading best agent from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Recreate agent with saved config
    config = A2CConfig(**checkpoint['config'])
    agent = A2CAgent(config)
    
    print(f"✓ Agent loaded successfully!")
    print(f"  - Episode: {checkpoint['episode']}")
    print(f"  - Best reward: ${checkpoint['reward']:.2f}")
    
    return agent


def evaluate_agent(agent, num_episodes=5):
    """Evaluate the trained agent on the environment."""
    env = ParkingPricingEnv(
        capacity=150,
        max_steps=288,
        target_occupancy=0.80,
        min_price=1.5,
        max_price=25.0
    )
    
    print(f"\n{'='*70}")
    print("AGENT EVALUATION")
    print(f"{'='*70}")
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action from trained agent (deterministic)
            action, _, _ = agent.select_action(state, training=False)
            action = np.clip(action, env.min_price, env.max_price)
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1}: ${episode_reward:.2f}")
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\n{'─'*70}")
    print(f"Average reward: ${avg_reward:.2f}")
    print(f"Std deviation: ${std_reward:.2f}")
    print(f"Min reward: ${min(episode_rewards):.2f}")
    print(f"Max reward: ${max(episode_rewards):.2f}")
    print(f"{'='*70}\n")
    
    return episode_rewards


def demo_pricing_decisions(agent, num_steps=50):
    """Show pricing decisions made by the trained agent."""
    env = ParkingPricingEnv(
        capacity=150,
        max_steps=288,
        target_occupancy=0.80,
        min_price=1.5,
        max_price=25.0
    )
    
    print(f"\n{'='*70}")
    print("PRICING DECISIONS DEMO")
    print(f"{'='*70}")
    print(f"{'Step':<8} {'Occupancy':<12} {'Demand':<10} {'Price':<10} {'Reward':<10}")
    print(f"{'-'*70}")
    
    state, _ = env.reset()
    
    for step in range(num_steps):
        # Get action from agent
        action, _, _ = agent.select_action(state, training=False)
        # action is numpy array [dim], clip and use directly
        action = np.clip(action, env.min_price, env.max_price)
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        occupancy = state[0] * 100  # Normalize back to percentage
        demand = state[2]           # Demand factor
        price_val = action[0] if isinstance(action, np.ndarray) else action
        
        print(f"{step+1:<8} {occupancy:<11.1f}% {demand:<9.3f} ${price_val:<8.2f} {reward:<9.2f}")
        
        state = next_state
        if terminated or truncated:
            break
    
    print(f"{'='*70}\n")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load and use the trained A2C parking pricing agent"
    )
    parser.add_argument(
        "--action",
        choices=["eval", "demo", "load"],
        default="eval",
        help="Action to perform: eval (evaluate), demo (show decisions), load (just load)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of steps for demo"
    )
    
    args = parser.parse_args()
    
    try:
        # Load agent
        agent = load_best_agent()
        
        if args.action == "eval":
            evaluate_agent(agent, num_episodes=args.episodes)
        elif args.action == "demo":
            demo_pricing_decisions(agent, num_steps=args.steps)
        else:
            print("\nAgent loaded and ready to use!")
            print("Use agent.select_action(state, training=False) to get pricing decisions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
