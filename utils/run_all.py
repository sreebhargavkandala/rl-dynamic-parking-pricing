#!/usr/bin/env python
"""
Master runner script - Execute all agents and generate results
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Fix path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("🚀 RUNNING ALL AGENTS - COMPLETE EVALUATION")
print("="*80)
print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

try:
    # Import all modules
    from role_1.env import ParkingPricingEnv
    from role_2.a2c_advanced import AdvancedA2CAgent, AdvancedA2CConfig
    print("✅ Imports successful\n")
    
    # Create environment
    print("📦 Initializing Parking Pricing Environment...")
    env = ParkingPricingEnv(capacity=100)
    print(f"✅ Environment initialized\n")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"   State dim: {state_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   Max steps: {env.max_steps}\n")
    
    # Create agent with config
    print("🤖 Initializing Advanced A2C Agent...")
    config = AdvancedA2CConfig()
    config.state_dim = state_dim
    config.action_dim = action_dim
    config.hidden_dim = 256
    config.policy_lr = 3e-4
    config.value_lr = 1e-3
    config.entropy_coef = 0.01
    config.device = 'cpu'
    
    agent = AdvancedA2CAgent(config)
    print("✅ Agent initialized\n")
    
    # Training configuration
    num_episodes = 50  # Quick run for demo
    print(f"📊 Training Configuration:")
    print(f"   Episodes: {num_episodes}")
    print(f"   Evaluation interval: 10\n")
    
    # Training loop
    all_rewards = []
    all_lengths = []
    best_reward = -np.inf
    
    print("🏃 Starting training...\n")
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action, log_prob = agent.select_action(state)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, done, log_prob)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if episode_length >= 288:  # Max steps
                break
        
        # Update agent
        if (episode + 1) % 5 == 0:
            agent.update()
        
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            avg_length = np.mean(all_lengths[-10:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(10): {avg_reward:7.2f} | "
                  f"Best: {best_reward:7.2f} | "
                  f"Length: {episode_length}")
    
    print("\n" + "="*80)
    print("📈 TRAINING COMPLETE")
    print("="*80 + "\n")
    
    # Calculate statistics
    final_10_avg = np.mean(all_rewards[-10:])
    final_20_avg = np.mean(all_rewards[-20:])
    overall_avg = np.mean(all_rewards)
    
    print("📊 FINAL RESULTS:\n")
    print(f"  Total Episodes:     {num_episodes}")
    print(f"  Best Reward:        {best_reward:.2f}")
    print(f"  Worst Reward:       {min(all_rewards):.2f}")
    print(f"  Average Reward:     {overall_avg:.2f}")
    print(f"  Last 10 Avg:        {final_10_avg:.2f}")
    print(f"  Last 20 Avg:        {final_20_avg:.2f}")
    print(f"  Avg Episode Length: {np.mean(all_lengths):.1f} steps")
    print(f"  Improvement:        {((final_10_avg - all_rewards[0]) / abs(all_rewards[0]) * 100):.1f}%" if all_rewards[0] != 0 else "")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "episodes": num_episodes,
        "best_reward": float(best_reward),
        "worst_reward": float(min(all_rewards)),
        "avg_reward": float(overall_avg),
        "final_10_avg": float(final_10_avg),
        "final_20_avg": float(final_20_avg),
        "episode_rewards": [float(r) for r in all_rewards],
        "episode_lengths": [int(l) for l in all_lengths],
        "agent_type": "AdvancedA2C",
        "environment": "EnhancedParkingEnvironment"
    }
    
    # Save to file
    output_dir = Path("training_results")
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "advanced_agent_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    print("\n" + "="*80)
    print("🎉 EXECUTION COMPLETE")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
