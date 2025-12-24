#!/usr/bin/env python3
import numpy as np
import torch
import json
import time
from pathlib import Path
from typing import List, Dict
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from role_1.env import ParkingPricingEnv
from role_2.a2c_new import A2CAgent, A2CConfig

print("[INFO] Training with improved A2C agent (Experience Replay + n-step returns)")
print("=" * 80)

# Setup
results_dir = Path("./training_results")
results_dir.mkdir(parents=True, exist_ok=True)

# Create environment
print("\n[1] Creating environment...")
env = ParkingPricingEnv(
    capacity=100,
    max_steps=288,
    target_occupancy=0.8,
    min_price=0.5,
    max_price=20.0,
    seed=42
)
print(f"    State dim: {env.observation_space.shape[0]}")
print(f"    Action dim: 1 (continuous price)")

# Create agent
print("\n[2] Creating A2C agent...")
config = A2CConfig(
    state_dim=env.observation_space.shape[0],
    action_dim=1,
    hidden_dim=256,
    num_hidden_layers=2,
    policy_lr=3e-4,
    value_lr=1e-3,
    gamma=0.99,
    entropy_coef=0.01,
    value_loss_coef=0.5,
    max_grad_norm=0.5,
    l2_reg=1e-5,
    device='cpu'
)
agent = A2CAgent(config)
print("    Agent initialized with n-step returns (n=3)")
print("    Agent has experience replay buffer (10K capacity)")

# Train
print("\n[3] Training for 500 episodes...")
print("=" * 80)
start_time = time.time()
episode_rewards = []
best_reward = -float('inf')

for episode in range(1, 501):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    # Collect trajectory
    states, actions, rewards, values, log_probs, dones, next_values = [], [], [], [], [], [], []
    
    while not done:
        action, log_prob, value = agent.select_action(state, training=True)
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        
        # Store trajectory
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value.item() if hasattr(value, 'item') else value)
        log_probs.append(log_prob)
        dones.append(done)
        
        # Get next value
        if not done:
            _, _, next_value = agent.select_action(next_state, training=True)
            next_values.append(next_value.item() if hasattr(next_value, 'item') else next_value)
        else:
            next_values.append(0.0)
        
        episode_reward += reward
        state = next_state
    
    # Train after episode
    if len(states) > 0:
        agent.update(states, actions, rewards, values, log_probs, dones, next_values)
    
    episode_rewards.append(episode_reward)
    best_reward = max(best_reward, episode_reward)
    
    if episode % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        elapsed = time.time() - start_time
        print(f"Episode {episode:3d} | Reward: {episode_reward:7.2f} | Avg(50): {avg_reward:7.2f} | Time: {elapsed:6.1f}s")

elapsed_total = time.time() - start_time
print("=" * 80)
print(f"[DONE] Training completed in {elapsed_total/60:.1f} minutes")
print(f"       Final avg reward (50 ep): {np.mean(episode_rewards[-50:]):.2f}")
print(f"       Best reward: {best_reward:.2f}")
print(f"       Overall avg: {np.mean(episode_rewards):.2f}")

# Evaluate
print("\n[4] Evaluating for 20 episodes...")
eval_rewards = []
for ep in range(20):
    state, _ = env.reset()
    ep_reward = 0
    done = False
    while not done:
        action, _, _ = agent.select_action(state, training=False)
        next_state, reward, done, truncated, _ = env.step(action)
        ep_reward += reward
        state = next_state
        done = done or truncated
    eval_rewards.append(ep_reward)
    if (ep + 1) % 5 == 0:
        print(f"    Episode {ep+1}: {ep_reward:.2f}, Avg: {np.mean(eval_rewards):.2f}")

print(f"\n[RESULTS] Evaluation average: {np.mean(eval_rewards):.2f}")
print(f"          Evaluation std:     {np.std(eval_rewards):.2f}")
print("=" * 80)

# Save results
metrics = {
    "final_avg_reward": float(np.mean(episode_rewards[-50:])),
    "best_reward": float(best_reward),
    "overall_avg": float(np.mean(episode_rewards)),
    "eval_avg": float(np.mean(eval_rewards)),
    "eval_std": float(np.std(eval_rewards)),
    "total_episodes": 500,
    "training_time_sec": elapsed_total
}

with open(results_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to: {results_dir / 'metrics.json'}")
print("\nTraining complete!")
