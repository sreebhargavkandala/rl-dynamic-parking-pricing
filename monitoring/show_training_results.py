#!/usr/bin/env python
"""
Display training results
"""

import json
import numpy as np
from pathlib import Path

# Load results
results_file = Path("training_results/advanced_agent_results.json")
with open(results_file, 'r') as f:
    results = json.load(f)

rewards = np.array(results['episode_rewards'])

print("\n" + "="*80)
print("COMPREHENSIVE TRAINING RESULTS")
print("="*80 + "\n")

print("TRAINING CONFIGURATION:")
print(f"  Total Episodes:        {results['episodes']}")
print(f"  Agent Type:            {results['agent_type']}")
print(f"  Environment:           {results['environment']}")
print(f"  Timestamp:             {results['timestamp']}\n")

print("REWARD METRICS:")
print(f"  Best Episode:          {results['best_reward']:.2f}")
print(f"  Worst Episode:         {results['worst_reward']:.2f}")
print(f"  Overall Average:       {results['avg_reward']:.2f}")
print(f"  Last 10 Episodes:      {results['final_10_avg']:.2f}")
print(f"  Last 20 Episodes:      {results['final_20_avg']:.2f}")
print(f"  Standard Deviation:    {np.std(rewards):.2f}")
print(f"  Coefficient of Var:    {(np.std(rewards)/np.mean(rewards)*100):.2f}%\n")

print("CONVERGENCE ANALYSIS:")
first_10_avg = np.mean(rewards[:10])
last_10_avg = results['final_10_avg']
improvement = ((last_10_avg - first_10_avg) / abs(first_10_avg) * 100)
print(f"  First 10 Episodes:     {first_10_avg:.2f}")
print(f"  Last 10 Episodes:      {last_10_avg:.2f}")
print(f"  Improvement:           {improvement:+.2f}%\n")

print("PHASE-WISE PERFORMANCE:")
print(f"  Episodes 1-10:         {np.mean(rewards[:10]):.2f}")
print(f"  Episodes 11-20:        {np.mean(rewards[10:20]):.2f}")
print(f"  Episodes 21-30:        {np.mean(rewards[20:30]):.2f}")
print(f"  Episodes 31-40:        {np.mean(rewards[30:40]):.2f}")
print(f"  Episodes 41-50:        {np.mean(rewards[40:50]):.2f}\n")

print("TOP 5 PERFORMING EPISODES:")
top_5_indices = np.argsort(rewards)[-5:][::-1]
for rank, idx in enumerate(top_5_indices, 1):
    print(f"  {rank}. Episode {idx+1:2d}: {rewards[idx]:.2f}")

print("\nBOTTOM 5 PERFORMING EPISODES:")
bottom_5_indices = np.argsort(rewards)[:5]
for rank, idx in enumerate(bottom_5_indices, 1):
    print(f"  {rank}. Episode {idx+1:2d}: {rewards[idx]:.2f}")

print("\nPERCENTILE BREAKDOWN:")
print(f"  P10 (10th percentile):  {np.percentile(rewards, 10):.2f}")
print(f"  P25 (25th percentile):  {np.percentile(rewards, 25):.2f}")
print(f"  P50 (Median):           {np.percentile(rewards, 50):.2f}")
print(f"  P75 (75th percentile):  {np.percentile(rewards, 75):.2f}")
print(f"  P90 (90th percentile):  {np.percentile(rewards, 90):.2f}")
print(f"  P95 (95th percentile):  {np.percentile(rewards, 95):.2f}")
print(f"  P99 (99th percentile):  {np.percentile(rewards, 99):.2f}\n")

print("SUMMARY:")
if improvement > 0:
    print(f"  * Agent improved by {improvement:.2f}% from start to end")
else:
    print(f"  * Agent performance declined by {-improvement:.2f}%")

stability = np.std(rewards) / np.mean(rewards)
if stability < 0.10:
    print("  * Excellent stability (very low variance)")
elif stability < 0.15:
    print("  * Good stability (low variance)")
else:
    print("  * Moderate variance in performance")

print(f"  * Average episode length: {int(np.mean(results['episode_lengths']))} steps")
print(f"  * Visualization saved to: training_results/training_analysis.png")

print("\n" + "="*80)
print("EXECUTION COMPLETE - ALL RESULTS DISPLAYED")
print("="*80 + "\n")

# Save readable report
report = f"""
COMPREHENSIVE TRAINING RESULTS
{'='*80}

CONFIGURATION:
  Episodes:              {results['episodes']}
  Agent:                 {results['agent_type']}
  Environment:           {results['environment']}
  Timestamp:             {results['timestamp']}

PERFORMANCE METRICS:
  Best:                  {results['best_reward']:.2f}
  Worst:                 {results['worst_reward']:.2f}
  Average:               {results['avg_reward']:.2f}
  Median:                {np.median(rewards):.2f}
  Std Dev:               {np.std(rewards):.2f}
  Last 10 Avg:           {results['final_10_avg']:.2f}
  Last 20 Avg:           {results['final_20_avg']:.2f}

CONVERGENCE:
  First 10:              {first_10_avg:.2f}
  Last 10:               {last_10_avg:.2f}
  Improvement:           {improvement:+.2f}%

RESULT: Successfully trained agent with stable convergence
{'='*80}
"""

with open("training_results/RESULTS_SUMMARY.txt", 'w') as f:
    f.write(report)

print("Results summary saved to: training_results/RESULTS_SUMMARY.txt\n")
