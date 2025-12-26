#!/usr/bin/env python3
"""
Role 4: Main Evaluation Script
==============================

Runs complete evaluation pipeline:
1. Load trained RL agent
2. Run all baseline strategies
3. Compare performance
4. Generate plots and tables
5. Create comparison visualizations

Author: Role 4 - Evaluation, Baselines & Presentation
"""

import sys
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Role 4 imports
from role_4.evaluation import (
    get_default_baselines,
    compare_strategies,
    generate_comparison_table,
    save_results,
    evaluate_strategy,
    FixedPriceStrategy,
    TimeBasedStrategy,
)
from role_4.evaluation.visualise import (
    plot_training_progress,
    plot_revenue_comparison,
    plot_occupancy_comparison,
    plot_price_volatility,
    plot_strategy_comparison_episode,
    create_summary_dashboard,
)
from role_4.evaluation.video_recording import (
    collect_episode_data,
    create_comparison_static,
    create_comparison_animation,
    create_episode_animation,
)

# Role 1 & 2 imports
from role_1.env import ParkingPricingEnv
from role_2.a2c_new import A2CAgent, A2CConfig


def load_trained_agent(
    checkpoint_dir: Path = None,
    device: str = 'cpu'
) -> A2CAgent:
    """
    Load trained A2C agent.
    
    Attempts to load from checkpoints directory.
    If not found, creates a fresh agent (for testing).
    """
    # Create agent with matching config
    config = A2CConfig(
        state_dim=5,
        action_dim=1,
        hidden_dim=256,
        policy_lr=3e-4,
        value_lr=1e-3,
        gamma=0.99,
        entropy_coef=0.01,
        device=device
    )
    
    agent = A2CAgent(config)
    
    # Try to load checkpoint
    if checkpoint_dir is None:
        checkpoint_dir = PROJECT_ROOT / "training_results" / "checkpoints"
    
    # Look for pickle checkpoints (.pkl files)
    checkpoint_files = list(checkpoint_dir.glob("*.pkl")) if checkpoint_dir.exists() else []
    
    if checkpoint_files:
        # Load most recent checkpoint
        latest = sorted(checkpoint_files, key=lambda x: x.stat().st_mtime)[-1]
        try:
            agent.load(str(latest))
            print(f"✓ Loaded trained agent from: {latest.name}")
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}")
            print("  Using fresh agent for evaluation")
    else:
        print("⚠ No checkpoints found. Using fresh agent for evaluation.")
        print("  (Run train_agent_complete.py first for proper evaluation)")
    
    return agent


def run_evaluation(
    num_episodes: int = 20,
    output_dir: Path = None,
    generate_animations: bool = False,
    show_plots: bool = False
) -> None:
    """
    Run complete evaluation pipeline.
    
    Args:
        num_episodes: Episodes per strategy for evaluation
        output_dir: Directory for saving results
        generate_animations: Whether to create animated comparisons
        show_plots: Whether to display plots interactively
    """
    print("\n" + "="*80)
    print("  ROLE 4: EVALUATION PIPELINE")
    print("="*80)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Episodes per strategy: {num_episodes}")
    print("="*80 + "\n")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # STEP 1: CREATE ENVIRONMENT
    # ==========================================================================
    print("\n[1/6] Setting up environment...")
    
    env = ParkingPricingEnv(
        capacity=100,
        max_steps=288,  # 24 hours
        target_occupancy=0.8,
        min_price=0.5,
        max_price=20.0,
        seed=42
    )
    print(f"  ✓ Environment: {env.capacity} spots, ${env.min_price}-${env.max_price} range")
    
    # ==========================================================================
    # STEP 2: LOAD TRAINED AGENT
    # ==========================================================================
    print("\n[2/6] Loading trained agent...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = load_trained_agent(device=device)
    print(f"  ✓ Agent device: {device}")
    
    # ==========================================================================
    # STEP 3: GET BASELINE STRATEGIES
    # ==========================================================================
    print("\n[3/6] Preparing baseline strategies...")
    
    baselines = get_default_baselines()
    print(f"  ✓ {len(baselines)} baseline strategies loaded:")
    for b in baselines:
        print(f"      - {b.name}")
    
    # ==========================================================================
    # STEP 4: RUN COMPARISON
    # ==========================================================================
    print("\n[4/6] Running strategy evaluation...")
    
    results = compare_strategies(
        env=env,
        strategies=baselines,
        rl_agent=agent,
        num_episodes=num_episodes,
        seed=42,
        verbose=True
    )
    
    # Print comparison table
    table = generate_comparison_table(results)
    print(table)
    
    # Save results
    save_results(results, output_dir)
    
    # ==========================================================================
    # STEP 5: GENERATE PLOTS
    # ==========================================================================
    print("\n[5/6] Generating visualizations...")
    
    # Training progress (if available)
    training_metrics_path = PROJECT_ROOT / "training_results" / "training_metrics.json"
    if training_metrics_path.exists():
        print("  → Training progress plot...")
        plot_training_progress(
            str(training_metrics_path),
            output_path=str(output_dir / "training_progress.png"),
            show=show_plots
        )
    else:
        print("  ⚠ Training metrics not found, skipping training progress plot")
    
    # Revenue comparison
    print("  → Revenue comparison plot...")
    plot_revenue_comparison(
        results,
        output_path=str(output_dir / "revenue_comparison.png"),
        show=show_plots
    )
    
    # Occupancy comparison
    print("  → Occupancy comparison plot...")
    plot_occupancy_comparison(
        results,
        output_path=str(output_dir / "occupancy_comparison.png"),
        show=show_plots
    )
    
    # Price volatility
    print("  → Price volatility plot...")
    plot_price_volatility(
        results,
        output_path=str(output_dir / "price_volatility.png"),
        show=show_plots
    )
    
    # Summary dashboard
    print("  → Summary dashboard...")
    create_summary_dashboard(
        results,
        training_metrics_path=str(training_metrics_path) if training_metrics_path.exists() else None,
        output_path=str(output_dir / "dashboard.png"),
        show=show_plots
    )
    
    # ==========================================================================
    # STEP 6: EPISODE COMPARISON
    # ==========================================================================
    print("\n[6/6] Generating episode comparison...")
    
    # Collect data for one episode from each strategy
    episode_data = {}
    
    # Fixed price baseline
    fixed_strategy = FixedPriceStrategy(price=5.0)
    prices_fixed, occ_fixed, _ = collect_episode_data(env, fixed_strategy, is_rl_agent=False, seed=123)
    episode_data["Fixed Price ($5.00)"] = (prices_fixed, occ_fixed)
    
    # Time-based baseline
    time_strategy = TimeBasedStrategy(peak_price=10.0, offpeak_price=3.0)
    prices_time, occ_time, _ = collect_episode_data(env, time_strategy, is_rl_agent=False, seed=123)
    episode_data["Time-Based"] = (prices_time, occ_time)
    
    # RL Agent
    prices_rl, occ_rl, _ = collect_episode_data(env, agent, is_rl_agent=True, seed=123)
    episode_data["RL Agent (A2C)"] = (prices_rl, occ_rl)
    
    # Multi-strategy comparison
    print("  → Strategy episode comparison plot...")
    plot_strategy_comparison_episode(
        episode_data,
        output_path=str(output_dir / "strategy_episode_comparison.png"),
        show=show_plots
    )
    
    # Static before/after comparison
    print("  → Before/after comparison image...")
    create_comparison_static(
        baseline_data=(prices_fixed, occ_fixed),
        rl_data=(prices_rl, occ_rl),
        baseline_name="Fixed Price Baseline",
        output_path=str(output_dir / "baseline_vs_rl.png"),
        show=show_plots
    )
    
    # Animated GIF comparisons
    print("  → Generating animated GIFs...")
    try:
        create_comparison_animation(
            baseline_data=(prices_fixed, occ_fixed),
            rl_data=(prices_rl, occ_rl),
            baseline_name="Fixed Price ($5)",
            output_path=str(output_dir / "comparison_animation.gif"),
            fps=15,
            duration_seconds=8
        )
        create_episode_animation(
            prices=prices_rl,
            occupancies=occ_rl,
            strategy_name="Trained RL Agent (A2C)",
            output_path=str(output_dir / "rl_agent_episode.gif"),
            fps=15,
            duration_seconds=8
        )
        print("  ✓ Animated GIFs saved!")
    except Exception as e:
        print(f"  ⚠ Could not generate GIFs: {e}")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "="*80)
    print("  EVALUATION COMPLETE!")
    print("="*80)
    print(f"\n✓ Results saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")
    
    # Find best performer
    best = max(results.items(), key=lambda x: x[1].avg_revenue if hasattr(x[1], 'avg_revenue') else x[1]['avg_revenue'])
    print(f"\n🏆 Best performing strategy: {best[0]}")
    if hasattr(best[1], 'avg_revenue'):
        print(f"   Average Revenue: ${best[1].avg_revenue:,.2f}")
    else:
        print(f"   Average Revenue: ${best[1]['avg_revenue']:,.2f}")
    
    print("\n" + "="*80 + "\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run parking pricing evaluation")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per strategy")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_evaluation(
        num_episodes=args.episodes,
        output_dir=output_dir,
        show_plots=args.show_plots
    )
