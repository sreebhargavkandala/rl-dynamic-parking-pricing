"""
Role 4: Visualization Module for Parking Pricing Evaluation
============================================================

Provides plotting functions for:
- Training progress visualization
- Strategy comparison plots
- Episode analysis
- Results dashboard

Author: Role 4 - Evaluation, Baselines & Presentation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Try to import seaborn for better aesthetics
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Set matplotlib style
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


# =============================================================================
# COLOR PALETTE
# =============================================================================

COLORS = {
    'rl_agent': '#2E86AB',       # Blue
    'fixed_price': '#A23B72',    # Pink
    'time_based': '#F18F01',     # Orange
    'random': '#C73E1D',         # Red
    'demand_based': '#3B1F2B',   # Dark
    'baseline_1': '#84A59D',     # Sage
    'baseline_2': '#F28482',     # Coral
}


def get_strategy_color(name: str) -> str:
    """Get consistent color for strategy name."""
    name_lower = name.lower()
    if 'rl' in name_lower or 'a2c' in name_lower:
        return COLORS['rl_agent']
    elif 'fixed' in name_lower and '5' in name_lower:
        return COLORS['fixed_price']
    elif 'fixed' in name_lower:
        return COLORS['baseline_1']
    elif 'time' in name_lower:
        return COLORS['time_based']
    elif 'random' in name_lower:
        return COLORS['random']
    elif 'demand' in name_lower:
        return COLORS['demand_based']
    else:
        return COLORS['baseline_2']


# =============================================================================
# TRAINING PROGRESS PLOTS
# =============================================================================

def plot_training_progress(
    metrics_path: str,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training progress from saved metrics.
    
    Args:
        metrics_path: Path to training_metrics.json
        output_path: Optional path to save figure
        show: Whether to display plot
        
    Returns:
        Matplotlib figure
    """
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    rewards = data['episode_rewards']
    losses = data.get('episode_losses', [])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Episode rewards
    ax1 = axes[0]
    episodes = range(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, alpha=0.3, color=COLORS['rl_agent'], label='Episode Reward')
    
    # Moving average
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(rewards) + 1), moving_avg, 
                color=COLORS['rl_agent'], linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key points
    max_reward = max(rewards)
    max_idx = rewards.index(max_reward)
    ax1.annotate(f'Best: ${max_reward:.0f}', xy=(max_idx+1, max_reward),
                xytext=(max_idx+1, max_reward*1.1),
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green', fontsize=10)
    
    # Loss curve (if available)
    ax2 = axes[1]
    if losses:
        ax2.plot(range(1, len(losses) + 1), losses, color=COLORS['random'], alpha=0.5)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.set_yscale('log')  # Log scale for loss
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Loss data not available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14, color='gray')
        ax2.set_title('Training Loss')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_reward_distribution(
    metrics_path: str,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot distribution of episode rewards."""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    rewards = data['episode_rewards']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(rewards, bins=50, color=COLORS['rl_agent'], alpha=0.7, edgecolor='white')
    
    # Add statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    ax.axvline(mean_reward, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: ${mean_reward:.0f}')
    
    ax.set_xlabel('Episode Reward ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Episode Rewards During Training')
    ax.legend()
    
    # Add text box with statistics
    textstr = f'Mean: ${mean_reward:.2f}\nStd: ${std_reward:.2f}\nMax: ${max(rewards):.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# STRATEGY COMPARISON PLOTS
# =============================================================================

def plot_revenue_comparison(
    results: Dict,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Bar chart comparing average revenue across strategies.
    
    Args:
        results: Dict from compare_strategies (name -> EvaluationResult or dict)
        output_path: Optional save path
        show: Display plot
        
    Returns:
        Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    names = list(results.keys())
    revenues = []
    errors = []
    colors = []
    
    for name, result in results.items():
        if hasattr(result, 'avg_revenue'):
            revenues.append(result.avg_revenue)
            errors.append(result.std_revenue)
        else:
            revenues.append(result['avg_revenue'])
            errors.append(result.get('std_revenue', 0))
        colors.append(get_strategy_color(name))
    
    x = np.arange(len(names))
    bars = ax.bar(x, revenues, yerr=errors, capsize=5, color=colors, 
                  edgecolor='white', linewidth=2, alpha=0.85)
    
    # Add value labels on bars
    for bar, revenue in zip(bars, revenues):
        height = bar.get_height()
        ax.annotate(f'${revenue:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Average Revenue ($)')
    ax.set_title('Revenue Comparison: RL Agent vs Baselines', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    
    # Highlight best performer
    best_idx = np.argmax(revenues)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_occupancy_comparison(
    results: Dict,
    target_occupancy: float = 0.8,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Bar chart comparing average occupancy across strategies.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    names = list(results.keys())
    occupancies = []
    colors = []
    
    for name, result in results.items():
        if hasattr(result, 'mean_occupancy'):
            occupancies.append(result.mean_occupancy)
        else:
            occupancies.append(result['mean_occupancy'])
        colors.append(get_strategy_color(name))
    
    x = np.arange(len(names))
    bars = ax.bar(x, [o * 100 for o in occupancies], color=colors, 
                  edgecolor='white', linewidth=2, alpha=0.85)
    
    # Target line
    ax.axhline(y=target_occupancy * 100, color='green', linestyle='--', 
               linewidth=2, label=f'Target: {target_occupancy:.0%}')
    
    # Value labels
    for bar, occ in zip(bars, occupancies):
        height = bar.get_height()
        ax.annotate(f'{occ:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Average Occupancy (%)')
    ax.set_title('Occupancy Comparison: RL Agent vs Baselines', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_price_volatility(
    results: Dict,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Bar chart comparing price volatility across strategies.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    names = list(results.keys())
    volatilities = []
    colors = []
    
    for name, result in results.items():
        if hasattr(result, 'price_volatility'):
            volatilities.append(result.price_volatility)
        else:
            volatilities.append(result.get('price_volatility', 0))
        colors.append(get_strategy_color(name))
    
    x = np.arange(len(names))
    bars = ax.bar(x, volatilities, color=colors, 
                  edgecolor='white', linewidth=2, alpha=0.85)
    
    # Value labels
    for bar, vol in zip(bars, volatilities):
        height = bar.get_height()
        ax.annotate(f'${vol:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Price Volatility (Std of Price Changes)')
    ax.set_title('Price Stability: Lower is More Stable', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# EPISODE ANALYSIS PLOTS
# =============================================================================

def plot_episode_trajectory(
    prices: List[float],
    occupancies: List[float],
    strategy_name: str = "Strategy",
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot price and occupancy trajectory over a single episode.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    steps = range(len(prices))
    hours = [s * 5 / 60 for s in steps]  # Convert to hours
    
    # Price trajectory
    ax1 = axes[0]
    ax1.plot(hours, prices, color=COLORS['rl_agent'], linewidth=2)
    ax1.fill_between(hours, prices, alpha=0.3, color=COLORS['rl_agent'])
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{strategy_name}: 24-Hour Episode Trajectory', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark peak hours
    ax1.axvspan(8, 18, alpha=0.1, color='orange', label='Peak Hours')
    ax1.legend()
    
    # Occupancy trajectory
    ax2 = axes[1]
    ax2.plot(hours, [o * 100 for o in occupancies], color=COLORS['time_based'], linewidth=2)
    ax2.fill_between(hours, [o * 100 for o in occupancies], alpha=0.3, color=COLORS['time_based'])
    ax2.axhline(y=80, color='green', linestyle='--', linewidth=1.5, label='Target (80%)')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Occupancy (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_strategy_comparison_episode(
    episode_data: Dict[str, Tuple[List[float], List[float]]],
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare multiple strategies on same episode (prices and occupancies).
    
    Args:
        episode_data: Dict mapping strategy name to (prices, occupancies) tuple
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Price comparison
    ax1 = axes[0]
    for name, (prices, _) in episode_data.items():
        hours = [i * 5 / 60 for i in range(len(prices))]
        ax1.plot(hours, prices, label=name, color=get_strategy_color(name), 
                linewidth=2, alpha=0.8)
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Strategy Comparison: Pricing Behavior Over 24 Hours', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(8, 18, alpha=0.05, color='orange')
    
    # Occupancy comparison
    ax2 = axes[1]
    for name, (_, occupancies) in episode_data.items():
        hours = [i * 5 / 60 for i in range(len(occupancies))]
        ax2.plot(hours, [o * 100 for o in occupancies], label=name, 
                color=get_strategy_color(name), linewidth=2, alpha=0.8)
    
    ax2.axhline(y=80, color='green', linestyle='--', linewidth=2, label='Target')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Occupancy (%)')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# SUMMARY DASHBOARD
# =============================================================================

def create_summary_dashboard(
    results: Dict,
    training_metrics_path: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create multi-panel dashboard summarizing all results.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # Panel 1: Revenue Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    names = list(results.keys())
    revenues = [r.avg_revenue if hasattr(r, 'avg_revenue') else r['avg_revenue'] 
                for r in results.values()]
    colors = [get_strategy_color(n) for n in names]
    bars = ax1.barh(names, revenues, color=colors, alpha=0.85)
    ax1.set_xlabel('Average Revenue ($)')
    ax1.set_title('Revenue Comparison', fontweight='bold')
    for bar, rev in zip(bars, revenues):
        ax1.text(bar.get_width() + max(revenues)*0.02, bar.get_y() + bar.get_height()/2,
                f'${rev:,.0f}', va='center', fontsize=10, fontweight='bold')
    
    # Panel 2: Occupancy Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    occupancies = [r.mean_occupancy if hasattr(r, 'mean_occupancy') else r['mean_occupancy'] 
                   for r in results.values()]
    bars = ax2.barh(names, [o * 100 for o in occupancies], color=colors, alpha=0.85)
    ax2.axvline(x=80, color='green', linestyle='--', linewidth=2)
    ax2.set_xlabel('Average Occupancy (%)')
    ax2.set_title('Occupancy vs Target (80%)', fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Panel 3: Training progress (if available)
    ax3 = fig.add_subplot(gs[1, 0])
    if training_metrics_path and Path(training_metrics_path).exists():
        with open(training_metrics_path, 'r') as f:
            data = json.load(f)
        rewards = data['episode_rewards']
        ax3.plot(rewards, alpha=0.3, color=COLORS['rl_agent'])
        window = 50
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(rewards)), ma, color=COLORS['rl_agent'], linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.set_title('Training Progress', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Training data not available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=14, color='gray')
    
    # Panel 4: Price Volatility
    ax4 = fig.add_subplot(gs[1, 1])
    volatilities = [r.price_volatility if hasattr(r, 'price_volatility') else r.get('price_volatility', 0) 
                    for r in results.values()]
    bars = ax4.barh(names, volatilities, color=colors, alpha=0.85)
    ax4.set_xlabel('Price Volatility')
    ax4.set_title('Price Stability (Lower = Better)', fontweight='bold')
    
    # Main title
    fig.suptitle('RL Parking Pricing: Evaluation Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("Plots module loaded successfully!")
    print("Available functions:")
    print("  - plot_training_progress(metrics_path, output_path)")
    print("  - plot_revenue_comparison(results, output_path)")
    print("  - plot_occupancy_comparison(results, output_path)")
    print("  - plot_price_volatility(results, output_path)")
    print("  - create_summary_dashboard(results, training_path, output_path)")
