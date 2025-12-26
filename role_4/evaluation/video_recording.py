"""
Role 4: Video Recording Module for Parking Pricing
===================================================

Generates animated visualizations of agent behavior over episodes.
Creates before/after comparisons of baseline vs trained RL agent.

Author: Role 4 - Evaluation, Baselines & Presentation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# EPISODE DATA COLLECTION
# =============================================================================

def collect_episode_data(
    env,
    strategy_or_agent,
    is_rl_agent: bool = False,
    seed: int = 42
) -> Tuple[List[float], List[float], List[float]]:
    """
    Collect price, occupancy, and revenue data for one episode.
    
    Args:
        env: ParkingPricingEnv instance
        strategy_or_agent: PricingStrategy or RL agent
        is_rl_agent: Whether this is an RL agent (changes action selection)
        seed: Random seed
        
    Returns:
        Tuple of (prices, occupancies, revenues)
    """
    import torch
    
    obs, _ = env.reset(seed=seed)
    
    prices = []
    occupancies = [obs[0]]  # Initial occupancy
    revenues = []
    
    done = False
    
    while not done:
        if is_rl_agent:
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(strategy_or_agent.device)
            action, _, _ = strategy_or_agent.select_action(state_tensor, training=False)
            
            if isinstance(action, torch.Tensor):
                price = action.cpu().detach().numpy().flatten()[0]
            elif isinstance(action, np.ndarray):
                price = action.flatten()[0]
            else:
                price = float(action)
        else:
            price = strategy_or_agent.get_price(obs, env)
        
        prices.append(price)
        
        action = np.array([price], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        occupancies.append(info['occupancy'])
        revenues.append(info['revenue'])
        
        done = terminated or truncated
    
    return prices, occupancies[:-1], revenues  # Align lengths


# =============================================================================
# ANIMATED EPISODE VISUALIZATION
# =============================================================================

def create_episode_animation(
    prices: List[float],
    occupancies: List[float],
    strategy_name: str = "Strategy",
    output_path: Optional[str] = None,
    fps: int = 30,
    duration_seconds: int = 10
) -> animation.FuncAnimation:
    """
    Create animated visualization of episode trajectory.
    
    Args:
        prices: List of prices over episode
        occupancies: List of occupancies over episode
        strategy_name: Name for title
        output_path: Path to save animation (GIF or MP4)
        fps: Frames per second
        duration_seconds: Target duration
        
    Returns:
        Animation object
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    total_frames = fps * duration_seconds
    steps_per_frame = max(1, len(prices) // total_frames)
    
    hours = [s * 5 / 60 for s in range(len(prices))]
    
    # Setup axes
    ax1, ax2 = axes
    
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, max(prices) * 1.2)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{strategy_name}: Real-Time Pricing', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(8, 18, alpha=0.1, color='orange', label='Peak Hours')
    
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Occupancy (%)', fontsize=12)
    ax2.axhline(y=80, color='green', linestyle='--', linewidth=2, label='Target')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Lines to animate
    line1, = ax1.plot([], [], color='#2E86AB', linewidth=2)
    line2, = ax2.plot([], [], color='#F18F01', linewidth=2)
    
    # Price marker
    marker1, = ax1.plot([], [], 'o', color='#2E86AB', markersize=10)
    marker2, = ax2.plot([], [], 'o', color='#F18F01', markersize=10)
    
    # Time indicator text
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        marker1.set_data([], [])
        marker2.set_data([], [])
        time_text.set_text('')
        return line1, line2, marker1, marker2, time_text
    
    def animate(frame):
        idx = min(frame * steps_per_frame, len(prices) - 1)
        
        # Update lines
        line1.set_data(hours[:idx+1], prices[:idx+1])
        line2.set_data(hours[:idx+1], [o * 100 for o in occupancies[:idx+1]])
        
        # Update markers
        marker1.set_data([hours[idx]], [prices[idx]])
        marker2.set_data([hours[idx]], [occupancies[idx] * 100])
        
        # Update time text
        current_hour = hours[idx]
        hour_int = int(current_hour)
        minutes = int((current_hour - hour_int) * 60)
        time_text.set_text(f'Time: {hour_int:02d}:{minutes:02d}\nPrice: ${prices[idx]:.2f}')
        
        return line1, line2, marker1, marker2, time_text
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=min(total_frames, len(prices) // steps_per_frame),
        interval=1000/fps, blit=True
    )
    
    plt.tight_layout()
    
    if output_path:
        # Save as GIF or MP4
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        else:
            anim.save(output_path, writer='ffmpeg', fps=fps)
        print(f"✓ Animation saved: {output_path}")
    
    return anim


# =============================================================================
# COMPARISON ANIMATION
# =============================================================================

def create_comparison_animation(
    baseline_data: Tuple[List[float], List[float]],
    rl_data: Tuple[List[float], List[float]],
    baseline_name: str = "Baseline",
    output_path: Optional[str] = None,
    fps: int = 30,
    duration_seconds: int = 15
) -> animation.FuncAnimation:
    """
    Create side-by-side comparison animation.
    
    Args:
        baseline_data: (prices, occupancies) for baseline
        rl_data: (prices, occupancies) for RL agent
        baseline_name: Name of baseline strategy
        output_path: Save path
        fps: Frames per second
        duration_seconds: Target duration
        
    Returns:
        Animation object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    prices_base, occ_base = baseline_data
    prices_rl, occ_rl = rl_data
    
    max_len = max(len(prices_base), len(prices_rl))
    total_frames = fps * duration_seconds
    steps_per_frame = max(1, max_len // total_frames)
    
    hours = [s * 5 / 60 for s in range(max_len)]
    
    # Configure axes
    for ax in axes.flat:
        ax.set_xlim(0, 24)
        ax.grid(True, alpha=0.3)
    
    # Row 1: Prices
    axes[0, 0].set_ylim(0, 55)
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title(f'{baseline_name}', fontsize=14, fontweight='bold')
    
    axes[0, 1].set_ylim(0, 55)
    axes[0, 1].set_title('RL Agent (A2C)', fontsize=14, fontweight='bold')
    
    # Row 2: Occupancy
    for ax in axes[1, :]:
        ax.set_ylim(0, 100)
        ax.set_xlabel('Hour of Day')
        ax.axhline(y=80, color='green', linestyle='--', linewidth=1.5)
    
    axes[1, 0].set_ylabel('Occupancy (%)')
    
    # Lines
    line_p_base, = axes[0, 0].plot([], [], color='#A23B72', linewidth=2)
    line_p_rl, = axes[0, 1].plot([], [], color='#2E86AB', linewidth=2)
    line_o_base, = axes[1, 0].plot([], [], color='#A23B72', linewidth=2)
    line_o_rl, = axes[1, 1].plot([], [], color='#2E86AB', linewidth=2)
    
    # Revenue text
    rev_text_base = axes[0, 0].text(0.02, 0.95, '', transform=axes[0, 0].transAxes,
                                     fontsize=11, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='white'))
    rev_text_rl = axes[0, 1].text(0.02, 0.95, '', transform=axes[0, 1].transAxes,
                                   fontsize=11, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white'))
    
    def init():
        for line in [line_p_base, line_p_rl, line_o_base, line_o_rl]:
            line.set_data([], [])
        rev_text_base.set_text('')
        rev_text_rl.set_text('')
        return line_p_base, line_p_rl, line_o_base, line_o_rl, rev_text_base, rev_text_rl
    
    def animate(frame):
        idx = min(frame * steps_per_frame, max_len - 1)
        idx_base = min(idx, len(prices_base) - 1)
        idx_rl = min(idx, len(prices_rl) - 1)
        
        # Update lines
        line_p_base.set_data(hours[:idx_base+1], prices_base[:idx_base+1])
        line_p_rl.set_data(hours[:idx_rl+1], prices_rl[:idx_rl+1])
        line_o_base.set_data(hours[:idx_base+1], [o * 100 for o in occ_base[:idx_base+1]])
        line_o_rl.set_data(hours[:idx_rl+1], [o * 100 for o in occ_rl[:idx_rl+1]])
        
        # Compute running revenue
        rev_base = sum([prices_base[i] * occ_base[i] * 100 for i in range(idx_base+1)])
        rev_rl = sum([prices_rl[i] * occ_rl[i] * 100 for i in range(idx_rl+1)])
        
        rev_text_base.set_text(f'Revenue: ${rev_base:,.0f}')
        rev_text_rl.set_text(f'Revenue: ${rev_rl:,.0f}')
        
        return line_p_base, line_p_rl, line_o_base, line_o_rl, rev_text_base, rev_text_rl
    
    fig.suptitle('Baseline vs RL Agent: Side-by-Side Comparison', fontsize=16, fontweight='bold')
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=min(total_frames, max_len // steps_per_frame),
        interval=1000/fps, blit=True
    )
    
    plt.tight_layout()
    
    if output_path:
        try:
            if output_path.endswith('.gif'):
                anim.save(output_path, writer='pillow', fps=fps)
            else:
                anim.save(output_path, writer='ffmpeg', fps=fps)
            print(f"✓ Comparison animation saved: {output_path}")
        except Exception as e:
            print(f"⚠ Could not save animation: {e}")
            print("  (ffmpeg or pillow may be required)")
    
    return anim


# =============================================================================
# STATIC COMPARISON IMAGE
# =============================================================================

def create_comparison_static(
    baseline_data: Tuple[List[float], List[float]],
    rl_data: Tuple[List[float], List[float]],
    baseline_name: str = "Baseline",
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create static side-by-side comparison (fallback if animation fails).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    prices_base, occ_base = baseline_data
    prices_rl, occ_rl = rl_data
    
    hours_base = [s * 5 / 60 for s in range(len(prices_base))]
    hours_rl = [s * 5 / 60 for s in range(len(prices_rl))]
    
    # Prices
    axes[0, 0].plot(hours_base, prices_base, color='#A23B72', linewidth=2)
    axes[0, 0].fill_between(hours_base, prices_base, alpha=0.3, color='#A23B72')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title(baseline_name, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(hours_rl, prices_rl, color='#2E86AB', linewidth=2)
    axes[0, 1].fill_between(hours_rl, prices_rl, alpha=0.3, color='#2E86AB')
    axes[0, 1].set_title('RL Agent (A2C)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Occupancy
    axes[1, 0].plot(hours_base, [o * 100 for o in occ_base], color='#A23B72', linewidth=2)
    axes[1, 0].axhline(y=80, color='green', linestyle='--', linewidth=1.5)
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Occupancy (%)')
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(hours_rl, [o * 100 for o in occ_rl], color='#2E86AB', linewidth=2)
    axes[1, 1].axhline(y=80, color='green', linestyle='--', linewidth=1.5)
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add revenue annotations
    rev_base = sum([prices_base[i] * occ_base[i] * 100 for i in range(len(prices_base))])
    rev_rl = sum([prices_rl[i] * occ_rl[i] * 100 for i in range(len(prices_rl))])
    
    axes[0, 0].text(0.98, 0.95, f'Total: ${rev_base:,.0f}', transform=axes[0, 0].transAxes,
                    ha='right', va='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white'))
    axes[0, 1].text(0.98, 0.95, f'Total: ${rev_rl:,.0f}', transform=axes[0, 1].transAxes,
                    ha='right', va='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white'))
    
    fig.suptitle('Baseline vs RL Agent Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison image saved: {output_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Video Recording Module")
    print("=" * 50)
    print("Available functions:")
    print("  - collect_episode_data(env, strategy)")
    print("  - create_episode_animation(prices, occupancies, name)")
    print("  - create_comparison_animation(baseline_data, rl_data)")
    print("  - create_comparison_static(baseline_data, rl_data)")
