"""
VISUALIZATION & ANALYSIS DASHBOARD
===================================

Advanced visualization and analysis tools for comparing agents and understanding performance.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class VisualizationDashboard:
    """Create comprehensive visualization dashboards."""
    
    def __init__(self, output_dir: str = "visualization_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.style = self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        return {
            'original': '#1f77b4',
            'advanced': '#ff7f0e',
            'baseline': '#2ca02c',
            'figsize': (14, 10),
            'dpi': 150
        }
    
    def plot_convergence_comparison(self, original_rewards: List, advanced_rewards: List,
                                    window: int = 10):
        """
        Plot convergence comparison with moving averages.
        
        Args:
            original_rewards: Original agent rewards
            advanced_rewards: Advanced agent rewards
            window: Moving average window
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.style['figsize'])
        
        # Raw rewards
        episodes = np.arange(len(original_rewards))
        ax1.plot(episodes, original_rewards, alpha=0.3, label='Original (raw)', 
                color=self.style['original'], linewidth=0.5)
        ax1.plot(episodes, advanced_rewards, alpha=0.3, label='Advanced (raw)',
                color=self.style['advanced'], linewidth=0.5)
        
        # Moving averages
        if len(original_rewards) >= window:
            orig_ma = self._moving_average(original_rewards, window)
            adv_ma = self._moving_average(advanced_rewards, window)
            ax1.plot(range(len(orig_ma)), orig_ma, label=f'Original (MA-{window})',
                    color=self.style['original'], linewidth=2)
            ax1.plot(range(len(adv_ma)), adv_ma, label=f'Advanced (MA-{window})',
                    color=self.style['advanced'], linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Convergence Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative rewards
        orig_cum = np.cumsum(original_rewards)
        adv_cum = np.cumsum(advanced_rewards)
        ax2.plot(episodes, orig_cum, label='Original', color=self.style['original'], linewidth=2)
        ax2.plot(episodes, adv_cum, label='Advanced', color=self.style['advanced'], linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('Cumulative Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        file_path = self.output_dir / "convergence_comparison.png"
        plt.savefig(file_path, dpi=self.style['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Convergence plot saved to {file_path}")
    
    def plot_metrics_dashboard(self, metrics_dict: Dict[str, List], agent_name: str = "Agent"):
        """
        Create comprehensive metrics dashboard.
        
        Args:
            metrics_dict: Dictionary with metric lists
            agent_name: Name of agent for title
        """
        fig, axes = plt.subplots(2, 3, figsize=self.style['figsize'])
        fig.suptitle(f'{agent_name} - Metrics Dashboard', fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('rewards', 'Reward per Episode', axes[0, 0]),
            ('prices', 'Average Price per Episode', axes[0, 1]),
            ('occupancies', 'Average Occupancy per Episode', axes[0, 2]),
            ('revenues', 'Revenue per Episode', axes[1, 0]),
            ('entropy', 'Entropy Decay', axes[1, 1]),
            ('value_loss', 'Value Network Loss', axes[1, 2])
        ]
        
        for metric_key, title, ax in metrics_to_plot:
            if metric_key in metrics_dict and metrics_dict[metric_key]:
                data = metrics_dict[metric_key]
                ax.plot(data, color=self.style['advanced'], alpha=0.7)
                
                # Add moving average
                if len(data) >= 10:
                    ma = self._moving_average(data, 10)
                    ax.plot(range(len(ma)), ma, color='red', linewidth=2, label='MA-10')
                    ax.legend()
                
                ax.set_title(title)
                ax.set_xlabel('Episode')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        file_path = self.output_dir / f"{agent_name.lower()}_dashboard.png"
        plt.savefig(file_path, dpi=self.style['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Metrics dashboard saved to {file_path}")
    
    def plot_statistics_comparison(self, original_metrics: Dict, advanced_metrics: Dict):
        """
        Plot statistical comparison.
        
        Args:
            original_metrics: Original agent metrics
            advanced_metrics: Advanced agent metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=self.style['figsize'])
        fig.suptitle('Statistical Comparison', fontsize=16, fontweight='bold')
        
        # Reward distribution
        ax = axes[0, 0]
        orig_rewards = original_metrics.get('rewards', [])
        adv_rewards = advanced_metrics.get('rewards', [])
        ax.hist([orig_rewards, adv_rewards], label=['Original', 'Advanced'], bins=20)
        ax.set_title('Reward Distribution')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Price distribution
        ax = axes[0, 1]
        orig_prices = original_metrics.get('prices', [])
        adv_prices = advanced_metrics.get('prices', [])
        ax.hist([orig_prices, adv_prices], label=['Original', 'Advanced'], bins=20)
        ax.set_title('Price Distribution')
        ax.set_xlabel('Average Price ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Occupancy distribution
        ax = axes[1, 0]
        orig_occ = np.array(original_metrics.get('occupancies', [])) * 100
        adv_occ = np.array(advanced_metrics.get('occupancies', [])) * 100
        ax.hist([orig_occ, adv_occ], label=['Original', 'Advanced'], bins=20)
        ax.set_title('Occupancy Distribution')
        ax.set_xlabel('Occupancy (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
        REWARD METRICS:
        Original: μ={np.mean(orig_rewards):.2f}, σ={np.std(orig_rewards):.2f}
        Advanced: μ={np.mean(adv_rewards):.2f}, σ={np.std(adv_rewards):.2f}
        
        PRICE METRICS:
        Original: μ=${np.mean(orig_prices):.2f}, σ=${np.std(orig_prices):.2f}
        Advanced: μ=${np.mean(adv_prices):.2f}, σ=${np.std(adv_prices):.2f}
        
        OCCUPANCY METRICS:
        Original: μ={np.mean(orig_occ):.1f}%, σ={np.std(orig_occ):.1f}%
        Advanced: μ={np.mean(adv_occ):.1f}%, σ={np.std(adv_occ):.1f}%
        
        IMPROVEMENT:
        Reward: {(np.mean(adv_rewards)/np.mean(orig_rewards)-1)*100:+.1f}%
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        file_path = self.output_dir / "statistics_comparison.png"
        plt.savefig(file_path, dpi=self.style['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Statistics comparison saved to {file_path}")
    
    @staticmethod
    def _moving_average(data: List, window: int) -> np.ndarray:
        """Compute moving average."""
        return np.convolve(data, np.ones(window)/window, mode='valid')


class PerformanceAnalyzer:
    """Advanced performance analysis."""
    
    @staticmethod
    def convergence_analysis(rewards: List[float]) -> Dict:
        """
        Analyze convergence characteristics.
        
        Returns:
            Dictionary with convergence metrics
        """
        rewards = np.array(rewards)
        best = np.max(rewards)
        
        # Find convergence point (when reaches 80% of best)
        conv_target = best * 0.8
        conv_episode = next((i for i, r in enumerate(rewards) if r >= conv_target), len(rewards))
        
        return {
            'best_reward': float(best),
            'convergence_episode': conv_episode,
            'convergence_efficiency': float(best / max(1, conv_episode)),
            'final_reward': float(rewards[-1]),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'reward_variance': float(np.var(rewards[-min(10, len(rewards)):]))
        }
    
    @staticmethod
    def stability_analysis(prices: List[float]) -> Dict:
        """
        Analyze pricing stability.
        
        Returns:
            Dictionary with stability metrics
        """
        prices = np.array(prices)
        price_changes = np.diff(prices)
        
        return {
            'avg_price': float(np.mean(prices)),
            'std_price': float(np.std(prices)),
            'price_range': (float(np.min(prices)), float(np.max(prices))),
            'avg_price_change': float(np.mean(np.abs(price_changes))),
            'max_price_change': float(np.max(np.abs(price_changes))),
            'price_stability': float(1.0 / (1.0 + np.mean(np.abs(price_changes))))
        }
    
    @staticmethod
    def occupancy_analysis(occupancies: List[float], target: float = 0.75) -> Dict:
        """
        Analyze occupancy control.
        
        Returns:
            Dictionary with occupancy metrics
        """
        occupancies = np.array(occupancies)
        
        return {
            'avg_occupancy': float(np.mean(occupancies)),
            'target_occupancy': float(target),
            'occupancy_error': float(np.mean(np.abs(occupancies - target))),
            'occupancy_std': float(np.std(occupancies)),
            'time_at_target': float(np.mean((occupancies >= target - 0.05) & (occupancies <= target + 0.05))),
            'occupancy_range': (float(np.min(occupancies)), float(np.max(occupancies)))
        }


def compare_agents(original_dir: str, advanced_dir: str, output_dir: str = "analysis_results"):
    """
    Complete comparison analysis of two agents.
    
    Args:
        original_dir: Path to original agent results
        advanced_dir: Path to advanced agent results
        output_dir: Output directory for analysis
    """
    dashboard = VisualizationDashboard(output_dir)
    analyzer = PerformanceAnalyzer()
    
    # Load metrics
    original_metrics = json.load(open(Path(original_dir) / "training_metrics.json"))
    advanced_metrics = json.load(open(Path(advanced_dir) / "advanced_metrics.json"))
    
    # Create visualizations
    dashboard.plot_convergence_comparison(
        original_metrics.get('rewards', []),
        advanced_metrics.get('rewards', [])
    )
    
    dashboard.plot_metrics_dashboard(advanced_metrics, "Advanced Agent")
    dashboard.plot_statistics_comparison(original_metrics, advanced_metrics)
    
    # Analysis
    orig_conv = analyzer.convergence_analysis(original_metrics.get('rewards', []))
    adv_conv = analyzer.convergence_analysis(advanced_metrics.get('rewards', []))
    
    orig_stab = analyzer.stability_analysis(original_metrics.get('prices', []))
    adv_stab = analyzer.stability_analysis(advanced_metrics.get('prices', []))
    
    orig_occ = analyzer.occupancy_analysis(original_metrics.get('occupancies', []))
    adv_occ = analyzer.occupancy_analysis(advanced_metrics.get('occupancies', []))
    
    # Save analysis
    analysis = {
        'convergence': {'original': orig_conv, 'advanced': adv_conv},
        'stability': {'original': orig_stab, 'advanced': adv_stab},
        'occupancy': {'original': orig_occ, 'advanced': adv_occ}
    }
    
    analysis_file = Path(output_dir) / "detailed_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"✓ Analysis saved to {analysis_file}")
    return analysis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("Visualization Dashboard initialized.")
    print("Use compare_agents() to create comprehensive analysis.")
