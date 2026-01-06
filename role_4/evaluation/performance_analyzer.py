 

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyze and compare agent performance."""
    
    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def compare_agents(self, original_metrics: Dict, advanced_metrics: Dict):
        """
        Compare original vs advanced agent.
        
        Args:
            original_metrics: Training metrics from original agent
            advanced_metrics: Training metrics from advanced agent
        """
        logger.info("\n" + "="*80)
        logger.info("COMPARING ORIGINAL VS ADVANCED AGENT")
        logger.info("="*80)
        
        comparison = {
            'convergence': self._analyze_convergence(original_metrics, advanced_metrics),
            'stability': self._analyze_stability(original_metrics, advanced_metrics),
            'revenue': self._analyze_revenue(original_metrics, advanced_metrics),
            'exploration': self._analyze_exploration(original_metrics, advanced_metrics)
        }
        
        # Log comparison
        logger.info("\nCONVERGENCE ANALYSIS:")
        logger.info(f"  Original - Episodes to 80% best reward: {comparison['convergence']['original_episodes']}")
        logger.info(f"  Advanced - Episodes to 80% best reward: {comparison['convergence']['advanced_episodes']}")
        improvement = (1 - comparison['convergence']['advanced_episodes'] / 
                      comparison['convergence']['original_episodes']) * 100
        logger.info(f"  Speedup: {improvement:.1f}% faster")
        
        logger.info("\nSTABILITY ANALYSIS:")
        logger.info(f"  Original - Reward variance: {comparison['stability']['original_var']:.2f}")
        logger.info(f"  Advanced - Reward variance: {comparison['stability']['advanced_var']:.2f}")
        logger.info(f"  Improvement: {(1-comparison['stability']['advanced_var']/comparison['stability']['original_var'])*100:.1f}% lower variance")
        
        logger.info("\nREVENUE ANALYSIS:")
        logger.info(f"  Original - Avg revenue: ${comparison['revenue']['original_avg']:.2f}")
        logger.info(f"  Advanced - Avg revenue: ${comparison['revenue']['advanced_avg']:.2f}")
        revenue_gain = (comparison['revenue']['advanced_avg'] / comparison['revenue']['original_avg'] - 1) * 100
        logger.info(f"  Improvement: ${comparison['revenue']['advanced_avg'] - comparison['revenue']['original_avg']:.2f} ({revenue_gain:.1f}%)")
        
        return comparison
    
    def _analyze_convergence(self, original: Dict, advanced: Dict) -> Dict:
        """Analyze convergence speed."""
        original_rewards = original.get('rewards', [])
        advanced_rewards = advanced.get('rewards', [])
        
        # Find when agent reaches 80% of best performance
        original_best = max(original_rewards) * 0.8
        advanced_best = max(advanced_rewards) * 0.8
        
        original_ep = next((i for i, r in enumerate(original_rewards) if r >= original_best), len(original_rewards))
        advanced_ep = next((i for i, r in enumerate(advanced_rewards) if r >= advanced_best), len(advanced_rewards))
        
        return {
            'original_episodes': original_ep,
            'advanced_episodes': advanced_ep,
            'original_best': max(original_rewards),
            'advanced_best': max(advanced_rewards)
        }
    
    def _analyze_stability(self, original: Dict, advanced: Dict) -> Dict:
        """Analyze reward stability."""
        original_rewards = np.array(original.get('rewards', []))
        advanced_rewards = np.array(advanced.get('rewards', []))
        
        return {
            'original_var': float(np.var(original_rewards[-50:])),
            'advanced_var': float(np.var(advanced_rewards[-50:])),
            'original_std': float(np.std(original_rewards[-50:])),
            'advanced_std': float(np.std(advanced_rewards[-50:]))
        }
    
    def _analyze_revenue(self, original: Dict, advanced: Dict) -> Dict:
        """Analyze revenue optimization."""
        original_revenues = original.get('revenues', [])
        advanced_revenues = advanced.get('revenues', [])
        
        return {
            'original_avg': float(np.mean(original_revenues[-30:])) if original_revenues else 0,
            'advanced_avg': float(np.mean(advanced_revenues[-30:])) if advanced_revenues else 0,
            'original_max': float(max(original_revenues)) if original_revenues else 0,
            'advanced_max': float(max(advanced_revenues)) if advanced_revenues else 0
        }
    
    def _analyze_exploration(self, original: Dict, advanced: Dict) -> Dict:
        """Analyze price exploration."""
        original_prices = original.get('prices', [])
        advanced_prices = advanced.get('prices', [])
        
        return {
            'original_std': float(np.std(original_prices)) if original_prices else 0,
            'advanced_std': float(np.std(advanced_prices)) if advanced_prices else 0,
            'original_mean': float(np.mean(original_prices)) if original_prices else 0,
            'advanced_mean': float(np.mean(advanced_prices)) if advanced_prices else 0
        }
    
    def analyze_robustness(self, evaluation_results: Dict) -> Dict:
        """
        Analyze agent robustness against different conditions.
        
        Args:
            evaluation_results: Evaluation results from pipeline
        """
        logger.info("\n" + "="*80)
        logger.info("ROBUSTNESS ANALYSIS")
        logger.info("="*80)
        
        agent_results = evaluation_results.get('Advanced RL Agent', {})
        baseline_results = {k: v for k, v in evaluation_results.items() 
                          if k != 'Advanced RL Agent'}
        
        logger.info("\nAGENT ROBUSTNESS METRICS:")
        logger.info(f"  Revenue: ${agent_results.get('avg_revenue', 0):.2f}")
        logger.info(f"  Occupancy: {agent_results.get('avg_occupancy', 0):.2%}")
        logger.info(f"  Price Volatility: ${agent_results.get('price_volatility', 0):.2f}")
        
        logger.info("\nCOMPARISON WITH BASELINES:")
        best_baseline_revenue = max(v.get('avg_revenue', 0) for v in baseline_results.values())
        improvement = (agent_results.get('avg_revenue', 0) / best_baseline_revenue - 1) * 100
        logger.info(f"  Best baseline: ${best_baseline_revenue:.2f}")
        logger.info(f"  Agent advantage: {improvement:+.1f}%")
        
        return {
            'agent': agent_results,
            'baselines': baseline_results,
            'advantage_over_best': improvement
        }
    
    def generate_report(self, comparison: Dict, robustness: Dict, original_metrics: Dict, 
                       advanced_metrics: Dict):
        """Generate comprehensive analysis report."""
        report = {
            'title': 'Advanced RL Parking Pricing - Performance Analysis',
            'timestamp': str(Path('.')),
            'sections': {
                'convergence': comparison['convergence'],
                'stability': comparison['stability'],
                'revenue': comparison['revenue'],
                'robustness': robustness
            }
        }
        
        report_file = self.output_dir / "performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Report saved to {report_file}")
        
        return report
    
    def plot_comparison(self, original_metrics: Dict, advanced_metrics: Dict):
        """Create comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Advanced vs Original Agent Performance', fontsize=16, fontweight='bold')
        
        # Rewards
        ax = axes[0, 0]
        ax.plot(original_metrics.get('rewards', []), label='Original', alpha=0.7)
        ax.plot(advanced_metrics.get('rewards', []), label='Advanced', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Prices
        ax = axes[0, 1]
        ax.plot(original_metrics.get('prices', []), label='Original', alpha=0.7)
        ax.plot(advanced_metrics.get('prices', []), label='Advanced', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Price ($)')
        ax.set_title('Average Pricing Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Occupancies
        ax = axes[1, 0]
        ax.plot(np.array(original_metrics.get('occupancies', [])) * 100, label='Original', alpha=0.7)
        ax.plot(np.array(advanced_metrics.get('occupancies', [])) * 100, label='Advanced', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Occupancy (%)')
        ax.set_title('Average Occupancy Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Revenues
        ax = axes[1, 1]
        ax.plot(original_metrics.get('revenues', []), label='Original', alpha=0.7)
        ax.plot(advanced_metrics.get('revenues', []), label='Advanced', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Revenue ($)')
        ax.set_title('Revenue Generation Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / "performance_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Comparison plots saved to {plot_file}")
        plt.close()


def main():
    """Example analysis pipeline."""
    analyzer = PerformanceAnalyzer()
    
    # Load metrics (placeholder)
    logger.info("Performance analyzer initialized. Use analyze_agents() and analyze_robustness().")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
