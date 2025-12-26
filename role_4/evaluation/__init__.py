"""
Evaluation Package
===================

Provides baseline strategies, evaluation framework, and visualization tools
for comparing RL agents against traditional pricing methods.

Modules:
- baselines: Baseline pricing strategies and evaluation runner
- metrics: Metric computation functions
- visualise: Visualization and plotting functions
- video_recording: Animation and comparison generation
- run_evaluation: Main script to run complete evaluation
"""

from .baselines import (
    PricingStrategy,
    FixedPriceStrategy,
    TimeBasedStrategy,
    RandomPriceStrategy,
    DemandBasedStrategy,
    evaluate_strategy,
    evaluate_rl_agent,
    compare_strategies,
    get_default_baselines,
    generate_comparison_table,
    save_results,
    EvaluationResult,
)

from .metrics import (
    compute_revenue_metrics,
    compute_occupancy_metrics,
    compute_price_volatility,
    compute_all_metrics,
)

from .visualise import (
    plot_training_progress,
    plot_revenue_comparison,
    plot_occupancy_comparison,
    plot_price_volatility,
    plot_strategy_comparison_episode,
    create_summary_dashboard,
)

from .video_recording import (
    create_comparison_static,
    create_comparison_animation,
    collect_episode_data,
)

__all__ = [
    # Strategies
    'PricingStrategy',
    'FixedPriceStrategy',
    'TimeBasedStrategy',
    'RandomPriceStrategy',
    'DemandBasedStrategy',
    # Evaluation
    'evaluate_strategy',
    'evaluate_rl_agent',
    'compare_strategies',
    'get_default_baselines',
    'generate_comparison_table',
    'save_results',
    'EvaluationResult',
    # Metrics
    'compute_revenue_metrics',
    'compute_occupancy_metrics',
    'compute_price_volatility',
    'compute_all_metrics',
    # Plots
    'plot_training_progress',
    'plot_revenue_comparison',
    'plot_occupancy_comparison',
    'plot_price_volatility',
    'plot_strategy_comparison_episode',
    'create_summary_dashboard',
    # Video
    'create_comparison_static',
    'create_comparison_animation',
    'collect_episode_data',
]
