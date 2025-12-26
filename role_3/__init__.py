"""
ROLE 3: DEMAND MODELING PACKAGE
=================================

Provides demand forecasting for dynamic parking pricing.

Modules:
- demand_model: Core demand prediction models
- data_processing: Data loading and preprocessing

Usage:
    from role_3 import DemandModelFactory, DataProcessor
    
    # Create a rule-based demand model
    from role_3.demand_model import DemandModelConfig
    config = DemandModelConfig(model_type='rule_based')
    model = DemandModelFactory.create(config)
    
    # Load and process data
    from role_3.data_processing import DataProcessor
    processor = DataProcessor()
    data = processor.load_csv('parking_data.csv')
    processed_data, metadata = processor.preprocess()
"""

from .demand_model import (
    DemandModelFactory,
    DemandModelConfig,
    BaseDemandModel,
    PolynomialDemandModel,
    RuleBasedDemandModel,
    NeuralDemandModel,
    generate_synthetic_demand_data,
    evaluate_demand_model
)

from .data_processing import (
    DataProcessor,
    DataConfig,
    generate_synthetic_parking_dataset,
    load_or_create_dataset
)

__all__ = [
    # Demand modeling
    'DemandModelFactory',
    'DemandModelConfig',
    'BaseDemandModel',
    'PolynomialDemandModel',
    'RuleBasedDemandModel',
    'NeuralDemandModel',
    'generate_synthetic_demand_data',
    'evaluate_demand_model',
    # Data processing
    'DataProcessor',
    'DataConfig',
    'generate_synthetic_parking_dataset',
    'load_or_create_dataset',
]

__version__ = '1.0.0'
__author__ = 'RL Parking Pricing Team'
