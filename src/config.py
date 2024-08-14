# 以下はconfig.pyのコード
# Configuration for synthetic data experiment
synthetic_config = {
    'n_channels': 8,
    'n_samples': 1000,
    'mean': 0,
    'variance': 1,
    'theta': 0.5,
    'change_point': 200,
    'new_mean': 2,
    'new_variance': 1.5,
    'reference_window_size': 100,
    'test_window_size': 50,
    'detection_method': 'B',
    'threshold': 5,
    'voting_threshold': 0.5,
    'aggregation_method': 'mean'
}

# Configuration for real data experiment
real_config = {
    'file_path': './data/LMP.csv',
    'bus_numbers': [115, 116, 117, 118, 119, 121, 135, 139],
    'num_samples': 855,
    'reference_window_size': 100,
    'test_window_size': 50,
    'detection_method': 'B',
    'threshold': 5,
    'voting_threshold': 0.5,
    'aggregation_method': 'mean'
}

# Parameter ranges for sensitivity analysis
param_ranges = {
    'reference_window_size': [1, 5, 10, 20, 50],
    'test_window_size': [1, 2, 5, 10, 20],
    'threshold': [0.001, 0.01, 0.1, 1, 10],
    'voting_threshold': [0.01, 0.1, 0.3, 0.5, 0.9]
}

# General configuration
general_config = {
    'random_seed': 42,
    'num_monte_carlo_runs': 100,
    'confidence_level': 0.95
}

# Visualization configuration
viz_config = {
    'figure_size': (12, 8),
    'font_size': 12,
    'line_width': 2,
    'marker_size': 6
}

# File paths
file_paths = {
    'results_dir': './results/',
    'figures_dir': './figures/',
    'log_file': './experiment.log'
}

# Performance metrics configuration
metrics_config = {
    'detection_delay_tolerance': 10,  # samples
    'false_alarm_rate_threshold': 0.05
}

# Method-specific configurations
method_a_config = {
    'voting_schemes': ['at_least_one', 'all_buses', 'majority']
}

method_b_config = {
    'aggregation_functions': ['mean', 'median', 'mad'],
    'sink_threshold_options': ['average', 'minimum', 'maximum', 'median']
}