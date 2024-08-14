# 以下はexperiment_runner.pyの内容
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from data_preprocessing import preprocess_data
from synthetic_data_generator import generate_dataset_with_labels
from decentralized_detection import decentralized_detection
from performance_evaluation import calculate_performance_metrics
import matplotlib.pyplot as plt

def run_synthetic_data_experiment(config: Dict) -> Dict:
    data, labels = generate_dataset_with_labels(
        n_channels=config['n_channels'],
        n_samples=config['n_samples'],
        mean=config['mean'],
        variance=config['variance'],
        theta=config['theta'],
        change_point=config['change_point'],
        new_mean=config['mean'] + 5 * np.sqrt(config['variance']),
        new_variance=config['variance'] * 5
    )

    change_detected, change_index, cusum_values = decentralized_detection(
        data=data,
        reference_window_size=config['reference_window_size'],
        test_window_size=config['test_window_size'],
        method=config['detection_method'],
        threshold=config['threshold'] * 0.1,
        voting_threshold=config['voting_threshold'],
        aggregation_method=config['aggregation_method']
    )

    metrics = calculate_performance_metrics(
        true_change_point=config['change_point'],
        detected_change_point=change_index,
        n_samples=config['n_samples']
    )

    visualize_data(data, config['change_point'], change_index, 'Synthetic Data')
    plot_cusum_statistic(cusum_values, config['threshold'] * 0.1, config['change_point'])

    return {
        'change_detected': change_detected,
        'detected_change_point': change_index,
        'metrics': metrics,
        'cusum_values': cusum_values
    }
    
def run_real_data_experiment(config: Dict) -> Dict:
    """
    Run experiment on real data.

    Args:
    config (Dict): Configuration parameters for the experiment.

    Returns:
    Dict: Results of the experiment.
    """
    data, attack_start, attack_end = preprocess_data(
        file_path=config['file_path'],
        bus_numbers=config['bus_numbers'],
        num_samples=config['num_samples']
    )

    change_detected, change_index, _ = decentralized_detection(
        data=data.drop(['Week', 'Label'], axis=1).values.T,
        reference_window_size=config['reference_window_size'],
        test_window_size=config['test_window_size'],
        method=config['detection_method'],
        threshold=config['threshold'],
        voting_threshold=config['voting_threshold'],
        aggregation_method=config['aggregation_method']
    )

    metrics = calculate_performance_metrics(
        true_change_point=attack_start,
        detected_change_point=change_index,
        n_samples=config['num_samples']
    )

    return {
        'change_detected': change_detected,
        'detected_change_point': change_index,
        'metrics': metrics
    }

def run_parameter_sensitivity_analysis(base_config: Dict, param_ranges: Dict) -> Dict:
    """
    Run parameter sensitivity analysis.

    Args:
    base_config (Dict): Base configuration for the experiment.
    param_ranges (Dict): Ranges of parameters to analyze.

    Returns:
    Dict: Results of the sensitivity analysis.
    """
    results = {}

    for param, values in param_ranges.items():
        param_results = []
        for value in values:
            config = base_config.copy()
            config[param] = value
            result = run_synthetic_data_experiment(config)
            param_results.append({
                'value': value,
                'metrics': result['metrics']
            })
        results[param] = param_results

    return results

def visualize_data(data, true_change_point, detected_change_point, title):
    plt.figure(figsize=(12, 6))
    for i in range(data.shape[0]):
        plt.plot(data[i], alpha=0.5, label=f'Channel {i+1}')
    plt.axvline(x=true_change_point, color='r', linestyle='--', label='True Change Point')
    plt.axvline(x=detected_change_point, color='g', linestyle='--', label='Detected Change Point')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_cusum_statistic(cusum_values, threshold, change_point):
    plt.figure(figsize=(12, 6))
    plt.plot(cusum_values)
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.axvline(x=change_point, color='g', linestyle='--', label='Change Point')
    plt.title('CUSUM Statistic Over Time')
    plt.xlabel('Sample')
    plt.ylabel('CUSUM Statistic')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Configuration for synthetic data experiment
    synthetic_config = {
        'n_channels': 8,
        'n_samples': 1000,
        'mean': 0,
        'variance': 1,
        'theta': 0.5,
        'change_point': 500,
        'new_mean': 2,
        'new_variance': 1.5,
        'reference_window_size': 100,
        'test_window_size': 50,
        'detection_method': 'B',
        'threshold': 5,
        'voting_threshold': 0.5,
        'aggregation_method': 'mean'
    }

    # Run synthetic data experiment
    synthetic_results = run_synthetic_data_experiment(synthetic_config)
    print("Synthetic Data Results:")
    print(synthetic_results)

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

    # Run real data experiment
    real_results = run_real_data_experiment(real_config)
    print("\nReal Data Results:")
    print(real_results)

    # Parameter sensitivity analysis
    param_ranges = {
        'reference_window_size': [50, 100, 150, 200],
        'test_window_size': [25, 50, 75, 100],
        'threshold': [3, 5, 7, 9],
        'voting_threshold': [0.3, 0.5, 0.7, 0.9]
    }

    sensitivity_results = run_parameter_sensitivity_analysis(synthetic_config, param_ranges)
    print("\nParameter Sensitivity Analysis Results:")
    print(sensitivity_results)