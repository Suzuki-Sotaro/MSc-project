# 以下はmain.pyのコードです。
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiment_runner import run_synthetic_data_experiment, run_real_data_experiment, run_parameter_sensitivity_analysis
from performance_evaluation import plot_roc_curve, plot_detection_delay_distribution, plot_parameter_sensitivity
from config import synthetic_config, real_config, param_ranges

def run_all_experiments():
    """
    Run all experiments and analyze results.
    """
    print("Running Synthetic Data Experiment...")
    synthetic_results = run_synthetic_data_experiment(synthetic_config)
    print("Synthetic Data Results:")
    print(synthetic_results)

    print("\nRunning Real Data Experiment...")
    real_results = run_real_data_experiment(real_config)
    print("Real Data Results:")
    print(real_results)

    print("\nRunning Parameter Sensitivity Analysis...")
    sensitivity_results = run_parameter_sensitivity_analysis(synthetic_config, param_ranges)
    print("Parameter Sensitivity Analysis Results:")
    print(sensitivity_results)

    return synthetic_results, real_results, sensitivity_results

def analyze_results(synthetic_results, real_results, sensitivity_results):
    """
    Analyze and visualize the results.
    """
    # Analyze synthetic data results
    print("\nSynthetic Data Analysis:")
    print(f"Change detected: {synthetic_results['change_detected']}")
    print(f"Detected change point: {synthetic_results['detected_change_point']}")
    print(f"Performance metrics: {synthetic_results['metrics']}")

    # Analyze real data results
    print("\nReal Data Analysis:")
    print(f"Change detected: {real_results['change_detected']}")
    print(f"Detected change point: {real_results['detected_change_point']}")
    print(f"Performance metrics: {real_results['metrics']}")

    print("\nParameter Sensitivity Analysis:")
    for param, results in sensitivity_results.items():
        print(f"\n{param}:")
        for result in results:
            print(f"  Value: {result['value']}")
            print(f"    Detection Delay: {result['metrics']['detection_delay']}")
            print(f"    False Alarm Rate: {result['metrics']['false_alarm_rate']}")
            print(f"    Detection Accuracy: {result['metrics']['detection_accuracy']}")
            print(f"    Missed Detection: {result['metrics']['missed_detection']}")

    # Generate ROC curve (example - you may need to adjust this based on your actual results)
    true_positives = [result['metrics']['detection_accuracy'] for result in sensitivity_results['threshold']]
    false_positives = [result['metrics']['false_alarm_rate'] for result in sensitivity_results['threshold']]
    plot_roc_curve(true_positives, false_positives)

    # Plot detection delay distribution (example - you may need to adjust this based on your actual results)
    delays = [result['metrics']['detection_delay'] for result in sensitivity_results['threshold'] if result['metrics']['detection_delay'] != np.inf]
    plot_detection_delay_distribution(delays)
    

def main():
    """
    Main function to run all experiments and analyze results.
    """
    synthetic_results, real_results, sensitivity_results = run_all_experiments()
    analyze_results(synthetic_results, real_results, sensitivity_results)

if __name__ == "__main__":
    main()