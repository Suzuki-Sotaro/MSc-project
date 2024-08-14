import numpy as np
import pandas as pd
import logging
from typing import List, Tuple

from data_loader import load_lmp_data, prepare_data_for_change_detection
from change_detection import gem_change_detection, qq_change_detection
from distributed_methods import method_a, method_b_average, method_b_median, method_b_mad
from utils import calculate_detection_delay, calculate_false_alarm_rate, evaluate_performance
from experiments import run_all_experiments
from visualization import visualize_results
from synthetic_data import create_synthetic_dataset
import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, filename=config.LOG_FILE, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_experiment_on_data(data: np.ndarray, true_changes: List[int]) -> pd.DataFrame:
    results = []

    # GEM method
    gem_changes, gem_stats = gem_change_detection(data, config.GEM_K, config.GEM_ALPHA, config.GEM_H)
    results.append(evaluate_method('GEM', true_changes, gem_changes, len(data)))

    # Q-Q method
    qq_changes, qq_distances = qq_change_detection(data, config.QQ_WINDOW_SIZE, config.QQ_H)
    results.append(evaluate_method('Q-Q', true_changes, qq_changes, len(data)))

    # Method A
    for p in config.METHOD_A_P_VALUES:
        a_changes, _ = method_a(data, p)
        results.append(evaluate_method(f'Method A (p={p})', true_changes, a_changes, len(data)))

    # Method B
    local_thresholds = np.random.uniform(1, 3, size=data.shape[1])
    for h_method in config.METHOD_B_H_METHODS:
        for agg_type, method_func in [('average', method_b_average), ('median', method_b_median), ('mad', method_b_mad)]:
            if agg_type == 'mad':
                b_changes, _ = method_func(data, h_method, local_thresholds, config.METHOD_B_MAD_THRESHOLD)
            else:
                b_changes, _ = method_func(data, h_method, local_thresholds)
            results.append(evaluate_method(f'Method B ({h_method}, {agg_type})', true_changes, b_changes, len(data)))

    return pd.DataFrame(results)

def evaluate_method(method_name: str, true_changes: List[int], detected_changes: List[int], data_length: int) -> dict:
    """
    Evaluate method results.

    Args:
    method_name (str): Name of the method
    true_changes (List[int]): List of true change points
    detected_changes (List[int]): List of detected change points
    data_length (int): Length of the data

    Returns:
    dict: Evaluation metrics
    """
    delay = calculate_detection_delay(true_changes, detected_changes)
    far = calculate_false_alarm_rate(true_changes, detected_changes, data_length)
    precision, recall, f1, _ = evaluate_performance(true_changes, detected_changes, data_length)
    return {
        'method': method_name,
        'detected_changes': detected_changes,
        'avg_delay': delay,
        'false_alarm_rate': far,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():
    print("Starting main execution...")
    logging.info("Starting main execution")

    # Load real data
    print("Loading real LMP data...")
    real_df = load_lmp_data(config.DATA_PATH, config.SELECTED_BUSES)
    real_data, real_labels = prepare_data_for_change_detection(real_df, config.WINDOW_SIZE, config.STEP_SIZE)
    real_changes = np.where(np.diff(real_labels) != 0)[0].tolist()
    print(f"Real data loaded. Shape: {real_data.shape}")
    print(f"Real change points: {real_changes}")

    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_df, synthetic_changes = create_synthetic_dataset()
    synthetic_data, synthetic_labels = prepare_data_for_change_detection(synthetic_df, config.WINDOW_SIZE, config.STEP_SIZE)
    print(f"Synthetic data generated. Shape: {synthetic_data.shape}")
    print(f"Synthetic change points: {synthetic_changes}")

    # Run experiments on real data
    print("Running experiments on real data...")
    real_results = run_experiment_on_data(real_data, real_changes)
    print("Experiments on real data completed.")

    # Run experiments on synthetic data
    print("Running experiments on synthetic data...")
    synthetic_results = run_experiment_on_data(synthetic_data, synthetic_changes)
    print("Experiments on synthetic data completed.")

    # Visualize results
    print("Visualizing results...")
    visualize_results(real_data, real_results, real_changes)
    visualize_results(synthetic_data, synthetic_results, synthetic_changes)
    print("Visualization completed.")

    # Save results
    print("Saving results...")
    real_results.to_csv(f"{config.RESULTS_DIR}/real_data_results.csv", index=False)
    synthetic_results.to_csv(f"{config.RESULTS_DIR}/synthetic_data_results.csv", index=False)
    print("Results saved.")

    print("Main execution completed.")
    logging.info("Main execution completed")

if __name__ == "__main__":
    main()