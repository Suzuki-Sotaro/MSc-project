import numpy as np
import pandas as pd
from typing import Dict, Any
from data_loader import load_lmp_data, prepare_data_for_change_detection
from change_detection import gem_change_detection, qq_change_detection
from distributed_methods import method_a, method_b_average, method_b_median, method_b_mad
from utils import calculate_detection_delay, calculate_false_alarm_rate, evaluate_performance
import config
import logging

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, filename=config.LOG_FILE, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_gem_experiment(data: np.ndarray, true_changes: list) -> Dict[str, Any]:
    """
    Run an experiment using the GEM change detection method.

    Args:
    data (np.ndarray): Input data
    true_changes (list): List of true change points

    Returns:
    Dict[str, Any]: Results of the experiment
    """
    print("Running GEM experiment...")
    logging.info("Starting GEM experiment")
    
    changes, stats = gem_change_detection(data, config.GEM_K, config.GEM_ALPHA, config.GEM_H)
    
    delay = calculate_detection_delay(true_changes, changes)
    far = calculate_false_alarm_rate(true_changes, changes, len(data))
    precision, recall, f1, _ = evaluate_performance(true_changes, changes, len(data))
    
    results = {
        "method": "GEM",
        "detected_changes": changes,
        "avg_delay": delay,
        "false_alarm_rate": far,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    print(f"GEM experiment results: {results}")
    logging.info(f"GEM experiment completed. Results: {results}")
    return results

def run_qq_experiment(data: np.ndarray, true_changes: list) -> Dict[str, Any]:
    """
    Run an experiment using the Q-Q distance change detection method.

    Args:
    data (np.ndarray): Input data
    true_changes (list): List of true change points

    Returns:
    Dict[str, Any]: Results of the experiment
    """
    print("Running Q-Q experiment...")
    logging.info("Starting Q-Q experiment")
    
    changes, distances = qq_change_detection(data, config.QQ_WINDOW_SIZE, config.QQ_H)
    
    delay = calculate_detection_delay(true_changes, changes)
    far = calculate_false_alarm_rate(true_changes, changes, len(data))
    precision, recall, f1, _ = evaluate_performance(true_changes, changes, len(data))
    
    results = {
        "method": "Q-Q",
        "detected_changes": changes,
        "avg_delay": delay,
        "false_alarm_rate": far,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    print(f"Q-Q experiment results: {results}")
    logging.info(f"Q-Q experiment completed. Results: {results}")
    return results

def run_method_a_experiment(data: np.ndarray, true_changes: list) -> Dict[str, Any]:
    """
    Run an experiment using Method A for distributed change detection.

    Args:
    data (np.ndarray): Input data
    true_changes (list): List of true change points

    Returns:
    Dict[str, Any]: Results of the experiment
    """
    print("Running Method A experiment...")
    logging.info("Starting Method A experiment")
    
    results = {}
    for p in config.METHOD_A_P_VALUES:
        print(f"Running Method A with p = {p}")
        changes, _ = method_a(data, p)
        
        delay = calculate_detection_delay(true_changes, changes)
        far = calculate_false_alarm_rate(true_changes, changes, len(data))
        precision, recall, f1, _ = evaluate_performance(true_changes, changes, len(data))
        
        results[p] = {
            "detected_changes": changes,
            "avg_delay": delay,
            "false_alarm_rate": far,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        print(f"Method A (p={p}) results: {results[p]}")
    
    logging.info(f"Method A experiment completed. Results: {results}")
    return {"method": "Method A", "results": results}

def run_method_b_experiment(data: np.ndarray, true_changes: list) -> Dict[str, Any]:
    """
    Run an experiment using Method B for distributed change detection.

    Args:
    data (np.ndarray): Input data
    true_changes (list): List of true change points

    Returns:
    Dict[str, Any]: Results of the experiment
    """
    print("Running Method B experiment...")
    logging.info("Starting Method B experiment")
    
    results = {}
    local_thresholds = np.random.uniform(1, 3, size=data.shape[1])
    
    for h_method in config.METHOD_B_H_METHODS:
        print(f"Running Method B with h_method = {h_method}")
        
        # Average aggregation
        changes_avg, _ = method_b_average(data, h_method, local_thresholds)
        delay_avg = calculate_detection_delay(true_changes, changes_avg)
        far_avg = calculate_false_alarm_rate(true_changes, changes_avg, len(data))
        precision_avg, recall_avg, f1_avg, _ = evaluate_performance(true_changes, changes_avg, len(data))
        
        # Median aggregation
        changes_med, _ = method_b_median(data, h_method, local_thresholds)
        delay_med = calculate_detection_delay(true_changes, changes_med)
        far_med = calculate_false_alarm_rate(true_changes, changes_med, len(data))
        precision_med, recall_med, f1_med, _ = evaluate_performance(true_changes, changes_med, len(data))
        
        # MAD-based aggregation
        changes_mad, _ = method_b_mad(data, h_method, local_thresholds, config.METHOD_B_MAD_THRESHOLD)
        delay_mad = calculate_detection_delay(true_changes, changes_mad)
        far_mad = calculate_false_alarm_rate(true_changes, changes_mad, len(data))
        precision_mad, recall_mad, f1_mad, _ = evaluate_performance(true_changes, changes_mad, len(data))
        
        results[h_method] = {
            "average": {
                "detected_changes": changes_avg,
                "avg_delay": delay_avg,
                "false_alarm_rate": far_avg,
                "precision": precision_avg,
                "recall": recall_avg,
                "f1_score": f1_avg
            },
            "median": {
                "detected_changes": changes_med,
                "avg_delay": delay_med,
                "false_alarm_rate": far_med,
                "precision": precision_med,
                "recall": recall_med,
                "f1_score": f1_med
            },
            "mad": {
                "detected_changes": changes_mad,
                "avg_delay": delay_mad,
                "false_alarm_rate": far_mad,
                "precision": precision_mad,
                "recall": recall_mad,
                "f1_score": f1_mad
            }
        }
        print(f"Method B (h_method={h_method}) results: {results[h_method]}")
    
    logging.info(f"Method B experiment completed. Results: {results}")
    return {"method": "Method B", "results": results}

def run_all_experiments():
    """
    Run all experiments and compile results.
    """
    print("Starting all experiments...")
    logging.info("Starting all experiments")
    
    # Load and preprocess data
    df = load_lmp_data(config.DATA_PATH, config.SELECTED_BUSES)
    data, labels = prepare_data_for_change_detection(df, config.WINDOW_SIZE, config.STEP_SIZE)
    
    # Get true change points
    true_changes = np.where(np.diff(labels) != 0)[0].tolist()
    print(f"True change points: {true_changes}")
    
    # Run experiments
    results = []
    results.append(run_gem_experiment(data, true_changes))
    results.append(run_qq_experiment(data, true_changes))
    results.append(run_method_a_experiment(data, true_changes))
    results.append(run_method_b_experiment(data, true_changes))
    
    # Compile and save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{config.RESULTS_DIR}/experiment_results.csv", index=False)
    print("All experiments completed. Results saved to CSV.")
    logging.info("All experiments completed. Results saved to CSV.")
    
    return results_df

if __name__ == "__main__":
    np.random.seed(config.RANDOM_SEED)
    results = run_all_experiments()
    print(results)