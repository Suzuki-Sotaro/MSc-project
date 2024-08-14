# 以下はperformance_evaluation.pyの内容です。
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def calculate_detection_delay(true_change_point: int, detected_change_point: int) -> int:
    """
    Calculate the detection delay.

    Args:
    true_change_point (int): The true change point.
    detected_change_point (int): The detected change point.

    Returns:
    int: The detection delay.
    """
    if detected_change_point == -1:  # No change detected
        return np.inf
    return max(0, detected_change_point - true_change_point)

def calculate_false_alarm_rate(detected_change_point: int, true_change_point: int, n_samples: int) -> float:
    """
    Calculate the false alarm rate.

    Args:
    detected_change_point (int): The detected change point.
    true_change_point (int): The true change point.
    n_samples (int): Total number of samples.

    Returns:
    float: The false alarm rate.
    """
    if detected_change_point == -1:  # No change detected
        return 0
    return max(0, (true_change_point - detected_change_point) / n_samples)

def calculate_detection_accuracy(true_change_point: int, detected_change_point: int, tolerance: int = 100) -> float:
    if detected_change_point == -1:  # No change detected
        return 0.0
    return float(abs(true_change_point - detected_change_point) <= tolerance)

def calculate_performance_metrics(true_change_point: int, detected_change_point: int, n_samples: int, tolerance: int = 50) -> Dict:
    if detected_change_point == -1:
        return {
            'detection_delay': np.inf,
            'false_alarm_rate': 0,
            'detection_accuracy': 0,
            'missed_detection': 1
        }
    
    detection_delay = max(0, detected_change_point - true_change_point)
    false_alarm_rate = 1 if detected_change_point < true_change_point else 0
    detection_accuracy = 1 - min(1, abs(true_change_point - detected_change_point) / tolerance)
    missed_detection = 0 if abs(true_change_point - detected_change_point) <= tolerance else 1

    return {
        'detection_delay': detection_delay,
        'false_alarm_rate': false_alarm_rate,
        'detection_accuracy': detection_accuracy,
        'missed_detection': missed_detection
    }

def plot_roc_curve(true_positives: List[float], false_positives: List[float]) -> None:
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Args:
    true_positives (List[float]): List of true positive rates.
    false_positives (List[float]): List of false positive rates.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(false_positives, true_positives, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.grid(True)
    plt.show()

def plot_detection_delay_distribution(delays: List[int]) -> None:
    """
    Plot the distribution of detection delays.

    Args:
    delays (List[int]): List of detection delays.
    """
    plt.figure(figsize=(10, 8))
    plt.hist(delays, bins=20, edgecolor='black')
    plt.xlabel('Detection Delay')
    plt.ylabel('Frequency')
    plt.title('Distribution of Detection Delays')
    plt.grid(True)
    plt.show()

def plot_parameter_sensitivity(parameter_name: str, parameter_values: List[float], metric_values: List[float]) -> None:
    """
    Plot the sensitivity of a performance metric to a parameter.

    Args:
    parameter_name (str): Name of the parameter.
    parameter_values (List[float]): List of parameter values.
    metric_values (List[float]): List of corresponding metric values.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(parameter_values, metric_values, marker='o')
    plt.xlabel(parameter_name)
    plt.ylabel('Performance Metric')
    plt.title(f'Sensitivity of Performance to {parameter_name}')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    true_change_point = 500
    detected_change_point = 510
    n_samples = 1000

    metrics = calculate_performance_metrics(true_change_point, detected_change_point, n_samples)
    print("Performance Metrics:")
    print(metrics)

    # Example ROC curve
    true_positives = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    false_positives = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    plot_roc_curve(true_positives, false_positives)

    # Example detection delay distribution
    delays = [5, 7, 10, 12, 15, 18, 20, 22, 25, 30]
    plot_detection_delay_distribution(delays)

    # Example parameter sensitivity
    window_sizes = [50, 100, 150, 200, 250]
    detection_delays = [15, 12, 10, 11, 13]
    plot_parameter_sensitivity("Window Size", window_sizes, detection_delays)