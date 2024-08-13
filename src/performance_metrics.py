# 以下はperformance_metrics.pyのコードです。
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_detection_delay(true_changes, detected_changes, tolerance=5):
    """
    Calculate the average detection delay.
    
    Args:
    true_changes (list): Indices of true change points.
    detected_changes (list): Indices of detected change points.
    tolerance (int): Maximum allowed delay to consider a detection as correct.
    
    Returns:
    float: Average detection delay.
    """
    delays = []
    for true_change in true_changes:
        for detected_change in detected_changes:
            if 0 <= detected_change - true_change <= tolerance:
                delays.append(detected_change - true_change)
                break
    return np.mean(delays) if delays else np.inf

def calculate_false_alarm_rate(true_changes, detected_changes, total_samples, tolerance=5):
    """
    Calculate the false alarm rate.
    
    Args:
    true_changes (list): Indices of true change points.
    detected_changes (list): Indices of detected change points.
    total_samples (int): Total number of samples in the dataset.
    tolerance (int): Maximum allowed delay to consider a detection as correct.
    
    Returns:
    float: False alarm rate.
    """
    false_alarms = 0
    for detected_change in detected_changes:
        if all(abs(detected_change - true_change) > tolerance for true_change in true_changes):
            false_alarms += 1
    return false_alarms / total_samples

def calculate_detection_rate(true_changes, detected_changes, tolerance=5):
    """
    Calculate the detection rate (recall).
    
    Args:
    true_changes (list or np.array): Indices of true change points.
    detected_changes (list): Indices of detected change points.
    tolerance (int): Maximum allowed delay to consider a detection as correct.
    
    Returns:
    float: Detection rate.
    """
    true_changes = np.array(true_changes)  # 確実にNumPy配列に変換
    detected = np.sum([np.any((detected_changes >= tc) & (detected_changes <= tc + tolerance)) for tc in true_changes])
    return detected / len(true_changes) if len(true_changes) > 0 else 0

def calculate_f1_score(true_changes, detected_changes, total_samples, tolerance=5):
    """
    Calculate the F1 score.
    
    Args:
    true_changes (list): Indices of true change points.
    detected_changes (list): Indices of detected change points.
    total_samples (int): Total number of samples in the dataset.
    tolerance (int): Maximum allowed delay to consider a detection as correct.
    
    Returns:
    float: F1 score.
    """
    y_true = np.zeros(total_samples)
    y_pred = np.zeros(total_samples)
    
    for change in true_changes:
        y_true[max(0, change - tolerance):min(total_samples, change + tolerance + 1)] = 1
    
    for change in detected_changes:
        y_pred[change] = 1
    
    return f1_score(y_true, y_pred)

def evaluate_performance(true_changes, detected_changes, total_samples, tolerance=5):
    """
    Evaluate the overall performance of the change detection algorithm.
    
    Args:
    true_changes (list): Indices of true change points.
    detected_changes (list): Indices of detected change points.
    total_samples (int): Total number of samples in the dataset.
    tolerance (int): Maximum allowed delay to consider a detection as correct.
    
    Returns:
    dict: A dictionary containing various performance metrics.
    """
    avg_delay = calculate_detection_delay(true_changes, detected_changes, tolerance)
    far = calculate_false_alarm_rate(true_changes, detected_changes, total_samples, tolerance)
    detection_rate = calculate_detection_rate(true_changes, detected_changes, tolerance)
    f1 = calculate_f1_score(true_changes, detected_changes, total_samples, tolerance)
    
    return {
        "Average Detection Delay": avg_delay,
        "False Alarm Rate": far,
        "Detection Rate": detection_rate,
        "F1 Score": f1
    }

if __name__ == "__main__":
    # Example usage
    true_changes = [100, 300, 500]
    detected_changes = [98, 305, 502, 700]
    total_samples = 1000
    
    performance = evaluate_performance(true_changes, detected_changes, total_samples)
    
    print("Performance Metrics:")
    for metric, value in performance.items():
        print(f"{metric}: {value}")