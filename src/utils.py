import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.metrics import confusion_matrix

def calculate_detection_delay(true_changes: List[int], detected_changes: List[int], tolerance: int = 5) -> float:
    delays = []
    for true_change in true_changes:
        matched_detections = [d for d in detected_changes if abs(d - true_change) <= tolerance]
        if matched_detections:
            delay = min(matched_detections) - true_change
            delays.append(delay)
    
    if delays:
        return np.mean(delays)
    else:
        return np.inf  # 変更点: -5.0 の代わりに np.inf を返す

def calculate_false_alarm_rate(true_changes: List[int], detected_changes: List[int], total_time: int, tolerance: int = 5) -> float:
    """
    Calculate the false alarm rate.

    Args:
    true_changes (List[int]): True change points
    detected_changes (List[int]): Detected change points
    total_time (int): Total time steps
    tolerance (int): Tolerance window for matching true and detected changes

    Returns:
    float: False alarm rate
    """
    print("Calculating false alarm rate...")
    false_alarms = 0
    for detected in detected_changes:
        if not any(abs(detected - true) <= tolerance for true in true_changes):
            false_alarms += 1
            print(f"False alarm at {detected}")
    
    far = false_alarms / total_time
    print(f"Total false alarms: {false_alarms}")
    print(f"False alarm rate: {far:.4f}")
    return far

def plot_detection_results(data: np.ndarray, true_changes: List[int], detected_changes: List[int], title: str):
    """
    Plot the data with true and detected change points.

    Args:
    data (np.ndarray): Input data
    true_changes (List[int]): True change points
    detected_changes (List[int]): Detected change points
    title (str): Plot title
    """
    print("Plotting detection results...")
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Data')
    for tc in true_changes:
        plt.axvline(x=tc, color='r', linestyle='--', label='True change' if tc == true_changes[0] else '')
    for dc in detected_changes:
        plt.axvline(x=dc, color='g', linestyle=':', label='Detected change' if dc == detected_changes[0] else '')
    plt.title(title)
    plt.legend()
    plt.show()
    print("Plot displayed.")

def evaluate_performance(true_changes: List[int], detected_changes: List[int], total_time: int, tolerance: int = 5) -> Tuple[float, float, float, float]:
    """
    Evaluate the performance of the change detection algorithm.

    Args:
    true_changes (List[int]): True change points
    detected_changes (List[int]): Detected change points
    total_time (int): Total time steps
    tolerance (int): Tolerance window for matching true and detected changes

    Returns:
    Tuple[float, float, float, float]: Precision, Recall, F1-score, and False Alarm Rate
    """
    print("Evaluating performance...")
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true_change in true_changes:
        if any(abs(true_change - detected) <= tolerance for detected in detected_changes):
            true_positives += 1
        else:
            false_negatives += 1

    for detected_change in detected_changes:
        if not any(abs(detected_change - true) <= tolerance for true in true_changes):
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    far = false_positives / total_time

    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    print(f"False Alarm Rate: {far:.4f}")

    return precision, recall, f1_score, far

def plot_roc_curve(true_changes: List[int], decision_function: np.ndarray, total_time: int, tolerance: int = 5):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Args:
    true_changes (List[int]): True change points
    decision_function (np.ndarray): Decision function values at each time step
    total_time (int): Total time steps
    tolerance (int): Tolerance window for matching true and detected changes
    """
    print("Plotting ROC curve...")
    thresholds = np.linspace(np.min(decision_function), np.max(decision_function), 100)
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        detected_changes = np.where(decision_function > threshold)[0].tolist()
        y_true = np.zeros(total_time)
        y_pred = np.zeros(total_time)

        for tc in true_changes:
            y_true[max(0, tc - tolerance):min(total_time, tc + tolerance + 1)] = 1

        for dc in detected_changes:
            y_pred[dc] = 1

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()
    print("ROC curve displayed.")

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    total_time = 1000
    data = np.cumsum(np.random.randn(total_time))
    true_changes = [250, 500, 750]
    data[true_changes[0]:] += 2
    data[true_changes[1]:] -= 3
    data[true_changes[2]:] += 1

    # Simulating detected changes
    detected_changes = [248, 505, 752, 800]  # 800 is a false alarm
    decision_function = np.abs(np.convolve(data, [1, -1], mode='same'))

    print("Example usage of utility functions:")
    
    avg_delay = calculate_detection_delay(true_changes, detected_changes)
    far = calculate_false_alarm_rate(true_changes, detected_changes, total_time)
    
    plot_detection_results(data, true_changes, detected_changes, "Change Detection Results")
    
    precision, recall, f1_score, far = evaluate_performance(true_changes, detected_changes, total_time)
    
    plot_roc_curve(true_changes, decision_function, total_time)