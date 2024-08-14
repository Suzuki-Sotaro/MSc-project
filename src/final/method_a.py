import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Function to execute non-parametric CuSum
def calculate_cusum(data, window_size, threshold):
    n = len(data)
    reference_window = data[:window_size]
    cusum = np.zeros(n)
    for t in range(window_size, n):
        test_window = data[t-window_size:t]
        distance = np.abs(np.mean(reference_window) - np.mean(test_window))
        cusum[t] = cusum[t-1] + distance
        if cusum[t] > threshold:
            return t, cusum  # Detect change point
    return -1, cusum  # No change point detected

# Method A - Analyze using voting scheme
def analyze_method_a(df, buses, window_size, p_values):
    results = []
    labels = df['Label'].values
    bus_data = {bus: df[bus].values for bus in buses}

    # Optimize local threshold h[k] for each bus
    local_thresholds = {}
    for bus in buses:
        best_threshold = 10  # Initial threshold
        best_f1_score = 0
        for threshold in range(1, 20):
            change_point, cusum = calculate_cusum(bus_data[bus], window_size, threshold)
            pred_labels = np.zeros(len(labels))
            if change_point != -1:
                pred_labels[change_point:] = 1
            f1 = f1_score(labels, pred_labels)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = threshold
        local_thresholds[bus] = best_threshold

    # Voting scheme implementation
    for p in p_values:
        votes = np.zeros(len(labels))
        detection_times = []
        for bus in buses:
            threshold = local_thresholds[bus]
            change_point, cusum = calculate_cusum(bus_data[bus], window_size, threshold)
            if change_point != -1:
                votes[change_point:] += 1

        for t in range(len(votes)):
            if votes[t] >= p * len(buses):
                detection_times.append(t)
                break

        if detection_times:
            detection_time = detection_times[0]
        else:
            detection_time = -1

        pred_labels = np.zeros(len(labels))
        if detection_time != -1:
            pred_labels[detection_time:] = 1

        accuracy = accuracy_score(labels, pred_labels)
        recall = recall_score(labels, pred_labels)
        precision = precision_score(labels, pred_labels)
        f1 = f1_score(labels, pred_labels)

        results.append({
            'p': p,
            'Detection Time': detection_time,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

    return pd.DataFrame(results)
