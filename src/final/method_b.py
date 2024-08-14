# 以下はmethod_b.pyの内容
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to execute non-parametric CuSum
def calculate_cusum(data, window_size, threshold):
    n = len(data)
    reference_window = data[:window_size]
    cusum = np.zeros(n)
    for t in range(window_size, n):
        test_window = data[t-window_size:t]
        distance = np.abs(np.mean(reference_window) - np.mean(test_window))
        cusum[t] = max(0, cusum[t-1] + distance - threshold)  # CuSumをリセットする処理を追加
        if cusum[t] > threshold:
            return t, cusum, distance  # Detect change point
    return -1, cusum, distance  # No change point detected

# Method B - Analyze using different aggregation methods
def analyze_method_b(df, buses, window_size):
    labels = df['Label'].values
    bus_data = {bus: df[bus].values for bus in buses}

    # Optimize local threshold h[k] for each bus
    local_thresholds = {}
    for bus in buses:
        best_threshold = 10  # Initial threshold
        best_f1_score = 0
        for threshold in range(1, 10):  # Test different thresholds in a range
            change_point, cusum, _ = calculate_cusum(bus_data[bus], window_size, threshold)
            pred_labels = np.zeros(len(labels))
            if change_point != -1:
                pred_labels[change_point:] = 1
            f1 = f1_score(labels, pred_labels, zero_division=0)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = threshold
        local_thresholds[bus] = best_threshold

    # Aggregation functions implementation
    def aggregate_statistics(statistics, method='average'):
        if method == 'average':
            return np.mean(statistics)
        elif method == 'median':
            return np.median(statistics)
        elif method == 'outlier_detection':
            # Use MAD for outlier detection
            median = np.median(statistics)
            mad = np.median(np.abs(statistics - median))
            non_outliers = [s for s in statistics if abs(s - median) <= 3 * mad]
            return np.mean(non_outliers) if non_outliers else median

    # Set the threshold H at the sink
    def determine_sink_threshold(local_thresholds, method='average'):
        if method == 'average':
            return np.mean(list(local_thresholds.values()))
        elif method == 'minimum':
            return np.min(list(local_thresholds.values()))
        elif method == 'maximum':
            return np.max(list(local_thresholds.values()))
        elif method == 'median':
            return np.median(list(local_thresholds.values()))

    # Experiment with different aggregation methods and sink thresholds
    aggregation_methods = ['average', 'median', 'outlier_detection']
    sink_threshold_methods = ['average', 'minimum', 'maximum', 'median']
    results = []

    for aggregation_method in aggregation_methods:
        for sink_threshold_method in sink_threshold_methods:
            H = determine_sink_threshold(local_thresholds, method=sink_threshold_method)
            detection_time = -1
            for t in range(window_size, len(labels)):
                statistics = []
                for bus in buses:
                    threshold = local_thresholds[bus]
                    _, _, stat = calculate_cusum(bus_data[bus], window_size, threshold)
                    statistics.append(stat)
                aggregated_stat = aggregate_statistics(statistics, method=aggregation_method)
                if aggregated_stat > H:
                    detection_time = t
                    break

            pred_labels = np.zeros(len(labels))
            if detection_time != -1:
                pred_labels[detection_time:] = 1

            accuracy = accuracy_score(labels, pred_labels)
            precision = precision_score(labels, pred_labels, zero_division=0)
            recall = recall_score(labels, pred_labels, zero_division=0)
            f1 = f1_score(labels, pred_labels, zero_division=0)

            results.append({
                'Aggregation Method': aggregation_method,
                'Sink Threshold Method': sink_threshold_method,
                'Sink Threshold': H,
                'Detection Time': detection_time,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })

    return pd.DataFrame(results)
