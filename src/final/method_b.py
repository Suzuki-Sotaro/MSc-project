# method_b.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_mad(data):
    median = np.median(data)
    return np.median(np.abs(data - median))

def detect_outliers_mad(data, threshold=3.5):
    if len(data) == 0:
        return np.array([], dtype=bool)
    median = np.median(data)
    mad = calculate_mad(data)
    if mad == 0:
        return np.full(len(data), False)
    modified_z_scores = 0.6745 * (data - median) / mad
    return np.abs(modified_z_scores) > threshold

def aggregate_statistics(statistics, method='average', mad_threshold=3.5):
    if len(statistics) == 0:
        return 0
    if method == 'average':
        return np.mean(statistics)
    elif method == 'median':
        return np.median(statistics)
    elif method == 'outlier_detection':
        outliers = detect_outliers_mad(statistics, mad_threshold)
        non_outliers = statistics[~outliers]
        return np.mean(non_outliers) if len(non_outliers) > 0 else np.median(statistics)

def determine_sink_threshold(local_thresholds, method='average'):
    if method == 'average':
        return np.mean(list(local_thresholds.values()))
    elif method == 'minimum':
        return np.min(list(local_thresholds.values()))
    elif method == 'maximum':
        return np.max(list(local_thresholds.values()))
    elif method == 'median':
        return np.median(list(local_thresholds.values()))

def analyze_method_b(df, buses, window_size, mad_thresholds=[2.5, 3.0, 3.5]):
    labels = df['Label'].values
    bus_data = {bus: df[bus].values for bus in buses}

    # Optimize local threshold h[k] for each bus
    local_thresholds = {}
    for bus in buses:
        best_threshold = 0
        best_f1_score = 0
        for threshold in np.arange(0.01, 1.0, 0.01):
            changes = np.abs(np.convolve(bus_data[bus], [-1]*window_size + [1]*window_size, mode='valid')) > threshold
            # Ensure that changes and labels have the same length
            f1 = f1_score(labels[window_size*2-1:], changes, zero_division=0)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = threshold
        local_thresholds[bus] = best_threshold

    aggregation_methods = ['average', 'median', 'outlier_detection']
    sink_threshold_methods = ['average', 'minimum', 'maximum', 'median']
    results = []

    for aggregation_method in aggregation_methods:
        for sink_threshold_method in sink_threshold_methods:
            for mad_threshold in mad_thresholds:
                H = determine_sink_threshold(local_thresholds, method=sink_threshold_method)
                detection_time = -1
                for t in range(window_size*2-1, len(labels)):  # Start from window_size*2-1
                    statistics = []
                    for bus in buses:
                        s_k = np.abs(np.mean(bus_data[bus][t-window_size:t]) - np.mean(bus_data[bus][t-window_size*2:t-window_size]))
                        statistics.append(s_k)
                    aggregated_stat = aggregate_statistics(np.array(statistics), method=aggregation_method, mad_threshold=mad_threshold)
                    if aggregated_stat > H:
                        detection_time = t
                        break

                pred_labels = np.zeros(len(labels) - (window_size*2-1))
                if detection_time != -1:
                    pred_labels[detection_time-(window_size*2-1):] = 1

                true_labels = labels[window_size*2-1:]
                accuracy = accuracy_score(true_labels, pred_labels)
                precision = precision_score(true_labels, pred_labels, zero_division=0)
                recall = recall_score(true_labels, pred_labels, zero_division=0)
                f1 = f1_score(true_labels, pred_labels, zero_division=0)

                results.append({
                    'Aggregation Method': aggregation_method,
                    'Sink Threshold Method': sink_threshold_method,
                    'MAD Threshold': mad_threshold,
                    'Sink Threshold': H,
                    'Detection Time': detection_time,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                })

    return pd.DataFrame(results)
