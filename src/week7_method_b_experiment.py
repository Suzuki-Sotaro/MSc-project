import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Create directory for Method B results
method_b_dir = os.path.join('results', 'MethodB')
os.makedirs(method_b_dir, exist_ok=True)

# Load data from the specified CSV file
file_path = './data/A_LMPFreq3_Labeled.csv'
data = pd.read_csv(file_path)

# Specify bus numbers to analyze
bus_numbers = [115, 116, 117, 118, 119, 121, 135, 139]

# Extract the last 855 values
data_last_855 = data.iloc[-855:]

# Extract data for each specified bus
bus_data = {bus: data_last_855[f'Bus{bus}'].values for bus in bus_numbers}

# Extract labels
labels = data_last_855['Label'].values

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
            return t, cusum, distance  # Detect change point
    return -1, cusum, distance  # No change point detected

# Set initial parameters
window_size = 50  # You may need to optimize this
initial_threshold = 10  # Initial threshold to be optimized for each bus

# Optimize local threshold h[k] for each bus
local_thresholds = {}
for bus in bus_numbers:
    best_threshold = initial_threshold
    best_f1_score = 0
    for threshold in range(1, 20):
        change_point, cusum, _ = calculate_cusum(bus_data[bus], window_size, threshold)
        pred_labels = np.zeros(len(labels))
        if change_point != -1:
            pred_labels[change_point:] = 1
        f1 = f1_score(labels, pred_labels, zero_division=0)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold
    local_thresholds[bus] = best_threshold
    print(f'Optimized threshold for Bus {bus}: {best_threshold}')

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
        statistics_over_time = []
        for t in range(window_size, len(labels)):
            statistics = []
            for bus in bus_numbers:
                threshold = local_thresholds[bus]
                _, _, stat = calculate_cusum(bus_data[bus], window_size, threshold)
                statistics.append(stat)
            aggregated_stat = aggregate_statistics(statistics, method=aggregation_method)
            statistics_over_time.append(aggregated_stat)
            if aggregated_stat > H:
                detection_time = t
                break
        else:
            detection_time = -1
        
        pred_labels = np.zeros(len(labels))
        if detection_time != -1:
            pred_labels[detection_time:] = 1
        
        cm = confusion_matrix(labels, pred_labels)
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

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(method_b_dir, 'results.csv'), index=False)
print(results_df)
