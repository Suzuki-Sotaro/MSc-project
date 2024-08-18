# The following is the code for qq_detection.py.
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def binarize_data(data, threshold):
    return (data >= threshold).astype(int)

def calculate_qq_distance(window1, window2):
    if len(window1) == 0 or len(window2) == 0:
        return 0  # または適切なデフォルト値
    quantiles = np.linspace(0.01, 0.99, 100)
    q_window1 = np.quantile(window1, quantiles)
    q_window2 = np.quantile(window2, quantiles)
    qq_distance = np.sqrt(2) / 2 * np.mean(np.abs(q_window1 - q_window2))
    return qq_distance

def detect_local_change(data, window_size, threshold):
    changes = np.zeros(len(data) - window_size * 2 + 1)
    for i in range(window_size * 2 - 1, len(data)):
        window1 = data[i - window_size * 2 + 1 : i - window_size + 1]
        window2 = data[i - window_size + 1 : i + 1]
        qq_distance = calculate_qq_distance(window1, window2)
        if qq_distance > threshold:
            changes[i - window_size * 2 + 1] = 1
    return changes

def learn_local_threshold(data, labels, window_size):
    best_threshold = 0
    best_f1 = 0
    for threshold in np.arange(0.1, 5.0, 0.1):
        changes = detect_local_change(data, window_size, threshold)
        if len(changes) < len(labels) - window_size * 2 + 1:
            continue  # データが不十分な場合はスキップ
        f1 = f1_score(labels[window_size * 2 - 1:], changes)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def method_a(bus_changes, p_values):
    results = []
    n_buses = len(bus_changes)
    
    for p in p_values:
        threshold = int(p * n_buses)
        combined_changes = np.zeros_like(list(bus_changes.values())[0])
        detection_time = -1
        
        # At least one bus
        one_bus_changes = np.any(list(bus_changes.values()), axis=0)
        one_bus_detection_time = np.argmax(one_bus_changes) if np.any(one_bus_changes) else -1
        
        # All buses
        all_buses_changes = np.all(list(bus_changes.values()), axis=0)
        all_buses_detection_time = np.argmax(all_buses_changes) if np.any(all_buses_changes) else -1
        
        # p% of buses
        for t in range(len(combined_changes)):
            votes = sum(changes[t] for changes in bus_changes.values())
            if votes >= threshold:
                combined_changes[t] = 1
                if detection_time == -1:
                    detection_time = t
        
        results.append({
            'Method': f'Method A (p={p})',
            'Changes': combined_changes,
            'Detection Time': detection_time,
            'One Bus Detection Time': one_bus_detection_time,
            'All Buses Detection Time': all_buses_detection_time
        })
    
    return results

def method_b(bus_statistics, aggregation_methods, sink_threshold_methods):
    results = []
    local_thresholds = [stats.max() for stats in bus_statistics.values()]
    
    for agg_method in aggregation_methods:
        for sink_method in sink_threshold_methods:
            combined_changes = np.zeros_like(list(bus_statistics.values())[0])
            detection_time = -1
            
            if sink_method == 'average':
                H = np.mean(local_thresholds)
            elif sink_method == 'minimum':
                H = np.min(local_thresholds)
            elif sink_method == 'maximum':
                H = np.max(local_thresholds)
            elif sink_method == 'median':
                H = np.median(local_thresholds)
            
            for t in range(len(combined_changes)):
                statistics = [stats[t] for stats in bus_statistics.values()]
                
                if agg_method == 'average':
                    agg_stat = np.mean(statistics)
                elif agg_method == 'median':
                    agg_stat = np.median(statistics)
                elif agg_method == 'outlier_detection':
                    median = np.median(statistics)
                    mad = np.median(np.abs(statistics - median))
                    if mad == 0:
                        agg_stat = np.mean(statistics)
                    else:
                        z_scores = 0.6745 * (statistics - median) / mad
                        non_outliers = [s for s, z in zip(statistics, z_scores) if abs(z) <= 3.5]
                        agg_stat = np.mean(non_outliers) if non_outliers else np.mean(statistics)
                
                if agg_stat > H:
                    combined_changes[t] = 1
                    if detection_time == -1:
                        detection_time = t
            
            results.append({
                'Method': f'Method B ({agg_method}, {sink_method})',
                'Changes': combined_changes,
                'Detection Time': detection_time
            })
    
    return results

def qq_detection(df, buses, window_size, p_values, aggregation_methods, sink_threshold_methods):
    bus_changes = {}
    bus_statistics = {}
    individual_bus_results = []
    
    # Learn local thresholds and detect changes for each bus
    for bus in buses:
        data = df[bus].values
        labels = df['Label'].values
        if len(data) < window_size * 2:
            print(f"Warning: Not enough data for bus {bus}. Skipping.")
            continue
        local_threshold = learn_local_threshold(data, labels, window_size)
        changes = detect_local_change(data, window_size, local_threshold)
        bus_changes[bus] = changes
        bus_statistics[bus] = np.cumsum(changes)

        # Calculate individual bus performance
        min_length = min(len(labels[window_size * 2 - 1:]), len(changes))
        labels_truncated = labels[window_size * 2 - 1:][:min_length]
        changes_truncated = changes[:min_length]
        
        accuracy = accuracy_score(labels_truncated, changes_truncated)
        precision = precision_score(labels_truncated, changes_truncated)
        recall = recall_score(labels_truncated, changes_truncated)
        f1 = f1_score(labels_truncated, changes_truncated)
        
        individual_bus_results.append({
            'Bus': bus,
            'Method': 'Individual Q-Q',
            'Threshold': local_threshold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Detection Time': np.argmax(changes) if np.any(changes) else -1
        })
    
    # Apply Method A
    method_a_results = method_a(bus_changes, p_values)
    
    # Apply Method B
    method_b_results = method_b(bus_statistics, aggregation_methods, sink_threshold_methods)
    
    # Evaluate results
    labels = df['Label'].values[window_size * 2 - 1:]  # Adjust labels to match the length of changes
    
    for result in method_a_results + method_b_results:
        changes = result['Changes']
        
        # Ensure that labels and changes have the same length
        min_length = min(len(labels), len(changes))
        labels_truncated = labels[:min_length]
        changes_truncated = changes[:min_length]
        
        accuracy = accuracy_score(labels_truncated, changes_truncated)
        precision = precision_score(labels_truncated, changes_truncated)
        recall = recall_score(labels_truncated, changes_truncated)
        f1 = f1_score(labels_truncated, changes_truncated)
        
        result.update({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    
    # method_a_resultsとmethod_b_resultsでChangesの列は削除して返す
    return (pd.DataFrame(method_a_results).drop(columns='Changes'), 
            pd.DataFrame(method_b_results).drop(columns='Changes'), 
            pd.DataFrame(individual_bus_results))