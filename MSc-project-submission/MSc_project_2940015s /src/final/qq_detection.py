# The following is the code for qq_detection.py.
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from method_a import apply_method_a, evaluate_method_a
from method_b import apply_method_b, evaluate_method_b
from utils import calculate_far_ed, evaluate_results

def calculate_qq_distance(window1, window2):
    if len(window1) == 0 or len(window2) == 0:
        return 0
    s = min(len(window1), len(window2))
    quantiles = np.linspace(0, 1, s)
    q_window1 = np.quantile(window1, quantiles)
    q_window2 = np.quantile(window2, quantiles)
    qq_distance = (1/s) * np.sum(np.sqrt(2)/2 * np.abs(q_window1 - q_window2))
    return qq_distance

def detect_local_change(data, window_size, thresholds):
    all_changes = {threshold: np.zeros(len(data) - window_size * 2 + 1) for threshold in thresholds}
    
    for i in range(window_size * 2 - 1, len(data)):
        window1 = data[i - window_size * 2 + 1 : i - window_size + 1]
        window2 = data[i - window_size + 1 : i + 1]
        qq_distance = calculate_qq_distance(window1, window2)
        
        for threshold in thresholds:
            if qq_distance > threshold:
                all_changes[threshold][i - window_size * 2 + 1] = 1
                
    return all_changes

def binary_detection(data, threshold):
    return (data >= threshold).astype(int)

def centralized_detection(qq_distances, h):
    cumulative_distance = np.cumsum(qq_distances)
    return np.argmax(cumulative_distance >= h)

def qq_detection(df, buses, window_sizes, thresholds, p_values, aggregation_methods, sink_threshold_methods):
    individual_bus_results = []
    method_a_results = []
    method_b_results = []
    
    for window_size in window_sizes:
        for threshold in thresholds:
            bus_changes = {}
            bus_statistics = {}
            
            for bus in buses:
                data = df[bus].values
                labels = df['Label'].values.astype(int).tolist()  # Convert to list of integers
                if len(data) < window_size * 2:
                    print(f"Warning: Not enough data for bus {bus} with window size {window_size}. Skipping.")
                    continue
                
                changes = detect_local_change(data, window_size, [threshold])
                changes = changes[threshold].astype(int).tolist()  # Convert to list of integers
                bus_changes[bus] = changes
                bus_statistics[bus] = np.cumsum(changes)
                
                # Q-Q距離の計算（集中型検出用）
                qq_distances = []
                for i in range(window_size * 2 - 1, len(data)):
                    window1 = data[i - window_size * 2 + 1 : i - window_size + 1]
                    window2 = data[i - window_size + 1 : i + 1]
                    qq_distances.append(calculate_qq_distance(window1, window2))
                
                # 個別バスのパフォーマンス計算
                min_length = min(len(labels[window_size * 2 - 1:]), len(changes))
                labels_truncated = labels[window_size * 2 - 1:][:min_length]
                changes_truncated = changes[:min_length]
                detection_time = np.argmax(changes) if np.any(changes) else -1
                cm, accuracy, precision, recall, f1 = evaluate_results(changes_truncated, labels_truncated)
                far, ed = calculate_far_ed(labels_truncated, changes_truncated, detection_time)
                
                individual_bus_results.append({
                    'Bus': bus,
                    'Data': data.tolist()[:min_length],  # Convert to list if needed
                    'Label': labels_truncated,
                    'Detection': changes_truncated,
                    'Window Size': window_size,
                    'QQ Threshold': threshold,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'False Alarm Rate': far,
                    'Expected Delay': ed,
                    'Detection Time': detection_time
                })
            
            # Method A と Method B の適用
            method_a_results_window = apply_method_a(bus_changes, p_values, df, buses)
            method_b_results_window = apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods, df, buses)
            
            # 結果の評価
            labels = df['Label'].values[window_size * 2 - 1:].astype(int).tolist()  # Convert to list of integers
            
            method_a_results_window = evaluate_method_a(method_a_results_window, labels)
            method_b_results_window = evaluate_method_b(method_b_results_window, labels)
            
            for result in method_a_results_window:
                result['Window Size'] = window_size
                result['Threshold'] = threshold
            for result in method_b_results_window:
                result['Window Size'] = window_size
                result['Threshold'] = threshold
            
            method_a_results.extend(method_a_results_window)
            method_b_results.extend(method_b_results_window)
    
    print("Q-Q detection analysis completed.")
    individual_bus_results = pd.DataFrame(individual_bus_results)
    individual_bus_results['Label'] = individual_bus_results['Label'].apply(lambda x: list(map(int, x)))
    individual_bus_results['Detection'] = individual_bus_results['Detection'].apply(lambda x: list(map(int, x)))

    return (individual_bus_results,
            pd.DataFrame(method_a_results),
            pd.DataFrame(method_b_results))
