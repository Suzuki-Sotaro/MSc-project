# The following is the code for qq_detection.py.
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from method_a import apply_method_a, evaluate_method_a, calculate_far_ed
from method_b import apply_method_b, evaluate_method_b

def calculate_qq_distance(window1, window2):
    if len(window1) == 0 or len(window2) == 0:
        return 0
    s = min(len(window1), len(window2))
    quantiles = np.linspace(0, 1, s)
    q_window1 = np.quantile(window1, quantiles)
    q_window2 = np.quantile(window2, quantiles)
    qq_distance = (1/s) * np.sum(np.sqrt(2)/2 * np.abs(q_window1 - q_window2))
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
            continue
        f1 = f1_score(labels[window_size * 2 - 1:], changes)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

def binary_detection(data, threshold):
    return (data >= threshold).astype(int)

def centralized_detection(qq_distances, h):
    cumulative_distance = np.cumsum(qq_distances)
    return np.argmax(cumulative_distance >= h)

def qq_detection(df, buses, window_sizes, p_values, aggregation_methods, sink_threshold_methods):
    results = []
    individual_bus_results = []
    centralized_results = []
    
    for window_size in window_sizes:
        bus_changes = {}
        bus_statistics = {}
        bus_binary_changes = {}
        bus_qq_distances = {}
        
        for bus in buses:
            data = df[bus].values
            labels = df['Label'].values
            if len(data) < window_size * 2:
                print(f"Warning: Not enough data for bus {bus} with window size {window_size}. Skipping.")
                continue
            
            local_threshold = learn_local_threshold(data, labels, window_size)
            changes = detect_local_change(data, window_size, local_threshold)
            bus_changes[bus] = changes
            bus_statistics[bus] = np.cumsum(changes)
            
            # バイナリ検出の実装
            binary_threshold = np.mean(data[:window_size])  # 事前変化の平均を閾値として使用
            bus_binary_changes[bus] = binary_detection(data, binary_threshold)
            
            # Q-Q距離の計算（集中型検出用）
            qq_distances = []
            for i in range(window_size * 2 - 1, len(data)):
                window1 = data[i - window_size * 2 + 1 : i - window_size + 1]
                window2 = data[i - window_size + 1 : i + 1]
                qq_distances.append(calculate_qq_distance(window1, window2))
            bus_qq_distances[bus] = np.array(qq_distances)

            # 個別バスのパフォーマンス計算
            min_length = min(len(labels[window_size * 2 - 1:]), len(changes))
            labels_truncated = labels[window_size * 2 - 1:][:min_length]
            changes_truncated = changes[:min_length]
            
            accuracy = accuracy_score(labels_truncated, changes_truncated)
            precision = precision_score(labels_truncated, changes_truncated, zero_division=0)
            recall = recall_score(labels_truncated, changes_truncated)
            f1 = f1_score(labels_truncated, changes_truncated)
            detection_time = np.argmax(changes) if np.any(changes) else -1
            far, ed = calculate_far_ed(labels_truncated, changes_truncated, detection_time)
            
            individual_bus_results.append({
                'Bus': bus,
                'Method': 'Individual Q-Q',
                'Window Size': window_size,
                'Threshold': local_threshold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'False Alarm Rate': far,
                'Expected Delay': ed,
                'Detection Time': detection_time
            })
        
        # 集中型の非パラメトリック検出
        total_qq_distances = np.sum([distances for distances in bus_qq_distances.values()], axis=0)
        for h in np.arange(0.1, 5.0, 0.1):
            centralized_detection_time = centralized_detection(total_qq_distances, h)
            centralized_results.append({
                'Method': 'Centralized Detection',
                'Window Size': window_size,
                'Threshold': h,
                'Detection Time': centralized_detection_time
            })
        
        # Method A と Method B の適用
        method_a_results = apply_method_a(bus_changes, p_values)
        method_b_results = apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods)
        
        # 結果の評価
        labels = df['Label'].values[window_size * 2 - 1:]
        
        method_a_results = evaluate_method_a(method_a_results, labels)
        method_b_results = evaluate_method_b(method_b_results, labels)
        
    return (pd.DataFrame(centralized_results), 
            pd.DataFrame(method_a_results),
            pd.DataFrame(method_b_results))