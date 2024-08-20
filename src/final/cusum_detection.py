# Below is the content of cusum_detection.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from method_a import apply_method_a, evaluate_method_a, calculate_far_ed
from method_b import apply_method_b, evaluate_method_b

def calculate_statistics(df, buses):
    statistics = {}
    for bus in buses:
        bus_data = df[bus].values
        label = df['Label'].values
        
        data_before = bus_data[label == 0]
        data_after = bus_data[label == 1]
        
        statistics[bus] = {
            'mean_before': np.mean(data_before),
            'sigma_before': np.std(data_before),
            'mean_after': np.mean(data_after),
            'sigma_after': np.std(data_after)
        }
    return statistics

def cusum_for_each_bus(data, mean_before, sigma_before, mean_after, sigma_after, threshold):
    n = len(data)
    cusum_scores = np.zeros(n)
    
    for i in range(1, n):
        likelihood_ratio = np.log((1 / (np.sqrt(2 * np.pi) * sigma_after)) * np.exp(-0.5 * ((data[i] - mean_after) / sigma_after) ** 2)) - \
                           np.log((1 / (np.sqrt(2 * np.pi) * sigma_before)) * np.exp(-0.5 * ((data[i] - mean_before) / sigma_before) ** 2))
        cusum_scores[i] = max(0, cusum_scores[i-1] + likelihood_ratio)
        
    return cusum_scores

def analyze_cusum_with_methods(df, buses, statistics, cusum_threshold_values, p_values, aggregation_methods, sink_threshold_methods):
    all_method_a_results = []
    all_method_b_results = []
    all_individual_bus_results = []
    
    for threshold in cusum_threshold_values:
        bus_detections = {}
        bus_statistics = {}
        individual_bus_results = []  # 各バスの個別結果を格納するリスト
        
        for bus in buses:
            data = df[bus].values
            mean_before = statistics[bus]['mean_before']
            sigma_before = statistics[bus]['sigma_before']
            mean_after = statistics[bus]['mean_after']
            sigma_after = statistics[bus]['sigma_after']
            
            cusum_scores = cusum_for_each_bus(data, mean_before, sigma_before, mean_after, sigma_after, threshold)
            detections = (cusum_scores > threshold).astype(int)
            bus_detections[bus] = detections
            bus_statistics[bus] = cusum_scores
            
            # 各バスの個別性能を評価
            labels = df['Label'].values
            accuracy = accuracy_score(labels, detections)
            precision = precision_score(labels, detections, zero_division=0)
            recall = recall_score(labels, detections)
            f1 = f1_score(labels, detections)
            far, ed = calculate_far_ed(labels, detections, np.argmax(detections) if np.any(detections) else -1)
            
            individual_bus_results.append({
                'Bus': bus,
                'Method': 'Individual CUSUM',
                'Cusum Threshold': threshold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'False Alarm Rate': far,
                'Expected Delay': ed,
                'Detection Time': np.argmax(detections) if np.any(detections) else -1
            })
        
        method_a_results = apply_method_a(bus_detections, p_values)
        method_b_results =  apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods)
        
        labels = df['Label'].values
        method_a_results = evaluate_method_a(method_a_results, labels)
        method_b_results = evaluate_method_b(method_b_results, labels)
        
        all_method_a_results.extend(method_a_results)
        all_method_b_results.extend(method_b_results)
        all_individual_bus_results.extend(individual_bus_results)
    
    return pd.DataFrame(all_method_a_results), pd.DataFrame(all_method_b_results), pd.DataFrame(all_individual_bus_results)
