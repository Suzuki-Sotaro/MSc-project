# Below is the content of cusum_detection.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from method_a import apply_method_a, evaluate_method_a
from method_b import apply_method_b, evaluate_method_b
from utils import calculate_far_ed, evaluate_results

def calculate_statistics(df, buses):
    statistics = {}
    for bus in buses:
        bus_data = df[bus].values
        statistics[bus] = {
            'mean': np.mean(bus_data)
        }
    return statistics

def nonparametric_cusum_for_each_bus(data, mean, threshold):
    n = len(data)
    cusum_scores = np.zeros(n)
    
    for i in range(1, n):
        cusum_scores[i] = max(0, cusum_scores[i-1] + data[i] - mean)
        
    return cusum_scores

def analyze_cusum_with_methods(df, buses, statistics, cusum_threshold_values, p_values, aggregation_methods, sink_threshold_methods):
    all_method_a_results = []
    all_method_b_results = []
    all_individual_bus_results = []
    
    for threshold in cusum_threshold_values:
        bus_detections = {}
        bus_statistics = {}
        individual_bus_results = [] 
        for bus in buses:
            data = df[bus].values
            mean = statistics[bus]['mean']
            
            cusum_scores = nonparametric_cusum_for_each_bus(data, mean, threshold)
            detections = (cusum_scores > threshold).astype(int)
            bus_detections[bus] = detections
            bus_statistics[bus] = cusum_scores
            
            # 各バスの個別性能を評価
            labels = df['Label'].values
            cm, accuracy, precision, recall, f1 = evaluate_results(detections, labels)
            far, ed = calculate_far_ed(labels, detections, np.argmax(detections) if np.any(detections) else -1)
            
            individual_bus_results.append({
                'Bus': bus,
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
        method_b_results = apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods)
        
        labels = df['Label'].values
        method_a_results = evaluate_method_a(method_a_results, labels)
        method_b_results = evaluate_method_b(method_b_results, labels)
        
        # Add the threshold to each record in method_a_results and method_b_results
        for record in method_a_results:
            record['Cusum Threshold'] = threshold
        for record in method_b_results:
            record['Cusum Threshold'] = threshold
        
        all_method_a_results.extend(method_a_results)
        all_method_b_results.extend(method_b_results)
        all_individual_bus_results.extend(individual_bus_results)
        
    print("Nonparametric CUSUM analysis completed.")
    
    return pd.DataFrame(all_method_a_results), pd.DataFrame(all_method_b_results), pd.DataFrame(all_individual_bus_results)