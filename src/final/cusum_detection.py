# Below is the content of cusum_detection.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    detection_point = -1
    
    for i in range(1, n):
        likelihood_ratio = np.log((1 / (np.sqrt(2 * np.pi) * sigma_after)) * np.exp(-0.5 * ((data[i] - mean_after) / sigma_after) ** 2)) - \
                           np.log((1 / (np.sqrt(2 * np.pi) * sigma_before)) * np.exp(-0.5 * ((data[i] - mean_before) / sigma_before) ** 2))
        cusum_scores[i] = max(0, cusum_scores[i-1] + likelihood_ratio)
        
    return cusum_scores

def method_a_cusum(bus_detections, p_values):
    results = []
    n_buses = len(bus_detections)
    
    # Scheme 1: At least one bus
    combined_detections_one = np.any(list(bus_detections.values()), axis=0)
    detection_time_one = np.argmax(combined_detections_one) if np.any(combined_detections_one) else -1
    
    results.append({
        'Method': 'Method A (At least one bus)',
        'Detections': combined_detections_one,
        'Detection Time': detection_time_one
    })
    
    # Scheme 2: All buses
    combined_detections_all = np.all(list(bus_detections.values()), axis=0)
    detection_time_all = np.argmax(combined_detections_all) if np.any(combined_detections_all) else -1
    
    results.append({
        'Method': 'Method A (All buses)',
        'Detections': combined_detections_all,
        'Detection Time': detection_time_all
    })
    
    # Scheme 3: p% of buses
    for p in p_values:
        threshold = int(p * n_buses)
        combined_detections = np.zeros_like(list(bus_detections.values())[0])
        detection_time = -1
        
        for t in range(len(combined_detections)):
            votes = sum(detections[t] for detections in bus_detections.values())
            if votes >= threshold:
                combined_detections[t] = 1
                if detection_time == -1:
                    detection_time = t
        
        results.append({
            'Method': f'Method A (p={p})',
            'Detections': combined_detections,
            'Detection Time': detection_time
        })
    
    return results

def method_b_cusum(bus_statistics, aggregation_methods, sink_threshold_methods):
    results = []
    local_thresholds = [stats.max() for stats in bus_statistics.values()]
    
    for agg_method in aggregation_methods:
        for sink_method in sink_threshold_methods:
            combined_detections = np.zeros_like(list(bus_statistics.values())[0])
            detection_time = -1
            
            if sink_method == 'average':
                H = np.mean(local_thresholds)
            elif sink_method == 'minimum':
                H = np.min(local_thresholds)
            elif sink_method == 'maximum':
                H = np.max(local_thresholds)
            elif sink_method == 'median':
                H = np.median(local_thresholds)
            
            for t in range(len(combined_detections)):
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
                    combined_detections[t] = 1
                    if detection_time == -1:
                        detection_time = t
            
            results.append({
                'Method': f'Method B ({agg_method}, {sink_method})',
                'Detections': combined_detections,
                'Detection Time': detection_time
            })
    
    return results

def analyze_cusum_with_methods(df, buses, statistics, threshold, p_values, aggregation_methods, sink_threshold_methods):
    bus_detections = {}
    bus_statistics = {}
    
    for bus in buses:
        data = df[bus].values
        mean_before = statistics[bus]['mean_before']
        sigma_before = statistics[bus]['sigma_before']
        mean_after = statistics[bus]['mean_after']
        sigma_after = statistics[bus]['sigma_after']
        
        cusum_scores = cusum_for_each_bus(data, mean_before, sigma_before, mean_after, sigma_after, threshold)
        bus_detections[bus] = (cusum_scores > threshold).astype(int)
        bus_statistics[bus] = cusum_scores
    
    method_a_results = method_a_cusum(bus_detections, p_values)
    method_b_results = method_b_cusum(bus_statistics, aggregation_methods, sink_threshold_methods)
    
    labels = df['Label'].values
    
    for result in method_a_results + method_b_results:
        detections = result['Detections']
        accuracy = accuracy_score(labels, detections)
        precision = precision_score(labels, detections)
        recall = recall_score(labels, detections)
        f1 = f1_score(labels, detections)
        
        result.update({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    
    return pd.DataFrame(method_a_results), pd.DataFrame(method_b_results)

def analyze_cusum(df, buses, statistics, threshold_values):
    results = []
    
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]
    aggregation_methods = ['average', 'median', 'outlier_detection']
    sink_threshold_methods = ['average', 'minimum', 'maximum', 'median']
    
    for threshold in threshold_values:
        method_a_results, method_b_results = analyze_cusum_with_methods(
            df, buses, statistics, threshold, p_values, aggregation_methods, sink_threshold_methods
        )
        
        results.extend(method_a_results.to_dict('records'))
        results.extend(method_b_results.to_dict('records'))
    
    return pd.DataFrame(results)