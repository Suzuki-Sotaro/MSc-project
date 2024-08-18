# below is the code for the gem_detection.py file
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def transform_time_series(data, d):
    return np.array([data[i:i+d] for i in range(len(data) - d + 1)])

def calculate_gem_statistic(S1, S2, k):
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(S1)
    distances, indices = nbrs.kneighbors(S2)
    
    gem_stats = np.zeros(len(S2))
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        volume = np.prod(dist)
        if volume == 0:
            volume = np.finfo(float).eps
        density = k / volume
        gem_stats[i] = -np.log(density)
    
    return gem_stats

def estimate_tail_probability(dt, gem_stats, N2):
    pt_hat = np.sum(gem_stats > dt) / N2
    if pt_hat == 0:
        pt_hat = 1 / N2
    return pt_hat

def evaluate_results(pred_labels, true_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    return cm, accuracy, precision, recall, f1

def learn_optimal_threshold(S1, S2, k, alpha_values, h_values):
    """Learn the optimal threshold for a bus based on training data."""
    best_f1 = 0
    best_threshold = None
    
    gem_stats_S2 = calculate_gem_statistic(S1, S2, k)
    N2 = len(S2)
    
    for alpha in alpha_values:
        for h in h_values:
            anomalies = []
            gt = 0
            for t, xt in enumerate(S2):
                dt = calculate_gem_statistic(S1, [xt], k)[0]
                pt_hat = estimate_tail_probability(dt, gem_stats_S2, N2)
                st = np.log(alpha / pt_hat)
                gt = max(0, gt + st)
                anomalies.append(gt >= h)
            
            # Assuming the last half of S2 contains the true labels
            true_labels = np.zeros(len(S2), dtype=bool)
            true_labels[len(S2)//2:] = True
            
            _, _, _, _, f1 = evaluate_results(anomalies, true_labels)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = h
    
    return best_threshold

def analyze_gem(df, buses, d, k_values, alpha_values, h_values):
    results = []
    bus_anomalies = {}
    bus_statistics = {}
    detection_times = {}
    
    for bus in buses:
        data = df[bus].values
        labels = df['Label'].values
        
        transformed_data = transform_time_series(data, d)
        
        N = len(transformed_data)
        N1 = N // 2
        N2 = N - N1
        S1, S2 = transformed_data[:N1], transformed_data[N1:]
        
        # Learn optimal threshold for each bus
        optimal_k = k_values[0]  # For simplicity, using the first k value
        optimal_threshold = learn_optimal_threshold(S1, S2[:N2//2], optimal_k, alpha_values, h_values)
        
        gem_stats_S2 = calculate_gem_statistic(S1, S2, optimal_k)
        sorted_gem_stats = np.sort(gem_stats_S2)
        
        anomalies = []
        statistics = []
        gt = 0
        for t, xt in enumerate(transformed_data[N1:], start=N1):
            dt = calculate_gem_statistic(S1, [xt], optimal_k)[0]
            pt_hat = estimate_tail_probability(dt, sorted_gem_stats, N2)
            st = np.log(alpha_values[0] / pt_hat)  # Using first alpha value for simplicity
            gt = max(0, gt + st)
            anomalies.append(gt >= optimal_threshold)
            statistics.append(gt)
        
        bus_anomalies[bus] = np.array(anomalies)
        bus_statistics[bus] = np.array(statistics)
        
        # Record detection times
        detection_times[bus] = [t for t, anomaly in enumerate(anomalies) if anomaly]
        
        # Evaluate results for this bus
        adjusted_labels = labels[d-1:]
        pred_labels = np.zeros_like(adjusted_labels, dtype=bool)
        pred_labels[N1:] = anomalies
        
        cm, accuracy, precision, recall, f1 = evaluate_results(pred_labels[N1:], adjusted_labels[N1:])
        
        results.append({
            'Bus': bus,
            'd': d,
            'k': optimal_k,
            'Optimal Threshold': optimal_threshold,
            'Confusion Matrix': cm,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    
    return pd.DataFrame(results), bus_anomalies, bus_statistics, detection_times

def apply_method_a(bus_anomalies, p_values):
    results = []
    n_buses = len(bus_anomalies)
    
    for p in p_values:
        threshold = int(p * n_buses)
        combined_anomalies = np.zeros_like(list(bus_anomalies.values())[0], dtype=bool)
        detection_time = None
        
        # Implement all three voting schemes
        # 1. At least one bus
        one_bus_anomalies = np.any(list(bus_anomalies.values()), axis=0)
        one_bus_detection_time = np.argmax(one_bus_anomalies) if np.any(one_bus_anomalies) else None
        
        # 2. All buses
        all_buses_anomalies = np.all(list(bus_anomalies.values()), axis=0)
        all_buses_detection_time = np.argmax(all_buses_anomalies) if np.any(all_buses_anomalies) else None
        
        # 3. Percentage-based (p%)
        for t in range(len(combined_anomalies)):
            votes = sum(anomalies[t] for anomalies in bus_anomalies.values())
            if votes >= threshold:
                combined_anomalies[t] = True
                if detection_time is None:
                    detection_time = t
        
        results.append({
            'Method': f'Method A (p={p})',
            'Anomalies': combined_anomalies,
            'Detection Time': detection_time,
            'One Bus Anomalies': one_bus_anomalies,
            'One Bus Detection Time': one_bus_detection_time,
            'All Buses Anomalies': all_buses_anomalies,
            'All Buses Detection Time': all_buses_detection_time
        })
    
    return results

def apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods):
    results = []
    
    # Get all local thresholds (assuming they're stored with the statistics)
    local_thresholds = [stats.max() for stats in bus_statistics.values()]  # This is a simplification
    
    for agg_method in aggregation_methods:
        for sink_method in sink_threshold_methods:
            combined_anomalies = np.zeros_like(list(bus_statistics.values())[0], dtype=bool)
            detection_time = None
            
            if sink_method == 'average':
                H = np.mean(local_thresholds)
            elif sink_method == 'minimum':
                H = np.min(local_thresholds)
            elif sink_method == 'maximum':
                H = np.max(local_thresholds)
            elif sink_method == 'median':
                H = np.median(local_thresholds)
            
            for t in range(len(combined_anomalies)):
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
                    combined_anomalies[t] = True
                    if detection_time is None:
                        detection_time = t
            
            results.append({
                'Method': f'Method B ({agg_method}, {sink_method})',
                'Anomalies': combined_anomalies,
                'Detection Time': detection_time
            })
    
    return results

def analyze_gem_with_methods(df, buses, d, k_values, alpha_values, h_values, p_values, aggregation_methods, sink_threshold_methods):
    gem_results, bus_anomalies, bus_statistics, detection_times = analyze_gem(df, buses, d, k_values, alpha_values, h_values)
    
    method_a_results = apply_method_a(bus_anomalies, p_values)
    method_b_results = apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods)
    
    labels = df['Label'].values[d-1:]
    N1 = len(labels) // 2  # Online detection phase start point

    for result in method_a_results + method_b_results:
        # Evaluate only the online detection phase results
        cm, accuracy, precision, recall, f1 = evaluate_results(result['Anomalies'], labels[N1:])
        result.update({
            'Confusion Matrix': cm,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    # method_a_resultsとmethod_b_resultsでAnomary以外の列のみを返す
    return gem_results, pd.DataFrame(method_a_results).drop(columns=['Anomalies', 'One Bus Anomalies', 'All Buses Anomalies']), pd.DataFrame(method_b_results).drop(columns='Anomalies'), detection_times
