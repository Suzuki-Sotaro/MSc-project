# below is the code for the gem_detection.py file
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2
from scipy.special import gamma
from method_a import apply_method_a, evaluate_method_a
from method_b import apply_method_b, evaluate_method_b
from utils import calculate_far_ed, evaluate_results

def estimate_intrinsic_dimension(data, k=10):
    """
    Estimate the intrinsic dimension of the data using the maximum likelihood method.
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(data)
    distances, _ = nbrs.kneighbors(data)
    distances = distances[:, 1:]  # Exclude self-distance
    
    N = len(data)
    d_mle = np.zeros(N)
    for i in range(N):
        d_mle[i] = 1 / (np.sum(np.log(distances[i][-1] / distances[i][:-1])) / (k - 1))
    
    return np.median(d_mle)

def transform_time_series(data, d):
    return np.array([data[i:i+d] for i in range(len(data) - d + 1)])

def calculate_gem_statistic(S1, S2, k, d):
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='ball_tree').fit(S1)
    distances, _ = nbrs.kneighbors(S2)
    
    gem_stats = np.zeros(len(S2))
    for i, dist in enumerate(distances):
        volume = np.prod(dist)
        if volume == 0:
            volume = np.finfo(float).eps
        density = k / (volume * np.pi**(d/2) / gamma(d/2 + 1))
        gem_stats[i] = -np.log(density)
    
    return gem_stats

def estimate_tail_probability(dt, gem_stats, N2):
    pt_hat = np.sum(gem_stats > dt) / N2
    if pt_hat == 0:
        pt_hat = 1 / (N2 * 10)  # Small non-zero value to prevent log(0)
    return pt_hat

def analyze_gem(df, buses, d, k_values, alpha_values, h_values):
    results = []
    bus_anomalies = {}
    bus_statistics = {}
    detection_times = {}
    binary_detections = {}
    
    for bus in buses:
        data = df[bus].values
        labels = df['Label'].values
        
        transformed_data = transform_time_series(data, d)
        
        N = len(transformed_data)
        N1 = N // 2
        N2 = N - N1
        S1, S2 = transformed_data[:N1], transformed_data[N1:]
        
        estimated_d = estimate_intrinsic_dimension(np.vstack((S1, S2)))
        
        for k in k_values:
            if k >= N1:
                continue
            
            gem_stats_S2 = calculate_gem_statistic(S1, S2, k, estimated_d)
            sorted_gem_stats = np.sort(gem_stats_S2)
            
            for alpha in alpha_values:
                for h in h_values:
                    # Online phase: Perform anomaly detection
                    anomalies = []
                    statistics = []
                    gt = 0
                    for t, xt in enumerate(transformed_data[N1:], start=N1):
                        dt = calculate_gem_statistic(S1, [xt], k, estimated_d)[0]
                        pt_hat = estimate_tail_probability(dt, sorted_gem_stats, N2)
                        st = np.log(alpha / pt_hat)
                        gt = max(0, gt + st)
                        anomalies.append(gt >= h)
                        statistics.append(gt)
                    
                    bus_anomalies[f"{bus}_k{k}_alpha{alpha}_h{h}"] = np.array(anomalies)
                    bus_statistics[f"{bus}_k{k}_alpha{alpha}_h{h}"] = np.array(statistics)
                    
                    detection_time = next((t for t, anomaly in enumerate(anomalies) if anomaly), -1)
                    detection_times[f"{bus}_k{k}_alpha{alpha}_h{h}"] = detection_time
                    
                    # Implement binary detection
                    binary_threshold = chi2.ppf(1 - alpha, df=estimated_d)
                    binary_detections[f"{bus}_k{k}_alpha{alpha}_h{h}"] = (gem_stats_S2 > binary_threshold).astype(int)
                    
                    adjusted_labels = labels[d-1:]
                    pred_labels = np.zeros_like(adjusted_labels, dtype=bool)
                    pred_labels[N1:] = anomalies
                    
                    cm, accuracy, precision, recall, f1 = evaluate_results(pred_labels[N1:], adjusted_labels[N1:])
                    far, ed = calculate_far_ed(adjusted_labels[N1:], pred_labels[N1:], detection_time)
                    
                    results.append({
                        'Bus': bus,
                        'K': k,
                        'Alpha': alpha,
                        'Threshold': h,
                        'd': d,
                        'Data': data.tolist()[d-1:][N1:],
                        'Label': adjusted_labels[N1:],
                        'Detection': pred_labels[N1:],
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        'False Alarm Rate': far,
                        'Expected Delay': ed,
                        'Detection Time': detection_time
                    })
    
    return pd.DataFrame(results), bus_anomalies, bus_statistics, detection_times, binary_detections

def analyze_gem_with_methods(df, buses, d, k_values, alpha_values, h_values, p_values, aggregation_methods, sink_threshold_methods):
    gem_results, bus_anomalies, bus_statistics, detection_times, binary_detections = analyze_gem(
        df, buses, d, k_values, alpha_values, h_values
    )
    
    method_a_results = apply_method_a(bus_anomalies, p_values, df, buses)
    method_b_results = apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods, df, buses)
    
    labels = df['Label'].values[d-1:].astype(int).tolist()
    N1 = len(labels) // 2  # Start of online detection phase
    
    method_a_results = evaluate_method_a(method_a_results, labels[N1:])
    method_b_results = evaluate_method_b(method_b_results, labels[N1:])
    
    # Convert the Label and Detection columns to the desired format
    gem_results['Label'] = gem_results['Label'].apply(lambda x: list(map(int, x)))
    gem_results['Detection'] = gem_results['Detection'].apply(lambda x: list(map(int, x)))
    
    method_a_results = pd.DataFrame(method_a_results)
    method_b_results = pd.DataFrame(method_b_results)

    print("GEM analysis completed.")
    
    return (
        gem_results,
        method_a_results,
        method_b_results
    )
