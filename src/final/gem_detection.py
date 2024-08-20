# below is the code for the gem_detection.py file
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from method_a import apply_method_a, evaluate_method_a
from method_b import apply_method_b, evaluate_method_b
from utils import calculate_far_ed

def transform_time_series(data, d):
    return np.array([data[i:i+d] for i in range(len(data) - d + 1)])

def calculate_gem_statistic(S1, S2, k):
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(S1)
    distances, _ = nbrs.kneighbors(S2)
    
    gem_stats = np.zeros(len(S2))
    for i, dist in enumerate(distances):
        volume = np.prod(dist)
        if volume == 0:
            volume = np.finfo(float).eps
        density = k / volume
        gem_stats[i] = -np.log(density)
    
    return gem_stats

def estimate_tail_probability(dt, gem_stats, N2):
    pt_hat = np.sum(gem_stats > dt) / N2
    if pt_hat == 0:
        pt_hat = 1 / (N2 * 10)  
    return pt_hat

def evaluate_results(pred_labels, true_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    return cm, accuracy, precision, recall, f1

def binary_detection(data, threshold):
    return (data >= threshold).astype(int)

def learn_optimal_threshold(S1, S2, k, alpha_values, h_values):
    best_f1 = 0
    best_threshold = None
    best_k = None
    best_alpha = None
    
    N1, N2 = len(S1), len(S2)
    
    for k_candidate in range(1, min(k, N1)):  # kの値を動的に調整
        gem_stats_S2 = calculate_gem_statistic(S1, S2, k_candidate)
        
        for alpha in alpha_values:
            for h in h_values:
                anomalies = []
                gt = 0
                for t, xt in enumerate(S2):
                    dt = calculate_gem_statistic(S1, [xt], k_candidate)[0]
                    pt_hat = estimate_tail_probability(dt, gem_stats_S2, N2)
                    st = np.log(alpha / pt_hat)
                    gt = max(0, gt + st)
                    anomalies.append(gt >= h)
                
                # S2の後半を真のラベルとして使用
                true_labels = np.zeros(len(S2), dtype=bool)
                true_labels[len(S2)//2:] = True
                
                _, _, _, _, f1 = evaluate_results(anomalies, true_labels)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = h
                    best_k = k_candidate
                    best_alpha = alpha
    
    return best_threshold, best_k, best_alpha

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
        
        # オフラインフェーズ: BP-GEMメソッドを用いて最適なパラメータを学習
        optimal_threshold, optimal_k, optimal_alpha = learn_optimal_threshold(S1, S2[:N2//2], max(k_values), alpha_values, h_values)
        
        gem_stats_S2 = calculate_gem_statistic(S1, S2, optimal_k)
        sorted_gem_stats = np.sort(gem_stats_S2)
        
        # オンラインフェーズ: 異常検出を実行
        anomalies = []
        statistics = []
        gt = 0
        for t, xt in enumerate(transformed_data[N1:], start=N1):
            dt = calculate_gem_statistic(S1, [xt], optimal_k)[0]
            pt_hat = estimate_tail_probability(dt, sorted_gem_stats, N2)
            st = np.log(optimal_alpha / pt_hat)
            gt = max(0, gt + st)
            anomalies.append(gt >= optimal_threshold)
            statistics.append(gt)
        
        bus_anomalies[bus] = np.array(anomalies)
        bus_statistics[bus] = np.array(statistics)

        detection_times[bus] = next((t for t, anomaly in enumerate(anomalies) if anomaly), -1)
        
        # バイナリ検出の実装
        binary_threshold = np.mean(data[:N1])  # 事前変化の平均を閾値として使用
        binary_detections[bus] = binary_detection(data[N1:], binary_threshold)
        
        adjusted_labels = labels[d-1:]
        pred_labels = np.zeros_like(adjusted_labels, dtype=bool)
        pred_labels[N1:] = anomalies
        
        cm, accuracy, precision, recall, f1 = evaluate_results(pred_labels[N1:], adjusted_labels[N1:])
        far, ed = calculate_far_ed(adjusted_labels[N1:], pred_labels[N1:], detection_times[bus])
        
        results.append({
            'Bus': bus,
            'd': d,
            'k': optimal_k,
            'alpha': optimal_alpha,
            'Optimal Threshold': optimal_threshold,
            'Confusion Matrix': cm,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'False Alarm Rate': far,
            'Expected Delay': ed
        })
    
    return pd.DataFrame(results), bus_anomalies, bus_statistics, detection_times, binary_detections

def analyze_gem_with_methods(df, buses, d, k_values, alpha_values, h_values, p_values, aggregation_methods, sink_threshold_methods):
    gem_results, bus_anomalies, bus_statistics, detection_times, binary_detections = analyze_gem(df, buses, d, k_values, alpha_values, h_values)
    
    method_a_results = apply_method_a(bus_anomalies, p_values)
    method_b_results = apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods)
    
    labels = df['Label'].values[d-1:]
    N1 = len(labels) // 2  # オンライン検出フェーズの開始点

    method_a_results = evaluate_method_a(method_a_results, labels[N1:])
    method_b_results = evaluate_method_b(method_b_results, labels[N1:])

    return (gem_results, 
            pd.DataFrame(method_a_results), 
            pd.DataFrame(method_b_results), 
            )