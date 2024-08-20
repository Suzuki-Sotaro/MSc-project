# pca_detection.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from method_a import apply_method_a, evaluate_method_a, calculate_far_ed
from method_b import apply_method_b, evaluate_method_b

def transform_time_series(data, d):
    return np.array([data[i:i+d] for i in range(len(data) - d + 1)])

def compute_pca_statistics(S1, S2, gamma):
    # Compute mean and covariance matrix
    mean = np.mean(S1, axis=0)
    cov_matrix = np.cov(S1.T)
    
    # Perform PCA
    pca = PCA()
    pca.fit(S1)
    
    # Determine the number of components to retain
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    r = np.argmax(cumulative_variance_ratio >= gamma) + 1
    
    # Form the matrix V
    V = pca.components_[:r].T
    
    # Compute residual terms for S2
    residuals = np.array([compute_residual(x, mean, V) for x in S2])
    
    return mean, V, residuals

def compute_residual(x, mean, V):
    return x - mean - V @ V.T @ (x - mean)

def estimate_tail_probability(rt, residuals, N2):
    pt_hat = np.sum(np.linalg.norm(residuals, axis=1) > np.linalg.norm(rt)) / N2
    if pt_hat == 0:
        pt_hat = 1 / (N2 * np.log(N2))  # Small non-zero value
    return pt_hat

def evaluate_results(pred_labels, true_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    return cm, accuracy, precision, recall, f1

def learn_optimal_parameters(S1, S2, gamma_values, h_values, alpha):
    best_f1 = 0
    best_gamma = gamma_values[0]  # デフォルト値を最初のgamma値に設定
    best_h = h_values[0]  # デフォルト値を最初のh値に設定
    
    N2 = len(S2)
    
    for gamma in gamma_values:
        mean, V, residuals = compute_pca_statistics(S1, S2[:N2//2], gamma)
        
        for h in h_values:
            anomalies = []
            gt = 0
            for t, xt in enumerate(S2[N2//2:]):
                rt = compute_residual(xt, mean, V)
                pt_hat = estimate_tail_probability(rt, residuals, len(residuals))
                st = np.log(alpha / pt_hat)
                gt = max(0, gt + st)
                anomalies.append(gt >= h)
            
            true_labels = np.zeros(len(S2[N2//2:]), dtype=bool)
            true_labels[len(S2[N2//2:])//2:] = True
            
            _, _, _, _, f1 = evaluate_results(anomalies, true_labels)
            
            if f1 > best_f1:
                best_f1 = f1
                best_gamma = gamma
                best_h = h
    
    return best_gamma, best_h

def analyze_pca(df, buses, d, gamma_values, h_values, alpha):
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
        
        # Learn optimal parameters
        optimal_gamma, optimal_h = learn_optimal_parameters(S1, S2[:N2//2], gamma_values, h_values, alpha)
        
        # optimal_gammaがNoneの場合のデフォルト値を設定
        if optimal_gamma is None:
            optimal_gamma = gamma_values[0]
        
        mean, V, residuals = compute_pca_statistics(S1, S2, optimal_gamma)
        
        anomalies = []
        statistics = []
        gt = 0
        for t, xt in enumerate(transformed_data[N1:], start=N1):
            rt = compute_residual(xt, mean, V)
            pt_hat = estimate_tail_probability(rt, residuals, N2)
            st = np.log(alpha / pt_hat)
            gt = max(0, gt + st)
            anomalies.append(gt >= optimal_h)
            statistics.append(gt)
        
        bus_anomalies[bus] = np.array(anomalies)
        bus_statistics[bus] = np.array(statistics)

        detection_times[bus] = next((t for t, anomaly in enumerate(anomalies) if anomaly), -1)
        
        adjusted_labels = labels[d-1:]
        pred_labels = np.zeros_like(adjusted_labels, dtype=bool)
        pred_labels[N1:] = anomalies
        
        cm, accuracy, precision, recall, f1 = evaluate_results(pred_labels[N1:], adjusted_labels[N1:])
        far, ed = calculate_far_ed(adjusted_labels[N1:], pred_labels[N1:], detection_times[bus])
        
        results.append({
            'Bus': bus,
            'd': d,
            'Optimal Gamma': optimal_gamma,
            'Optimal h': optimal_h,
            'Confusion Matrix': cm,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'False Alarm Rate': far,
            'Expected Delay': ed
        })
    
    return pd.DataFrame(results), bus_anomalies, bus_statistics, detection_times

def analyze_pca_with_methods(df, buses, d, gamma_values, h_values, alpha, p_values, aggregation_methods, sink_threshold_methods):
    pca_results, bus_anomalies, bus_statistics, detection_times = analyze_pca(df, buses, d, gamma_values, h_values, alpha)
    
    method_a_results = apply_method_a(bus_anomalies, p_values)
    method_b_results = apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods)
    
    labels = df['Label'].values[d-1:]
    N1 = len(labels) // 2  # Online detection phase start point

    method_a_results = evaluate_method_a(method_a_results, labels[N1:])
    method_b_results = evaluate_method_b(method_b_results, labels[N1:])

    return (pca_results, 
            pd.DataFrame(method_a_results), 
            pd.DataFrame(method_b_results), 
            )
