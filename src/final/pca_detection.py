# pca_detection.py
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from method_a import apply_method_a, evaluate_method_a
from method_b import apply_method_b, evaluate_method_b
from utils import calculate_far_ed, evaluate_results

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

def offline_phase(X, N1, N2, d, gamma):
    # Step 1: Choose subsets S1 and S2
    S1, S2 = X[:N1], X[N1:N1+N2]
    
    # Step 2: Compute x̄ and Q over S1
    x_bar = np.mean(S1, axis=0)
    Q = np.cov(S1.T)
    
    # Step 3: Compute eigenvalues and eigenvectors of Q
    eigenvalues, eigenvectors = np.linalg.eig(Q)
    
    # Step 4: Determine r and form matrix V
    cumulative_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    r = np.argmax(cumulative_variance_ratio >= gamma) + 1
    V = eigenvectors[:, :r]
    
    # Step 5-8: Compute residuals for S2
    residuals = []
    for x in S2:
        r = x - x_bar - V @ V.T @ (x - x_bar)
        residuals.append(np.linalg.norm(r))
    
    # Step 9: Sort residuals
    sorted_residuals = np.sort(residuals)
    
    return x_bar, V, sorted_residuals

def online_phase(x_bar, V, sorted_residuals, X_online, alpha, h):
    N2 = len(sorted_residuals)
    gt = 0
    anomalies = []
    
    for xt in X_online:
        # Step 5: Compute residual
        rt = xt - x_bar - V @ V.T @ (xt - x_bar)
        rt_norm = np.linalg.norm(rt)
        
        # Step 6: Estimate tail probability
        pt = np.sum(sorted_residuals > rt_norm) / N2
        if pt == 0:
            pt = 1 / (N2 * np.log(N2))  # Small non-zero value
        
        # Step 7: Compute statistical evidence
        st = np.log(alpha / pt)
        
        # Step 8: Update decision statistic
        gt = max(0, gt + st)
        
        anomalies.append(gt >= h)
    
    return np.array(anomalies)

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
        
        # Offline phase
        x_bar, V, sorted_residuals = offline_phase(transformed_data, N1, N2, d, gamma_values[0])
        
        # Online phase
        anomalies = online_phase(x_bar, V, sorted_residuals, transformed_data[N1:], alpha, h_values[0])
        
        bus_anomalies[bus] = anomalies
        bus_statistics[bus] = sorted_residuals
        detection_times[bus] = next((t for t, anomaly in enumerate(anomalies) if anomaly), -1)
        
        adjusted_labels = labels[d-1+N1:]
        cm, accuracy, precision, recall, f1 = evaluate_results(anomalies, adjusted_labels)
        far, ed = calculate_far_ed(adjusted_labels, anomalies, detection_times[bus])
        
        results.append({
            'Bus': bus,
            'd': d,
            'Gamma': gamma_values[0],
            'h': h_values[0],
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
            pd.DataFrame(method_b_results))