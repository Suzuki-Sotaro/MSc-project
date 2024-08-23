# pca_detection.py
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from method_a import apply_method_a, evaluate_method_a
from method_b import apply_method_b, evaluate_method_b
from utils import calculate_far_ed, evaluate_results

def transform_time_series(data, d):
    return np.array([data[i:i+d] for i in range(len(data) - d + 1)])

def offline_phase(S1, S2, gamma):
    # Step 2-3: Compute x̄ and Q over S1
    x_bar = np.mean(S1, axis=0)
    Q = np.cov(S1.T)
    
    # Step 4: Compute eigenvalues and eigenvectors of Q
    eigenvalues, eigenvectors = np.linalg.eig(Q)
    
    # Determine r based on gamma
    cumulative_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    r = np.argmax(cumulative_variance_ratio >= gamma) + 1
    V = eigenvectors[:, :r]
    
    # Compute residuals for S2
    residuals = []
    for x in S2:
        r = x - x_bar - V @ V.T @ (x - x_bar)
        residuals.append(np.linalg.norm(r))
    
    sorted_residuals = np.sort(residuals)
    
    return x_bar, V, sorted_residuals

def online_phase(x_bar, V, sorted_residuals, X_online, alpha, h):
    N2 = len(sorted_residuals)
    gt = 0
    anomalies = []
    
    for xt in X_online:
        # Compute residual
        rt = xt - x_bar - V @ V.T @ (xt - x_bar)
        rt_norm = np.linalg.norm(rt)
        
        # Estimate tail probability
        pt = np.sum(sorted_residuals > rt_norm) / N2
        if pt == 0:
            pt = 1 / (N2 * np.log(N2))  # Small non-zero value
        
        # Compute statistical evidence
        st = np.log(alpha / pt)
        
        # Update decision statistic
        gt = max(0, gt + st)
        
        anomalies.append(gt >= h)
    
    return np.array(anomalies)

def analyze_pca_with_methods(df, buses, d, gamma_values, h_values, alpha, p_values, aggregation_methods, sink_threshold_methods):
    all_individual_bus_results = []
    all_method_a_results = []
    all_method_b_results = []
    
    for gamma in gamma_values:
        for h in h_values:
            bus_anomalies = {}
            bus_statistics = {}
            
            for bus in buses:
                data = df[bus].values
                transformed_data = transform_time_series(data, d)
                
                N = len(transformed_data)
                N1 = N // 2
                N2 = N - N1
                
                # Offline phase
                S1, S2 = transformed_data[:N1], transformed_data[N1:]
                x_bar, V, sorted_residuals = offline_phase(S1, S2, gamma)
                
                # Online phase
                anomalies = online_phase(x_bar, V, sorted_residuals, transformed_data[N1:], alpha, h)
                
                bus_anomalies[bus] = anomalies
                bus_statistics[bus] = sorted_residuals
                
                labels = df['Label'].values[d-1+N1:]
                cm, accuracy, precision, recall, f1 = evaluate_results(anomalies, labels)
                far, ed = calculate_far_ed(labels, anomalies, np.argmax(anomalies) if np.any(anomalies) else -1)
                
                all_individual_bus_results.append({
                    'Bus': bus,
                    'd': d,
                    'Gamma': gamma,
                    'h': h,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'False Alarm Rate': far,
                    'Expected Delay': ed,
                    'Detection Time': np.argmax(anomalies) if np.any(anomalies) else -1
                })
            
            method_a_results = apply_method_a(bus_anomalies, p_values)
            method_b_results = apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods)
            
            labels = df['Label'].values[d-1+N1:]
            method_a_results = evaluate_method_a(method_a_results, labels)
            method_b_results = evaluate_method_b(method_b_results, labels)
            
            for result in method_a_results + method_b_results:
                result['Gamma'] = gamma
                result['h'] = h
            
            all_method_a_results.extend(method_a_results)
            all_method_b_results.extend(method_b_results)
    
    print("PCA analysis completed.")
    return (pd.DataFrame(all_individual_bus_results), 
            pd.DataFrame(all_method_a_results), 
            pd.DataFrame(all_method_b_results))