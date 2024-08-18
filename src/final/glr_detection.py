# below is the content of glr_detection.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def glr_detect(data, theta0, sigma, h, nu_min=0):
    n = len(data)
    g = np.zeros(n)
    t_hat = 0
    
    for k in range(1, n):
        max_likelihood = -np.inf
        max_j = 0
        
        for j in range(1, k+1):
            mean_diff = np.mean(data[j-1:k]) - theta0
            nu = max(mean_diff, nu_min)
            
            likelihood = np.sum((nu/sigma**2) * (data[j-1:k] - theta0) - 0.5 * (nu**2/sigma**2))
            
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                max_j = j
        
        g[k-1] = max_likelihood
        
        if g[k-1] > h:
            t_hat = max_j
            break
    
    return t_hat, g

def plot_glr_results(data, change_point, glr_scores, detection_points, threshold_values, method_name, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(data, label='Data')
    ax1.axvline(change_point, color='red', linestyle='--', label='True Change Point')
    ax1.set_ylabel('Data Value')
    ax1.set_title(f'{method_name} Change Detection - Data and Change Points')
    ax1.legend()

    for i, (threshold, glr_score) in enumerate(zip(threshold_values, glr_scores)):
        if detection_points[i] != -1:
            ax2.axvline(detection_points[i], color=f'C{i}', linestyle='--', label=f'Detected Change (h={threshold:.2f})')
        ax2.plot(glr_score, color=f'C{i}', label=f'GLR Scores (h={threshold:.2f})')
        ax2.axhline(threshold, color=f'C{i}', linestyle=':', label=f'Threshold (h={threshold:.2f})')

    ax2.set_xlabel('Sample')
    ax2.set_ylabel('GLR Score')
    ax2.set_title('GLR Scores and Detection Points')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def calculate_glr_scores(df, buses, statistics, threshold):
    bus_detections = {}
    bus_scores = {}
    
    for bus in buses:
        data = df[bus].values
        theta0 = statistics[bus]['mean_before']
        sigma = statistics[bus]['sigma_before']
        
        detection_point, glr_scores = glr_detect(data, theta0, sigma, threshold)
        bus_detections[bus] = (glr_scores > threshold).astype(int)
        bus_scores[bus] = glr_scores
    
    return bus_detections, bus_scores

def method_a_glr(bus_detections, p_values):
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

def method_b_glr(bus_scores, aggregation_methods, sink_threshold_methods):
    results = []
    local_thresholds = [scores.max() for scores in bus_scores.values()]
    
    for agg_method in aggregation_methods:
        for sink_method in sink_threshold_methods:
            combined_detections = np.zeros_like(list(bus_scores.values())[0])
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
                statistics = [scores[t] for scores in bus_scores.values()]
                
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

def analyze_glr_with_methods(df, buses, statistics, threshold, p_values, aggregation_methods, sink_threshold_methods):
    bus_detections, bus_scores = calculate_glr_scores(df, buses, statistics, threshold)
    
    method_a_results = method_a_glr(bus_detections, p_values)
    method_b_results = method_b_glr(bus_scores, aggregation_methods, sink_threshold_methods)
    
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

def analyze_glr(df, buses, statistics, threshold_values):
    results = []
    
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]
    aggregation_methods = ['average', 'median', 'outlier_detection']
    sink_threshold_methods = ['average', 'minimum', 'maximum', 'median']
    
    for threshold in threshold_values:
        method_a_results, method_b_results = analyze_glr_with_methods(
            df, buses, statistics, threshold, p_values, aggregation_methods, sink_threshold_methods
        )
        
        results.extend(method_a_results.to_dict('records'))
        results.extend(method_b_results.to_dict('records'))
    
    return pd.DataFrame(results)

def find_optimal_threshold(df, buses, statistics, threshold_range):
    best_threshold = None
    best_f1 = -1
    
    for threshold in threshold_range:
        results_df = analyze_glr(df, buses, statistics, [threshold])
        avg_f1 = results_df['F1 Score'].mean()
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = threshold
    
    return best_threshold, best_f1

def main(df, buses, statistics):
    # Find optimal threshold
    threshold_range = np.arange(0.1, 5.1, 0.1)
    optimal_threshold, best_f1 = find_optimal_threshold(df, buses, statistics, threshold_range)
    print(f"Optimal threshold: {optimal_threshold:.2f} (F1 Score: {best_f1:.4f})")
    
    # Analyze using optimal threshold
    results = analyze_glr(df, buses, statistics, [optimal_threshold])
    
    # Print results
    print("\nResults:")
    print(results.to_string(index=False))
    
    # Plot results for the best performing method
    best_method = results.loc[results['F1 Score'].idxmax()]
    plot_glr_results(
        df[buses[0]].values,  # Assuming the first bus is representative
        df['Label'].values.argmax(),  # Assuming the first 1 in Label column is the true change point
        [best_method['Detections']],
        [best_method['Detection Time']],
        [optimal_threshold],
        best_method['Method'],
        'best_glr_result.png'
    )
