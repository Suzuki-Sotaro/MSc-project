# below is the content of glr_detection.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from method_a import apply_method_a, evaluate_method_a
from method_b import apply_method_b, evaluate_method_b
from utils import calculate_far_ed

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

def analyze_glr_with_methods(df, buses, statistics, glr_threshold_values, p_values, aggregation_methods, sink_threshold_methods):
    bus_detections, bus_scores = calculate_glr_scores(df, buses, statistics, glr_threshold_values)
    
    method_a_results = apply_method_a(bus_detections, p_values)
    method_b_results = apply_method_b(bus_scores, aggregation_methods, sink_threshold_methods)
    
    labels = df['Label'].values
    
    method_a_results = evaluate_method_a(method_a_results, labels)
    method_b_results = evaluate_method_b(method_b_results, labels)
    
    # 各バスの個別性能を評価
    individual_bus_results = []
    for bus, detections in bus_detections.items():
        accuracy = accuracy_score(labels, detections)
        precision = precision_score(labels, detections, zero_division=0)
        recall = recall_score(labels, detections)
        f1 = f1_score(labels, detections)
        detection_time = np.argmax(detections) if np.any(detections) else -1
        far, ed = calculate_far_ed(labels, detections, detection_time)
        
        individual_bus_results.append({
            'Bus': bus,
            'Method': 'Individual GLR',
            'GLR Threshold Values': glr_threshold_values,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'False Alarm Rate': far,
            'Expected Delay': ed,
            'Detection Time': detection_time
        })
    
    return (pd.DataFrame(individual_bus_results),
            pd.DataFrame(method_a_results).drop(columns='Detections'), 
            pd.DataFrame(method_b_results).drop(columns='Detections'), 
            )

def analyze_glr(df, buses, statistics, threshold_values):
    results_a = []
    results_b = []
    individual_results = []
    
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]
    aggregation_methods = ['average', 'median', 'outlier_detection']
    sink_threshold_methods = ['average', 'minimum', 'maximum', 'median']
    
    for threshold in threshold_values:
        bus_results, method_a_results, method_b_results = analyze_glr_with_methods(
            df, buses, statistics, threshold, p_values, aggregation_methods, sink_threshold_methods
        )
        
        results_a.extend(method_a_results.to_dict('records'))
        results_b.extend(method_b_results.to_dict('records'))
        individual_results.extend(bus_results.to_dict('records'))
    
    return pd.DataFrame(results_a), pd.DataFrame(results_b), pd.DataFrame(individual_results)

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