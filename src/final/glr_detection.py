# below is the content of glr_detection.py
import numpy as np
import pandas as pd
from scipy import stats
from method_a import apply_method_a, evaluate_method_a
from method_b import apply_method_b, evaluate_method_b
from utils import calculate_far_ed, evaluate_results

def nonparametric_glr_detect(data, window_size, h):
    n = len(data)
    g = np.zeros(n)
    t_hat = 0

    # Pre-compute ranks for efficiency
    ranks = np.array([stats.rankdata(data[i:i+window_size]) for i in range(n-window_size+1)])

    for k in range(window_size * 2, n):
        max_statistic = -np.inf
        max_j = 0

        for j in range(k - window_size + 1, k + 1):
            if j >= len(ranks):  # Ensure j is within bounds
                continue
            
            before_rank = ranks[j-window_size]
            after_rank = ranks[j]

            # Calculate the GLR statistic using rank differences
            statistic = np.sum((before_rank - after_rank) ** 2)

            if statistic > max_statistic:
                max_statistic = statistic
                max_j = j

        g[k-1] = max_statistic

        # Check if the GLR statistic exceeds the threshold
        if g[k-1] > h:
            t_hat = max_j
            break

    return t_hat, g

def calculate_glr_scores(df, buses, window_size, threshold):
    bus_detections = {}
    bus_scores = {}
    bus_change_magnitudes = {}

    for bus in buses:
        data = df[bus].values
        detection_point, glr_scores = nonparametric_glr_detect(data, window_size, threshold)
        bus_detections[bus] = np.zeros(len(data), dtype=int)
        if detection_point > 0:
            bus_detections[bus][detection_point:] = 1
        bus_scores[bus] = glr_scores
        
        if detection_point > 0:
            change_magnitude = estimate_change_magnitude(data, detection_point, window_size)
            bus_change_magnitudes[bus] = change_magnitude
        else:
            bus_change_magnitudes[bus] = 0

    return bus_detections, bus_scores, bus_change_magnitudes

def estimate_change_magnitude(data, change_point, window_size):
    before_change = data[max(0, change_point-window_size):change_point]
    after_change = data[change_point:min(len(data), change_point+window_size)]
    
    change_magnitude = np.median(after_change) - np.median(before_change)
    return change_magnitude

def analyze_glr_with_methods(df, buses, window_size, glr_threshold_values, p_values, aggregation_methods, sink_threshold_methods):
    all_method_a_results = []
    all_method_b_results = []
    all_individual_bus_results = []

    for threshold in glr_threshold_values:
        bus_detections, bus_scores, bus_change_magnitudes = calculate_glr_scores(df, buses, window_size, threshold)

        method_a_results = apply_method_a(bus_detections, p_values, df, buses)
        method_b_results = apply_method_b(bus_scores, aggregation_methods, sink_threshold_methods, df, buses)

        labels = df['Label'].values.astype(int).tolist()  # Convert to list of integers

        method_a_results = evaluate_method_a(method_a_results, labels)
        method_b_results = evaluate_method_b(method_b_results, labels)
        
        for bus, detections in bus_detections.items():
            detections = detections.astype(int).tolist()  # Convert to list of integers
            cm, accuracy, precision, recall, f1 = evaluate_results(detections, labels)
            detection_time = np.argmax(detections) if np.any(detections) else -1
            far, ed = calculate_far_ed(labels, detections, detection_time)

            all_individual_bus_results.append({
                'Bus': bus,
                'Data': df[bus].values.tolist(),  # Convert to list if needed
                'Label' : labels,
                'Detection': detections,
                'GLR Threshold': threshold,
                'Window Size': window_size,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'False Alarm Rate': far,
                'Expected Delay': ed,
                'Detection Time': detection_time,
                'Change Magnitude': bus_change_magnitudes[bus]
            })

        for result in method_a_results + method_b_results:
            result['GLR Threshold'] = threshold
            result['Window Size'] = window_size

        all_method_a_results.extend(method_a_results)
        all_method_b_results.extend(method_b_results)

    print("Nonparametric GLR analysis completed.")
    all_individual_bus_results = pd.DataFrame(all_individual_bus_results)
    all_individual_bus_results['Label'] = all_individual_bus_results['Label'].apply(lambda x: list(map(int, x)))
    all_individual_bus_results['Detection'] = all_individual_bus_results['Detection'].apply(lambda x: list(map(int, x)))

    return all_individual_bus_results, pd.DataFrame(all_method_a_results), pd.DataFrame(all_method_b_results)

def analyze_glr(df, buses, window_size, threshold_values):
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]
    aggregation_methods = ['average', 'median', 'outlier_detection']
    sink_threshold_methods = ['average', 'minimum', 'maximum', 'median']

    return analyze_glr_with_methods(df, buses, window_size, threshold_values, p_values, aggregation_methods, sink_threshold_methods)
