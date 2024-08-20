# method_b.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods):
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

def evaluate_method_b(results, labels):
    for result in results:
        detections = result['Detections']
        accuracy = accuracy_score(labels, detections)
        precision = precision_score(labels, detections, zero_division=0)
        recall = recall_score(labels, detections)
        f1 = f1_score(labels, detections)
        far, ed = calculate_far_ed(labels, detections, result['Detection Time'])
        
        result.update({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'False Alarm Rate': far,
            'Expected Delay': ed
        })
    
    return results

def calculate_far_ed(true_labels, predicted_labels, detection_time):
    far = np.sum((predicted_labels == 1) & (true_labels == 0)) / np.sum(true_labels == 0)
    
    if detection_time != -1 and np.any(true_labels == 1):
        true_change_point = np.argmax(true_labels)
        ed = max(0, detection_time - true_change_point)
    else:
        ed = np.inf
    
    return far, ed