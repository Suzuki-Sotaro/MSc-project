# method_b.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import calculate_far_ed
# Set the threshold to a higher number or use `np.inf` to show all elements
np.set_printoptions(threshold=np.inf)


def apply_method_b(bus_statistics, aggregation_methods, sink_threshold_methods, df, buses):
    results = []
    local_thresholds = [stats.max() for stats in bus_statistics.values()]
    
    combined_data_length = len(list(bus_statistics.values())[0])
    
    for agg_method in aggregation_methods:
        for sink_method in sink_threshold_methods:
            combined_detections = np.zeros_like(list(bus_statistics.values())[0]).astype(int).tolist()  # Initialize as list of integers
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
                'Detection': combined_detections,
                'Detection Time': detection_time,
                'Data': df[buses].values.tolist()[-combined_data_length:],  # Convert to list if needed
                'Label': df['Label'].values.astype(int).tolist()[-combined_data_length:]  # Convert to list of integers
            })
    # resultsのDataのshapeを表示する。
    print(np.array(results[0]['Detection']).shape)
    print(np.array(results[0]['Data']).shape)
    print(np.array(results[0]['Label']).shape)
    return results

def evaluate_method_b(results, labels):
    labels = np.array(labels).astype(int).tolist()  # Convert to list of integers if needed
    for result in results:
        detections = np.array(result['Detection']).astype(int).tolist()  # Convert to list of integers
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