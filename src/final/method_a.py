# below is the implementation of method A
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import calculate_far_ed

def apply_method_a(bus_detections, p_values):
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

def evaluate_method_a(results, labels):
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
