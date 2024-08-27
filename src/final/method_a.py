# below is the implementation of method A
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import calculate_far_ed
# Set the threshold to a higher number or use `np.inf` to show all elements
np.set_printoptions(threshold=np.inf)


def apply_method_a(bus_detections, p_values, df, buses):
    results = []
    n_buses = len(bus_detections)
    
    combined_data_length = len(list(bus_detections.values())[0])
    
    # Scheme 1: At least one bus
    combined_detections_one = np.any(list(bus_detections.values()), axis=0).astype(int).tolist()  # Convert to list of integers
    detection_time_one = np.argmax(combined_detections_one) if np.any(combined_detections_one) else -1
    
    results.append({
        'Method': 'Method A (At least one bus)',
        'Detection': combined_detections_one,
        'Detection Time': detection_time_one,
        'Data': df[buses].values.tolist()[-combined_data_length:],  # Convert to list if needed
        'Label': df['Label'].values.astype(int).tolist()[-combined_data_length:]  # Convert to list of integers
    })
    
    # Scheme 2: All buses
    combined_detections_all = np.all(list(bus_detections.values()), axis=0).astype(int).tolist()  # Convert to list of integers
    detection_time_all = np.argmax(combined_detections_all) if np.any(combined_detections_all) else -1
    
    results.append({
        'Method': 'Method A (All buses)',
        'Detection': combined_detections_all,
        'Detection Time': detection_time_all,
        'Data': df[buses].values.tolist()[-combined_data_length:],  # Convert to list if needed
        'Label': df['Label'].values.astype(int).tolist()[-combined_data_length:]  # Convert to list of integers
    })
    
    # Scheme 3: p% of buses
    for p in p_values:
        threshold = int(p * n_buses)
        combined_detections = np.zeros_like(list(bus_detections.values())[0]).astype(int).tolist()  # Initialize as list of integers
        detection_time = -1
        
        for t in range(len(combined_detections)):
            votes = sum(detections[t] for detections in bus_detections.values())
            if votes >= threshold:
                combined_detections[t] = 1
                if detection_time == -1:
                    detection_time = t
        
        results.append({
            'Method': f'Method A (p={p})',
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

def evaluate_method_a(results, labels):
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