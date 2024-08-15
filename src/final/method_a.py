# below is the implementation of method A
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def learn_local_threshold(data, labels, window_size):
    best_threshold = 0
    best_f1 = 0
    for threshold in np.arange(0.1, 5.0, 0.1):
        changes = detect_local_change(data, window_size, threshold)
        f1 = f1_score(labels[window_size:], changes)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

def detect_local_change(data, window_size, threshold):
    changes = np.zeros(len(data) - window_size)
    for i in range(window_size, len(data)):
        if abs(np.mean(data[i-window_size:i]) - np.mean(data[i-window_size*2:i-window_size])) > threshold:
            changes[i-window_size] = 1
    return changes

def analyze_method_a(df, buses, window_size, p_values):
    results = []
    labels = df['Label'].values[window_size:]
    bus_data = {bus: df[bus].values for bus in buses}

    # Learn local thresholds using initial data
    local_thresholds = {bus: learn_local_threshold(bus_data[bus][:len(bus_data[bus])//2], 
                                                   df['Label'].values[:len(bus_data[bus])//2], 
                                                   window_size) for bus in buses}

    voting_schemes = ['at_least_one', 'all_buses'] + [f'p_{p}' for p in p_values]
    
    for scheme in voting_schemes:
        votes = np.zeros(len(labels))
        for bus in buses:
            changes = detect_local_change(bus_data[bus], window_size, local_thresholds[bus])
            votes += changes

        if scheme == 'at_least_one':
            detection_time = np.argmax(votes > 0) if np.any(votes > 0) else -1
        elif scheme == 'all_buses':
            detection_time = np.argmax(votes == len(buses)) if np.any(votes == len(buses)) else -1
        else:
            p = float(scheme.split('_')[1])
            detection_time = np.argmax(votes >= p * len(buses)) if np.any(votes >= p * len(buses)) else -1

        pred_labels = np.zeros(len(labels))
        if detection_time != -1:
            pred_labels[detection_time:] = 1

        accuracy = accuracy_score(labels, pred_labels)
        recall = recall_score(labels, pred_labels)
        precision = precision_score(labels, pred_labels)
        f1 = f1_score(labels, pred_labels)

        results.append({
            'Voting Scheme': scheme,
            'Detection Time': detection_time,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

    return pd.DataFrame(results)