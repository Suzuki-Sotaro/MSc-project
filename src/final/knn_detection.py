# below is the code for the knn_detection.py file
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def transform_time_series(data, d):
    return np.array([data[i:i+d] for i in range(len(data) - d + 1)])

def calculate_knn_distances(S1, S2, k):
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(S1)
    distances, _ = nbrs.kneighbors(S2)
    return np.sum(distances, axis=1)

def estimate_tail_probability(dt, distances, N2):
    pt_hat = np.sum(distances > dt) / N2
    if pt_hat == 0:
        pt_hat = 1 / N2  # Prevent division by zero as suggested in the paper
    return pt_hat

def evaluate_results(pred_labels, true_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    return cm, accuracy, precision, recall, f1

def analyze_knn(df, buses, d, k_values, alpha_values, h_values):
    results = []
    
    for bus in buses:
        data = df[bus].values
        labels = df['Label'].values
        
        # Transform the time series into d-dimensional vectors
        transformed_data = transform_time_series(data, d)
        
        # Partition the data into S1 and S2
        N = len(transformed_data)
        N1 = N // 2
        N2 = N - N1
        S1, S2 = transformed_data[:N1], transformed_data[N1:]
        
        for k in k_values:
            for alpha in alpha_values:
                for h in h_values:
                    # Offline Phase
                    distances_S2 = calculate_knn_distances(S1, S2, k)
                    sorted_distances = np.sort(distances_S2)
                    
                    # Online Detection Phase
                    anomalies = []
                    gt = 0
                    for t, xt in enumerate(transformed_data[N1:], start=1):
                        dt = calculate_knn_distances(S1, [xt], k)[0]
                        pt_hat = estimate_tail_probability(dt, sorted_distances, N2)
                        st = np.log(alpha / pt_hat)
                        gt = max(0, gt + st)
                        if gt >= h:
                            anomalies.append(True)
                        else:
                            anomalies.append(False)
                    
                    # Adjust the length of the labels to match the transformed data
                    adjusted_labels = labels[d-1:]
                    pred_labels = np.zeros_like(adjusted_labels, dtype=bool)
                    pred_labels[N1:] = anomalies
                    
                    # Evaluate results
                    cm, accuracy, precision, recall, f1 = evaluate_results(pred_labels[N1:], adjusted_labels[N1:])
                    
                    results.append({
                        'Bus': bus,
                        'd': d,
                        'k': k,
                        'alpha': alpha,
                        'h': h,
                        'Confusion Matrix': cm,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1
                    })
    
    return pd.DataFrame(results)