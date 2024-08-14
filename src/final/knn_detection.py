# knn_detection.pyの内容。
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def transform_time_series(data, d):
    return np.array([data[i:i+d] for i in range(len(data) - d + 1)])

def calculate_knn_distances(S1, S2, k):
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(S1)
    distances, _ = nbrs.kneighbors(S2)
    return np.sum(distances, axis=1)

def detect_outliers(distances, alpha):
    sorted_distances = np.sort(distances)
    threshold = sorted_distances[int((1 - alpha) * len(distances))]
    return distances > threshold

def estimate_tail_probability(dt, distances):
    return np.mean(distances > dt)

def cusum_algorithm(statistics, h):
    gt = 0
    anomalies = []
    for st in statistics:
        gt = max(0, gt + st)
        if gt >= h:
            anomalies.append(True)
            gt = 0  # Reset after detecting an anomaly
        else:
            anomalies.append(False)
    return np.array(anomalies)

def evaluate_results(pred_labels, true_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    return cm, accuracy, precision, recall, f1

def analyze_knn(data, labels, d, k, alpha, h):
    # Transform the time series into d-dimensional vectors
    transformed_data = transform_time_series(data, d)
    
    # Partition the data into S1 and S2
    N = len(transformed_data)
    N1 = N // 2
    S1, S2 = transformed_data[:N1], transformed_data[N1:]
    
    # Offline Phase
    distances_S2 = calculate_knn_distances(S1, S2, k)
    sorted_distances = np.sort(distances_S2)
    
    # Online Detection Phase
    anomalies = []
    gt = 0
    for xt in transformed_data[N1:]:
        dt = calculate_knn_distances(S1, [xt], k)[0]
        pt_hat = estimate_tail_probability(dt, sorted_distances)
        st = np.log(alpha / max(pt_hat, 1e-10))  # Avoid division by zero
        gt = max(0, gt + st)
        if gt >= h:
            anomalies.append(True)
            gt = 0  # Reset after detecting an anomaly
        else:
            anomalies.append(False)
    
    # Adjust the length of the labels to match the transformed data
    adjusted_labels = labels[d-1:]
    pred_labels = np.zeros_like(adjusted_labels)
    pred_labels[N1:] = anomalies
    
    # Evaluate results
    cm, accuracy, precision, recall, f1 = evaluate_results(pred_labels[N1:], adjusted_labels[N1:])
    
    return {
        'd': d,
        'k': k,
        'alpha': alpha,
        'h': h,
        'Confusion Matrix': cm,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Example usage
def main():
    # Generate some example data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    anomaly_data = np.random.normal(3, 1, 100)
    data = np.concatenate([normal_data, anomaly_data])
    labels = np.concatenate([np.zeros(1000), np.ones(100)])
    
    # Run the analysis
    result = analyze_knn(data, labels, d=5, k=5, alpha=0.05, h=2)
    print(result)

if __name__ == "__main__":
    main()