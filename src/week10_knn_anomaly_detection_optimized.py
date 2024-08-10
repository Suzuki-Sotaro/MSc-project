import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Create directory for results
results_dir = os.path.join('results', 'knn_outlier_detection_optimized_v2')
os.makedirs(results_dir, exist_ok=True)

# Load data from the specified CSV file
file_path = './data/A_LMPFreq3_Labeled.csv'
data = pd.read_csv(file_path)

# Specify bus numbers to analyze
bus_numbers = [115, 116, 117, 118, 119, 121, 135, 139]

# Extract the last 855 values
data_last_855 = data.iloc[-855:]

# Extract data for each specified bus
bus_data = {bus: data_last_855[f'Bus{bus}'].values for bus in bus_numbers}

# Extract labels
labels = data_last_855['Label'].values

# Function to transform time series into d-dimensional vectors
def transform_time_series(data, d):
    transformed_data = []
    for i in range(len(data) - d + 1):
        transformed_data.append(data[i:i+d])
    return np.array(transformed_data)

# Example: Transform the time series into d-dimensional vectors (e.g., d = 3)
d = 3
transformed_data = {bus: transform_time_series(bus_data[bus], d) for bus in bus_numbers}

# Function to calculate k-NN distances
def calculate_knn_distances(transformed_data, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(transformed_data)
    distances, indices = nbrs.kneighbors(transformed_data)
    return distances.sum(axis=1)  # Sum of distances to k-NNs

# Function to detect outliers using tail probability
def detect_outliers(distances, alpha):
    sorted_distances = np.sort(distances)
    threshold = sorted_distances[int((1 - alpha) * len(distances))]
    return distances > threshold

# Function to implement the CUSUM-like algorithm
def cusum_algorithm(statistics, h):
    gt = 0
    anomalies = []
    for s in statistics:
        gt = max(0, gt + s)
        if gt >= h:
            anomalies.append(True)
            gt = 0  # Reset after detecting an anomaly
        else:
            anomalies.append(False)
    return np.array(anomalies)

# Function to evaluate results
def evaluate_results(pred_labels, true_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    return cm, accuracy, precision, recall, f1

# Experiment with different parameters
k_values = [10, 15, 20]  # Adjust k to be larger for more robust detection
alpha_values = [0.01, 0.05, 0.1]  # Use a range of alpha to explore different thresholds
h_values = [10, 20, 30]  # Adjust the CUSUM threshold h

best_results = []

for k in k_values:
    for alpha in alpha_values:
        for h in h_values:
            outliers = {bus: detect_outliers(calculate_knn_distances(transformed_data[bus], k), alpha) for bus in bus_numbers}
            anomaly_detected = {bus: cusum_algorithm(outliers[bus], h) for bus in bus_numbers}

            # Apply the evaluation for each bus
            results = []
            for bus in bus_numbers:
                pred_labels = anomaly_detected[bus].astype(int)

                cm, accuracy, precision, recall, f1 = evaluate_results(pred_labels, labels)
                results.append({
                    'Bus': bus,
                    'k': k,
                    'alpha': alpha,
                    'h': h,
                    'Confusion Matrix': cm,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                })

            best_results.append(results)

# Flatten and save the results
flattened_results = [item for sublist in best_results for item in sublist]
results_df = pd.DataFrame(flattened_results)
results_df.to_csv(os.path.join(results_dir, 'knn_outlier_detection_results_optimized_v2.csv'), index=False)
print(results_df)
